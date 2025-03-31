import glob
import os
import random
import gzip
import struct
import numpy as np

import torch

V4_VERSION = struct.pack('i', 4)
V3_VERSION = struct.pack('i', 3)

V4_STRUCT_STRING = '4s7432s832sBBBBBBBbffff'
V3_STRUCT_STRING = '4s7432s832sBBBBBBBb'

v4_struct = struct.Struct(V4_STRUCT_STRING)
v3_struct = struct.Struct(V3_STRUCT_STRING)

flat_planes = []
for i in range(2):
    flat_planes.append(np.zeros(64, dtype=np.float32) + i)

def parse_record(record):
    (ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q, root_d, best_d) = v4_struct.unpack(record)
    move_count = 0

    planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)
    rule50_plane = (np.zeros(8*8, dtype=np.float32) + rule50_count) / 99

    planes = planes.tobytes() + \
             flat_planes[us_ooo].tobytes() + \
             flat_planes[us_oo].tobytes() + \
             flat_planes[them_ooo].tobytes() + \
             flat_planes[them_oo].tobytes() + \
             flat_planes[stm].tobytes() + \
             rule50_plane.tobytes() + \
             flat_planes[move_count].tobytes() + \
             flat_planes[1].tobytes()

    assert len(planes) == ((8*13*1 + 8*1*1) * 8 * 8 * 4)
    winner = float(winner)
    assert winner == 1.0 or winner == -1.0 or winner == 0.0
    winner = struct.pack('fff', winner == 1.0, winner == 0.0, winner == -1.0)

    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)
    assert -1.0 <= best_q <= 1.0 and 0.0 <= best_d <= 1.0
    best_q = struct.pack('fff', best_q_w, best_d, best_q_l)

    return (planes, probs, winner, best_q)

def sample_record(chunkdata, is_white):
    """
    Randomly sample through the v4 chunk data and select records
    """
    version = chunkdata[0:4]
    if version == V4_VERSION:
        record_size = v4_struct.size
    elif version == V3_VERSION:
        record_size = v3_struct.size
    else:
        return []

    records = []
    for i in range(0, len(chunkdata), record_size):
        # if 32 > 1:
        #     # Downsample, using only 1/Nth of the items.
        #     if random.randint(0, 32-1) != 0:
        #         continue  # Skip this record.
        record = chunkdata[i:i+record_size]
        if version == V3_VERSION:
            # add 16 bytes of fake root_q, best_q, root_d, best_d to match V4 format
            record += 16 * b'\x00'

        (ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q, root_d, best_d) = v4_struct.unpack(record)
        
        if is_white and not stm:
            records.append(record)
        elif not is_white and stm:
            records.append(record)
        else:
            continue

    return records

def get_latest_chunks(path):
    whites = []
    blacks = []

    for d in glob.glob(path):
        for root, dirs, files in os.walk(d):
            for fpath in files:
                if fpath.endswith('.gz'):
                    if 'black' in root:
                        blacks.append(os.path.join(root, fpath))
                    elif 'white' in root:
                        whites.append(os.path.join(root, fpath))
                    else:
                        raise RuntimeError(
                            f"invalid chunk path found:{os.path.join(root, fpath)}")

    if len(whites) < 1 or len(blacks) < 1:
        return None, None

    whites.sort()
    blacks.sort()
    # random.shuffle(blacks)
    # random.shuffle(whites)

    return whites, blacks

def byte_to_np(record):
    planes, probs, winner, best_q = parse_record(record)

    planes = np.frombuffer(planes, dtype=np.float32).reshape(112, 8, 8)  # 将字节流转换回 np.array
    probs = np.frombuffer(probs, dtype=np.float32)    # 将字节流转换回 np.array
    winner = np.frombuffer(winner, dtype=np.float32)  # 三元 winner
    best_q = np.frombuffer(best_q, dtype=np.float32)  # 三元 best_q
    
    return planes, probs, winner, best_q

def chunk_to_records(chunk_file_lst):
    record = []
    for chunk_name in chunk_file_lst.copy():
        with gzip.open(chunk_name, 'rb') as chunk_file:
            chunk = chunk_file.read()
            lst = sample_record(chunk, True)
            record.extend(lst)
            
    return record

def chunk_to_trainingdata(player_name):
    whites, blacks = get_latest_chunks(f"./players/{player_name}")
    white_records = chunk_to_records(whites)
    black_records = chunk_to_records(blacks)

    first_plane = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    dataset = { }
    for color, records in zip(["white", "black"], [white_records, black_records]):
        matching_indices = []
        for i, record in enumerate(records):
            planes, probs, winner, best_q = byte_to_np(record)
            if np.allclose(planes[0], first_plane):
                matching_indices.append(i)

        print(f"[{player_name}][{color}] matched games:", len(matching_indices))

        train_data = []
        all_indices = matching_indices + [len(records)]

        for i in range(len(matching_indices)):
            start_idx = matching_indices[i]
            end_idx = all_indices[i + 1]
            game_records = records[start_idx:end_idx]

            states, actions, masks = [], [], []

            for record in game_records:
                planes, probs, winner, best_q = byte_to_np(record)
                # action_idx = np.argmax(probs)

                states.append(torch.tensor(planes, dtype=torch.float16))
                actions.append(torch.tensor(probs, dtype=torch.float16))
                # masks.append(torch.tensor(True, dtype=torch.bool))

            train_data.append({
                "state": torch.stack(states),
                "action": torch.stack(actions)
            })
        
        dataset[color] = train_data

    return dataset
    # save_path = f"./dataset_4/{player_name}.pt"
    # torch.save(dataset, save_path)
    # print(f"Saved games to {save_path}")

from multiprocessing import Pool, cpu_count

def process_player_wrapper(player_name):
    try:
        print(f"[•] Processing {player_name}")
        dataset = chunk_to_trainingdata(player_name)
        return player_name, dataset
    except Exception as e:
        print(f"[✗] Failed: {player_name} — {e}")

def batch_process_all_players(player_root="./players", num_workers=None, batch_size=32):
    # 取所有玩家名
    player_names = [
        name for name in sorted(os.listdir(player_root))
        if os.path.isdir(os.path.join(player_root, name))
    ]

    # 默认的并行进程数
    if num_workers is None:
        num_workers = min(8, int(cpu_count() / 2))

    # 分批处理，一次处理 batch_size 个玩家，并保存到一个文件
    for batch_idx in range(0, len(player_names), batch_size):
        # 只取当前批次的玩家
        current_batch_names = player_names[batch_idx : batch_idx + batch_size]

        with Pool(num_workers) as pool:
            # 并行处理这 32 个玩家
            results = pool.map(process_player_wrapper, current_batch_names)

        # 过滤可能出现错误导致的 None
        results = [res for res in results if res is not None]

        # 如果这一批全失败，直接跳过
        if not results:
            print(f"[!] Batch {batch_idx//batch_size} is empty, skipping.")
            continue

        # 组装这一批的数据
        batch_dict = {}
        for player_name, dataset in results:
            batch_dict[player_name] = dataset

        # 保存这一批为一个文件
        save_path = f"./dataset_4/batch_{batch_idx // batch_size}.pt.gz"
        with gzip.open(save_path, "wb") as f:
            torch.save(batch_dict, f)

        print(f"[✓] Saved batch {batch_idx // batch_size} "
              f"({len(results)} players) to {save_path}")
        
if __name__ == '__main__':
    batch_process_all_players()
