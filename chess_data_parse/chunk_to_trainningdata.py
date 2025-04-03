import glob
import os
import random
import gzip
import struct
import numpy as np

import torch

from init_states import first_planes
from multiprocessing import Pool, cpu_count

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
    (
        ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q,
        root_d,
        best_d) = v4_struct.unpack(record)
    move_count = 0

    planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)
    rule50_plane = (np.zeros(8 * 8, dtype=np.float32) + rule50_count) / 99

    planes = planes.tobytes() + \
             flat_planes[us_ooo].tobytes() + \
             flat_planes[us_oo].tobytes() + \
             flat_planes[them_ooo].tobytes() + \
             flat_planes[them_oo].tobytes() + \
             flat_planes[stm].tobytes() + \
             rule50_plane.tobytes() + \
             flat_planes[move_count].tobytes() + \
             flat_planes[1].tobytes()

    assert len(planes) == ((8 * 13 * 1 + 8 * 1 * 1) * 8 * 8 * 4)
    winner = float(winner)
    assert winner == 1.0 or winner == -1.0 or winner == 0.0
    winner = struct.pack('fff', winner == 1.0, winner == 0.0, winner == -1.0)

    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)
    assert -1.0 <= best_q <= 1.0 and 0.0 <= best_d <= 1.0
    best_q = struct.pack('fff', best_q_w, best_d, best_q_l)

    return (planes, probs, winner, best_q, stm)


def sample_record(chunkdata):
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
        record = chunkdata[i:i + record_size]
        if version == V3_VERSION:
            # add 16 bytes of fake root_q, best_q, root_d, best_d to match V4 format
            record += 16 * b'\x00'

        # (ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q,
        #  root_d, best_d) = v4_struct.unpack(record)

        # if is_white and not stm:
        #     records.append(record)
        # elif not is_white and stm:
        #     records.append(record)
        # else:
        #     continue

        records.append(record)

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

    return whites, blacks


def byte_to_np(record):
    planes, probs, winner, best_q, stm = parse_record(record)

    planes = np.frombuffer(planes, dtype=np.float32).reshape(112, 8, 8)  # 将字节流转换回 np.array
    probs = np.frombuffer(probs, dtype=np.float32)  # 将字节流转换回 np.array
    winner = np.frombuffer(winner, dtype=np.float32)  # 三元 winner
    best_q = np.frombuffer(best_q, dtype=np.float32)  # 三元 best_q

    return planes, probs, winner, best_q, stm


def chunk_to_records(chunk_file_lst):
    record = []
    for chunk_name in chunk_file_lst.copy():
        with gzip.open(chunk_name, 'rb') as chunk_file:
            chunk = chunk_file.read()
            lst = sample_record(chunk)
            record.extend(lst)

    return record


def chunk_to_trainingdata(player_name):
    whites, blacks = get_latest_chunks(f"./players/{player_name}")
    if whites is None or blacks is None:
        return

    white_records = chunk_to_records(whites)
    black_records = chunk_to_records(blacks)

    dataset = {}
    for color, records in zip(["white", "black"], [white_records, black_records]):
        matching_indices = []

        for i, record in enumerate(records):
            planes, probs, winner, best_q, stm = byte_to_np(record)

            for j in range(12):
                if not np.allclose(planes[j], first_planes[j]):
                    j = -1
                    break
            if j != -1:
                matching_indices.append(i)

        print(f"[{player_name}][{color}] matched games:", len(matching_indices))

        train_data = []
        all_indices = matching_indices + [len(records)]

        for i in range(len(matching_indices)):
            start_idx = matching_indices[i]
            end_idx = all_indices[i + 1]
            game_records = records[start_idx:end_idx]

            states, actions = [], []

            for record in game_records:
                planes, probs, winner, best_q, stm = byte_to_np(record)

                states.append(torch.tensor(planes, dtype=torch.float16))
                actions.append(torch.tensor(probs, dtype=torch.float16))

            if len(states) >= 60:
                train_data.append({
                    "states": torch.stack(states),
                    "actions": torch.stack(actions)
                })

        print(f"[{player_name}][{color}] train data(state count > 60):", len(train_data))

        dataset[color] = train_data

    torch.save(dataset, "./dataset/" + player_name + ".pt")


def process_all_players(player_root="./players"):
    player_names = [
        name for name in sorted(os.listdir(player_root))
        if os.path.isdir(os.path.join(player_root, name))
    ]

    num_workers = min(8, int(cpu_count() / 2))

    with Pool(num_workers) as pool:
        pool.map(chunk_to_trainingdata, player_names)



if __name__ == '__main__':
    # chunk_to_trainingdata("Demo")
    process_all_players()
