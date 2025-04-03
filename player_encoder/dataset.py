import json
import random
import subprocess
import os
from glob import glob
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset


class MetaStyleDataset(Dataset):
    def __init__(self, player_files, N=5, K=5, Q=5, max_len=100):
        self.player_dataset = PlayerDataset(player_files)
        self.N = N
        self.K = K
        self.Q = Q
        self.max_len = max_len

    def __len__(self):
        return 1000

    def set_creator(self, sub_games, label_id):
        states, masks, labels = [], [], []
        for (s, m, _) in sub_games:
            s = pad_or_truncate(s, self.max_len, dim=0)  # s: [T, 224, 8, 8]
            m = pad_or_truncate(m, self.max_len, pad_value=True, dim=0)  # m: [T]
            states.append(s)
            masks.append(m)
            labels.append(label_id)

        return states, masks, labels

    def __getitem__(self, index):
        sampled_ids = random.sample(self.player_dataset.player_ids, self.N)

        support_pos, support_mask, support_labels = [], [], []
        query_pos, query_mask, query_labels = [], [], []

        for label_id, pid in enumerate(sampled_ids):
            games = self.player_dataset.get_player_data_by_id(pid, self.K + self.Q)
            random.shuffle(games)

            assert len(games) >= self.K + self.Q, f"玩家 {pid} 样本不够"

            support_games = games[:self.K]
            query_games = games[self.K:self.K + self.Q]

            states, masks, labels = self.set_creator(support_games, label_id)
            support_pos.extend(states)
            support_mask.extend(masks)
            support_labels.extend(labels)

            states, masks, labels = self.set_creator(query_games, label_id)
            query_pos.extend(states)
            query_mask.extend(masks)
            query_labels.extend(labels)

        return {
            'support_pos': torch.stack(support_pos),  # [N*K, T, 112, 8, 8]
            'support_mask': torch.stack(support_mask),
            'support_labels': torch.tensor(support_labels),

            'query_pos': torch.stack(query_pos),  # [N*Q, T, 112, 8, 8]
            'query_mask': torch.stack(query_mask),
            'query_labels': torch.tensor(query_labels),
        }


def pad_or_truncate(tensor, target_len, pad_value=0, dim=0):
    """
    自动补全/截断 tensor 到 target_len
    """
    T = tensor.size(dim)
    if T == target_len:
        return tensor
    elif T > target_len:
        return tensor.narrow(dim, 0, target_len)
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = target_len - T
        pad_tensor = torch.full(pad_size, pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad_tensor], dim=dim)


class PlayerDataset(Dataset):
    def __init__(self, player_files, transform=None):
        """
        player_files: Dict[int, str] → 玩家ID映射到文件路径
        games_per_player: 每次采样一个玩家的几局棋
        """
        self.player_files = player_files  # {player_id: file_path}
        self.player_ids = list(player_files.keys())  # 所有玩家ID列表
        self.transform = transform

    def __len__(self):
        return len(self.player_ids)

    def extract_segments_from_game(self, state, action, mask, num_segments=3, segment_len=30):
        """
        从一盘棋中抽取 num_segments 个片段，每段 segment_len 步。
        state: [T, C, 8, 8]
        action: [T]
        mask: [T]
        """
        T = state.size(0)
        segments = []

        if T < segment_len:
            return []  # 太短的局面，跳过

        # 计算采样起点范围（避免越界）
        margin = T - segment_len
        if margin <= 0:
            return []

        # 如果总长度太短，最多采样一段
        if T < segment_len * 2:
            start_idxs = [random.randint(0, margin)]
        else:
            # 取 num_segments 个均匀分布的采样点（避免全落在前段）
            ratios = torch.linspace(0.2, 0.8, num_segments)  # 相对时间点
            start_idxs = [min(int(r * T), margin) for r in ratios]

        for start in start_idxs:
            seg_state = state[start: start + segment_len]
            seg_action = action[start: start + segment_len]
            seg_mask = mask[start: start + segment_len]
            segments.append((seg_state, seg_action, seg_mask))

        return segments

    def get_player_data_by_id(self, player_id, game_num):
        folder_path = self.player_files[player_id]
        game_files = glob(os.path.join(folder_path, "*.pt"))
        random.shuffle(game_files)

        selected_files = game_files[:game_num]

        # 使用线程池并行处理文件
        processed = []
        with ThreadPoolExecutor(max_workers=min(cpu_count(), len(selected_files))) as executor:
            future_to_file = {executor.submit(process_file, file): file for file in selected_files}
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    if result is not None:
                        states, mask = result
                        processed.append((states, mask, int(player_id)))
                except Exception as e:
                    print(f"Error processing file {future_to_file[future]}: {e}")

        return processed


def process_file(file):
    """
    单独处理一个文件的逻辑，作为多进程的目标函数。
    """
    game = torch.load(file, weights_only=True)
    states = game['states']  # [T, 112, 8, 8]
    T = states.size(0)
    mask = torch.zeros(T, dtype=torch.bool)

    paired_states = []
    paired_mask = []

    # 根据文件名判断颜色
    color = "white" if "white" in file else "black"
    if color == 'white':
        indices = range(0, T - 1, 2)  # 白方下在偶数步
    else:
        indices = range(1, T - 1, 2)  # 黑方下在奇数步

    for idx, i in enumerate(indices):
        s_t = states[i]
        s_tp1 = states[i + 1]
        s_pair = torch.cat([s_t, s_tp1], dim=0).half()  # 压缩为 float16
        paired_states.append(s_pair)
        paired_mask.append(mask[i])

    if len(paired_states) < 1:
        return None  # 跳过无效文件

    paired_states = torch.stack(paired_states)  # [T', 224, 8, 8]
    paired_mask = torch.tensor(paired_mask, dtype=torch.bool)  # [T']

    # 返回处理后的结果
    return (paired_states, paired_mask)


def lc0_eval_fens(fens, lc0_path="lc0", weights_path="your_network.pb.gz"):
    evals = []
    proc = subprocess.Popen(
        [lc0_path, f"--weights={weights_path}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    for fen in fens:
        proc.stdin.write(f"position fen {fen}\ngo depth 1\n")
        proc.stdin.flush()
        score = None
        for line in proc.stdout:
            if "score cp" in line:
                try:
                    score = int(line.split("score cp")[1].split()[0])
                except:
                    score = 0
                break
        evals.append(score if score is not None else 0)
    proc.stdin.close()
    proc.wait()
    return evals


if __name__ == '__main__':
    with open(f"chess_data_parse/train_players.json", "r", encoding="utf-8") as f:
        files = json.load(f)

    for file in files:
        games = torch.load(file, weights_only=True)

        white_game_list = games["white"]
        black_game_list = games["black"]

        game = white_game_list[0]
