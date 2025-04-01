import random
import time

from torch.utils.data import Dataset
import torch

class PlayerDataset(Dataset):
    def __init__(self, player_files, games_per_player=4, max_len=100, transform=None, desired_length=30):
        """
        player_files: Dict[int, str] → 玩家ID映射到文件路径
        games_per_player: 每次采样一个玩家的几局棋
        """
        self.player_files = player_files              # {player_id: file_path}
        self.player_ids = list(player_files.keys())   # 所有玩家ID列表
        self.games_per_player = games_per_player
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.player_ids)

    def pad_or_truncate(self, tensor, target_len, pad_value=0, dim=0):
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

    def extract_segments_from_game(self, state, action, mask, num_segments=3, segment_len=10):
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

    def __getitem__(self, index):
        player_id = self.player_ids[index]

        file_path = self.player_files[player_id]
        
        games = torch.load(file_path, weights_only=True)
        
        white_game_list = games["white"]
        black_game_list = games["black"]

        # desired_length = 30
        # valid_white_games = [g for g in white_game_list if g['state'].size(0) >= desired_length]
        # valid_black_games = [g for g in black_game_list if g['state'].size(0) >= desired_length]

        selected_white_games = random.sample(white_game_list, self.games_per_player)
        selected_black_games = random.sample(black_game_list, self.games_per_player)

        selected_games = selected_white_games + selected_black_games

        processed = []
        for game in selected_games:
            state = game['state']   # [T, 112, 8, 8]
            action = game['action'] # [T]
            mask = game['mask']     # [T]

            # segments = self.extract_segments_from_game(state, action, mask, num_segments=3, segment_len=10)
            #
            # for seg_state, seg_action, seg_mask in segments:
            #
            #     if self.transform:
            #         seg_state, seg_action, seg_mask = self.transform(seg_state, seg_action, seg_mask)

                # processed.append((seg_state, seg_action, seg_mask, int(player_id)))
            processed.append((state, action, mask, int(player_id)))

        return processed  # List[(state, action, mask, player_id)]
        