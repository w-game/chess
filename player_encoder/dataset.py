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

            state = self.pad_or_truncate(state, self.max_len, pad_value=0)              # [max_len, 112, 8, 8]
            action = self.pad_or_truncate(action, self.max_len, pad_value=0)            # [max_len]
            mask = self.pad_or_truncate(mask, self.max_len, pad_value=False)

            if self.transform:
                state, action, mask = self.transform(state, action, mask)
            processed.append((state, action, mask, int(player_id)))

        return processed  # List[(state, action, mask, player_id)]
        