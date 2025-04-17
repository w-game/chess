from glob import glob
import os
import random
import numpy as np
import torch

from player_encoder.encoder import TransformerEncoder
from style_zero.alphazero import AlphaZeroTrainer, Game
from style_zero.mcts import MCTS
from style_zero.model import AlphaZeroNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def style_reward(game, color=True):
    """
    Compute cosine‑similarity style reward in the range [-1, 1].
    `color=True`  -> evaluate white;  `False` -> evaluate black.
    """
    T = len(game)
    indices = range(0, T - 1, 2) if color else range(1, T - 1, 2)

    paired = []
    for k, i in enumerate(indices):
        if k >= 100:                # truncate to 100 move‑pairs
            break
        pair = np.concatenate([game[i], game[i + 1]], axis=0)  # [224,8,8]
        paired.append(pair)

    if not paired:
        return 0.0

    states = torch.tensor(np.stack(paired), dtype=torch.float32, device=device).unsqueeze(0)  # (1,T',224,8,8)
    mask   = torch.zeros(states.size(1), dtype=torch.bool, device=device).unsqueeze(0)        # (1,T')

    style_emb = target_a_style_embedding if color else target_b_style_embedding
    with torch.no_grad():
        _, pred_emb = emb_net(states, mask)

    return torch.cosine_similarity(pred_emb, style_emb.unsqueeze(0), dim=1).mean().item()

def calc_target_emb(player_name):
    target_a_path = f"../chess_data_parse/dataset/{player_name}"
    game_files = glob(os.path.join(target_a_path, "*.pt"))
    random.shuffle(game_files)
    selected_files = game_files[:5]

    paired_states = []
    paired_mask = []

    for file_path in selected_files:
        data = torch.load(file_path)
        states = data['states']
        T = states.size(0)
        mask = torch.zeros(T, dtype=torch.bool)
        color = "white" if "white" in file_path else "black"
        if color == 'white':
            indices = range(0, T - 1, 2)  # 白方下在偶数步
        else:
            indices = range(1, T - 1, 2)  # 黑方下在奇数步

        for idx, i in enumerate(indices):
            s_t = states[i]
            s_tp1 = states[i + 1]
            s_pair = torch.cat([s_t, s_tp1], dim=0)  # 压缩为 float16
            paired_states.append(s_pair)
            paired_mask.append(mask[i])

        paired_states = torch.stack(paired_states).unsqueeze(0).float().to(device)  # [T', 224, 8, 8]
        paired_mask = torch.tensor(paired_mask, dtype=torch.bool).unsqueeze(0).to(device)  # [T']
        with torch.no_grad():
            _, pred_emb = emb_net(paired_states, paired_mask)
            return pred_emb

if __name__ == "__main__":
    config = {
        'lr': 0.001,
        'memory_size': 10000,
        'batch_size': 64,
        'num_iterations': 1000,
        'num_self_play_games': 25,
        'temperature': 1.0,
        'num_simulations': 20,
        'save_interval': 5
        # その他のハイパーパラメータ
    }

    emb_net = TransformerEncoder(
        cnn_in_channels=224,
        state_embed_dim=256,
        transformer_d_model=256,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
        max_seq_len=100
    )


    checkpoint = torch.load("../models/trained_model/player_encoder_60.pt")

    emb_net.to(device)

    target_a_style_embedding = calc_target_emb("xugal").to(device)
    target_b_style_embedding = calc_target_emb("DrMarlonsky").to(device)

    net = AlphaZeroNet().to(device)
    # net_b = AlphaZeroNet().to(device)

    trainer = AlphaZeroTrainer(Game, net, style_reward, MCTS, config, device)
    trainer.run()
