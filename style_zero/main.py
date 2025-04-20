from glob import glob
import os
import random
import numpy as np
import torch

from player_encoder.encoder import TransformerEncoder
from alphazero import AlphaZeroTrainer, Game
from mcts import MCTS
from model import AlphaZeroNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def style_reward(game, color=True, min_value=0.2, max_value=0.9):
    """
    Compute cosine‑similarity style reward in the range [-1, 1].
    `color=True`  -> evaluate white;  `False` -> evaluate black.
    """
    T = min(game.shape[0], 200)  # max sequence length
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

    states = pad_or_truncate(states, 100, pad_value=0.0, dim=1)  # (1,100,224,8,8)
    mask   = pad_or_truncate(mask, 100, pad_value=True, dim=1)   # (1,100)

    style_emb = target_a_style_embedding if color else target_b_style_embedding
    with torch.no_grad():
        _, pred_emb = emb_net(states, mask)

    raw_sim = torch.cosine_similarity(pred_emb, style_emb.unsqueeze(0), dim=1).mean().item()
    # Normalize to [-1, 1]
    norm_sim = (raw_sim - min_value) / (max_value - min_value)
    return norm_sim

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

def calc_target_emb(player_name):
    target_a_path = f"../chess_data_parse/dataset/{player_name}"
    game_files = glob(os.path.join(target_a_path, "*.pt"))
    random.shuffle(game_files)
    selected_files = game_files[:5]

    embs = []

    for file_path in selected_files:
        paired_states = []
        paired_mask = []
        data = torch.load(file_path)
        states = data['states']
        T = min(states.shape[0], 200)
        mask = torch.zeros(T, dtype=torch.bool)
        color = True if "white" in file_path else False
        indices = range(0, T - 1, 2) if color else range(1, T - 1, 2)

        for k, i in enumerate(indices):
            pair = np.concatenate([states[i], states[i + 1]], axis=0)  # [224,8,8]
            pair = torch.tensor(pair, dtype=torch.float32, device=device)  # [224,8,8]
            paired_states.append(pair)
            paired_mask.append(mask[i])

        processed_states = torch.stack(paired_states).unsqueeze(0).float().to(device)  # [T', 224, 8, 8]
        processed_masks = torch.tensor(paired_mask, dtype=torch.bool).unsqueeze(0).to(device)  # [T']

        processed_states = pad_or_truncate(processed_states, 100, pad_value=0.0, dim=1)
        processed_masks = pad_or_truncate(processed_masks, 100, pad_value=True, dim=1)
        with torch.no_grad():
            _, pred_emb = emb_net(processed_states, processed_masks)
        embs.append(pred_emb)

    embs = torch.stack(embs).mean(dim=0)
    return embs

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
    emb_net.load_state_dict(checkpoint['model_state_dict'])

    target_a_style_embedding = calc_target_emb("xugal").to(device)
    target_b_style_embedding = calc_target_emb("clparagao123").to(device)

    print(f"target_a_style_embedding: {target_a_style_embedding.shape}")
    print(f"target_b_style_embedding: {target_b_style_embedding.shape}")

    target_sim = torch.cosine_similarity(target_a_style_embedding, target_b_style_embedding, dim=1).mean().item()
    print(f"target_sim: {target_sim:.4f}")

    net = AlphaZeroNet().to(device)
    net_b = AlphaZeroNet().to(device)

    trainer = AlphaZeroTrainer(Game, net, net_b, style_reward, MCTS, config, device)
    trainer.run()
