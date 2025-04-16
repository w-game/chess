import numpy as np
import torch

from alphazero import AlphaZeroTrainer, Game
from model import AlphaZeroNet
from mcts import MCTS
from player_encoder.encoder import TransformerEncoder

def style_reward(game, color=True, s_min=0.2, s_max=0.9):
    """
    计算风格相似度并进行Min-Max缩放映射到[0,1]。
    :param game: 棋局特征序列
    :param color: True表示对白方动作做风格分析，False表示对黑方
    :param s_min: 相似度分布可能出现的“最小值”
    :param s_max: 相似度分布可能出现的“最大值”
    :return: 缩放后的风格奖励 (0~1)
    """
    T = len(game)
    if color:
        indices = range(0, T - 1, 2)
    else:
        indices = range(1, T - 1, 2)

    states = []
    for _, i in enumerate(indices):
        s_t = game[i]
        s_tp1 = game[i + 1]
        s_pair = np.concatenate([s_t, s_tp1], axis=0)
        states.append(s_pair)

    states = np.stack(states, axis=0)  

    state_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device)
    style_emb =  target_a_style_embedding if color else target_b_style_embedding
    _, pred_emb = emb_net(state_tensor)

    # 计算原相似度
    raw_sim = torch.cosine_similarity(pred_emb, style_emb.unsqueeze(0), dim=1).mean()
    raw_sim_val = raw_sim.item()

    clipped_sim = max(min(raw_sim_val, s_max), s_min)  
    scaled_sim = (clipped_sim - s_min) / (s_max - s_min)
    print(f"Raw similarity: {raw_sim_val}, Clipped similarity: {clipped_sim}, Scaled similarity: {scaled_sim}")

    return scaled_sim

if __name__ == "__main__":
    config = {
        'lr': 0.001,
        'memory_size': 10000,
        'batch_size': 64,
        'num_iterations': 1000,
        'num_self_play_games': 2,
        'temperature': 1.0,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_net.to(device)

    target_a_style_embedding = torch.randn(256).to(device)
    target_b_style_embedding = torch.randn(256).to(device)

    net_a = AlphaZeroNet().to(device)
    net_b = AlphaZeroNet().to(device)

    trainer = AlphaZeroTrainer(Game, net_a, net_b, style_reward, MCTS, config, device)
    trainer.run()
