import torch

from alphazero import AlphaZeroTrainer, Game
from model import AlphaZeroNet
from mcts import MCTS
from player_encoder.encoder import TransformerEncoder

if __name__ == "__main__":
    config = {
        'lr': 0.001,
        'memory_size': 10000,
        'batch_size': 64,
        'num_iterations': 1000,
        'num_self_play_games': 25,
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

    alpha_zero_net = AlphaZeroNet()

    checkpoint = torch.load("../models/trained_model/player_encoder_60.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_net.to(device)

    trainer = AlphaZeroTrainer(Game, alpha_zero_net, emb_net, MCTS, config, device)
    trainer.run()
