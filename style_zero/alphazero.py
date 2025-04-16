import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import chess
from leela_board import LeelaBoard

from torch.nn import functional as F


class Game:
    def __init__(self):
        self.board = LeelaBoard()
        self.current_player = 0

    def get_current_state(self):
        return self.board.copy()

    def play_action(self, action):
        move = chess.Move.from_uci(action)
        if move in self.board.generate_legal_moves():
            self.board.push(move)
            self.current_player = 1 - self.current_player
        else:
            raise ValueError("Illegal move")

    def is_game_over(self):
        return self.board.is_game_over()

    def get_winner(self):
        if self.board.is_checkmate():
            return 1 if self.current_player == 0 else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        else:
            return None

    def generate_legal_moves(self):
        return list(self.board.generate_legal_moves())


class AlphaZeroTrainer:
    def __init__(self, game_cls, net_a, net_b, reward_fc, mcts_cls, config, device=None):
        self.game_cls = game_cls
        self.net_a = net_a
        self.net_b = net_b
        self.reward_fc = reward_fc
        self.mcts_cls = mcts_cls
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_a = optim.Adam(self.net_a.parameters(), lr=config['lr'])
        self.optimizer_b = optim.Adam(self.net_b.parameters(), lr=config['lr'])
        self.memory = deque(maxlen=config['memory_size'])

    def self_play(self):
        game = self.game_cls()
        mcts = self.mcts_cls(self.net_a, self.net_b, self.reward_fc, self.device, num_simulations=20, c_puct=1.0)
        states, pis = [], []

        step = 0

        while not game.is_game_over() and step < 100:
            step += 1

            legal_move = game.generate_legal_moves()

            state = game.get_current_state()
            pi = mcts.get_action_probabilities(state, temp=self.config['temperature'])

            legal_indices = [game.board.move_to_index(move, game.board.turn, game.board.is_castling(move)) for move in legal_move]
            legal_p = pi[legal_indices]
            legal_p = legal_p / np.sum(legal_p)
            action_idx = np.random.choice(legal_indices, p=legal_p)
            action = game.board.idx_to_move(action_idx, game.board.turn)

            game.play_action(action)

            states.append(state.lcz_features())
            pis.append(pi)

        sim_a = self.reward_fc(game.board.get_feature_sequence(), True)
        sim_b = self.reward_fc(game.board.get_feature_sequence(), False)
        for idx, (state, pi) in enumerate(zip(states, pis)):
            sim = sim_a if idx % 2 == 0 else sim_b
            self.memory.append((state, pi, sim, idx % 2 == 0))  # winner should be replaced with sim

    def train(self):
        if len(self.memory) < self.config['batch_size']:
            return

        batch = random.sample(self.memory, self.config['batch_size'])
        states, pis, zs, is_white = zip(*batch)

        states_white = [torch.tensor(state) for state, is_white in zip(states, is_white) if is_white]
        pis_white = [torch.tensor(pi) for pi, is_white in zip(pis, is_white) if is_white]
        zs_white = [torch.tensor(z) for z, is_white in zip(zs, is_white) if is_white]

        states_white = torch.stack(states_white, dim=0).to(dtype=torch.float32, device=self.device)
        target_pis = torch.stack(pis_white, dim=0).to(dtype=torch.float32, device=self.device)
        target_zs = torch.stack(zs_white, dim=0).to(dtype=torch.float32, device=self.device).view(-1, 1)

        logits, pred_zs = self.net_a(states_white)
        log_pis = F.log_softmax(logits, dim=1)
        loss_pi = F.kl_div(log_pis, target_pis, reduction='batchmean')
        loss_z = F.mse_loss(pred_zs, target_zs)
        loss = loss_pi + loss_z

        self.optimizer_a.zero_grad()
        loss.backward()
        self.optimizer_a.step()
        print(f"Loss A: {loss.item()}")

        # 训练B网络
        states_black = [torch.tensor(state) for state, is_white in zip(states, is_white) if not is_white]
        pis_black = [torch.tensor(pi) for pi, is_white in zip(pis, is_white) if not is_white]
        zs_black = [torch.tensor(z) for z, is_white in zip(zs, is_white) if not is_white]

        states_black = torch.stack(states_black, dim=0).to(dtype=torch.float32, device=self.device)
        target_pis = torch.stack(pis_black, dim=0).to(dtype=torch.float32, device=self.device)
        target_zs = torch.stack(zs_black, dim=0).to(dtype=torch.float32, device=self.device).view(-1, 1)

        print(states_black.shape, target_pis.shape, target_zs.shape)
        logits, pred_zs = self.net_b(states_black)
        log_pis = F.log_softmax(logits, dim=1)
        loss_pi = F.kl_div(log_pis, target_pis, reduction='batchmean')
        loss_z = F.mse_loss(pred_zs, target_zs)
        loss = loss_pi + loss_z

        self.optimizer_b.zero_grad()
        loss.backward()
        self.optimizer_b.step()
        print(f"Loss B: {loss.item()}")


    def run(self):
        for iteration in range(self.config['num_iterations']):
            print(f"Iteration {iteration + 1}/{self.config['num_iterations']}")
            for idx in range(self.config['num_self_play_games']):
                self.self_play()
                print(f"Self-play game {idx + 1}/{self.config['num_self_play_games']} completed.")
            self.train()
            # 必要に应对模型的保存或评估进行处理

