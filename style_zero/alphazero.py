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
    def __init__(self, game_cls, net, reward_fc, mcts_cls, config, device=None):
        self.game_cls = game_cls
        self.net = net
        self.reward_fc = reward_fc
        self.mcts_cls = mcts_cls
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.net.parameters(), lr=config['lr'])
        self.memory = deque(maxlen=config['memory_size'])

    def self_play(self):
        game = self.game_cls()
        mcts = self.mcts_cls(self.net, self.reward_fc, self.device, num_simulations=50, c_puct=3.0)
        states, pis = [], []

        step = 0

        while not game.is_game_over() and step < 200:
            step += 1

            legal_move = game.generate_legal_moves()

            state = game.get_current_state()

            temp = self.config['temperature'] if step < 30 else 0
            pi = mcts.get_action_probabilities(state, step, temp=temp)

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
        for state, pi in zip(states, pis):
            self.memory.append((state, pi, sim_a, sim_b))
        
        return step, sim_a, sim_b


    def train(self):
        if len(self.memory) < self.config['batch_size']:
            return

        batch = random.sample(self.memory, self.config['batch_size'])
        states, pis, sim_ws, sim_bs = zip(*batch)

        states_np = np.stack(states).astype(np.float32)
        pis_np    = np.stack(pis).astype(np.float32)
        sim_ws_np  = np.asarray(sim_ws, dtype=np.float32).reshape(-1,1)
        sim_bs_np  = np.asarray(sim_bs, dtype=np.float32).reshape(-1,1)
        target_vw = torch.from_numpy(sim_ws_np).to(self.device)
        target_vb = torch.from_numpy(sim_bs_np).to(self.device)

        states_t  = torch.from_numpy(states_np).to(self.device)
        target_pi = torch.from_numpy(pis_np).to(self.device)

        # ---------- monitoring ----------
        # We'll record per‑batch statistics right after forward pass.
        logits, pred_vw, pred_vb = self.net(states_t)
        # policy loss
        loss_pi = F.kl_div(F.log_softmax(logits,1), target_pi, reduction='batchmean')

        # value losses
        loss_vw = F.mse_loss(pred_vw, target_vw)
        loss_vb = F.mse_loss(pred_vb, target_vb)
        loss_z  = loss_vw + loss_vb

        # total
        loss = loss_pi + loss_z

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ---------- print metrics ----------
        mean_vw = pred_vw.mean().item()
        mean_vb = pred_vb.mean().item()
        print(f"[TRAIN] batch={self.config['batch_size']} "
              f"Loss_pi={loss_pi.item():.4f} "
              f"Loss_vw={loss_vw.item():.4f} "
              f"Loss_vb={loss_vb.item():.4f} "
              f"Total={loss.item():.4f} "
              f"mean_pred_vw={mean_vw:.3f} "
              f"mean_pred_vb={mean_vb:.3f}")

    def run(self):
        for iteration in range(self.config['num_iterations']):
            print(f"Iteration {iteration + 1}/{self.config['num_iterations']}")
            for idx in range(self.config['num_self_play_games']):
                step, sim_a, sim_b = self.self_play()
                print(f"Self-play game {idx + 1}/{self.config['num_self_play_games']} completed with {step} steps. sim_a: {sim_a}, sim_b: {sim_b}")

            self.train()

            if (iteration + 1) % self.config['save_interval'] == 0:
                torch.save(self.net.state_dict(), f"./models/model_{iteration+1}.pth")
                print(f"Models saved at iteration {iteration + 1}.")
