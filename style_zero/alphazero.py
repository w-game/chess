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
        # 原有的检查
        if self.board.is_game_over():
            return True
        # 三次重复局面
        if self.board.pc_board.can_claim_threefold_repetition():
            return True
        # 50 步走子不吃子不走兵
        if self.board.pc_board.halfmove_clock >= 100:
            return True
        return False

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
    def __init__(self, game_cls, net, net_b, reward_fc, mcts_cls, config, device=None):
        self.game_cls = game_cls
        self.net = net
        self.net_b = net_b
        self.reward_fc = reward_fc
        self.mcts_cls = mcts_cls
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.net.parameters(), lr=config['lr'])
        self.optimizer_b = optim.Adam(self.net_b.parameters(), lr=config['lr'])
        self.memory = deque(maxlen=config['memory_size'])

    def self_play(self):
        game = self.game_cls()
        root = None               # root of the search tree (kept across moves)
        mcts = self.mcts_cls(self.net, self.net_b, self.reward_fc, self.device,
                             num_simulations=400, c_init=1.25, c_base=19652, c_factor=2.0)
        states, pis, turns = [], [], []

        step = 0

        while not game.is_game_over():
            step += 1

            legal_move = game.generate_legal_moves()

            state = game.get_current_state()

            temp = self.config['temperature'] if step < 30 else 0
            pi, root, visits = mcts.get_action_probabilities(state, step, temp=temp, root=root)

            legal_indices = [game.board.move_to_index(move, game.board.turn, game.board.is_castling(move)) for move in legal_move]
            legal_p = pi[legal_indices]
            legal_p = legal_p / np.sum(legal_p)
            action_idx = np.random.choice(legal_indices, p=legal_p)
            action = game.board.idx_to_move(action_idx, game.board.turn)

            # Reuse the subtree corresponding to chosen action
            # if root is not None and action_idx in root.children:
            #     root = root.children[action_idx]
            #     root.parent = None
            # else:
            #     root = None

            
            # Show top‑3 visit counts for quick sanity check
            if len(visits) > 0:
                top3 = sorted(visits.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"[MCTS - Step {step}] top‑3 root visits: {top3}, Selected action: {action}({action_idx})")


            root = None  # Reset root for the next move

            turns.append(game.board.turn)
            game.play_action(action)

            states.append(state.lcz_features())
            pis.append(pi)

        sim_a = self.reward_fc(game.board.get_feature_sequence(), True)
        sim_b = self.reward_fc(game.board.get_feature_sequence(), False)
        outcome = game.board.pc_board.outcome()
        if outcome is not None:
            winner = outcome.winner
            if winner:
                sim_a = sim_a * 0.8 + 0.2
                sim_b = sim_b * 0.8 - 0.2
            elif not winner:
                sim_a = sim_a * 0.8 - 0.2
                sim_b = sim_b * 0.8 + 0.2

        for state, pi, turn in zip(states, pis, turns):
            sim = sim_a if turn else sim_b
            self.memory.append((state, pi, sim, turn))
        
        return step, sim_a, sim_b


    def train(self):
        if len(self.memory) < self.config['batch_size']:
            return

        batch = random.sample(self.memory, self.config['batch_size'])
        states, pis, sims, turns = zip(*batch)

        states_w = []
        states_b = []
        pis_w = []
        pis_b = []
        vs_w = []
        vs_b = []
        for state, pi, sim, turn in zip(states, pis, sims, turns):
            if turn:
                states_w.append(state)
                pis_w.append(pi)
                vs_w.append(sim)
            else:
                states_b.append(state)
                pis_b.append(pi)
                vs_b.append(sim)

        states_w_np = np.stack(states_w).astype(np.float32)
        pis_w_np    = np.stack(pis_w).astype(np.float32)
        vs_w_np  = np.asarray(vs_w, dtype=np.float32).reshape(-1,1)

        states_w_t = torch.from_numpy(states_w_np).to(self.device)
        pis_w_t    = torch.from_numpy(pis_w_np).to(self.device)
        vs_w_t  = torch.from_numpy(vs_w_np).to(self.device)

        states_b_np = np.stack(states_b).astype(np.float32)
        pis_b_np    = np.stack(pis_b).astype(np.float32)
        vs_b_np  = np.asarray(vs_b, dtype=np.float32).reshape(-1,1)

        states_b_t = torch.from_numpy(states_b_np).to(self.device)
        pis_b_t    = torch.from_numpy(pis_b_np).to(self.device)
        vs_b_t  = torch.from_numpy(vs_b_np).to(self.device)


        # ---------- monitoring ----------
        # We'll record per‑batch statistics right after forward pass.
        logits, pred_v = self.net(states_w_t)
        # policy loss

        logp = F.log_softmax(logits, dim=1)
        loss_pi = -(pis_w_t * logp).sum(dim=1).mean()
        # loss_pi = F.kl_div(F.log_softmax(logits,1), target_pi, reduction='batchmean')

        # value losses
        loss_z = F.mse_loss(pred_v, vs_w_t)

        # total
        loss = loss_pi + loss_z

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ---------- monitoring ----------
        logits, pred_v = self.net_b(states_b_t)
        # policy loss

        logp = F.log_softmax(logits, dim=1)
        loss_pi = -(pis_b_t * logp).sum(dim=1).mean()
        # loss_pi = F.kl_div(F.log_softmax(logits,1), target_pi, reduction='batchmean')

        # value losses
        loss_b_z = F.mse_loss(pred_v, vs_b_t)

        # total
        loss_b = loss_pi + loss_b_z

        self.optimizer_b.zero_grad()
        loss_b.backward()
        self.optimizer_b.step()

        # ---------- print metrics ----------
        mean_v = pred_v.mean().item()
        print(f"[TRAIN] batch={self.config['batch_size']} "
              f"Loss_pi={loss_pi.item():.4f} "
              f"Loss_z={loss_z.item():.4f} "
              f"Loss_b_pi={loss_b.item():.4f} "
              f"Loss_b_z={loss_b_z.item():.4f} "
              f"Total={loss.item():.4f} "
              f"mean_pred_v={mean_v:.3f}")

    def run(self):
        for iteration in range(self.config['num_iterations']):
            print(f"Iteration {iteration + 1}/{self.config['num_iterations']}")
            for idx in range(self.config['num_self_play_games']):
                step, sim_a, sim_b = self.self_play()
                print(f"Self-play game {idx + 1}/{self.config['num_self_play_games']} completed with {step} steps. sim_a: {sim_a}, sim_b: {sim_b}")

            # train_steps = self.config.get('train_steps_per_iter', 1)
            # for _ in range(train_steps):
                self.train()

            if (iteration + 1) % self.config['save_interval'] == 0:
                torch.save(self.net.state_dict(), f"./models/model_{iteration+1}.pth")
                print(f"Models saved at iteration {iteration + 1}.")
