import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import chess
from leela_board import LeelaBoard


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
    def __init__(self, game_cls, network_cls, emb_net, mcts_cls, config, device=None):
        self.game_cls = game_cls
        self.network = network_cls
        self.emb_net = emb_net
        self.mcts_cls = mcts_cls
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.network.parameters(), lr=config['lr'])
        self.memory = deque(maxlen=config['memory_size'])
        self.target_style_embedding = torch.randn(256).to(self.device)  # 示例目标风格，可改为真实风格向量

    def style_reward(self, game, color=True, s_min=0.2, s_max=0.9):
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

        state_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(self.device)
        style_emb = self.target_style_embedding
        _, pred_emb = self.emb_net(state_tensor)

        # 计算原相似度
        raw_sim = torch.cosine_similarity(pred_emb, style_emb.unsqueeze(0), dim=1).mean()
        raw_sim_val = raw_sim.item()

        clipped_sim = max(min(raw_sim_val, s_max), s_min)  
        scaled_sim = (clipped_sim - s_min) / (s_max - s_min)

        return scaled_sim

    def self_play(self):
        game = self.game_cls()
        mcts = self.mcts_cls(self.network, self.style_reward, self.device, num_simulations=20, c_puct=1.0)
        states, pis = [], []

        step = 0

        while not game.is_game_over() and step < 100:
            step += 1
            state = game.get_current_state()
            pi = mcts.get_action_probabilities(game, temp=self.config['temperature'])
            # action_idx = np.random.choice(np.arange(1858), p=pi)

            legal_move = game.generate_legal_moves()
            legal_indices = [game.board.move_to_index(move, game.board.turn, game.board.is_castling(move)) for move in legal_move]
            legal_p = pi[legal_indices]
            legal_p = legal_p / np.sum(legal_p)  # 只对合法动作对应的概率做归一化
            action_idx = np.random.choice(legal_indices, p=legal_p)
            action = game.board.idx_to_move(action_idx, game.board.turn)
            game.play_action(action)

            states.append(state.lcz_features())
            pis.append(pi)

        sim = self.style_reward(game.board.get_feature_sequence())
        for state, pi in zip(states, pis):
            self.memory.append((state, pi, sim))  # winner should be replaced with sim

    def train(self):
        if len(self.memory) < self.config['batch_size']:
            return

        batch = random.sample(self.memory, self.config['batch_size'])
        states, pis, zs = zip(*batch)

        states_np = np.array(states)
        states = torch.from_numpy(states_np, dtype=torch.float32).to(self.device)
        target_pis = torch.tensor(pis, dtype=torch.float32).to(self.device)
        target_zs = torch.tensor(zs, dtype=torch.float32).view(-1, 1).to(self.device)

        pred_pis, pred_zs = self.network(states)
        loss_pi = nn.KLDivLoss()(torch.log(pred_pis), target_pis, reduction='batchmean')
        loss_z = nn.MSELoss()(pred_zs, target_zs)
        loss = loss_pi + loss_z

        print(f"Loss: {loss.item()}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        for iteration in range(self.config['num_iterations']):
            print(f"Iteration {iteration + 1}/{self.config['num_iterations']}")
            for idx in range(self.config['num_self_play_games']):
                self.self_play()
                print(f"Self-play game {idx + 1}/{self.config['num_self_play_games']} completed.")
            self.train()
            # 必要に応じてモデルの保存や評価を行う


# if __name__ == "__main__":

    # game = torch.load('../chess_data_parse/dataset/a_ndre/black_0000.pt')
    # states = game['states']  # [T, 112, 8, 8]
    # actions = game['actions']  # [T]
    # print(actions.shape, actions[0])
    # lb = LeelaBoard()
    # for action in actions:
    #     move_idx = torch.argmax(action).item()
    #     move = idx_to_move(move_idx, lb.turn)
    #     print(move)
    #     lb.push(chess.Move.from_uci(move))
