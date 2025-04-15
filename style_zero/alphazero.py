import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from mcts import MCTS  # モンテカルロ木探索を実装したモジュール
from network import AlphaZeroNet  # ポリシーとバリューヘッドを持つニューラルネットワーク
import chess  # python-chessライブラリを使用してチェスのルールを実装
from ..chess_data_parse.policy_index import policy_index

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.history = []
        self.result = None

    def get_current_state(self):
        # 将当前棋盘状态转换为 [112, 8, 8] 的张量
        return encode_board(self.board)

    def index_to_move(self, index):
        return policy_index[index]

    def get_legal_actions(self):
        # 返回当前合法动作的索引列表
        return [move_to_index(move) for move in self.board.legal_moves]

    def play_action(self, action):
        # 将动作索引转换为棋步并执行
        torch.softmax(action, dim=-1)
        move = self.index_to_move(action_index)
        self.board.push(move)
        self.history.append(move)

    def is_game_over(self):
        return self.board.is_game_over()

    def get_winner(self):
        result = self.board.result()
        if result == '1-0':
            return 1  # 白方胜
        elif result == '0-1':
            return -1  # 黑方胜
        else:
            return 0  # 平局

class AlphaZeroTrainer:
    def __init__(self, game_cls, network_cls, mcts_cls, config):
        self.game_cls = game_cls
        self.network = network_cls()
        self.mcts_cls = mcts_cls
        self.config = config
        self.optimizer = optim.Adam(self.network.parameters(), lr=config['lr'])
        self.memory = deque(maxlen=config['memory_size'])

    def self_play(self):
        game = self.game_cls()
        mcts = self.mcts_cls(self.network, self.config)
        states, pis, zs = [], [], []

        while not game.is_game_over():
            state = game.get_current_state()
            pi = mcts.get_action_probabilities(game, temp=self.config['temperature'])
            action = np.random.choice(len(pi), p=pi)
            game.play_action(action)

            states.append(state)
            pis.append(pi)

        winner = game.get_winner()
        for state, pi in zip(states, pis):
            zs.append(winner)
            self.memory.append((state, pi, winner))

    def train(self):
        if len(self.memory) < self.config['batch_size']:
            return

        batch = random.sample(self.memory, self.config['batch_size'])
        states, pis, zs = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        target_pis = torch.tensor(pis, dtype=torch.float32)
        target_zs = torch.tensor(zs, dtype=torch.float32).view(-1, 1)

        pred_pis, pred_zs = self.network(states)
        loss_pi = nn.KLDivLoss()(torch.log(pred_pis), target_pis)
        loss_z = nn.MSELoss()(pred_zs, target_zs)
        loss = loss_pi + loss_z

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        for iteration in range(self.config['num_iterations']):
            print(f"Iteration {iteration + 1}/{self.config['num_iterations']}")
            for _ in range(self.config['num_self_play_games']):
                self.self_play()
            self.train()
            # 必要に応じてモデルの保存や評価を行う

if __name__ == "__main__":
    # config = {
    #     'lr': 0.001,
    #     'memory_size': 10000,
    #     'batch_size': 64,
    #     'num_iterations': 1000,
    #     'num_self_play_games': 25,
    #     'temperature': 1.0,
    #     # その他のハイパーパラメータ
    # }

    # trainer = AlphaZeroTrainer(Game, AlphaZeroNet, MCTS, config)
    # trainer.run()
    game = torch.load('../chess_data_parse/dataset/demo/black_0000.pth')
    states = game['states']  # [T, 112, 8, 8]
    actions = game['actions']  # [T]
    print(actions.shape, actions[0])
    # chess_game = Game()
    # chess_game.play_action(actions[0])