import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import chess
from player_encoder.encoder import BoardCNNEncoder


class Game:
    def __init__(self, evaluator_black=None, evaluator_white=None):
        self.board = chess.Board()
        self.evaluator_black = evaluator_black
        self.evaluator_white = evaluator_white

    def encode_tensor(self):
        tensor = torch.zeros((14, 8, 8), dtype=torch.float32)
        piece_map = self.board.piece_map()
        for square, piece in piece_map.items():
            i = 0
            if piece.color == chess.WHITE:
                i = "PNBRQK".index(piece.symbol().upper())
            else:
                i = 6 + "PNBRQK".index(piece.symbol().upper())
            row = 7 - (square // 8)
            col = square % 8
            tensor[i, row, col] = 1.0
        tensor[12] = 1.0 if self.board.turn == chess.WHITE else 0.0
        tensor[13] = 1.0 if self.board.has_kingside_castling_rights(chess.WHITE) else 0.0
        return tensor

    def legal_moves(self):
        return list(self.board.legal_moves)

    def play(self, move):
        new_game = self.clone()
        new_game.board.push(move)
        return new_game

    def is_terminal(self):
        return self.board.is_game_over()

    def reward(self):
        player = 'white' if self.board.turn == chess.WHITE else 'black'
        evaluator = self.evaluator_white if player == 'white' else self.evaluator_black
        return evaluator(self) if evaluator else 0.0

    def clone(self):
        new_game = Game(self.evaluator_black, self.evaluator_white)
        new_game.board = self.board.copy(stack=True)
        return new_game


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        out = F.relu(out)
        return out

class AlphaZeroNet(nn.Module):
    def __init__(self, in_channels=112, action_size=1858):
        super().__init__()
        self.bonenet = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            *nn.ModuleList([ResidualBlock(256) for _ in range(6)]),
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(2 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

        self.value_head = nn.Sequential(
            nn.Linear(2 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.bonenet(x).view(x.size(0), -1)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
class StyleRewardEvaluator:
    def __init__(self, encoder_path='...', prototype_path='...'):
        self.encoder = BoardCNNEncoder(in_channels=112, out_dim=256)
        self.encoder.load_state_dict(torch.load(encoder_path))  # 如有模型权重
        self.encoder.eval()
        self.prototype = F.normalize(torch.load(prototype_path), p=2, dim=1)

    def __call__(self, game):
        with torch.no_grad():
            state_tensor = game.encode_tensor().unsqueeze(0)
            style_vector = self.encoder(state_tensor)
            sim, _ = compute_similarity_percent_cosine(style_vector, self.prototype)
        return sim.item() / 100.0
    

class TreeNode:
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior = prior

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class MCTS:
    def __init__(self, game, net, num_simulations=50, c_puct=1.0):
        self.game = game
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, state):
        root = TreeNode(None, 1.0)
        for _ in range(self.num_simulations):
            self.simulate(state.clone(), root)
        visits = {a: child.visit_count for a, child in root.children.items()}
        return visits

    def simulate(self, state, node):
        if state.is_terminal():
            reward = state.reward()
            return reward

        if not node.children:
            policy_logits, value = self.net(state.encode_tensor().unsqueeze(0))
            policy = softmax(policy_logits.detach().numpy().flatten())
            legal_moves = state.legal_moves()
            for a in legal_moves:
                node.children[a] = TreeNode(node, policy[a])
            return value.item()

        best_score, best_action = -float('inf'), None
        for a, child in node.children.items():
            ucb = child.value() + self.c_puct * child.prior * (np.sqrt(node.visit_count + 1) / (child.visit_count + 1))
            if ucb > best_score:
                best_score = ucb
                best_action = a

        next_state = state.play(best_action)
        v = self.simulate(next_state, node.children[best_action])

        child = node.children[best_action]
        child.visit_count += 1
        child.total_value += v
        node.visit_count += 1
        return v


def generate_self_play(game_cls, net, mcts_simulations, evaluator_black, evaluator_white):
    memory = []
    game = game_cls(evaluator_black, evaluator_white)
    mcts = MCTS(game_cls(evaluator_black, evaluator_white), net, mcts_simulations)

    while not game.is_terminal():
        visits = mcts.run(game)
        actions, counts = zip(*visits.items())
        probs = [c / sum(counts) for c in counts]
        action = random.choices(actions, probs)[0]

        current_player = game.current_player
        memory.append((game.encode_tensor(), probs, current_player))
        game = game.play(action)

    reward_black = evaluator_black(game)
    reward_white = evaluator_white(game)

    labeled_memory = []
    for s, p, player in memory:
        reward = reward_black if player == 'black' else reward_white
        labeled_memory.append((s, p, reward))

    return labeled_memory

def train_net(net, optimizer, memory, epochs=10, batch_size=64):
    states, policies, values = zip(*memory)
    states = torch.stack(states)
    policy_targets = torch.tensor(policies, dtype=torch.float)
    value_targets = torch.tensor(values, dtype=torch.float).unsqueeze(1)

    dataset = TensorDataset(states, policy_targets, value_targets)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    net.train()
    for _ in range(epochs):
        for s, p, v in dataloader:
            pred_p, pred_v = net(s)
            loss_policy = F.cross_entropy(pred_p, p)
            loss_value = F.mse_loss(pred_v, v)
            loss = loss_policy + loss_value
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def generate_adapted_evaluator(support_games, encoder_path):
    with torch.no_grad():
        encoder = BoardCNNEncoder(in_channels=112, out_dim=256)
        encoder.load_state_dict(torch.load(encoder_path))
        encoder.eval()
        support_tensors = [g.encode_tensor().unsqueeze(0) for g in support_games]
        z_list = [encoder(t) for t in support_tensors]
        z_mean = torch.mean(torch.cat(z_list, dim=0), dim=0, keepdim=True)
        z_norm = F.normalize(z_mean, p=2, dim=1)

    class AdaptedEvaluator:
        def __call__(self, game):
            with torch.no_grad():
                query_tensor = game.encode_tensor().unsqueeze(0)
                z = encoder(query_tensor)
                sim, _ = compute_similarity_percent_cosine(z, z_norm)
            return sim.item() / 100.0

    return AdaptedEvaluator()

net = AlphaZeroNet(in_channels=112, action_size=1858)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

def compute_similarity_percent_cosine(query_z, prototype):
    cos_sims = F.cosine_similarity(query_z, prototype, dim=1)

    similarity_percent = (cos_sims + 1.0) / 2.0 * 100.0

    avg_similarity = similarity_percent.mean().item()
    return similarity_percent, avg_similarity

def target_player_prototype():
    prototype = torch.load('style_zero/prototype.pt')

    prototype_vector = F.normalize(prototype, p=2, dim=1)
    return prototype_vector

def dummy_style_reward_fn(game):
    evaluator = StyleRewardEvaluator("", "./")
    return  evaluator(game)

target_support_games = [...]  # 5局对象玩家棋谱
opponent_sets = [...]         # 若干个对手，每个包含5局棋谱

target_eval = generate_adapted_evaluator(target_support_games, "style_zero/encoder.pt")

candidates = []
for opponent_support_games in opponent_sets:
    opp_eval = generate_adapted_evaluator(opponent_support_games, "style_zero/encoder.pt")

    prob = random.random()
    if prob < 0.5:
        candidates += generate_self_play(Game, net, mcts_simulations=30,
                                     evaluator_black=opp_eval, evaluator_white=target_eval)
    else:
        candidates += generate_self_play(Game, net, mcts_simulations=30,
                                        evaluator_black=target_eval, evaluator_white=opp_eval)

memory = random.sample(candidates, 5)

evaluator_black = StyleRewardEvaluator("style_zero/encoder.pt", "style_zero/black.pt")
evaluator_white = StyleRewardEvaluator("style_zero/encoder.pt", "style_zero/white.pt")

for iteration in range(100):
    train_net(net, optimizer, memory)
    print(f"Iteration {iteration} finished.")