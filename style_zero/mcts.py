import numpy as np
import random

import torch


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
    def __init__(self, net, style_reward, num_simulations=50, c_puct=1.0):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.reward = style_reward

    def run(self, state):
        root = TreeNode(None, 1.0)
        for _ in range(self.num_simulations):
            self.simulate(state.copy(), root)
        visits = {a: child.visit_count for a, child in root.children.items()}
        return visits  # 不再使用 all_actions 填充

    def get_action_probabilities(self, game, temp=1.0):
        state = game.get_current_state()
        visits = self.run(state)
        legal_moves = game.generate_legal_moves()
        legal_indices = list(visits.keys())
        if temp == 0:
            best_action = max(visits.items(), key=lambda x: x[1])[0]
            probs = np.zeros(1858)
            probs[best_action] = 1.0
            return probs, legal_indices
        else:
            legal_indices = list(visits.keys())
            counts = np.array([visits[a] for a in legal_indices], dtype=np.float32)
            if counts.sum() == 0:
                return np.ones(1858) / 1858, legal_indices
            counts = counts ** (1.0 / temp)
            probs = np.zeros(1858, dtype=np.float32)
            for idx, count in zip(legal_indices, counts):
                probs[idx] = count
            probs /= probs.sum()
            return probs, legal_indices

    def simulate(self, state, node):
        if state.is_game_over():
            features = state.get_feature_sequence()
            reward = self.reward(features)
            print(f"Game over: {reward}")
            return reward

        if not node.children:
            features = state.lcz_features()
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            policy_logits, value = self.net.forward(features)
            policy = softmax(policy_logits.detach().numpy().flatten())
            legal_moves = state.generate_legal_moves()
            for a in legal_moves:
                a_idx = state.move_to_index(a, state.turn, state.is_castling(a))
                node.children[a_idx] = TreeNode(node, policy[a_idx])
            return value.item()

        best_score, best_action = -float('inf'), None
        for a, child in node.children.items():
            ucb = child.value() + self.c_puct * child.prior * (np.sqrt(node.visit_count + 1) / (child.visit_count + 1))
            if ucb > best_score:
                best_score = ucb
                best_action = a

        real_action = state.idx_to_move(best_action, state.turn)
        state.push_uci(real_action)
        next_state = state.copy()
        v = self.simulate(next_state, node.children[best_action])

        child = node.children[best_action]
        child.visit_count += 1
        child.total_value += v
        node.visit_count += 1
        return v
