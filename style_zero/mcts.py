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
    def __init__(self, net_a, net_b, reward_fc, device=None, num_simulations=20, c_puct=1.0):
        self.net_a = net_a
        self.net_b = net_b
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.reward_fc = reward_fc
    
    
    def simulate(self, state, node):
        if state.is_game_over():
            features = state.get_feature_sequence()
            if state.turn:
                # 如果当前是白方的回合，则黑胜利，最后一手为黑方
                reward = self.reward_fc(features, False)
            else:
                # 如果当前是黑方的回合，则白胜利，最后一手为白方
                reward = self.reward_fc(features, True)
            print(f"Game over: value = {reward}")
            return (reward + 0.5), not state.turn

        if not node.children:
            features = state.lcz_features()
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            if state.turn:
                policy_logits, v = self.net_a.forward(features)
            else:
                policy_logits, v = self.net_b.forward(features)

            policy = softmax(policy_logits.detach().cpu().numpy().flatten())
            legal_moves = state.generate_legal_moves()
            for a in legal_moves:
                a_idx = state.move_to_index(a, state.turn, state.is_castling(a))
                node.children[a_idx] = TreeNode(node, policy[a_idx])

            return v.item(), state.turn

        best_score, best_action_idx = -float('inf'), None
        for a, child in node.children.items():
            ucb = child.value() + self.c_puct * child.prior * (np.sqrt(node.visit_count + 1) / (child.visit_count + 1))
            if ucb > best_score:
                best_score = ucb
                best_action_idx = a

        real_action = state.idx_to_move(best_action_idx, state.turn)
        state.push_uci(real_action)
        next_state = state.copy()
        v, is_white = self.simulate(next_state, node.children[best_action_idx])

        child = node.children[best_action_idx]
        if state.turn == is_white:
            child.total_value += v
            child.visit_count += 1
        return v, is_white
    
    def run(self, state):
        root = TreeNode(None, 1.0)
        for _ in range(self.num_simulations):
            self.simulate(state.copy(), root)
        visits = {a: child.visit_count for a, child in root.children.items()}
        return visits

    def get_action_probabilities(self, state, temp=1.0):
        visits = self.run(state)

        if temp == 0:
            best_action = max(visits.items(), key=lambda x: x[1])[0]
            probs = np.zeros(1858)
            probs[best_action] = 1.0
            return probs
        else:
            legal_indices = list(visits.keys())
            counts = np.array([visits[a] for a in legal_indices], dtype=np.float32)
            if counts.sum() == 0:
                return np.ones(1858) / 1858
            counts = counts ** (1.0 / temp)
            probs = np.zeros(1858, dtype=np.float32)
            for idx, count in zip(legal_indices, counts):
                probs[idx] = count
            probs /= probs.sum()
            return probs
