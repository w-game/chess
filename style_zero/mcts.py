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
    def __init__(self, net, reward_fc, device=None, num_simulations=20, c_puct=1.0, gamma=0.85):
        self.net = net
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.reward_fc = reward_fc
        self.gamma = gamma
    
    
    def simulate(self, state, node, step=0):
        if state.is_game_over():
            features = state.get_feature_sequence()
            reward_white = self.reward_fc(features, True) * (self.gamma ** step)
            reward_black = self.reward_fc(features, False) * (self.gamma ** step)

            node.visit_count += 1

            if state.turn:
                node.total_value += reward_white
            else:
                node.total_value += reward_black
            
            return reward_white, reward_black

        if not node.children:
            features = state.lcz_features()
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, v_w, v_b = self.net.forward(features)
                
            policy = softmax(policy_logits.detach().cpu().numpy().flatten())
            legal_moves = state.generate_legal_moves()
            legal_idxs = [state.move_to_index(a, state.turn, state.is_castling(a)) for a in legal_moves]

            # --- create child nodes ---
            if node.parent is None:
                noise = np.random.dirichlet([0.3] * len(legal_idxs))

            for i, a_idx in enumerate(legal_idxs):
                prior = policy[a_idx]
                if node.parent is None:
                    prior = 0.75 * prior + 0.25 * noise[i]
                node.children[a_idx] = TreeNode(node, float(prior))

            # if step > 150:
            #     penalty = -0.05 * (step - 150)
            #     node.visit_count += 1
            #     node.total_value += penalty
            #     return penalty, penalty

            node.visit_count += 1

            if state.turn:
                node.total_value += v_w.item()
            else:
                node.total_value += v_b.item()

            return v_w, v_b

        best_score, best_action_idx = -float('inf'), None
        for a, child in node.children.items():
            ucb = child.value() + self.c_puct * child.prior * (np.sqrt(node.visit_count) / (child.visit_count + 1))
            if ucb > best_score:
                best_score = ucb
                best_action_idx = a

        real_action = state.idx_to_move(best_action_idx, state.turn)
        next_state = state.copy()
        next_state.push_uci(real_action)
        v_w, v_b = self.simulate(next_state, node.children[best_action_idx], step + 1)

        node.visit_count += 1
        if state.turn:
            node.total_value += v_w
        else:
            node.total_value += v_b

        return v_w, v_b
    
    def run(self, state, step=0):
        root = TreeNode(None, 1.0)
        for _ in range(self.num_simulations):
            v_w, v_b = self.simulate(state.copy(), root, step)

        visits = {a: child.visit_count for a, child in root.children.items()}
        return visits

    def get_action_probabilities(self, state, step, temp=1.0):
        visits = self.run(state, step)

        # Show top‑3 visit counts for quick sanity check
        if len(visits) > 0:
            top3 = sorted(visits.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"[MCTS - Step {step}] top‑3 root visits: {top3}")

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
