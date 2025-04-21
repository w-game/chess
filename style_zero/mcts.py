import numpy as np
import random

import torch


class TreeNode:
    def __init__(self, parent, prior, turn):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior = prior
        self.turn = turn 

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class MCTS:
    def __init__(self, net, net_b, reward_fc, turn, device=None, *,
                 num_simulations=20,
                 c_init=1.25,          # AlphaZero‑style dynamic cpuct
                 c_base=19652,
                 c_factor=2.0,
                 gamma=0.995):
        self.net = net
        self.net_b = net_b
        self.turn = turn
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_simulations = num_simulations
        self.c_init   = c_init
        self.c_base   = c_base
        self.c_factor = c_factor
        self.gamma = gamma
        self.reward_fc = reward_fc

    def _dyn_cpuct(self, parent_visits: int) -> float:
        """
        AlphaZero dynamic cpuct coefficient:
            c_puct = c_init + c_factor * log((N + c_base + 1) / c_base)
        """
        return self.c_init + self.c_factor * np.log((parent_visits + self.c_base + 1) / self.c_base)
    
    def backup(self, node, v_w, v_b):
        """
        Back up the value of the node to its parent.
        """
        while node is not None:
            if node.turn:
                node.total_value += v_w
            else:
                node.total_value += v_b
            node.visit_count += 1
            node = node.parent
    
    
    def simulate(self, state, node, step=0):
        while True:
            if state.is_game_over():
                features = state.get_feature_sequence()
                reward_white = self.reward_fc(features, True) * (self.gamma ** step)
                reward_black = self.reward_fc(features, False) * (self.gamma ** step)

                outcome = state.pc_board.outcome()  
                # outcome 是 None（游戏未结束）或一个 chess.Outcome 对象
                if outcome is not None:
                    winner = outcome.winner    # True 白胜，False 黑胜，None 平局
                    if self.turn and winner == True:
                        reward_white = reward_white * 0.8 + 0.2

                    if not self.turn and winner == False:
                        reward_black = reward_black * 0.8 + 0.2

                self.backup(node, reward_white, reward_black)
                return reward_white, reward_black

            if not node.children:
                features = state.lcz_features()
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                v_w, v_b = 0.0, 0.0
                with torch.no_grad():
                    if state.turn:
                        policy_logits, v_w = self.net.forward(features)
                    else:
                        policy_logits, v_b = self.net_b.forward(features)
                    
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
                    node.children[a_idx] = TreeNode(node, float(prior), not node.turn)

                self.backup(node, v_w, v_b)
                
                return v_w, v_b

            best_score, best_action_idx = -float('inf'), None
            for a, child in node.children.items():
                cpuct_coeff = self._dyn_cpuct(node.visit_count)
                ucb = child.value() + cpuct_coeff * child.prior * (np.sqrt(node.visit_count) / (child.visit_count + 1))
                if ucb > best_score:
                    best_score = ucb
                    best_action_idx = a

            real_action = state.idx_to_move(best_action_idx, state.turn)
            next_state = state.copy()
            next_state.push_uci(real_action)
            state = next_state
            node = node.children[best_action_idx]
            step += 1
    
    def run(self, state, root=None, step=0):
        if root is None:
            root = TreeNode(None, 1.0, True)
        for _ in range(self.num_simulations):
            v_w, v_b = self.simulate(state.copy(), root)

        visits = {a: child.visit_count for a, child in root.children.items()}
        # return both visits and the updated root so callers can reuse subtree
        return visits, root

    def get_action_probabilities(self, state, step, temp=1.0, root=None):
        visits, root = self.run(state, root, step)

        if temp == 0:
            best_action = max(visits.items(), key=lambda x: x[1])[0]
            probs = np.zeros(1858)
            probs[best_action] = 1.0
            return probs, root, visits
        else:
            legal_indices = list(visits.keys())
            counts = np.array([visits[a] for a in legal_indices], dtype=np.float32)
            if counts.sum() == 0:
                return np.ones(1858) / 1858, root
            counts = counts ** (1.0 / temp)
            probs = np.zeros(1858, dtype=np.float32)
            for idx, count in zip(legal_indices, counts):
                probs[idx] = count
            probs /= probs.sum()
            return probs, root, visits
