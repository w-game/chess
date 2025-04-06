import numpy as np

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