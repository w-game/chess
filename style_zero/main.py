import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
from model import AlphaZeroNet
from self_play import generate_self_play
from game import Game  # 你需要自定义自己的棋类规则
import torch.optim as optim

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


net = AlphaZeroNet(board_size=8, action_size=64)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for iteration in range(100):
    memory = []
    for _ in range(20):  # 20局自我对弈
        memory += generate_self_play(Game, net, mcts_simulations=30)

    train_net(net, optimizer, memory)
    print(f"Iteration {iteration} finished.")