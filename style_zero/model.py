import torch
import torch.nn as nn
import torch.nn.functional as F

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
            *nn.ModuleList([ResidualBlock(256) for _ in range(20)]),
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
        x = self.conv(x).view(x.size(0), -1)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value