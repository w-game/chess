import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.models.resnet import Bottleneck

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
    
class BoardCNNEncoder(nn.Module):
    def __init__(self, in_channels=112, out_dim=256):
        super(BoardCNNEncoder, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            *nn.ModuleList([ResidualBlock(256) for _ in range(6)]),
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, out_dim),
        )

    def forward(self, x):
        # x: [B, 224, 8, 8]
        x = self.backbone(x)
        return x  


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        device = x.device

        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)  # [seq_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        x = x + pe.unsqueeze(0) # [1, seq_len, d_model]
        return self.dropout(x)  # [batch_size, seq_len, d_model]
    

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()

        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)  # 只做标准化，不学 γβ
        )

        # 推荐初始化：Kaiming 正态
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        z = self.net(x)           # [B, out_dim]
        return F.normalize(z, p=2, dim=1)   # 保证对比时在单位球面
    
class TransformerEncoder(nn.Module):
    def __init__(self, cnn_in_channels=112, state_embed_dim=256, transformer_d_model=256,
                 num_heads=8, num_layers=3, dropout=0.1):
        """
        参数说明：
          cnn_in_channels: 状态输入通道数（例如112）
          action_size: 总动作数（1858）
          state_embed_dim, action_embed_dim: 分别为状态和动作编码维度
          fusion_out_dim: 融合后的维度
          transformer_d_model: Transformer 模型维度（建议与 fusion_out_dim 保持一致）
          num_heads, num_layers, dropout: Transformer 超参数
          max_seq_len: 序列最大长度（用于位置编码）
        """
        super(TransformerEncoder, self).__init__()
        self.state_encoder = BoardCNNEncoder(in_channels=cnn_in_channels, out_dim=state_embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=transformer_d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(transformer_d_model, transformer_d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_d_model))
        # self.projection_head = nn.Sequential(
        #     nn.Linear(transformer_d_model, transformer_d_model),
        #     nn.ReLU(),
        #     nn.Linear(transformer_d_model, transformer_d_model)
        # )
        self.proj_head = ProjectionHead(in_dim=transformer_d_model, hidden_dim=transformer_d_model, out_dim=transformer_d_model)
        self.layernorm = nn.LayerNorm(transformer_d_model)
        self.temperature = 0.07

    def forward(self, states, mask=None):
        """
        参数:
          states: [batch, seq_len, 112, 8, 8]，状态序列
          actions: [batch, seq_len]，每步的动作索引
          mask: [batch, seq_len]，Bool 类型掩码，True 表示 padding（无效位置）
        返回:
          整局风格向量: [batch, transformer_d_model]
        """
        B, N, T, C, H, W = states.size()

        mask = mask.view(B * N, T) if mask is not None else None  # [batch, seq_len]
        
        # 编码状态
        states = states.view(B * N * T, C, H, W)
        state_emb = self.state_encoder(states)  # [batch*N, state_embed_dim]
        state_emb = state_emb.view(B * N, T, -1)  # Reshape to [B, N, state_embed_dim]

        token_embeddings = self.pos_encoder(state_emb) # [batch, seq_len, d_model]

        # Transformer 编码
        transformer_output = self.transformer_encoder(
            token_embeddings, src_key_padding_mask=mask
        )

        # 序列池化（聚合）
        valid_mask = (~mask).unsqueeze(-1).float()  # [B*N, T, 1]
        masked_output = transformer_output * valid_mask  # [B*N, T, D]
        sum_output = masked_output.sum(dim=1)  # [B*N, D]
        count = valid_mask.sum(dim=1)  # [B*N, 1]
        pooled = sum_output / (count + 1e-6)

        # 输出风格向量
        final_embedding = self.fc(pooled)
        final_embedding = self.dropout(final_embedding)

        # 防止除以 0 导致 nan
        norm = final_embedding.norm(p=2, dim=-1, keepdim=True)
        final_embedding = final_embedding / (norm + 1e-6)

        final_embedding = final_embedding / self.temperature
        final_embedding = final_embedding.view(B, N, -1)  # Reshape to [B, N, d_model]

        # contrastive_embedding = self.projection_head(final_embedding)

        # contrastive_embedding = contrastive_embedding.view(B, N, -1)  # Reshape to [B, N, d_model]

        h = self.proj_head(final_embedding.view(-1, self.fc.out_features)).view(B, N, -1)

        return h, final_embedding
    

if __name__ == "__main__":
    # 测试代码
    model = TransformerEncoder(cnn_in_channels=224, state_embed_dim=256, transformer_d_model=256,
                             num_heads=8, num_layers=3, dropout=0.1, max_seq_len=100)
    states = torch.randn(32, 100, 224, 8, 8)  # [batch, seq_len, C, H, W]
    mask = torch.zeros(32, 100).bool()  # [batch, seq_len]
    mask[:, :50] = True
    output = model(states, mask=mask)
    print(output.shape)