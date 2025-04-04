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
            *nn.ModuleList([ResidualBlock(256) for _ in range(20)]),
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, out_dim),
        )
        
    # def _make_layer(self, block, inplanes, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )

    #     layers = [block(inplanes, planes, stride, downsample)]
    #     inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(inplanes, planes))
    #     return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 112, 8, 8]
        x = self.backbone(x)                 # → [B, 2048, 1, 1]
        # x = F.adaptive_avg_pool2d(x, 1)      # → [B, 2048, 1, 1]
        # x = x.view(x.size(0), -1)            # → [B, 2048]
        # x = self.fc(x)                       # → [B, 256]
        return x  


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, cnn_in_channels=112, state_embed_dim=256, transformer_d_model=256,
                 num_heads=8, num_layers=3, dropout=0.1, max_seq_len=100):
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
        self.pos_encoder = PositionalEncoding(d_model=transformer_d_model, dropout=dropout, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(transformer_d_model, transformer_d_model)

    def forward(self, states, mask=None):
        """
        参数:
          states: [batch, seq_len, 112, 8, 8]，状态序列
          actions: [batch, seq_len]，每步的动作索引
          mask: [batch, seq_len]，Bool 类型掩码，True 表示 padding（无效位置）
        返回:
          整局风格向量: [batch, transformer_d_model]
        """
        batch_size, seq_len, C, H, W = states.size()

        # 编码状态
        states = states.view(batch_size * seq_len, C, H, W)
        state_emb = self.state_encoder(states)  # [batch*seq_len, state_embed_dim]
        state_emb = state_emb.view(batch_size, seq_len, -1)

        token_embeddings = self.pos_encoder(state_emb)

        # Transformer 编码
        transformer_output = self.transformer_encoder(
            token_embeddings, src_key_padding_mask=mask
        )

        # 序列池化（聚合）
        if mask is not None:
            # False 表示有效位置
            valid_mask = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            transformer_output = transformer_output * valid_mask
            valid_counts = valid_mask.sum(dim=1)
            pooled = transformer_output.sum(dim=1)

            # 对于全 padding 的样本，设置为 small vector，避免 pooled=0
            pooled[valid_counts.squeeze(-1) == 0] = 1e-6
            pooled = pooled / valid_counts.clamp(min=1)
        else:
            pooled = transformer_output.mean(dim=1)

        # 输出风格向量
        final_embedding = self.fc(pooled)

        # 防止除以 0 导致 nan
        norm = final_embedding.norm(p=2, dim=-1, keepdim=True)
        final_embedding = final_embedding / (norm + 1e-6)

        return final_embedding
    

if __name__ == "__main__":
    # 测试代码
    model = TransformerEncoder(cnn_in_channels=224, state_embed_dim=256, transformer_d_model=256,
                             num_heads=8, num_layers=3, dropout=0.1, max_seq_len=100)
    states = torch.randn(32, 100, 224, 8, 8)  # [batch, seq_len, C, H, W]
    mask = torch.zeros(32, 100).bool()  # [batch, seq_len]
    mask[:, :50] = True
    output = model(states, mask=mask)
    print(output.shape)