import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        B, C, H, W = x.size()
        xx = torch.linspace(0, 1, W, device=x.device).repeat(B, 1, H, 1).transpose(2, 3)
        yy = torch.linspace(0, 1, H, device=x.device).repeat(B, 1, W, 1)
        coords = torch.cat([xx, yy], dim=1)
        x = torch.cat([x, coords], dim=1)
        return self.conv(x)

class BoardCNNEncoder(nn.Module):
    def __init__(self, in_channels=112, out_dim=256):
        super(BoardCNNEncoder, self).__init__()

        base_resnet = models.resnet34(weights=None)
        base_resnet.conv1 = CoordConv(in_channels, 64, kernel_size=7, padding=3)
        base_resnet.fc = nn.Identity()
        self.backbone = nn.Sequential(
            base_resnet.conv1, base_resnet.bn1, base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4
        )
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        # x: [batch, 112, 8, 8]
        x = self.backbone(x) 
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
    def __init__(self, cnn_in_channels=112, action_size=1858,
                 state_embed_dim=256, action_embed_dim=256,
                 fusion_out_dim=256, transformer_d_model=256,
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
        # self.action_encoder = ActionEncoder(action_size=action_size, embed_dim=action_embed_dim)
        # self.fusion = StateActionFusion(state_dim=state_embed_dim, action_dim=action_embed_dim,
        #                                 out_dim=fusion_out_dim, fusion_method='concat')
        self.pos_encoder = PositionalEncoding(d_model=transformer_d_model, dropout=dropout, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(transformer_d_model, transformer_d_model)

    def forward(self, states, actions, mask=None):
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

        # 编码动作
        # actions = actions.view(batch_size * seq_len)
        # action_emb = self.action_encoder(actions)  # [batch*seq_len, action_embed_dim]
        # action_emb = action_emb.view(batch_size, seq_len, -1)

        # 融合状态和动作
        # token_embeddings = self.fusion(state_emb, action_emb)  # [batch, seq_len, fusion_out_dim]

        # 添加位置编码
        token_embeddings = self.pos_encoder(state_emb)

        # Transformer 编码
        transformer_output = self.transformer_encoder(
            token_embeddings, src_key_padding_mask=~mask
        )

        # 序列池化（聚合）
        if mask is not None:
            # False 表示有效位置
            valid_mask = (~mask).unsqueeze(-1).float()  # [batch, seq_len, 1]
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