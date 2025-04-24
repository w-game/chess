import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from player_encoder_infine.dataset import MetaStyleDataset
from player_encoder_infine.encoder import TransformerEncoder


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss as in: https://arxiv.org/pdf/2004.11362.pdf"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [N, D] embedding vectors
        labels: [N] with integer labels
        """
        device = features.device
        labels = labels.contiguous().view(-1, 1)  # [N, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [N, N]

        contrast_count = 1
        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast cases
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss

def compute_similarity_percent_cosine(query_z, prototypes, labels):
    """
    使用余弦相似度，将 (query, prototype) 的相似度映射到 [0, 100%]。
    - 对每个 query，先找出其对应的原型 prototype[labels[i]]。
    - 计算余弦相似度 cos_sim ∈ [-1, 1]。
    - 再把 [-1, 1] 线性映射到 [0, 1]，最后乘以 100%。

    返回:
      similarity_percent: shape (N,)，每个 query 样本的相似度百分比
      avg_similarity: 平均相似度百分比 (float)
    """

    chosen_proto = prototypes[labels]  # (N, d)
    # 计算余弦相似度 (N,)
    cos_sims = F.cosine_similarity(query_z, chosen_proto, dim=1)

    # 如果你确信 cos_sims 都是 >= 0，也可直接 cos_sims*100。
    # 通用做法: 把 [-1, 1] -> [0, 1]，再 -> [0, 100]
    similarity_percent = (cos_sims + 1.0) / 2.0 * 100.0

    # 平均值
    avg_similarity = similarity_percent.mean().item()
    return similarity_percent, avg_similarity

def compute_proto_similarity(prototypes):
    """
    输入: prototypes: Tensor [N, D]
    返回: 原型之间的平均余弦相似度（不含对角线）
    """
    norms = F.normalize(prototypes, dim=1)
    sim_matrix = norms @ norms.T
    N = sim_matrix.size(0)
    mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    return sim_matrix[mask].mean().item()


class EncoderTrainer:
    def __init__(self, train_loader, val_loader, test_loader=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.encoder = TransformerEncoder(cnn_in_channels=224, state_embed_dim=256, transformer_d_model=256,
                                          num_heads=8, num_layers=3, dropout=0.1).to(self.device)
        model_params = self.encoder.parameters()
        self.optimizer = torch.optim.AdamW(
            model_params,
            lr=1e-4,
            weight_decay=1e-4
        )
        self.supcon_loss = SupConLoss(temperature=0.07)
        # 新建一个 SGD 优化器（你可以指定新的lr、momentum等）
        # self.optimizer = torch.optim.SGD(
        #     model_params,
        #     lr=1e-4,
        #     momentum=0.9,
        #     weight_decay=1e-4
        # )
            
    @torch.no_grad()
    def val(self):
        self.encoder.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_score_cosine = 0.0
        total_proto_sim = 0.0

        with torch.autocast(device_type="cuda"):
            for batch in self.val_loader:
                # 1) Move to device
                query_games   = batch['query']['games'].to(self.device)    # [B, N*Q, T, C, H, W]
                query_masks   = batch['query']['masks'].to(self.device)    # [B, N*Q, T]

                # 2) Build prototypes from support set
                prototypes, _ = self.build_prototypes_from_support(batch)   # prototypes: [B, N, D]
                # 3) Encode queries
                _, query_z = self.encoder(query_games, query_masks)        # query_z: [B, N*Q, D]

                B, N, D = prototypes.shape
                Q = query_z.size(1) // N  # 动态计算每类 query 数量

                # 4) 计算 logits: [B, N*Q, N] → [B*N*Q, N]
                logits = F.cosine_similarity(
                    query_z.unsqueeze(2),        # [B, N*Q, 1, D]
                    prototypes.unsqueeze(1),     # [B, 1, N, D]
                    dim=-1
                ).view(-1, N)

                # 5) 构造 targets: [0,0…0,1,1…1,…,N-1×Q] 重复 B 次 → [B*N*Q]
                labels = torch.arange(N, device=self.device).repeat_interleave(Q)
                targets = labels.unsqueeze(0).expand(B, -1).reshape(-1)

                # 6) 计算 CE loss 和 accuracy
                loss = F.cross_entropy(logits, targets)
                preds = logits.argmax(dim=1)
                correct = (preds == targets).sum().item()

                total_loss   += loss.item()
                total_correct+= correct
                total_samples+= targets.numel()

                # 7) 计算“风格相似度”指标（Cosine %）
                #    flatten 后 batch*K*Q 样本
                proto_rep = prototypes.unsqueeze(1).repeat(1, Q, 1, 1) \
                                                .view(-1, D)  # [B*N*Q, D]
                query_rep = query_z.view(-1, D)                  # [B*N*Q, D]
                _, avg_cos = compute_similarity_percent_cosine(query_rep, proto_rep, targets)
                total_score_cosine += avg_cos

                # 8) 计算 prototype 之间的平均余弦相似度
                #    对每个样本（batch 中的每个 N 组原型）都算一次，再求和
                for b in range(B):
                    total_proto_sim += compute_proto_similarity(prototypes[b])

        num_batches = len(self.val_loader)
        avg_loss = total_loss   / num_batches
        avg_acc  = total_correct/ total_samples
        avg_score_cosine = total_score_cosine / num_batches
        # 平均到每个“样本组”（batch×N）
        avg_proto_sim = total_proto_sim / (num_batches * B)

        print(
            f"🔴 [Validation] "
            f"Loss: {avg_loss:.4f}, "
            f"Acc: {avg_acc:.4f}, "
            f"Cosine Similarity: {avg_score_cosine:.2f}%, "
            f"Prototype Similarity: {avg_proto_sim:.4f}"
        )
        return avg_loss

    def build_prototypes_from_support(self, batch):
        prototypes = []
        support_games = batch['support']['games'].to(self.device) # [B, N * K, T, C, H, W]
        support_masks = batch['support']['masks'].to(self.device) # [B, N * K, T]

        contrastive_z, support_z = self.encoder(support_games, support_masks) # [B, N * K, d]

        support_z = support_z.view(support_z.size(0), N, K, -1)  # [B, N, K, d]
        prototypes = support_z.mean(dim=2)  # [B, N, d]
        return prototypes, contrastive_z
    
    def train_one_epoch(self, epoch, dataloader, scaler):
        total_loss = 0
        batch_count = 0

        with torch.autocast(device_type="cuda"):
            for idx, batch in enumerate(dataloader):
                prototypes, contrastive_z = self.build_prototypes_from_support(batch)

                B, N_K, D = contrastive_z.size()  # [B, N * K, D]
                supcon_labels = torch.arange(B, device=self.device).repeat_interleave(N_K)  # [B * N * K]
                sup_loss = self.supcon_loss(contrastive_z.view(B * N_K, D), supcon_labels)

                query_games = batch['query']['games'].to(self.device)  # [B, N * Q, T, C, H, W]
                query_masks = batch['query']['masks'].to(self.device)  # [B, N * Q, T]

                contrastive_z, query_z = self.encoder(query_games, query_masks)  # [B, N * Q, d]

                prototypes = F.normalize(prototypes, dim=-1)  # [B, N, d]
                query_z = F.normalize(query_z, dim=-1)  # [B, N * Q, d]

                prototypes = prototypes.unsqueeze(1)         # [B, 1, N, D]
                query_z_flat = query_z.unsqueeze(2)          # [B, N*Q, 1, D]
                logits = F.cosine_similarity(query_z_flat, prototypes, dim=-1) / 0.05  # [B, N*Q, N]
                logits = logits.view(-1, N)  # [B * N * Q, N]

                targets = torch.arange(N, device=self.device, dtype=torch.long).repeat_interleave(Q)
                targets = targets.unsqueeze(0).repeat(B, 1).view(-1)

                loss_ce = F.cross_entropy(logits, targets)
                loss = loss_ce + sup_loss * 0.1

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                print(f"🔵 [Batch {idx + 1}/{len(dataloader)}] Loss: {loss.item():.4f} | CE: {loss_ce:.4f} | SupCon: {sup_loss.item():.4f}")

                total_loss += loss.item()
                batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"🔵 [Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}")

        return avg_loss
        
    def plot_loss_curve(self, train_losses, val_losses, save_path):
        train_losses_cpu = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in train_losses]
        val_losses_cpu = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in val_losses]

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses_cpu, label="Train Loss")
        plt.plot(val_losses_cpu, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/loss_curve.png")
        print(f"📉 Loss curve saved to {save_path}/loss_curve.png")

    def train(self, epochs=60, save_path="./models", model_idx=0):
        scaler = torch.GradScaler('cuda')
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.encoder.train()

            print(f"\n🟢 Epoch {epoch + 1}/{epochs} started...")

            avg_loss = self.train_one_epoch(epoch, self.train_loader, scaler)
            # avg_val_loss = self.val()
            
            train_losses.append(avg_loss)
            # val_losses.append(avg_val_loss)

            if (epoch + 1) % 2 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.encoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    # 'avg_val_loss': avg_val_loss
                }, f"{save_path}/player_encoder_{epoch + 1 + model_idx}.pt")
                print(f"📦 Model saved to {save_path}")

            # self.plot_loss_curve(train_losses, val_losses, save_path)

    def load_model(self, model_path):
        if os.path.exists(model_path):
            d = torch.load(model_path, weights_only=True)
            self.encoder.load_state_dict(d["model_state_dict"])
            # self.optimizer.load_state_dict(d["optimizer_state_dict"])

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1e-5
            print(f"load model from {model_path}")

def load_dataset_file(path):
    with open(f"chess_data_parse/{path}.json", "r", encoding="utf-8") as f:
        return json.load(f)

from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    # 这里的 batch 是一个列表，包含了多个样本
    # 你可以根据需要进行处理，比如将它们拼接成一个大的 tensor

    splits = ('support', 'query')
    batched = {}

    batch_size = len(batch)

    for split in splits:
        # 收集这一 split 下所有样本的 game/mask
        all_games = []
        all_masks = []
        for sample in batch:
            all_games.extend(sample[split]['games']) # [N, T, C, H, W]
            all_masks.extend(sample[split]['masks'])

        # 对所有序列进行统一 padding
        # padded_games: [batch_size*K_or_Q, T_max, ...]
        padded_games = pad_sequence(all_games, batch_first=True)
        # padded_masks: [batch_size*K_or_Q, T_max]
        padded_masks = pad_sequence(all_masks, batch_first=True, padding_value=True)

        # [batch_size, K_or_Q, T_max, ...]
        padded_games = padded_games.view(batch_size, padded_games.size(0) // batch_size, *padded_games.size()[1:])
        # [batch_size, K_or_Q, T_max]   
        padded_masks = padded_masks.view(batch_size, padded_masks.size(0) // batch_size, *padded_masks.size()[1:])

        # 将拼接好的 tensor 放入 batched 字典中
        batched[split] = {
            'games': padded_games,
            'masks': padded_masks,
            'len': padded_games.size(1)  # K_or_Q
        }

    return batched
    
if __name__ == '__main__':
    N, K, Q = 5, 5, 5
    train_dataset = MetaStyleDataset(load_dataset_file("train_players"), 1000)
    val_dataset = MetaStyleDataset(load_dataset_file("val_players"), 150)

    data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    trainer = EncoderTrainer(data_loader, val_data_loader)

    save_path = "./models/model_2025_04_024_N_5_K_5_Q_5_supconlosss"
    model_idx = 0
    os.makedirs(save_path, exist_ok=True)

    trainer.load_model(f"{save_path}/player_encoder_{model_idx}.pt")

    trainer.train(save_path=save_path, model_idx=model_idx)
