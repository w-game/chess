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
        total_loss = 0
        total_correct = 0
        total_total = 0
        total_score_cosine = 0
        total_similarity = 0

        for idx in range(len(self.val_loader)):
            data = self.val_loader.__getitem__(idx)
            with torch.autocast(device_type="cuda"):
                prototypes, labels = self.build_prototypes_from_support(data)
                all_query_zs = []
                all_targets = []

                for id, player_data in data.items():
                    query_games = player_data['query']['games']
                    query_masks = player_data['query']['masks']
                    for game, mask in zip(query_games, query_masks):
                        game = game.to(self.device)
                        mask = mask.to(self.device)
                        _, query_z = self.encoder(game)
                        all_query_zs.append(query_z.squeeze(0))
                        all_targets.append(id)
                query_zs = torch.stack(all_query_zs, dim=0)
                targets = torch.tensor(all_targets, dtype=torch.long, device=self.device)
                logits = -torch.cdist(query_zs, prototypes).to(torch.float32)
                loss = F.cross_entropy(logits, targets)
                pred = torch.argmax(logits, dim=1)
                correct = (pred == targets).sum().item()
                total_loss += loss.item()
                total_correct += correct
                total_total += targets.size(0)
                total_similarity += compute_proto_similarity(prototypes)
                similarity_percent, avg_similarity_cosine = compute_similarity_percent_cosine(query_zs, prototypes, targets)
                total_score_cosine += avg_similarity_cosine
                del query_zs, logits, prototypes
                torch.cuda.empty_cache()
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_total
        avg_score_cosine = total_score_cosine / len(self.val_loader)
        avg_similarity = total_similarity / len(self.val_loader)
        print(f"🔴 [Validation] Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Cosine Similarity: {avg_score_cosine:.4f}, Prototype Similarity: {avg_similarity:.4f}")

        return avg_loss

    def build_prototypes_from_support(self, data):
        prototypes = []
        labels = []
        for id, player_data in data.items():
            support_games = player_data['support']['games']
            support_masks = player_data['support']['masks']

            prototype_embeddings = []
            for game, mask in zip(support_games, support_masks):
                game = game.to(self.device)
                mask = mask.to(self.device)
                _, support_z = self.encoder(game)
                support_z = support_z.squeeze(0)  # [1, d] -> [d]
                prototype_embeddings.append(support_z)

            prototype = torch.stack(prototype_embeddings, dim=0).mean(dim=0)
            prototypes.append(prototype)
            labels.append(torch.tensor(id, device=self.device))
        return torch.stack(prototypes), torch.stack(labels)
    
    def train_one_epoch(self, epoch, dataloader, scaler):
        total_loss = 0
        batch_count = 0

        loss_ce = 0
        loss_supcon = 0
        for idx in range(len(dataloader)):
            data = dataloader.__getitem__(idx)
            with torch.autocast(device_type="cuda"):
                prototypes, _ = self.build_prototypes_from_support(data)

                all_query_zs = []
                all_contrastive_zs = []
                all_targets = []

                for id, player_data in data.items():
                    query_games = player_data['query']['games']
                    query_masks = player_data['query']['masks']
                    for game, mask in zip(query_games, query_masks):
                        game = game.to(self.device)
                        mask = mask.to(self.device)
                        contrastive_z, query_z = self.encoder(game)
                        all_query_zs.append(query_z.squeeze(0))  # [d]
                        all_contrastive_zs.append(contrastive_z.squeeze(0))
                        all_targets.append(id)

                query_zs = torch.stack(all_query_zs, dim=0)  # [Q_total, d]
                contrastive_zs = torch.stack(all_contrastive_zs, dim=0)
                targets = torch.tensor(all_targets, dtype=torch.long, device=self.device)  # [Q_total]

                logits = -torch.cdist(query_zs, prototypes).to(torch.float32)  # [Q_total, N]
                loss_ce += F.cross_entropy(logits, targets)
                loss_supcon += self.supcon_loss(F.normalize(contrastive_zs, dim=-1), targets)

                if (idx + 1) % 4 == 0:
                    loss_ce /= 4
                    loss_supcon /= 4
                    mean_loss = loss_ce + loss_supcon * 0.1
                    self.optimizer.zero_grad()
                    scaler.scale(mean_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    print(f"🔵 [Player {idx + 1}] Loss: {mean_loss.item():.4f} | CE: {loss_ce:.4f} | SupCon: {loss_supcon:.4f}")

                    total_loss += mean_loss.item()
                    batch_count += 1
                    loss_ce = 0
                    loss_supcon = 0

        
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
            avg_val_loss = self.val()
            
            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)

            if (epoch + 1) % 2 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.encoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'avg_val_loss': avg_val_loss
                }, f"{save_path}/player_encoder_{epoch + 1 + model_idx}.pt")
                print(f"📦 Model saved to {save_path}")

            self.plot_loss_curve(train_losses, val_losses, save_path)

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
    
if __name__ == '__main__':
    train_dataset = MetaStyleDataset(load_dataset_file("train_players"), 1000)
    val_dataset = MetaStyleDataset(load_dataset_file("val_players"), 150)

    trainer = EncoderTrainer(train_dataset, val_dataset)

    save_path = "./models/model_2025_04_022_N_5_K_5_Q_5_supconlosss"
    model_idx = 0
    os.makedirs(save_path, exist_ok=True)

    trainer.load_model(f"{save_path}/player_encoder_{model_idx}.pt")

    trainer.train(save_path=save_path, model_idx=model_idx)
