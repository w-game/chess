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
    def __init__(self, train_loader, val_loader, test_loader=None, max_len=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.encoder = TransformerEncoder(cnn_in_channels=224, state_embed_dim=256, transformer_d_model=256,
                                          num_heads=8, num_layers=3, dropout=0.1, max_seq_len=max_len).to(self.device)
        # self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
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

    def unpack_batch(self, batch):
        return (
            batch['support_pos'].to(self.device),
            batch['support_mask'].to(self.device),
            batch['support_labels'].to(self.device),
            batch['query_pos'].to(self.device),
            batch['query_mask'].to(self.device),
            batch['query_labels'].to(self.device)
        )

    def task_proto_loss_and_acc(self, support_pos, support_mask, support_labels, query_pos, query_mask, query_labels):
        B = support_pos.shape[0]
        total_loss = 0
        total_correct = 0
        total_total = 0
        total_score_cosine = 0
        total_similarity = 0

        for i in range(B):
            _, support_z = self.encoder(support_pos[i], support_mask[i])

            N = support_labels[i].max().item() + 1
            D = support_z.shape[-1]
            # Initialize tensors for summing embeddings and counting occurrences per class
            prototypes_sum = torch.zeros(N, D, device=support_z.device)
            counts = torch.zeros(N, device=support_z.device)
            
            # Compute the sum of embeddings for each class in a vectorized manner
            prototypes_sum = prototypes_sum.scatter_add(0, support_labels[i].unsqueeze(1).expand(-1, D), support_z)
            # Count the number of samples for each class
            counts = counts.scatter_add(0, support_labels[i], torch.ones_like(support_labels[i], dtype=torch.float))
            
            # Compute the mean (prototype) for each class
            prototypes = prototypes_sum / counts.unsqueeze(1)

            contrastive_z, query_z = self.encoder(query_pos[i], query_mask[i])

            supcon_loss = self.supcon_loss(contrastive_z, query_labels[i])
            logits = -torch.cdist(query_z, prototypes)
            loss = F.cross_entropy(logits, query_labels[i])
            pred = torch.argmax(logits, dim=1)
            correct = (pred == query_labels[i]).sum().item()

            total_loss += loss + 0.1 * supcon_loss
            total_correct += correct
            total_total += query_labels[i].size(0)
            total_similarity += compute_proto_similarity(prototypes)

            similarity_percent, avg_similarity_cosine = compute_similarity_percent_cosine(query_z, prototypes, query_labels[i])
            total_score_cosine += avg_similarity_cosine

            del support_z, query_z, logits, prototypes
            torch.cuda.empty_cache()

        return total_loss / B, total_correct, total_total, total_score_cosine / B, total_similarity / B

    @torch.no_grad()
    def val(self):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        total_score_cosine = 0
        total_similarity = 0
        for batch in self.val_loader:
            batch_count += 1

            support_pos, support_mask, support_labels, query_pos, query_mask, query_labels = self.unpack_batch(batch)
            with torch.autocast(device_type="cuda"):
                loss, correct, total, score_cosin, similarity = self.task_proto_loss_and_acc(
                    support_pos, support_mask, support_labels,
                    query_pos, query_mask, query_labels
                )

            total_loss += loss.item()
            total_correct += correct
            total_samples += total
            total_score_cosine += score_cosin
            total_similarity += similarity

        avg_loss = total_loss / batch_count
        accuracy = total_correct / total_samples
        avg_score_cosine = total_score_cosine / batch_count
        return avg_loss, accuracy, avg_score_cosine, total_similarity / batch_count

    def train(self, epochs=60, save_path="./models", model_idx=0):
        scaler = torch.GradScaler('cuda')
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.encoder.train()
            total_loss = 0
            batch_count = 0

            print(f"\n🟢 Epoch {epoch + 1}/{epochs} started...")

            for batch in self.train_loader:
                batch_count += 1

                support_pos, support_mask, support_labels, query_pos, query_mask, query_labels = self.unpack_batch(
                    batch)
                with torch.autocast(device_type="cuda"):
                    loss, _, _, _, _ = self.task_proto_loss_and_acc(
                        support_pos, support_mask, support_labels,
                        query_pos, query_mask, query_labels
                    )

                if torch.isnan(loss):
                    print("❌ Loss is NaN!")
                    print("support_labels:", support_labels.tolist())
                    print("query_labels:", query_labels.tolist())
                    exit()

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                total_loss += loss.detach().item()
                if (batch_count + 1) % 20 == 0:
                    print(f"  ├─ Batch {batch_count} Loss: {loss.item():.4f}")

                del support_pos, support_mask, support_labels
                del query_pos, query_mask, query_labels
                del batch, loss
                torch.cuda.empty_cache()

            avg_loss = total_loss / batch_count
            avg_val_loss, val_acc, val_score_cosin, val_similarity = self.val()

            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)

            print(f"✅ [Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val Cosine Score: {val_score_cosin:.4f}, Val Similarity: {val_similarity:.4f}")

            if (epoch + 1) % 2 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.encoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'avg_val_loss': avg_val_loss
                }, f"{save_path}/player_encoder_{epoch + 1 + model_idx}.pt")
                print(f"📦 Model saved to {save_path}")

            # if (epoch + 1) % 10 == 0:
            #     trainer.style_consistency_analysis(
            #         num_trials=50,
            #         support_size=5,
            #         save_path="./style_consistency.png"
            #     )

        # 绘制 loss 曲线图
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

    def load_model(self, model_path):
        if os.path.exists(model_path):
            d = torch.load(model_path, weights_only=True)
            self.encoder.load_state_dict(d["model_state_dict"])
            # self.optimizer.load_state_dict(d["optimizer_state_dict"])

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1e-5
            print(f"load model from {model_path}")

    def style_consistency_analysis(self, num_trials=50, support_size=5, save_path="./style_consistency.png"):
        self.encoder.eval()
        results = defaultdict(list)

        with torch.no_grad():
            for trial in range(num_trials):
                # 从测试集中随机选取一个 batch（包含多个任务，每个任务代表一名玩家的数据）
                batch = next(iter(self.test_loader))
                support_pos, support_mask, support_labels, query_pos, query_mask, query_labels = self.unpack_batch(batch)

                # 针对每个任务（batch 内的每个样本）分别处理
                for i in range(support_pos.size(0)):
                    task_support_labels = support_labels[i]  # 形状: (num_support,)
                    task_query_labels = query_labels[i]       # 形状: (num_query,)

                    for player_id in task_support_labels.unique():
                        # 从支持集中获取对应玩家的样本索引，至少需要 support_size 个样本
                        support_indices = (task_support_labels == player_id).nonzero(as_tuple=True)[0].tolist()
                        if len(support_indices) < support_size:
                            continue
                        sampled_support_indices = random.sample(support_indices, support_size)

                        support_p = support_pos[i][sampled_support_indices]  # 形状: (support_size, seq_len, C, H, W)
                        support_m = support_mask[i][sampled_support_indices]

                        # 构造原型
                        _, support_z = self.encoder(support_p, support_m)
                        proto = support_z.mean(dim=0, keepdim=True)  # 形状: (1, D)

                        # 从 query 集中获取该玩家的所有样本索引
                        query_indices = (task_query_labels == player_id).nonzero(as_tuple=True)[0].tolist()
                        if not query_indices:
                            continue

                        query_p = query_pos[i][query_indices]  # 形状: (num_query_samples, seq_len, C, H, W)
                        query_m = query_mask[i][query_indices]
                        _, query_z = self.encoder(query_p, query_m)  # 形状: (num_query_samples, D)

                        # 计算每个 query 样本与原型之间的距离
                        distances = torch.norm(query_z - proto, dim=1)
                        avg_distance = distances.mean().item()
                        results[player_id.item()].append(avg_distance)

        # 绘制柱状图：每个玩家所有 trial 的平均距离再次取平均
        plt.figure(figsize=(10, 5))
        sorted_ids = sorted(results.keys())
        avg_dists = [sum(results[k]) / len(results[k]) for k in sorted_ids]
        plt.bar([f"Player {k}" for k in sorted_ids], avg_dists)
        plt.ylabel("Avg Distance to Prototype")
        plt.title(f"Style Consistency Check (Each player: {num_trials} trials, Support size: {support_size})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📊 Style consistency chart saved to {save_path}")


def load_dataset_file(path):
    with open(f"chess_data_parse/{path}.json", "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == '__main__':
    max_len = 100
    num_workers = 4
    batch_size = 4  # or 8 depending on memory

    train_dataset = MetaStyleDataset(load_dataset_file("train_players"), 1000, N=3, max_len=max_len)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=False,
                              num_workers=num_workers,
                              persistent_workers=True
                              )

    val_dataset = MetaStyleDataset(load_dataset_file("val_players"), 150, N=3, max_len=max_len)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=False,
                            num_workers=num_workers,
                            persistent_workers=True
                            )

    # test_dataset = MetaStyleDataset(load_dataset_file("test_players"), 150, max_len=max_len)

    # test_loader = DataLoader(test_dataset,
    #                         batch_size=batch_size,
    #                         shuffle=True,
    #                         pin_memory=False,
    #                         num_workers=num_workers,s
    #                         persistent_workers=True
    #                          )

    trainer = EncoderTrainer(train_loader, val_loader, max_len=max_len)

    save_path = "./models/model_2025_04_07_N_5_K_5_Q_5_supconlosss"
    model_idx = 0
    os.makedirs(save_path, exist_ok=True)

    trainer.load_model(f"{save_path}/player_encoder_{model_idx}.pt")

    trainer.train(save_path=save_path, model_idx=model_idx)
