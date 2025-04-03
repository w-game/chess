import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from player_encoder.dataset import MetaStyleDataset
from player_encoder.encoder import TransformerEncoder


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        """
        temperature: 温度系数（通常为 0.07）
        contrast_mode: 'all' 表示 anchor 来自整个 batch
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """
        features: [batch_size, embed_dim] 或 [batch_size, n_views, embed_dim]
        labels:   [batch_size]，每个样本的类别（例如玩家 ID）
        mask:     [batch_size, batch_size]，手动指定哪些是正样本（可选）

        如果 labels 和 mask 都没有给出，就退化为 SimCLR（无监督对比学习）
        """
        device = features.device

        if len(features.shape) < 3:
            features = features.unsqueeze(1)  # [batch, 1, dim]

        batch_size = features.shape[0]
        n_views = features.shape[1]
        # features = F.normalize(features, dim=2)

        # [batch * n_views, dim]
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)  # [batch, 1]
            if labels.shape[0] != batch_size:
                raise ValueError("Mismatch between labels and features")
            mask = torch.eq(labels, labels.T).float().to(device)  # [batch, batch]
        elif mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = n_views
        contrast_mask = mask.repeat(contrast_count, contrast_count)  # [batch*n, batch*n]

        anchor_feature = contrast_features
        anchor_count = contrast_count if self.contrast_mode == 'all' else 1

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_features.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask out self-contrast cases
        logits_mask = torch.ones_like(contrast_mask).fill_diagonal_(0)
        mask = contrast_mask * logits_mask

        # log-softmax
        exp_logits = torch.exp(logits.clamp(min=-20, max=20)) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # mean of log-likelihood over positive
        denom = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (denom + 1e-12)
        mean_log_prob_pos[denom == 0] = 0.0

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class EncoderTrainer:
    def __init__(self, train_loader, val_loader, test_loader, max_len=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.loss_fn = SupConLoss(temperature=0.07)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.encoder = TransformerEncoder(
            cnn_in_channels=224, action_size=1858,
            state_embed_dim=256, action_embed_dim=256, fusion_out_dim=256,
            transformer_d_model=256, num_heads=8, num_layers=4,
            dropout=0.1, max_seq_len=max_len
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)

    def unpack_batch(self, batch):
        return (
            batch['support_pos'].to(self.device).squeeze(0),
            batch['support_mask'].to(self.device).squeeze(0),
            batch['support_labels'].to(self.device).squeeze(0),
            batch['query_pos'].to(self.device).squeeze(0),
            batch['query_mask'].to(self.device).squeeze(0),
            batch['query_labels'].to(self.device).squeeze(0)
        )

    def get_prototypes(self, support_pos, support_mask, support_labels):
        support_z = self.encoder(support_pos, support_mask)
        N = support_labels.max().item() + 1
        prototypes = [
            support_z[support_labels == i].mean(dim=0)
            for i in range(N)
            if (support_labels == i).sum() > 0
        ]
        return torch.stack(prototypes)

    def proto_loss(self, query_pos, query_mask, prototypes, query_labels):
        query_z = self.encoder(query_pos, query_mask)
        logits = -torch.cdist(query_z, prototypes)
        return F.cross_entropy(logits, query_labels)

    @torch.no_grad()
    def val(self):
        total_loss = 0
        batch_count = 0
        for batch in self.val_loader:
            batch_count += 1

            support_pos, support_mask, support_labels, query_pos, query_mask, query_labels = self.unpack_batch(batch)
            prototypes = self.get_prototypes(support_pos, support_mask, support_labels)
            loss = self.proto_loss(query_pos, query_mask, prototypes, query_labels)

            total_loss += loss.item()

        return total_loss / batch_count

    def train(self, epochs=60, save_path="./models"):
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
                prototypes = self.get_prototypes(support_pos, support_mask, support_labels)
                loss = self.proto_loss(query_pos, query_mask, prototypes, query_labels)

                if torch.isnan(loss):
                    print("❌ Loss is NaN!")
                    print("support_labels:", support_labels.tolist())
                    print("query_labels:", query_labels.tolist())
                    exit()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                if (batch_count + 1) % 20 == 0:
                    print(f"  ├─ Batch {batch_count} Loss: {loss.item():.4f}")

            avg_loss = total_loss / batch_count
            avg_val_loss = self.val()

            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)

            print(f"✅ [Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

            if (epoch + 1) % 2 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.encoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'avg_val_loss': avg_val_loss
                }, f"{save_path}/player_encoder_{epoch + 1}.pt")
                print(f"📦 Model saved to {save_path}")

            if (epoch + 1) % 10 == 0:
                trainer.style_consistency_analysis(
                    num_trials=50,
                    support_size=5,
                    save_path="./style_consistency.png"
                )

        # 绘制 loss 曲线图
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
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
            d = torch.load(model_path)
            self.encoder.load_state_dict(d["model_state_dict"])
            self.optimizer.load_state_dict(d["optimizer_state_dict"])

            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = 0.0001
            print(f"load model from {model_path}")

    def style_consistency_analysis(self, num_trials=50, support_size=5, save_path="./style_consistency.png"):
        self.encoder.eval()
        results = defaultdict(list)

        with torch.no_grad():
            for trial in range(num_trials):
                # 从测试集随机选一个 batch（即一名玩家的数据）
                batch = next(iter(self.test_loader))
                support_pos, support_mask, support_labels, query_pos, query_mask, query_labels = self.unpack_batch(
                    batch)

                unique_ids = support_labels.unique()
                for player_id in unique_ids:
                    indices = (support_labels == player_id).nonzero(as_tuple=True)[0].tolist()
                    if len(indices) <= support_size:
                        continue

                    sampled_indices = random.sample(indices, support_size + 1)
                    support_idx = sampled_indices[:support_size]
                    query_idx = sampled_indices[-1:]

                    support_p = support_pos[support_idx]
                    support_m = support_mask[support_idx]
                    query_p = support_pos[query_idx]
                    query_m = support_mask[query_idx]

                    # 构造原型
                    support_z = self.encoder(support_p, support_m)
                    proto = support_z.mean(dim=0, keepdim=True)  # [1, D]

                    # 新棋谱编码
                    query_z = self.encoder(query_p, query_m)  # [1, D]
                    dist = torch.norm(query_z - proto, dim=1).item()
                    results[player_id.item()].append(dist)

        # 可视化柱状图
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

    train_dataset = MetaStyleDataset(load_dataset_file("train_players"), max_len=max_len)

    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4,
                              persistent_workers=True
                              )

    val_dataset = MetaStyleDataset(load_dataset_file("val_players"), max_len=max_len)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4,
                            persistent_workers=True
                            )

    test_dataset = MetaStyleDataset(load_dataset_file("test_players"), max_len=max_len)

    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4,
                            persistent_workers=True
                             )

    trainer = EncoderTrainer(train_loader, val_loader, test_loader, max_len=max_len)

    save_path = "./models/model_0"
    os.makedirs(save_path, exist_ok=True)

    trainer.load_model(f"{save_path}/player_encoder_6.pt")

    trainer.train(save_path=save_path)
