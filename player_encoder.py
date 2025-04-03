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


class EncoderTrainer:
    def __init__(self, train_loader, val_loader, test_loader, max_len=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

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

        for i in range(B):
            support_z = self.encoder(support_pos[i], support_mask[i])
            query_z = self.encoder(query_pos[i], query_mask[i])

            N = support_labels[i].max().item() + 1
            prototypes = []
            for j in range(N):
                mask = (support_labels[i] == j)
                if mask.sum() > 0:
                    prototypes.append(support_z[mask].mean(dim=0))
            prototypes = torch.stack(prototypes)

            logits = -torch.cdist(query_z, prototypes)
            loss = F.cross_entropy(logits, query_labels[i])
            pred = torch.argmax(logits, dim=1)
            correct = (pred == query_labels[i]).sum().item()

            total_loss += loss
            total_correct += correct
            total_total += query_labels[i].size(0)

        return total_loss / B, total_correct, total_total

    @torch.no_grad()
    def val(self):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        for batch in self.val_loader:
            batch_count += 1

            support_pos, support_mask, support_labels, query_pos, query_mask, query_labels = self.unpack_batch(batch)
            loss, correct, total = self.task_proto_loss_and_acc(
                support_pos, support_mask, support_labels,
                query_pos, query_mask, query_labels
            )

            total_loss += loss
            total_correct += correct
            total_samples += total

        avg_loss = total_loss / batch_count
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

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
                loss, _, _ = self.task_proto_loss_and_acc(
                    support_pos, support_mask, support_labels,
                    query_pos, query_mask, query_labels
                )

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
            avg_val_loss, val_acc = self.val()

            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)

            print(f"✅ [Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

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
    num_workers = 8
    batch_size = 4  # or 8 depending on memory

    train_dataset = MetaStyleDataset(load_dataset_file("train_players"), max_len=max_len)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers,
                              persistent_workers=True
                              )

    val_dataset = MetaStyleDataset(load_dataset_file("val_players"), max_len=max_len)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=num_workers,
                            persistent_workers=True
                            )

    test_dataset = MetaStyleDataset(load_dataset_file("test_players"), max_len=max_len)

    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=num_workers,
                            persistent_workers=True
                             )

    trainer = EncoderTrainer(train_loader, val_loader, test_loader, max_len=max_len)

    save_path = "./models/model_1"
    os.makedirs(save_path, exist_ok=True)

    trainer.load_model(f"{save_path}/player_encoder_0.pt")

    trainer.train(save_path=save_path)
