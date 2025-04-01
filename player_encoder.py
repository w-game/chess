import torch
import torch.nn as nn

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


from player_encoder.encoder import TransformerEncoder
from player_encoder.dataset import PlayerDataset

from sklearn.decomposition import PCA
import json

import os

from torch.utils.data import DataLoader
import multiprocessing as mp
import torch.nn.functional as F


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

def worker_init_fn(worker_id):
    import os
    # 确保每个 worker 都允许重复加载 OpenMP 运行时（如果你必须这么做）
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 限制每个 worker 使用单线程，防止多线程冲突
    os.environ["OMP_NUM_THREADS"] = "1"


def flatten_collate(batch):
    flat_batch = [x for sample in batch for x in sample]  # flatten 所有片段
    # 现在 flat_batch 是 [(state, action, mask, player_id), ...]，可以正常处理了

    max_len = max(x[0].shape[0] for x in flat_batch)  # x[0] 是 state: [T, 112, 8, 8]
    max_len = min(100, max_len)

    padded_states, padded_actions, padded_masks, player_ids = [], [], [], []
    for state, action, mask, player_id in flat_batch:
        pad_len = max_len - state.shape[0]
        if pad_len > 0:
            state = F.pad(state, (0, 0, 0, 0, 0, 0, 0, pad_len))  # pad T dim
            action = F.pad(action, (0, pad_len))
            mask = F.pad(mask, (0, pad_len), value=False)

        if state.shape[0] > max_len:
            state = state[:max_len]
            action = action[:max_len]
            mask = mask[:max_len]

        padded_states.append(state)
        padded_actions.append(action)
        padded_masks.append(mask)
        player_ids.append(player_id)

    return (
        torch.stack(padded_states),     # [B, T, 112, 8, 8]
        torch.stack(padded_actions),    # [B, T]
        torch.stack(padded_masks),      # [B, T]
        torch.tensor(player_ids)        # [B]
    )


class EncoderTrainer:
    def __init__(self, train_loader, val_loader, test_loader, max_len=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.loss_fn = SupConLoss(temperature=0.07)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.encoder = TransformerEncoder(
            cnn_in_channels=112, action_size=1858,
            state_embed_dim=256, action_embed_dim=256, fusion_out_dim=256,
            transformer_d_model=256, num_heads=8, num_layers=4,
            dropout=0.1, max_seq_len=max_len
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)

    @torch.no_grad()
    def val(self):
        total_loss = 0
        batch_count = 0
        for states, actions, masks, player_ids in self.val_loader:
            batch_count += 1
            style_vector = self.encoder(states.to(self.device), actions.to(self.device), masks.to(self.device))
            loss = self.loss_fn(style_vector, labels=player_ids.to(self.device))

            total_loss += loss.item()

        return total_loss / batch_count

    def train(self, epochs=60, save_path="./models"):
        for epoch in range(epochs):
            self.encoder.train()
            total_loss = 0
            batch_count = 0
    
            print(f"\n🟢 Epoch {epoch+1}/{epochs} started...")
    
            for states, actions, masks, player_ids in self.train_loader:
                batch_count += 1
                style_vector = self.encoder(states.to(self.device), actions.to(self.device), masks.to(self.device))
                loss = self.loss_fn(style_vector, labels=player_ids.to(self.device))
    
                if torch.isnan(loss):
                    print("❌ Loss is NaN!")
                    print("player_ids:", player_ids.tolist())
                    print("style_vector:", style_vector)
                    print("Any NaN in style_vector?", torch.isnan(style_vector).any().item())
                    print("Any Inf in style_vector?", torch.isinf(style_vector).any().item())
                    exit()
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                total_loss += loss.item()
                print(f"  ├─ Batch {batch_count} Loss: {loss.item():.4f}")

            avg_val_loss = self.val()

            avg_loss = total_loss / batch_count
            print(f"✅ [Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss}")

            if (epoch + 1) % 2 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.encoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'avg_loss': avg_loss,
                }, f"{save_path}/player_encoder_{epoch + 1}.pt")
                print(f"📦 Model saved to {save_path}")
            
            if (epoch + 1) % 10 == 0:
                self.visualize_embeddings("./demo.png", "./demo_tsne.png")
            
    def load_model(self, model_path):
        if os.path.exists(model_path):
            d = torch.load(model_path)
            self.encoder.load_state_dict(d["model_state_dict"])
            self.optimizer.load_state_dict(d["optimizer_state_dict"])

            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = 0.0001
            print(f"load model from {model_path}")

    def visualize_embeddings_tsne(self, embeddings, labels):
        print("🔍 Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        print(embeddings_2d.shape)  # (N, 2)

        # 可视化 scatter plot
        plt.figure(figsize=(16, 8))
        num_classes = len(set(labels))
        palette = sns.husl_palette(num_classes)

        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=labels,
            palette=palette,
            legend="full",
            s=50,
            alpha=0.6
        )

        plt.title("t-SNE Visualization of Style Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Player ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        return plt\

    def visualize_embeddings_pca(self, embeddings, labels):
        # self.encoder.eval()
        print("🔍 Running PCA...")
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)

        # jitter = np.random.normal(0, 0.1, embeddings_2d.shape)
        # embeddings_2d += jitter
    
        # 可视化 scatter plot
        plt.figure(figsize=(16, 8))
        num_classes = len(set(labels))
        palette = sns.husl_palette(num_classes)
    
        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=labels,
            palette=palette,
            legend="full",
            s=50,
            alpha=0.6
        )
    
        plt.title("PCA Visualization of Style Embeddings")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.legend(title="Player ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
        return plt
            
    def visualize_embeddings(self, pca_path, tsne_path):
        with torch.no_grad():
            for states, actions, masks, player_ids in self.test_loader:
                style_vector = self.encoder(states.to(self.device), actions.to(self.device), masks.to(self.device))
                loss = self.loss_fn(style_vector, labels=player_ids.to(self.device))
                print(f"{loss.item()}")
                break
                # if loss.item() < 3.6:
                #     break

        labels = player_ids.numpy().tolist()  # list
        embeddings = style_vector.cpu().numpy()

        print(f"📊 Visualizing {len(labels)} samples across {len(set(labels))} players")
        plt = self.visualize_embeddings_pca(embeddings, labels)
        
        plt.savefig(pca_path, dpi=300)
        
        plt = self.visualize_embeddings_tsne(embeddings, labels)
        plt.savefig(tsne_path, dpi=300)

def load_dataset_file(path):
    with open(f"chess_data_parse/{path}.json", "r", encoding="utf-8") as f:
        return json.load(f)
        
if __name__ == '__main__':
    max_len = 100

    train_dataset = PlayerDataset(load_dataset_file("train_players"), games_per_player=8, max_len=max_len)

    train_loader = DataLoader(train_dataset,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=flatten_collate,
                             num_workers=16,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn
                             )

    val_dataset = PlayerDataset(load_dataset_file("val_players"), games_per_player=8, max_len=max_len)

    val_loader = DataLoader(val_dataset,
                              batch_size=16,
                              shuffle=True,
                              collate_fn=flatten_collate,
                              num_workers=16,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn
                              )

    test_dataset = PlayerDataset(load_dataset_file("test_players"), games_per_player=8, max_len=max_len)

    test_loader = DataLoader(test_dataset,
                            batch_size=16,
                            shuffle=True,
                            collate_fn=flatten_collate,
                            num_workers=16,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn
                            )

    trainer = EncoderTrainer(train_loader, val_loader, test_loader, max_len=max_len)

    save_path = "./models/model_0"
    os.makedirs(save_path, exist_ok=True)

    trainer.load_model(f"{save_path}/player_encoder_1.pt")

    trainer.train(save_path=save_path)
