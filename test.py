import json
import torch

from player_encoder.dataset import MetaStyleDataset
from player_encoder.encoder import TransformerEncoder

from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

def unpack_batch(batch):
        return (
            batch['support_pos'].to(device),
            batch['support_mask'].to(device),
            batch['support_labels'].to(device),
            batch['query_pos'].to(device),
            batch['query_mask'].to(device),
            batch['query_labels'].to(device)
        )

def plot_data(within_players, between_players, title):
    plt.figure(figsize=(6, 4))
    plt.hist(within_players, bins=30, density=True, alpha=0.5, color='C0', label="within players")
    plt.hist(between_players, bins=30, density=True, alpha=0.5, color='C1', label="between players")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Similarity Distribution - Training Set")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    max_len = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = TransformerEncoder(cnn_in_channels=224, state_embed_dim=256, transformer_d_model=256,
                                            num_heads=8, num_layers=3, dropout=0.1, max_seq_len=max_len).to(device)

    d = torch.load('./models/trained_model/player_encoder_60.pt')
    encoder.load_state_dict(d["model_state_dict"])

    def load_dataset_file(path):
        with open(f"chess_data_parse/{path}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    test_dataset = MetaStyleDataset(load_dataset_file("test_players"), 150, max_len=max_len)

    test_loader = DataLoader(test_dataset,
                            batch_size=4,
                            shuffle=True,
                            pin_memory=False,
                            num_workers=4,
                            persistent_workers=True)

    prototypes = []
    prototype_ids = []  # 每个 prototype 对应的 player_id

    query_sims = []             # self 相似度
    retrieval_labels = []      # GT label
    retrieval_preds = []       # 预测的 top-1 prototype 的 player_id
    non_self_sims = []         # query 和其他玩家 prototype 的平均相似度

    for batch in test_loader:
        support_pos, support_mask, support_labels, query_pos, query_mask, query_labels = unpack_batch(batch)
        B = support_pos.shape[0]

        with torch.no_grad():
            for i in range(B):
                # --- Support set ---
                task_support_pos = support_pos[i]      # [25, ...]
                task_support_mask = support_mask[i]
                task_support_labels = support_labels[i].to(device)

                all_support_embeddings = []
                for j in range(task_support_pos.shape[0]):
                    pos = task_support_pos[j].unsqueeze(0).to(device)
                    mask = task_support_mask[j].unsqueeze(0).to(device)

                    with torch.autocast(device_type="cuda"):
                        _, emb = encoder(pos, mask)
                        all_support_embeddings.append(emb.squeeze(0))  # [D]
                all_support_embeddings = torch.stack(all_support_embeddings, dim=0)  # [25, D]

                # --- Query set ---
                task_query_pos = query_pos[i]
                task_query_mask = query_mask[i]
                task_query_labels = query_labels[i].to(device)

                query_embeddings = []
                for j in range(task_query_pos.shape[0]):
                    pos = task_query_pos[j].unsqueeze(0).to(device)
                    mask = task_query_mask[j].unsqueeze(0).to(device)

                    with torch.autocast(device_type="cuda"):
                        _, emb = encoder(pos, mask)
                        query_embeddings.append(emb.squeeze(0))
                query_embeddings = torch.stack(query_embeddings, dim=0)  # [Q, D]

                # --- Per player ---
                unique_players = task_support_labels.unique()

                for player_id in unique_players:
                    pid = player_id.item()

                    # Build prototype
                    mask_support = task_support_labels == player_id
                    proto = all_support_embeddings[mask_support].mean(dim=0)  # [D]
                    prototypes.append(proto)
                    prototype_ids.append(pid)

                    # Find query embeddings for this player
                    mask_query = task_query_labels == player_id
                    if mask_query.sum() == 0:
                        continue
                    query_emb = query_embeddings[mask_query]  # [Q, D]

                    # 1. Self similarity
                    sim = F.cosine_similarity(query_emb, proto.unsqueeze(0), dim=1)  # [Q]
                    avg_sim = sim.mean().item()
                    query_sims.append(avg_sim)

                    # 2. Retrieval & Soft Matching
                    for emb in query_emb:
                        retrieval_labels.append(pid)

                        all_sims = F.cosine_similarity(
                            emb.unsqueeze(0),                      # [1, D]
                            torch.stack(prototypes).to(device),   # [N, D]
                            dim=1
                        )  # [N]

                        all_ids_tensor = torch.tensor(prototype_ids, device=all_sims.device)
                        top1_idx = torch.argmax(all_sims).item()
                        pred_id = prototype_ids[top1_idx]
                        retrieval_preds.append(pred_id)

                        # 3. Non-self prototype similarity
                        mask_not_self = all_ids_tensor != pid
                        if mask_not_self.sum() > 0:
                            non_self_avg = all_sims[mask_not_self].mean().item()
                            non_self_sims.append(non_self_avg)

    # --- Evaluation Metrics ---

    # (1) prototype-prototype similarity
    prototypes_tensor = torch.stack(prototypes, dim=0)
    N = prototypes_tensor.shape[0]
    sim_matrix = F.cosine_similarity(
        prototypes_tensor.unsqueeze(1), prototypes_tensor.unsqueeze(0), dim=2
    )
    mask = ~torch.eye(N, dtype=torch.bool, device=prototypes_tensor.device)
    pairwise_sim = sim_matrix[mask]
    avg_similarity = pairwise_sim.mean().item()
    print(f"Average similarity between prototypes: {avg_similarity:.4f}")

    # (2) query-prototype (self) similarity
    avg_query_sim = sum(query_sims) / len(query_sims)
    print(f"Average similarity between query and prototype: {avg_query_sim:.4f}")

    # (3) retrieval accuracy
    correct = sum([pred == gt for pred, gt in zip(retrieval_preds, retrieval_labels)])
    retrieval_acc = correct / len(retrieval_labels)
    print(f"Top-1 retrieval accuracy: {retrieval_acc:.4f}")

    # (4) query and non-self prototype similarity
    avg_non_self_sim = sum(non_self_sims) / len(non_self_sims)
    print(f"Average similarity between query and non-self prototypes: {avg_non_self_sim:.4f}")
