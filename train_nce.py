from __future__ import annotations
import os
"""
encoder_trainer_refactored.py
=============================
重构版训练脚本，将原有的 *数据准备 → 模型定义 → 训练循环* 拆分为
可维护的若干组件，并聚合在同一文件便于用户一次性查看。

主要改动
--------
1. **配置集中管理**：使用 `@dataclass` 保存超参数；便于命令行/Notebook 修改。
2. **模块化**：聚合器、损失函数、数据工具分别封装，Trainer 聚焦训练逻辑。
3. **去硬编码**：N / K / Q 与维度信息全部从 `Config` 注入，杜绝魔法数字。
4. **AMP 与 autocast**：统一封装在 `_amp_ctx` 上下文，避免重复书写。
5. **类型提示 & 文档**：为主要函数与方法补充类型提示，阅读更友好。
6. **性能细节**：
   - `with torch.no_grad()` 与 `model.eval()` 协同，减少不必要的 autograd。
   - 验证时批量归一化放到批维度之外，避免重复 `F.normalize`。
   - 视图裁剪逻辑抽成独立生成器，可并发 (未来可改为多线程)。
7. **日志 & 可视化**：抽象出 `MetricLogger`，未来可替换为 TensorBoard / WandB。

后续如需拆分多个文件，只需 `from encoder_trainer_refactored import X` 即可。
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

###############################################################################
# ⚙️  Config
###############################################################################


@dataclass
class Config:
    """集中管理所有超参数与路径。"""

    # ---- data ---- #
    n_way: int = 5  # N
    k_shot: int = 10  # K
    q_query: int = 10  # Q

    train_json: str = "train_players"
    val_json: str = "val_players"
    dataset_root: str = "chess_data_parse"

    batch_size: int = 1  # episodic ⇒ 每次一个 episode
    num_workers: int = 4

    # ---- memory ---- #
    chunk_size: int = 4  # 每次送入 encoder 的子批量大小 (N*K 维度)

    # ---- model ---- #
    d_model: int = 256
    cnn_channels: int = 224

    # ---- optim ---- #
    lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 30

    # ---- loss ---- #
    supcon_temp: float = 0.07
    info_nce_tau: float = 0.1
    info_nce_coef: float = 0.01

    # view sampling
    k_view: int = 2
    min_len: int = 10
    max_len: int = 40

    # ---- misc ---- #
    model_idx: int = 0
    best: bool = True
    out_dir: str = "./models/model_2025_05_10_2"
    seed: int = 999
    amp_dtype: torch.dtype = torch.float16  # 自动混合精度类型

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))


###############################################################################
# 🧩 模型组件
###############################################################################


class DeepSetAgg(nn.Module):
    """简单平均聚合版本。"""

    def __init__(self, d: int):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.rho = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d, elementwise_affine=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, N, K, D]
        return self.rho(self.phi(x).mean(dim=2))  # [B, N, D]


class AttnSetAgg(nn.Module):
    """注意力加权聚合 (Set Transformer 风格)。"""

    def __init__(self, d: int):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True))
        self.attn_score = nn.Linear(d, 1, bias=False)
        self.rho = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d, elementwise_affine=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, N, K, D]
        h = self.phi(x)  # [B, N, K, D]
        α = torch.softmax(self.attn_score(torch.tanh(h)), dim=2)  # 权重
        pooled = (α * h).sum(dim=2)
        return self.rho(pooled)  # [B, N, D]


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Chen+ 2020)。"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # z:[N,D]
        z = F.normalize(z, dim=1)
        sim = z @ z.T / self.t  # [N,N]
        label_mask = y.view(-1, 1).eq(y.view(1, -1)).float()
        logits_mask = torch.ones_like(label_mask) - torch.eye(label_mask.size(0), device=z.device)
        label_mask *= logits_mask  # remove diagonal

        sim = sim - sim.max(dim=1, keepdim=True)[0]  # 数值稳定
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        mean_log_prob = (label_mask * log_prob).sum(1) / label_mask.sum(1).clamp_min(1.0)
        return -mean_log_prob.mean()
    

###############################################################################
# 🔬  View sampler
###############################################################################


def temporal_crop(game: torch.Tensor, mask: torch.Tensor, min_len: int, max_len: int):
    valid = (~mask).sum().item()
    if valid < min_len:
        return game, mask
    seg_len = random.randint(min_len, min(max_len, valid))
    start = random.randint(0, valid - seg_len)
    end = start + seg_len
    return game[start:end], mask[start:end]


def sample_views(raw: Dict[str, Any], cfg: Config):
    game_views, mask_views, counts, labels_views = [], [], [], []
    for sp in ("support", "query"):
        for g, m, l in zip(raw[sp]["games"].squeeze(0), raw[sp]["masks"].squeeze(0), raw[sp]["labels"].squeeze(0)):
            cnt = 0
            for _ in range(cfg.k_view):
                gv, mv = temporal_crop(g, m.bool(), cfg.min_len, cfg.max_len)
                game_views.append(gv)
                mask_views.append(mv)
                labels_views.append(l)
                cnt += 1
            counts.append(cnt)
    gv = pad_sequence(game_views, batch_first=True).unsqueeze(0)  # [1,M,T,C,H,W]
    mv = pad_sequence(mask_views, batch_first=True, padding_value=True).unsqueeze(0)  # [1,M,T]
    labels = torch.stack(labels_views, dim=0)

    return gv, mv, counts, labels


###############################################################################
# 🛠️ 工具函数
###############################################################################


def info_nce_loss(z: torch.Tensor, view_counts: List[int], tau: float) -> torch.Tensor:
    """批内 InfoNCE：同一局不同裁剪为正样本。"""
    z = F.normalize(z.squeeze(0), dim=1)  # [M,d]
    sim = z @ z.T / tau

    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    start = 0
    for c in view_counts:
        pos_mask[start : start + c, start : start + c] = True
        start += c
    pos_mask.fill_diagonal_(False)

    diag_mask = torch.eye(sim.size(0), dtype=torch.bool, device=z.device)
    sim = sim - sim.max(dim=1, keepdim=True)[0]
    log_prob = sim - torch.log(torch.exp(sim).masked_fill(diag_mask, 0).sum(1, keepdim=True) + 1e-8)
    return -(log_prob.masked_select(pos_mask).mean())


def load_json(root: str, name: str) -> Any:
    return json.load(Path(root).joinpath(f"{name}.json").open())


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """原始代码中 batch 仅包含 1 个 episodic 样本，这里保持一致。"""
    sample = batch[0]  # type: ignore[index]
    out: Dict[str, Any] = {}
    
    for split in ("support", "query"):
        games = []
        masks = []
        labels = []
        for g, m, l in zip(sample[split]["games"], sample[split]["masks"], sample[split]["labels"].squeeze(0)):
            g = g.squeeze(0)  # [T,C,H,W]
            m = m.squeeze(0)  # [T]
            games.append(g)
            masks.append(m)
            labels.append(l)

        B = 1
        games = pad_sequence(games, batch_first=True)
        masks = pad_sequence(masks, batch_first=True, padding_value=True)
        labels = torch.stack(labels, dim=0).view(B, -1)

        out[split] = {
            "games": games.view(B, -1, *games.shape[1:]),  # [B,N*K,C,H,W]
            "masks": masks.view(B, -1, *masks.shape[1:]),
            "labels": labels,
        }

    return out

# ---------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------

def gpu_mem_str() -> str:
    """Return current & peak GPU memory usage (MB) as formatted string."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e6
        peak = torch.cuda.max_memory_allocated() / 1e6
        return f" | mem_alloc={alloc:.0f}MB | mem_peak={peak:.0f}MB"
    return ""

# ---------------------------------------------------------------------
# Detailed memory snapshot (optional)
# ---------------------------------------------------------------------
def mem_summary_if(tag: str) -> None:
    """
    Print an extended CUDA memory summary when DEBUG_MEM=1 is set in env.
    Very verbose → enable only for short debug runs:
        $ DEBUG_MEM=1 python train_nce.py
    """
    if os.getenv("DEBUG_MEM", "0") == "1" and torch.cuda.is_available():
        print(f"\n===== CUDA memory summary ({tag}) =====")
        print(torch.cuda.memory_summary(device=0, abbreviated=False))
        print("===== end memory summary =====\n")


###############################################################################
# 🚂 Trainer
###############################################################################


class EncoderTrainer:
    """封装完整训练流程。"""

    def __init__(self, cfg: Config, train_dl: DataLoader, val_dl: DataLoader, test_dl: DataLoader, encoder: nn.Module):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = encoder.to(self.device)
        self.setagg = AttnSetAgg(cfg.d_model).to(self.device)
        self.logit_scale = nn.Parameter(torch.ones([], device=self.device) * 2.3)

        params = list(self.encoder.parameters()) + list(self.setagg.parameters()) + [self.logit_scale]
        self.optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.GradScaler(enabled=self.device.type == "cuda")

        self.supcon_loss = SupConLoss(cfg.supcon_temp)
        self.train_dl, self.val_dl, self.test_dl = train_dl, val_dl, test_dl

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.out_dir / f"epoch{self.cfg.model_idx:03d}_{'best' if self.cfg.best else 'last'}.pt"
        if os.path.exists(self.model_path):
            self.ckpt = torch.load(self.model_path, map_location=self.device)
            self.encoder.load_state_dict(self.ckpt["encoder"])
            self.setagg.load_state_dict(self.ckpt["setagg"])
            self.logit_scale.data.copy_(self.ckpt["logit_scale"])
            self.optim.load_state_dict(self.ckpt["optimizer"])
            self.scaler.load_state_dict(self.ckpt["scaler"])
            print(f"Loaded checkpoint from {self.model_path}")
        else:
            self.ckpt = None
            print(f"Checkpoint not found at {self.model_path}, starting from scratch.")

    # ---------------------------------------------------------------------
    # util
    # ---------------------------------------------------------------------

    def _amp_ctx(self):
        return torch.autocast(device_type=self.device.type, dtype=self.cfg.amp_dtype, enabled=self.device.type == "cuda")

    # ---------------------------------------------------------------------
    # core logic
    # ---------------------------------------------------------------------

    def _encode(self, games: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        games, masks = games.to(self.device), masks.to(self.device)

        # contrastive_zs, zs = [], []
        # chunk_size = self.cfg.chunk_size  # 用于控制单次前向的显存

        # for i in range(0, games.size(1), chunk_size):  # games: [1, N*K, ...]
        #     mini_games = games[:, i:i+chunk_size]      # [1, c, T, C, H, W]
        #     mini_masks = masks[:, i:i+chunk_size]      # [1, c, T]

        #     with self._amp_ctx():
        #         c_z, z = self.encoder(mini_games, mini_masks)

        #     contrastive_zs.append(c_z)
        #     zs.append(z)

        # return torch.cat(contrastive_zs, dim=1), torch.cat(zs, dim=1)

        with self._amp_ctx():
            contrastive_z, z = self.encoder(games, masks)
        
        return contrastive_z, z

    def _build_proto(self, g: torch.Tensor, m: torch.Tensor, support_views: Dict[str, torch.Tensor] | None = None):
        """构建原型；可选地将裁剪视图嵌入拼接到对应类别。"""
        with torch.no_grad():
            s_contrastive_z, z = self._encode(g, m)            # [B,N*K,D]

        z = z.view(self.cfg.batch_size, self.cfg.n_way, self.cfg.k_shot, self.cfg.d_model)  # [B,N,K,D]

        # ---- (可选) 追加裁剪视图 ---- #
        if support_views is not None:
            v_games = support_views["game_views"]
            v_masks = support_views["mask_views"]
            v_labels = support_views["label_views"].to(self.device)  # [1,M]
            # 计算裁剪视图的原型
            with torch.no_grad():
                _, v_z = self._encode(v_games, v_masks)  # [1,M,D]
            v_z = v_z.squeeze(0)  # [M,D]

            merged: List[torch.Tensor] = []
            for n in range(self.cfg.n_way):
                sup_emb = z[:, n, :, :]                 # [B,K,D]
                idx = (v_labels == n).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    seg_emb = v_z[idx].unsqueeze(0)     # [1,k_view,D]
                    sup_emb = torch.cat([sup_emb, seg_emb], dim=1)
                
                merged.append(sup_emb)
            # pad to same K'
            max_k = max(t.size(1) for t in merged)
            for i, t in enumerate(merged):
                if t.size(1) < max_k:
                    pad = torch.zeros((self.cfg.batch_size, max_k - t.size(1), self.cfg.d_model), dtype=t.dtype, device=t.device)
                    merged[i] = torch.cat([t, pad], dim=1)
            z = torch.stack(merged, dim=1)  # [B,N,max_k,D]

        return s_contrastive_z, self.setagg(z)               # [B,N,D](self, g: torch.Tensor, m: torch.Tensor):

    # ---------------- validation ---------------- #

    @torch.no_grad()
    def _eval(self, fuse_views: bool = False) -> float:
        self.encoder.eval()
        self.setagg.eval()
        total_loss, total_n = 0.0, 0
        for sample in self.val_dl:
            sample = collate_fn([sample])
            q_games, q_masks = sample["query"]["games"], sample["query"]["masks"]
            targets = sample["query"]["labels"].view(-1).to(self.device)  # [1,N*Q]

            if fuse_views and self.cfg.info_nce_coef > 0:
                v_g, v_m, v_c, v_l = sample_views(sample, self.cfg)
                _, protos = self._build_proto(
                    sample["support"]["games"],
                    sample["support"]["masks"],
                    support_views={
                        "game_views": v_g[:self.cfg.n_way * self.cfg.k_shot * self.cfg.k_view],
                        "mask_views": v_m[:self.cfg.n_way * self.cfg.k_shot * self.cfg.k_view],
                        "label_views": v_l.view(-1)[:self.cfg.n_way * self.cfg.k_shot * self.cfg.k_view]
                    }
                )
            else:
                # 直接使用原始数据
                _, protos = self._build_proto(
                    sample["support"]["games"],
                    sample["support"]["masks"]
                )

            _, q_z = self._encode(q_games, q_masks)
            q_z, protos = map(lambda t: F.normalize(t, dim=-1), (q_z, protos))
            logits = (q_z.unsqueeze(2) * protos.unsqueeze(1)).sum(-1) * self.logit_scale.exp().clamp(1, 50)
            loss = F.cross_entropy(logits.view(-1, self.cfg.n_way), targets)
            total_loss += loss.item() * targets.numel()
            total_n += targets.numel()
        return total_loss / total_n

    # ---------------- training ---------------- #

    def save_model(self, epoch: int, best_loss, path: Path, tag: str) -> None:
        """保存模型参数。"""
        ckpt = {
            "epoch": epoch,
            "best_loss": best_loss,
            "encoder": self.encoder.state_dict(),
            "setagg": self.setagg.state_dict(),
            "logit_scale": self.logit_scale.data,   # 仅 tensor
            "optimizer": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(ckpt, path / f"epoch{epoch:03d}_{tag}.pt")


    def fit(self):
        best_loss = self.ckpt["best_loss"] if self.ckpt else float("inf")
        train_curve, val_curve = [], []
        start_epoch = self.ckpt.get("epoch", 0) + 1 if self.ckpt else 1
        end_epoch = start_epoch + self.cfg.epochs
        for epoch in range(start_epoch, end_epoch):
            self.encoder.train()
            self.setagg.train()
            epoch_loss = 0.0
            
            for batch_idx, raw in enumerate(self.train_dl):
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                sample = collate_fn([raw])
                self.optim.zero_grad()

                # --- forward & loss --- #
                if self.cfg.info_nce_coef > 0:
                    v_g, v_m, v_c, v_l = sample_views(sample, self.cfg)

                    _, protos = self._build_proto(
                        sample["support"]["games"], # [1,N*K,T,C,H,W]
                        sample["support"]["masks"], # [1,N*K,T]
                        support_views={
                            "game_views": v_g[:self.cfg.n_way * self.cfg.k_shot * self.cfg.k_view],
                            "mask_views": v_m[:self.cfg.n_way * self.cfg.k_shot * self.cfg.k_view],
                            "label_views": v_l.view(-1)[:self.cfg.n_way * self.cfg.k_shot * self.cfg.k_view]
                        }
                    )
                else:
                    _, protos = self._build_proto(
                        sample["support"]["games"], # [1,N*K,T,C,H,W]
                        sample["support"]["masks"]  # [1,N*K,T]
                    )
                
                q_games, q_masks = sample["query"]["games"], sample["query"]["masks"] # [1,N*Q,T,C,H,W], [1,N*Q,T]
                targets = sample["query"]["labels"].view(-1).to(self.device) # [1,N*Q]

                contrastive_z, q_z = self._encode(q_games, q_masks) # [1,N*Q,D], [1,N*Q,D]
                q_z, protos = map(lambda t: F.normalize(t, dim=-1), (q_z, protos))

                # CE
                # logits = (q_z.unsqueeze(2) * protos.unsqueeze(1)).sum(-1) * self.logit_scale.exp().clamp(1, 50)
                # ce = F.cross_entropy(logits.view(-1, self.cfg.n_way), targets)

                logits = torch.matmul(
                    q_z.unsqueeze(2),                    # [B, N*Q, 1, D]
                    protos.unsqueeze(1).transpose(2, 3)  # [B, 1, D, N]
                ).squeeze(3)                                  # [B, N*Q, N]

                logit_scale = self.logit_scale.exp().clamp(1, 50)
                logits = logits * logit_scale                 # [B, N*Q, N]
                ce = F.cross_entropy(logits.view(-1, self.cfg.n_way), targets)

                # SupCon (支持集)
                query_labels = sample["query"]["labels"].view(-1).to(self.device)
                supcon = self.supcon_loss(contrastive_z.view(-1, contrastive_z.size(-1)), query_labels)

                # 总损失
                loss = ce + supcon * 0.1  # 比例按需调整

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                epoch_loss += ce.item()

                if batch_idx % 20 == 0:
                    mem = gpu_mem_str()
                    print(
                        f"[Epoch {epoch}/{end_epoch}] "
                        f"Batch {batch_idx:4d}/{len(self.train_dl)} | "
                        f"loss={loss.item():.4f} | ce={ce.item():.4f} | "
                        f"supcon={supcon.item():.4f} | "
                        f"logit_scale={self.logit_scale.item():.4f}{mem}"
                    )
                    mem_summary_if(f"e{epoch}_b{batch_idx}")

            # ---- epoch end ---- #
            val_loss = self._eval(True)
            train_curve.append(epoch_loss / len(self.train_dl))
            val_curve.append(val_loss)
            print(f"[Epoch {epoch}/{end_epoch}] train={train_curve[-1]:.4f} | val={val_loss:.4f}")

            # checkpoint
            tag = "last"
            if val_loss < best_loss:
                best_loss = val_loss
                tag = "best"

            self.save_model(epoch, best_loss, self.out_dir, tag)

            self._plot_curve(train_curve, val_curve)

    # ---------------------------------------------------------------------
    # plot utils
    # ---------------------------------------------------------------------

    def _plot_curve(self, train: List[float], val: List[float]):
        plt.figure(figsize=(8, 4))
        plt.plot(train, label="train")
        plt.plot(val, label="val")
        plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(self.out_dir / "loss_curve.png")
        plt.close()

    def test_protoes_qs(self, dl):
        protoes = []
        q_zs = []
        for idx, sample in enumerate(dl):
            sample = collate_fn([sample])
            v_g, v_m, v_c, v_l = sample_views(sample, self.cfg)
            _, protoes_task = self._build_proto(
                sample["support"]["games"],
                sample["support"]["masks"],
                support_views={
                    "game_views": v_g[:self.cfg.n_way * self.cfg.k_shot * self.cfg.k_view],
                    "mask_views": v_m[:self.cfg.n_way * self.cfg.k_shot * self.cfg.k_view],
                    "label_views": v_l.view(-1)[:self.cfg.n_way * self.cfg.k_shot * self.cfg.k_view]
                }
            )  # [1,N,D]

            q_games, q_masks = sample["query"]["games"], sample["query"]["masks"]
            targets = sample["query"]["labels"].view(-1).to(self.device)
            _, q_z = self._encode(q_games, q_masks) # [1,N*Q,D]

            q_z, protoes_task = map(lambda t: F.normalize(t, dim=-1), (q_z, protoes_task))
            logits = (q_z.unsqueeze(2) * protoes_task.unsqueeze(1)).sum(-1) * self.logit_scale.exp().clamp(1, 50)
            loss = F.cross_entropy(logits.view(-1, self.cfg.n_way), targets)
            print(f"[Epoch {idx+1}/{len(dl)}] test_loss={loss.item():.4f}")

            protoes.append(protoes_task)
            q_zs.append(q_z)

        return protoes, q_zs

    @torch.no_grad()
    def test(self):
        self.encoder.eval()
        test_protoes, test_q_zs = self.test_protoes_qs(self.test_dl)

        test_protoes = torch.cat(test_protoes, dim=0)
        test_q_zs =  torch.cat(test_q_zs, dim=0)

        test_protoes = test_protoes.view(-1, self.cfg.d_model)  # [B*N,D]
        test_q_zs = test_q_zs.view(-1, self.cfg.q_query, self.cfg.d_model) # [B,N,Q,D]

        train_protoes, train_q_zs = self.test_protoes_qs(self.train_dl)
        train_protoes = torch.cat(train_protoes, dim=0)
        train_q_zs =  torch.cat(train_q_zs, dim=0)
        train_protoes = train_protoes.view(-1, self.cfg.d_model)  # [B*N,D]
        train_q_zs = train_q_zs.view(-1, self.cfg.q_query, self.cfg.d_model)

        offset = train_protoes.size(0)
        acc = self.calc_query_z_class_accuracy(torch.cat([train_protoes, test_protoes]), test_q_zs, offset) # [B*N]

        print("Accuracy:", acc.mean().item())

        # protoes = protoes.unsqueeze(1) # [B*N,1,D]

        # cos_sim = (protoes * q_zs).sum(-1) # [B*N,Q]

        # mean_sim = cos_sim.mean(dim=1) # [B*N]

        # print("⟂ proto‑proto mean:", (protoes @ protoes.T).mean().item())  
        # print("⟂ query‑query mean:", (q_zs.view(-1, q_zs.size(-1)) @  
        #                             q_zs.view(-1, q_zs.size(-1)).T).mean().item()) 

        # print(cos_sim.shape, mean_sim.shape)
        # print(mean_sim)

        # acc = self.calc_query_z_class_accuracy(protoes, q_zs)
        # print(acc.mean().item())

    @torch.no_grad()
    def calc_query_z_class_accuracy(self, all_protos: torch.Tensor,
                                    query_z: torch.Tensor, offset) -> torch.Tensor:
        """
        计算每个 player 的 Q 条 query 的分类准确率。

        Args:
            all_protos: Tensor of shape [N, d], N 个 prototype
            query_z:    Tensor of shape [N, Q, d], 每个 player 的 Q 条 query

        Returns:
            accuracies: Tensor of shape [N], 每个 player 的准确率 = (预测正确数 / Q)
        """
        _, d = all_protos.shape
        N, Q, d = query_z.size()

        # 1. L2 归一化
        # protos_norm = F.normalize(all_protos, dim=1)       # [N, d]
        # queries_norm = F.normalize(query_z, dim=2)         # [N, Q, d]
        q_z, protoes_task = map(lambda t: F.normalize(t, dim=-1), (query_z, all_protos))
        

        # 2. 展平所有 query 并计算相似度
        queries_flat = q_z.view(-1, d)         # [N*Q, d]
        sims = torch.matmul(queries_flat, protoes_task.T)   # [N*Q, N]

        # 3. 预测标签：取相似度最高的 prototype 下标
        preds = sims.argmax(dim=1)                         # [N*Q]

        # 4. 构造真实标签：第 i 个 player 的 Q 条 query 的标签都是 i
        true_labels = torch.arange(N, device=all_protos.device) \
                        .repeat_interleave(Q) + offset  # [N*Q]
        # 5. 计算每条 query 的是否预测正确
        correct = (preds == true_labels).view(N, Q)        # [N, Q], bool

        # 6. 计算每个 player 的准确率
        accuracies = correct.float().mean(dim=1)           # [N]

        return accuracies

    


###############################################################################
# 🔨 入口
###############################################################################


def main() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # -------- 数据集 -------- #
    from player_encoder_infine.dataset import MetaStyleDataset  # 延迟导入减少启动开销
    train_ds = MetaStyleDataset(load_json(cfg.dataset_root, cfg.train_json), 1000, N=cfg.n_way, K=cfg.k_shot, Q=cfg.q_query)
    val_ds = MetaStyleDataset(load_json(cfg.dataset_root, cfg.val_json), 150, N=cfg.n_way, K=cfg.k_shot, Q=cfg.q_query)
    test_ds = MetaStyleDataset(load_json(cfg.dataset_root, cfg.val_json), 60, N=cfg.n_way, K=cfg.k_shot, Q=cfg.q_query, rand=False)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # -------- 模型 -------- #
    from player_encoder_infine.encoder import TransformerEncoder

    encoder = TransformerEncoder(
        cnn_in_channels=cfg.cnn_channels,
        state_embed_dim=cfg.d_model,
        transformer_d_model=cfg.d_model,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
    )

    # -------- 训练器 -------- #
    trainer = EncoderTrainer(cfg, train_dl, val_dl, test_dl, encoder)
    trainer.fit()

def test():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # -------- 数据集 -------- #
    from player_encoder_infine.dataset import MetaStyleDataset  # 延迟导入减少启动开销
    train_ds = MetaStyleDataset(load_json(cfg.dataset_root, cfg.train_json), 2300 // cfg.n_way, rand=False, N=cfg.n_way, K=cfg.k_shot, Q=cfg.q_query)
    val_ds = MetaStyleDataset(load_json(cfg.dataset_root, cfg.val_json), 300 // cfg.n_way, rand=False, N=cfg.n_way, K=cfg.k_shot, Q=cfg.q_query)
    test_ds = MetaStyleDataset(load_json(cfg.dataset_root, cfg.val_json), 300 // cfg.n_way, rand=False, N=cfg.n_way, K=cfg.k_shot, Q=cfg.q_query)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # -------- 模型 -------- #
    from player_encoder_infine.encoder import TransformerEncoder

    encoder = TransformerEncoder(
        cnn_in_channels=cfg.cnn_channels,
        state_embed_dim=cfg.d_model,
        transformer_d_model=cfg.d_model,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
    )

    # -------- 训练器 -------- #
    trainer = EncoderTrainer(cfg, train_dl, val_dl, test_dl, encoder)
    trainer.test()


if __name__ == "__main__":
    main()
    # test()
