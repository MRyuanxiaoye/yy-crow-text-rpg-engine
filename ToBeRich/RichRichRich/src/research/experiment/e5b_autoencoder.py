# -*- coding: utf-8 -*-
"""
E5b：自编码器 + 最近邻检索

用自编码器学习"局面"的压缩表示，在低维空间中找相似局面。
主要评估级别：Level 3（号码级）+ Level 2（区间级）
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
from collections import defaultdict

from research.data_loader import LotteryData
from research.experiment.utils import log, Timer
from research.experiment.e5_framework import (
    E5_RESULTS_DIR, SPLIT_CONFIG, build_situation_vectors,
    knn_to_weights, save_json
)


# ============================================================
# 自编码器模型
# ============================================================

class Autoencoder(nn.Module):
    """简单 MLP 自编码器"""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)


# ============================================================
# 训练函数
# ============================================================

def train_autoencoder(train_vecs: np.ndarray, latent_dim: int = 16,
                      epochs: int = 500, lr: float = 1e-3,
                      batch_size: int = 64) -> Autoencoder:
    """训练自编码器"""
    device = torch.device('cpu')
    input_dim = train_vecs.shape[1]
    model = Autoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.FloatTensor(train_vecs).to(device)
    n = len(X)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            batch = X[idx]
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 200 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            log(f"      AE epoch {epoch+1}: loss={avg_loss:.6f}")

    model.eval()
    return model


# ============================================================
# 主运行函数
# ============================================================

def run_e5b(data: LotteryData, test_indices: np.ndarray,
            max_train_idx: int, **kwargs) -> List[List[Dict[int, float]]]:
    """E5b 主函数：多参数扫描，选最优"""

    n_pos = data.red_count
    red_range = data.red_range

    # 超参数候选
    windows = [5, 10]
    latent_dims = [8, 16, 32]
    k_values = [20, 50]
    metrics = ['cosine', 'euclidean']

    best_auc = -1.0
    best_weights = None
    best_params = None

    for W in windows:
        vectors, valid_indices, _ = build_situation_vectors(
            data, max_train_idx, window=W)

        train_mask = valid_indices <= max_train_idx
        test_mask = np.isin(valid_indices, test_indices)

        train_vecs = vectors[train_mask]
        test_vecs = vectors[test_mask]
        train_valid_idx = valid_indices[train_mask]
        test_valid_idx = valid_indices[test_mask]

        for ld in latent_dims:
            log(f"    训练 AE: W={W}, latent={ld}")
            model = train_autoencoder(train_vecs, latent_dim=ld, epochs=500)

            # 编码
            with torch.no_grad():
                train_latent = model.encode(
                    torch.FloatTensor(train_vecs)).numpy()
                test_latent = model.encode(
                    torch.FloatTensor(test_vecs)).numpy()

            for k in k_values:
                for metric in metrics:
                    all_weights = _predict_all_pos(
                        train_latent, test_latent,
                        train_valid_idx, data, n_pos,
                        k, red_range, metric)

                    auc = _quick_auc(all_weights, data, test_valid_idx)
                    log(f"      W={W} ld={ld} k={k} "
                        f"m={metric}: AUC={auc:.4f}")

                    if auc > best_auc:
                        best_auc = auc
                        best_weights = all_weights
                        best_params = {
                            "W": W, "latent_dim": ld,
                            "k": k, "metric": metric
                        }

    log(f"  E5b 最优参数: {best_params}, AUC={best_auc:.4f}")
    return _align_weights(best_weights, data, test_indices, n_pos, red_range)


def _predict_all_pos(train_latent, test_latent, train_valid_idx,
                     data, n_pos, k, red_range, metric):
    """逐位置 KNN 预测"""
    n_test = len(test_latent)
    all_weights = [[] for _ in range(n_test)]

    for pos in range(n_pos):
        train_next = np.array([
            int(data.red_matrix[int(t)+1, pos])
            for t in train_valid_idx
            if int(t)+1 < data.n_draws
        ])
        tl = train_latent[:len(train_next)]

        for i in range(n_test):
            w = knn_to_weights(
                test_latent[i], tl, train_next,
                k=k, red_range=red_range, metric=metric)
            all_weights[i].append(w)

    return all_weights


def _align_weights(weights, data, test_indices, n_pos, red_range):
    """对齐权重到 test_indices"""
    uniform = {v: 1.0/red_range for v in range(1, red_range+1)}
    if weights is None:
        return [[uniform.copy() for _ in range(n_pos)]
                for _ in range(len(test_indices))]
    if len(weights) == len(test_indices):
        return weights
    result = []
    wi = 0
    for i in range(len(test_indices)):
        if wi < len(weights):
            result.append(weights[wi])
            wi += 1
        else:
            result.append([uniform.copy() for _ in range(n_pos)])
    return result


def _quick_auc(all_weights, data, valid_indices):
    """快速计算整体 AUC"""
    n_pos = data.red_count
    red_range = data.red_range
    total_rank = 0
    count = 0

    for i, t in enumerate(valid_indices):
        t = int(t)
        if t >= data.n_draws - 1 or i >= len(all_weights):
            continue
        for pos in range(n_pos):
            true_val = int(data.red_matrix[t + 1, pos])
            w_dict = all_weights[i][pos]
            sorted_nums = sorted(w_dict.items(),
                                 key=lambda x: x[1], reverse=True)
            for r, (v, _) in enumerate(sorted_nums, 1):
                if v == true_val:
                    total_rank += r
                    count += 1
                    break

    if count == 0:
        return 0.5
    mean_rank = total_rank / count
    return 1.0 - (mean_rank - 1) / (red_range - 1)
