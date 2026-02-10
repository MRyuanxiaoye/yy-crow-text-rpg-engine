# -*- coding: utf-8 -*-
"""
E5e：对比学习（结果导向的表示学习）

学习一个嵌入空间，使得"下期走势相似的局面"在空间中接近。
主要评估级别：Level 1（方向级）+ Level 3（号码级）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from collections import defaultdict

from research.data_loader import LotteryData
from research.experiment.utils import log, Timer
from research.experiment.e5_framework import (
    E5_RESULTS_DIR, SPLIT_CONFIG, build_situation_vectors,
    knn_to_weights, save_json
)


# ============================================================
# 对比学习编码器
# ============================================================

class ContrastiveEncoder(nn.Module):
    """对比学习编码器"""

    def __init__(self, input_dim: int, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ============================================================
# 样本对构造
# ============================================================

def build_pairs(train_vecs: np.ndarray, train_directions: np.ndarray,
                pos: int, n_neg: int = 5):
    """构造正负样本对

    Args:
        train_vecs: (n_train, d) 局面向量
        train_directions: (n_train,) 方向标签 (-1/0/1)
        pos: 位置索引（用于选择方向标签列）
        n_neg: 每个 anchor 的负样本数

    Returns:
        anchors, positives, negatives: 索引数组
    """
    rng = np.random.RandomState(42)
    n = len(train_vecs)
    dirs = train_directions

    # 按方向分组
    groups = defaultdict(list)
    for i in range(n):
        groups[int(dirs[i])].append(i)

    anchors = []
    positives = []
    neg_list = []

    for i in range(n):
        d = int(dirs[i])
        same = groups[d]
        diff = []
        for dd in groups:
            if dd != d:
                diff.extend(groups[dd])

        if len(same) < 2 or len(diff) < n_neg:
            continue

        # 正样本：同方向随机选一个
        pos_idx = same[rng.randint(len(same))]
        while pos_idx == i and len(same) > 1:
            pos_idx = same[rng.randint(len(same))]

        # 负样本
        neg_idx = rng.choice(diff, size=min(n_neg, len(diff)), replace=False)

        anchors.append(i)
        positives.append(pos_idx)
        neg_list.append(neg_idx)

    return anchors, positives, neg_list


# ============================================================
# 训练函数
# ============================================================

def train_contrastive(train_vecs: np.ndarray, train_directions: np.ndarray,
                      pos: int, embed_dim: int = 32, tau: float = 0.1,
                      epochs: int = 300, lr: float = 1e-3) -> ContrastiveEncoder:
    """训练对比学习编码器（InfoNCE 损失）"""
    device = torch.device('cpu')
    input_dim = train_vecs.shape[1]
    model = ContrastiveEncoder(input_dim, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.FloatTensor(train_vecs).to(device)
    anchors, positives, neg_list = build_pairs(
        train_vecs, train_directions, pos, n_neg=5)

    if len(anchors) == 0:
        return model

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        # mini-batch（随机采样 256 个 anchor）
        rng = np.random.RandomState(epoch)
        batch_idx = rng.choice(len(anchors), size=min(256, len(anchors)),
                               replace=False)

        for bi in batch_idx:
            a_emb = model(X[anchors[bi]].unsqueeze(0))
            p_emb = model(X[positives[bi]].unsqueeze(0))
            n_idx = neg_list[bi]
            n_emb = model(X[n_idx])

            # InfoNCE
            pos_sim = (a_emb * p_emb).sum() / tau
            neg_sims = (a_emb * n_emb).sum(dim=-1) / tau
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            labels = torch.zeros(1, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits.unsqueeze(0), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            avg = total_loss / len(batch_idx)
            log(f"      CL epoch {epoch+1}: loss={avg:.4f}")

    model.eval()
    return model


# ============================================================
# 主运行函数
# ============================================================

def run_e5e(data: LotteryData, test_indices: np.ndarray,
            max_train_idx: int, **kwargs) -> List[List[Dict[int, float]]]:
    """E5e 主函数：对比学习 + KNN 预测"""

    n_pos = data.red_count
    red_range = data.red_range

    # 超参数候选
    windows = [5, 10]
    embed_dims = [16, 32]
    taus = [0.1, 0.2]
    k_values = [20, 50]

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

        for ed in embed_dims:
            for tau in taus:
                log(f"    训练 CL: W={W}, embed={ed}, tau={tau}")

                # 逐位置训练编码器
                pos_embeddings_train = []
                pos_embeddings_test = []

                for pos in range(n_pos):
                    # 方向标签
                    train_dirs = np.array([
                        int(data.direction_series[pos][int(t)])
                        for t in train_valid_idx
                        if int(t) < len(data.direction_series[pos])
                    ])
                    tv = train_vecs[:len(train_dirs)]

                    model = train_contrastive(
                        tv, train_dirs, pos,
                        embed_dim=ed, tau=tau, epochs=300)

                    with torch.no_grad():
                        tr_emb = model(
                            torch.FloatTensor(tv)).numpy()
                        te_emb = model(
                            torch.FloatTensor(test_vecs)).numpy()

                    pos_embeddings_train.append(tr_emb)
                    pos_embeddings_test.append(te_emb)

                for k in k_values:
                    all_weights = _predict_from_embeddings(
                        pos_embeddings_train, pos_embeddings_test,
                        train_valid_idx, data, n_pos, k, red_range)

                    auc = _quick_auc(
                        all_weights, data, test_valid_idx)
                    log(f"      k={k}: AUC={auc:.4f}")

                    if auc > best_auc:
                        best_auc = auc
                        best_weights = all_weights
                        best_params = {
                            "W": W, "embed_dim": ed,
                            "tau": tau, "k": k
                        }

    log(f"  E5e 最优参数: {best_params}, AUC={best_auc:.4f}")
    return _align(best_weights, test_indices, n_pos, red_range)


def _predict_from_embeddings(pos_emb_train, pos_emb_test,
                             train_valid_idx, data, n_pos, k, red_range):
    """从各位置嵌入做 KNN 预测"""
    n_test = len(pos_emb_test[0])
    all_weights = [[] for _ in range(n_test)]

    for pos in range(n_pos):
        train_next = np.array([
            int(data.red_matrix[int(t)+1, pos])
            for t in train_valid_idx
            if int(t)+1 < data.n_draws
        ])
        te = pos_emb_train[pos][:len(train_next)]

        for i in range(n_test):
            w = knn_to_weights(
                pos_emb_test[pos][i], te, train_next,
                k=k, red_range=red_range, metric='cosine')
            all_weights[i].append(w)

    return all_weights


def _align(weights, test_indices, n_pos, red_range):
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
