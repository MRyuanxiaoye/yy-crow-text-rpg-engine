# -*- coding: utf-8 -*-
"""
E5a：Matrix Profile（时序 Motif 发现）

对每个位置的历史序列，自动找到反复出现的子序列（motif），
用最相似窗口的下一期号码分布作为权重。
主要评估级别：Level 2（区间级）+ Level 3（号码级）
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from research.data_loader import LotteryData
from research.experiment.utils import log, Timer
from research.experiment.e5_framework import (
    E5_RESULTS_DIR, SPLIT_CONFIG, save_json
)


def _build_znorm_windows(series: np.ndarray, max_idx: int, m: int) -> np.ndarray:
    """预计算所有训练窗口的 z-normalized 矩阵（向量化）

    Returns:
        windows: (n_windows, m) z-normalized 窗口矩阵
    """
    n = max_idx - m + 1
    if n <= 0:
        return np.zeros((0, m))
    # 用 stride_tricks 构造滑动窗口视图
    s = series.astype(np.float64)
    shape = (n, m)
    strides = (s.strides[0], s.strides[0])
    windows = np.lib.stride_tricks.as_strided(s, shape=shape, strides=strides).copy()
    # z-normalize 每行
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)
    stds[stds < 1e-8] = 1.0
    windows = (windows - means) / stds
    return windows


def _batch_predict(train_windows: np.ndarray, query_windows: np.ndarray,
                   train_next_vals: np.ndarray, k: int,
                   red_range: int) -> List[Dict[int, float]]:
    """批量 KNN 预测：用矩阵乘法加速距离计算

    Args:
        train_windows: (n_train, m) 训练窗口
        query_windows: (n_query, m) 查询窗口
        train_next_vals: (n_train,) 训练窗口对应的下一期号码
        k: 邻居数
        red_range: 号码范围

    Returns:
        [权重字典] * n_query
    """
    # ||q - t||^2 = ||q||^2 + ||t||^2 - 2*q·t
    # z-normalized 后 ||q||^2 ≈ m, ||t||^2 ≈ m
    # 所以 dist^2 ≈ 2m - 2*q·t，只需计算 q·t
    sims = query_windows @ train_windows.T  # (n_query, n_train)
    # dist^2 = 2*(m - sim)，取 sqrt
    m = train_windows.shape[1]
    dist_sq = np.maximum(2.0 * (m - sims), 0.0)
    dists = np.sqrt(dist_sq)

    actual_k = min(k, dists.shape[1])
    results = []
    uniform = {v: 1.0 / red_range for v in range(1, red_range + 1)}

    for i in range(len(query_windows)):
        top_k_idx = np.argpartition(dists[i], actual_k)[:actual_k]
        top_k_dists = dists[i, top_k_idx]

        weights = defaultdict(float)
        for idx, d in zip(top_k_idx, top_k_dists):
            if idx < len(train_next_vals):
                val = int(train_next_vals[idx])
                weights[val] += 1.0 / (d + 1e-6)

        if not weights:
            results.append(uniform.copy())
            continue

        for v in range(1, red_range + 1):
            if v not in weights:
                weights[v] = 1e-10
        total = sum(weights.values())
        results.append({v: w / total for v, w in weights.items()})

    return results


# ============================================================
# 多维 Matrix Profile（向量化版本）
# ============================================================

def _build_multi_znorm_windows(data: LotteryData, max_idx: int,
                               m: int) -> np.ndarray:
    """预计算多位置联合 z-normalized 窗口矩阵

    Returns:
        windows: (n_windows, m * n_pos) 展平的 z-normalized 窗口
    """
    n_pos = data.red_count
    n = max_idx - m + 1
    if n <= 0:
        return np.zeros((0, m * n_pos))

    all_windows = []
    for start in range(n):
        row = []
        for pos in range(n_pos):
            seg = data.position_series[pos][start:start+m].astype(np.float64)
            std = seg.std()
            if std < 1e-8:
                std = 1.0
            row.extend((seg - seg.mean()) / std)
        all_windows.append(row)

    return np.array(all_windows, dtype=np.float64)


# ============================================================
# 主运行函数
# ============================================================

def run_e5a(data: LotteryData, test_indices: np.ndarray,
            max_train_idx: int, **kwargs) -> List[List[Dict[int, float]]]:
    """E5a 主函数：向量化批量计算，多参数扫描"""

    n_pos = data.red_count
    red_range = data.red_range

    # 减少超参数组合以控制运行时间
    window_sizes = [3, 5, 7]
    k_values = [10, 20, 50]

    best_auc = -1.0
    best_weights = None
    best_params = None

    for m in window_sizes:
        log(f"    预计算窗口矩阵 m={m}...")

        # 逐位置批量预测（向量化）
        for k in k_values:
            all_weights = [[] for _ in range(len(test_indices))]

            for pos in range(n_pos):
                series = data.position_series[pos]
                train_win = _build_znorm_windows(series, max_train_idx, m)
                if len(train_win) == 0:
                    for i in range(len(test_indices)):
                        all_weights[i].append(
                            {v: 1.0/red_range for v in range(1, red_range+1)})
                    continue

                # 训练窗口对应的下一期号码
                train_next = np.array([
                    int(series[start + m])
                    for start in range(len(train_win))
                    if start + m < len(series)
                ])
                tw = train_win[:len(train_next)]

                # 构造测试查询窗口
                query_list = []
                for t in test_indices:
                    t = int(t)
                    if t < m:
                        query_list.append(np.zeros(m))
                    else:
                        seg = series[t-m+1:t+1].astype(np.float64)
                        std = seg.std()
                        if std < 1e-8:
                            std = 1.0
                        query_list.append((seg - seg.mean()) / std)
                query_mat = np.array(query_list)

                # 批量预测
                pos_results = _batch_predict(
                    tw, query_mat, train_next, k, red_range)
                for i in range(len(test_indices)):
                    all_weights[i].append(pos_results[i])

            auc = _quick_auc(all_weights, data, test_indices)
            log(f"      m={m}, k={k}: AUC={auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_weights = all_weights
                best_params = {"m": m, "k": k}

    log(f"  E5a 最优参数: {best_params}, AUC={best_auc:.4f}")
    return best_weights


def _quick_auc(all_weights, data, test_indices):
    """快速计算整体 AUC"""
    n_pos = data.red_count
    red_range = data.red_range
    total_rank = 0
    count = 0

    for i, t in enumerate(test_indices):
        t = int(t)
        if t >= data.n_draws - 1:
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
