# -*- coding: utf-8 -*-
"""
E5d：Shapelet Discovery（判别性子序列发现）

自动搜索能区分"下期升/降/平"的子序列片段。
主要评估级别：Level 1（方向级）+ Level 2（区间级）
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from research.data_loader import LotteryData
from research.experiment.utils import log, Timer
from research.experiment.e5_framework import (
    E5_RESULTS_DIR, SPLIT_CONFIG, save_json
)


# ============================================================
# Shapelet 搜索
# ============================================================

def extract_candidate_shapelets(series: np.ndarray, max_idx: int,
                                shapelet_len: int,
                                n_candidates: int = 1000) -> np.ndarray:
    """从训练期随机采样候选 shapelet

    Returns:
        candidates: (n_candidates, shapelet_len) z-normalized 子序列
    """
    rng = np.random.RandomState(42)
    candidates = []
    max_start = max_idx - shapelet_len + 1

    if max_start <= 0:
        return np.zeros((0, shapelet_len))

    starts = rng.randint(0, max_start, size=n_candidates)
    for s in starts:
        seg = series[s:s + shapelet_len].astype(np.float64)
        std = seg.std()
        if std < 1e-8:
            std = 1.0
        candidates.append((seg - seg.mean()) / std)

    return np.array(candidates)


# ============================================================
# 信息增益计算
# ============================================================

def _entropy(labels):
    """计算熵"""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-10))


def compute_shapelet_distances(series: np.ndarray, shapelet: np.ndarray,
                               max_idx: int, shapelet_len: int) -> np.ndarray:
    """计算每个时刻窗口与 shapelet 的距离

    Returns:
        dists: (max_idx - shapelet_len + 2,) 距离数组
    """
    n = max_idx - shapelet_len + 2
    dists = np.zeros(n)
    for t in range(shapelet_len - 1, max_idx + 1):
        seg = series[t - shapelet_len + 1: t + 1].astype(np.float64)
        std = seg.std()
        if std < 1e-8:
            std = 1.0
        seg_norm = (seg - seg.mean()) / std
        dists[t - shapelet_len + 1] = np.sqrt(np.sum((seg_norm - shapelet)**2))
    return dists


def select_top_shapelets(series: np.ndarray, direction_labels: np.ndarray,
                         max_idx: int, shapelet_len: int,
                         n_candidates: int = 1000,
                         top_m: int = 100) -> List[np.ndarray]:
    """用信息增益选择 top-M 个判别性 shapelet

    Args:
        series: 位置序列
        direction_labels: 方向标签（-1/0/1），长度 = len(series)-1
        max_idx: 训练期最大索引
        shapelet_len: shapelet 长度
        n_candidates: 候选数
        top_m: 保留数

    Returns:
        top shapelets 列表
    """
    candidates = extract_candidate_shapelets(
        series, max_idx, shapelet_len, n_candidates)

    if len(candidates) == 0:
        return []

    # 训练期方向标签（对齐到窗口结束位置）
    train_labels = direction_labels[shapelet_len - 1: max_idx + 1]
    n_train = len(train_labels)
    base_entropy = _entropy(train_labels)

    scored = []
    for ci in range(len(candidates)):
        shapelet = candidates[ci]
        dists = compute_shapelet_distances(
            series, shapelet, max_idx, shapelet_len)
        dists = dists[:n_train]

        # 用中位数作为分割点
        threshold = np.median(dists)
        left_mask = dists <= threshold
        right_mask = ~left_mask

        left_labels = train_labels[left_mask]
        right_labels = train_labels[right_mask]

        n_left = len(left_labels)
        n_right = len(right_labels)
        if n_left == 0 or n_right == 0:
            continue

        ig = base_entropy - (
            n_left / n_train * _entropy(left_labels) +
            n_right / n_train * _entropy(right_labels)
        )
        scored.append((ig, ci))

    # 选 top-M
    scored.sort(reverse=True)
    top_shapelets = [candidates[ci] for _, ci in scored[:top_m]]
    return top_shapelets


# ============================================================
# Shapelet 特征化 + KNN 预测
# ============================================================

def shapelet_featurize(series: np.ndarray, shapelets: List[np.ndarray],
                       indices: np.ndarray,
                       shapelet_len: int) -> np.ndarray:
    """将每个时刻转为 shapelet 距离特征向量

    Returns:
        features: (len(indices), len(shapelets))
    """
    n = len(indices)
    m = len(shapelets)
    features = np.zeros((n, m), dtype=np.float32)

    for i, t in enumerate(indices):
        t = int(t)
        if t < shapelet_len - 1:
            continue
        seg = series[t - shapelet_len + 1: t + 1].astype(np.float64)
        std = seg.std()
        if std < 1e-8:
            std = 1.0
        seg_norm = (seg - seg.mean()) / std
        for j, sp in enumerate(shapelets):
            features[i, j] = np.sqrt(np.sum((seg_norm - sp)**2))

    return features


# ============================================================
# 主运行函数
# ============================================================

def run_e5d(data: LotteryData, test_indices: np.ndarray,
            max_train_idx: int, **kwargs) -> List[List[Dict[int, float]]]:
    """E5d 主函数：Shapelet 发现 + KNN 预测"""

    from research.experiment.e5_framework import knn_to_weights

    n_pos = data.red_count
    red_range = data.red_range

    # 超参数
    shapelet_lens = [3, 5, 7]
    n_candidates = 1000
    top_m = 100
    k_values = [20, 50]

    best_auc = -1.0
    best_weights = None
    best_params = None

    # 训练期索引
    train_indices = np.arange(max(shapelet_lens), max_train_idx + 1)

    for sl in shapelet_lens:
        for k in k_values:
            log(f"    尝试 shapelet_len={sl}, k={k}")
            all_weights = [[] for _ in range(len(test_indices))]

            for pos in range(n_pos):
                series = data.position_series[pos]
                dir_labels = data.direction_series[pos]

                # 选择 shapelet
                shapelets = select_top_shapelets(
                    series, dir_labels, max_train_idx,
                    sl, n_candidates, top_m)

                if len(shapelets) == 0:
                    for i in range(len(test_indices)):
                        all_weights[i].append(
                            {v: 1.0/red_range
                             for v in range(1, red_range+1)})
                    continue

                # 特征化
                train_feats = shapelet_featurize(
                    series, shapelets, train_indices, sl)
                test_feats = shapelet_featurize(
                    series, shapelets, test_indices, sl)

                # 训练集下一期号码
                train_next = np.array([
                    int(data.red_matrix[int(t)+1, pos])
                    for t in train_indices
                    if int(t)+1 < data.n_draws
                ])
                tf = train_feats[:len(train_next)]

                # KNN 预测
                for i in range(len(test_indices)):
                    w = knn_to_weights(
                        test_feats[i], tf, train_next,
                        k=k, red_range=red_range,
                        metric='euclidean')
                    all_weights[i].append(w)

            auc = _quick_auc(all_weights, data, test_indices)
            log(f"      AUC={auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_weights = all_weights
                best_params = {"shapelet_len": sl, "k": k}

    log(f"  E5d 最优参数: {best_params}, AUC={best_auc:.4f}")
    return best_weights


def _quick_auc(all_weights, data, test_indices):
    """快速计算整体 AUC"""
    n_pos = data.red_count
    red_range = data.red_range
    total_rank = 0
    count = 0

    for i, t in enumerate(test_indices):
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
