# -*- coding: utf-8 -*-
"""
E5g：位置自适应特征 KNN

基于自动特征搜索发现的最优特征组合，每个位置使用独立的特征集。
混合策略：P0/P4/P5 用统一3特征（cross_dp-1, min_15, big_in_5），
         P1/P2/P3 用位置自适应特征。
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict

from research.data_loader import LotteryData
from research.experiment.utils import log


# === 特征定义 ===

def _feat_cross_dp(data, t, pos, dp):
    pp = pos + dp
    if 0 <= pp < data.red_count:
        return float(data.position_series[pp][t] - data.position_series[pos][t])
    return 0.0

def _feat_min_w(data, t, pos, w):
    return float(np.min(data.position_series[pos][max(0, t - w + 1):t + 1]))

def _feat_max_w(data, t, pos, w):
    return float(np.max(data.position_series[pos][max(0, t - w + 1):t + 1]))

def _feat_big_in_w(data, t, pos, w):
    return float(np.sum(data.position_series[pos][max(0, t - w + 1):t + 1] > 16))

def _feat_std_w(data, t, pos, w):
    return float(np.std(data.position_series[pos][max(0, t - w + 1):t + 1]))

def _feat_odd_in_w(data, t, pos, w):
    return float(np.sum(data.position_series[pos][max(0, t - w + 1):t + 1] % 2 == 1))

def _feat_val_t1(data, t, pos):
    return float(data.position_series[pos][t - 1]) if t >= 1 else 0.0

def _feat_range_w(data, t, pos, w):
    return float(np.ptp(data.position_series[pos][max(0, t - w + 1):t + 1]))

def _feat_div10(data, t, pos):
    return float(data.position_series[pos][t] // 10)

def _feat_zone3(data, t, pos):
    return float(data.position_series[pos][t] // 11)

def _feat_range_15(data, t, pos):
    return _feat_range_w(data, t, pos, 15)


# === 位置特征配置 ===
# 混合策略：取每个位置在自适应 vs 统一中更优的方案

POSITION_FEATURES = {
    # P0: 统一3特征 rank=5.7 (优于自适应6.0)
    0: [
        lambda d, t, p: _feat_cross_dp(d, t, p, -1),
        lambda d, t, p: _feat_min_w(d, t, p, 15),
        lambda d, t, p: _feat_big_in_w(d, t, p, 5),
    ],
    # P1: 自适应 rank=11.0 (优于统一13.9)
    1: [
        _feat_val_t1,
    ],
    # P2: 自适应 rank=11.9 (优于统一12.2)
    2: [
        lambda d, t, p: _feat_range_w(d, t, p, 5),
    ],
    # P3: 自适应 rank=11.1 (优于统一13.3)
    3: [
        _feat_div10,
        _feat_range_15,
        _feat_zone3,
    ],
    # P4: 统一3特征 rank=10.3 (优于自适应11.6)
    4: [
        lambda d, t, p: _feat_cross_dp(d, t, p, -1),
        lambda d, t, p: _feat_min_w(d, t, p, 15),
        lambda d, t, p: _feat_big_in_w(d, t, p, 5),
    ],
    # P5: 统一3特征 rank=6.5 (优于自适应6.8)
    5: [
        lambda d, t, p: _feat_cross_dp(d, t, p, -1),
        lambda d, t, p: _feat_min_w(d, t, p, 15),
        lambda d, t, p: _feat_big_in_w(d, t, p, 5),
    ],
}


def _compute_features(data, t, pos):
    """计算指定位置的特征向量"""
    feat_funcs = POSITION_FEATURES[pos]
    return np.array([f(data, t, pos) for f in feat_funcs], dtype=np.float64)


def run_e5g(data: LotteryData, test_indices: np.ndarray,
            max_train_idx: int, **kwargs) -> List[List[Dict[int, float]]]:
    """E5g 主函数：位置自适应特征 KNN"""

    n_pos = data.red_count
    red_range = data.red_range
    K = kwargs.get('k', 20)
    min_train_t = 20  # 需要足够历史

    all_weights = [[] for _ in range(len(test_indices))]

    for pos in range(n_pos):
        series = data.position_series[pos]
        n_feats = len(POSITION_FEATURES[pos])

        # 构造训练集特征矩阵
        train_feats = []
        train_next = []
        for t in range(min_train_t, max_train_idx):
            train_feats.append(_compute_features(data, t, pos))
            train_next.append(int(series[t + 1]))
        train_feats = np.array(train_feats)
        train_next = np.array(train_next)

        # z-normalize（用训练集统计量）
        mu = train_feats.mean(axis=0)
        sigma = train_feats.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        train_normed = (train_feats - mu) / sigma

        # 对每个测试样本做 KNN 预测
        for i, t in enumerate(test_indices):
            t = int(t)
            query = _compute_features(data, t, pos)
            query_normed = (query - mu) / sigma

            # 欧氏距离
            if n_feats > 1:
                dists = np.linalg.norm(train_normed - query_normed, axis=1)
            else:
                dists = np.abs(train_normed.flatten() - query_normed[0])

            actual_k = min(K, len(train_normed))
            top_k_idx = np.argpartition(dists, actual_k)[:actual_k]

            # 距离倒数加权投票
            weights = defaultdict(float)
            for idx in top_k_idx:
                val = train_next[idx]
                weights[val] += 1.0 / (dists[idx] + 1e-6)

            # 补全 + 归一化
            for v in range(1, red_range + 1):
                if v not in weights:
                    weights[v] = 1e-10
            total = sum(weights.values())
            all_weights[i].append({v: w / total for v, w in weights.items()})

        log(f"    P{pos}: {n_feats} features, K={K}")

    return all_weights
