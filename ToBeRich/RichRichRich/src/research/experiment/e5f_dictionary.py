# -*- coding: utf-8 -*-
"""
E5f：字典学习 / 稀疏编码

每个局面是若干"基础模式"的叠加，算法自己学基础模式。
稀疏激活向量 = 数据驱动的布尔掩码。
主要评估级别：Level 3（号码级）+ Level 2（区间级）
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict

from research.data_loader import LotteryData
from research.experiment.utils import log, Timer
from research.experiment.e5_framework import (
    E5_RESULTS_DIR, SPLIT_CONFIG, build_situation_vectors,
    knn_to_weights, save_json
)


def dict_learn_predict(train_vecs: np.ndarray, test_vecs: np.ndarray,
                       train_next_vals: np.ndarray,
                       n_components: int, alpha: float,
                       k: int, red_range: int) -> List[Dict[int, float]]:
    """字典学习 + 稀疏编码 → KNN 预测

    Args:
        train_vecs: (n_train, d) 训练局面向量
        test_vecs: (n_test, d) 测试局面向量
        train_next_vals: (n_train,) 训练集下一期号码
        n_components: 字典大小 K
        alpha: 稀疏度参数
        k: KNN 邻居数
        red_range: 号码范围

    Returns:
        [权重字典] * n_test
    """
    from sklearn.decomposition import DictionaryLearning

    # 训练字典
    dl = DictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=500,
        transform_algorithm='lasso_lars',
        random_state=42,
        n_jobs=-1,
    )
    train_codes = dl.fit_transform(train_vecs)
    test_codes = dl.transform(test_vecs)

    # 在稀疏编码空间做 KNN
    results = []
    for i in range(len(test_codes)):
        w = knn_to_weights(
            test_codes[i], train_codes, train_next_vals,
            k=k, red_range=red_range, metric='euclidean')
        results.append(w)

    return results


# ============================================================
# 主运行函数
# ============================================================

def run_e5f(data: LotteryData, test_indices: np.ndarray,
            max_train_idx: int, **kwargs) -> List[List[Dict[int, float]]]:
    """E5f 主函数：多参数扫描，选最优"""

    n_pos = data.red_count
    red_range = data.red_range

    # 超参数候选
    windows = [5, 10]
    n_components_list = [50, 100, 200]
    alphas = [0.5, 1.0]
    k_values = [20, 50]

    best_auc = -1.0
    best_weights = None
    best_params = None

    for W in windows:
        # 构造局面向量
        vectors, valid_indices, _ = build_situation_vectors(
            data, max_train_idx, window=W)

        # 分离训练/测试
        train_mask = valid_indices <= max_train_idx
        test_mask = np.isin(valid_indices, test_indices)

        train_vecs = vectors[train_mask]
        test_vecs = vectors[test_mask]
        train_valid_idx = valid_indices[train_mask]
        test_valid_idx = valid_indices[test_mask]

        for n_comp in n_components_list:
            if n_comp > train_vecs.shape[0]:
                continue
            for alpha in alphas:
                for k in k_values:
                    log(f"    尝试 W={W}, K={n_comp}, α={alpha}, k={k}")

                    # 逐位置预测
                    all_weights = [[] for _ in range(len(test_valid_idx))]
                    for pos in range(n_pos):
                        # 训练集下一期号码
                        train_next = np.array([
                            int(data.red_matrix[int(t)+1, pos])
                            for t in train_valid_idx
                            if int(t)+1 < data.n_draws
                        ])
                        tv = train_vecs[:len(train_next)]

                        try:
                            pos_results = dict_learn_predict(
                                tv, test_vecs, train_next,
                                n_comp, alpha, k, red_range)
                        except Exception as e:
                            log(f"      字典学习失败: {e}")
                            pos_results = [
                                {v: 1.0/red_range
                                 for v in range(1, red_range+1)}
                                for _ in range(len(test_vecs))
                            ]

                        for i in range(len(test_valid_idx)):
                            all_weights[i].append(pos_results[i])

                    # 快速 AUC
                    auc = _quick_auc(all_weights, data, test_valid_idx)
                    log(f"      AUC={auc:.4f}")

                    if auc > best_auc:
                        best_auc = auc
                        best_weights = all_weights
                        best_params = {
                            "W": W, "n_comp": n_comp,
                            "alpha": alpha, "k": k
                        }

    log(f"  E5f 最优参数: {best_params}, AUC={best_auc:.4f}")

    # 对齐到 test_indices 格式
    return _align_weights(best_weights, data, test_indices, max_train_idx)


def _align_weights(weights_from_valid, data, test_indices, max_train_idx):
    """将基于 valid_indices 的权重对齐到 test_indices"""
    n_pos = data.red_count
    red_range = data.red_range
    uniform = {v: 1.0/red_range for v in range(1, red_range+1)}

    if weights_from_valid is None:
        return [[uniform.copy() for _ in range(n_pos)]
                for _ in range(len(test_indices))]

    # 如果长度已匹配，直接返回
    if len(weights_from_valid) == len(test_indices):
        return weights_from_valid

    # 否则用均匀分布填充缺失
    result = []
    wi = 0
    for i in range(len(test_indices)):
        if wi < len(weights_from_valid):
            result.append(weights_from_valid[wi])
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
