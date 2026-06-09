# -*- coding: utf-8 -*-
"""
E5：数据驱动模式发现实验（无假设局面匹配）

统一框架：局面向量构造、KNN预测、多粒度评估、信号融合
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_json, RESULTS_DIR
)
from research.experiment.e1_combo_evaluation import get_best_probs
from research.experiment.e3_rule_framework import compute_auc_and_survival

# === 常量 ===
E5_RESULTS_DIR = RESULTS_DIR / "e5_pattern_discovery"
E5_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 时间切分（与现有实验一致）
SPLIT_CONFIG = {
    "daletou": {"n_draws": 2833, "train_end": 2400, "test_start": 2401},
    "shuangseqiu": {"n_draws": 3413, "train_end": 2900, "test_start": 2901},
}


# ============================================================
# 共用局面向量构造
# ============================================================

def build_situation_vectors(data: LotteryData, max_idx: int, window: int = 10):
    """构造局面向量矩阵，E5b/d/e/f 共用

    每行 = [W期号码(展平) + (W-1)期方向 + (W-1)期差分]

    Args:
        data: LotteryData 实例
        max_idx: 最大可用索引（含），用于标准化
        window: 窗口期数 W

    Returns:
        vectors: (n_valid, feature_dim) 局面向量矩阵
        valid_indices: (n_valid,) 对应的期数索引（从 window-1 开始）
        scaler_params: (mean, std) 仅用训练期计算
    """
    n_pos = data.red_count
    red_range = data.red_range
    n_draws = data.n_draws

    # 收集所有有效索引（需要 window 期历史）
    all_vecs = []
    all_indices = []

    for t in range(window - 1, n_draws - 1):
        vec = []
        # W 期号码（归一化到 0-1）
        for offset in range(window):
            idx = t - window + 1 + offset
            for pos in range(n_pos):
                vec.append(data.red_matrix[idx, pos] / red_range)

        # (W-1) 期方向（-1/0/1）
        for offset in range(window - 1):
            idx = t - window + 1 + offset  # 方向序列比号码序列短1
            for pos in range(n_pos):
                vec.append(data.direction_series[pos][idx])

        # (W-1) 期差分（归一化）
        for offset in range(window - 1):
            idx = t - window + 1 + offset
            for pos in range(n_pos):
                diff_val = (data.position_series[pos][idx + 1]
                            - data.position_series[pos][idx])
                vec.append(diff_val / red_range)

        all_vecs.append(vec)
        all_indices.append(t)

    vectors = np.array(all_vecs, dtype=np.float32)
    valid_indices = np.array(all_indices, dtype=np.int32)

    # 用训练期数据计算标准化参数
    train_mask = valid_indices <= max_idx
    train_vecs = vectors[train_mask]
    mean = train_vecs.mean(axis=0)
    std = train_vecs.std(axis=0)
    std[std < 1e-8] = 1.0  # 避免除零

    # 标准化
    vectors = (vectors - mean) / std

    return vectors, valid_indices, (mean, std)


# ============================================================
# 共用 KNN 预测转权重
# ============================================================

def knn_to_weights(query_vec: np.ndarray, train_vecs: np.ndarray,
                   train_next_vals: np.ndarray, k: int = 20,
                   red_range: int = 35, metric: str = 'cosine') -> Dict[int, float]:
    """通用 KNN → 号码权重转换

    Args:
        query_vec: (d,) 查询向量
        train_vecs: (n_train, d) 训练集向量
        train_next_vals: (n_train,) 训练集下一期号码值
        k: 邻居数
        red_range: 号码范围上限
        metric: 距离度量 ('cosine' 或 'euclidean')

    Returns:
        {号码: 权重} 归一化字典
    """
    if metric == 'cosine':
        # 余弦距离
        q_norm = np.linalg.norm(query_vec)
        t_norms = np.linalg.norm(train_vecs, axis=1)
        if q_norm < 1e-10:
            dists = np.ones(len(train_vecs))
        else:
            sims = train_vecs @ query_vec / (t_norms * q_norm + 1e-10)
            dists = 1.0 - sims
    else:
        dists = np.linalg.norm(train_vecs - query_vec, axis=1)

    # 找 top-k 最近邻
    k = min(k, len(train_vecs))
    top_k_idx = np.argpartition(dists, k)[:k]
    top_k_dists = dists[top_k_idx]

    # 距离倒数加权
    weights = {}
    for idx, dist in zip(top_k_idx, top_k_dists):
        w = 1.0 / (dist + 1e-6)
        val = int(train_next_vals[idx])
        weights[val] = weights.get(val, 0.0) + w

    # 补全所有号码（未出现的给极小权重）
    for v in range(1, red_range + 1):
        if v not in weights:
            weights[v] = 1e-10

    # 归一化
    total = sum(weights.values())
    if total > 0:
        weights = {v: w / total for v, w in weights.items()}

    return weights


# ============================================================
# 多粒度评估函数
# ============================================================

def evaluate_direction(direction_preds: np.ndarray, data: LotteryData,
                       test_indices: np.ndarray) -> Dict:
    """Level 1：方向级评估

    Args:
        direction_preds: (n_samples, n_pos, 3) 方向概率 [P(D), P(E), P(U)]
        data: LotteryData
        test_indices: 测试期索引

    Returns:
        {"overall_acc": float, "per_pos": {pos: acc}, "confusion": ...}
    """
    n_samples = len(test_indices)
    n_pos = data.red_count
    correct = np.zeros(n_pos)
    total = 0

    for i in range(n_samples):
        t = int(test_indices[i])
        if t >= data.n_draws - 1:
            continue
        total += 1
        for pos in range(n_pos):
            # 真实方向：direction_series[pos][t] 是 t→t+1 的方向
            true_dir = int(data.direction_series[pos][t])  # -1/0/1
            # 转为索引：D=0, E=1, U=2
            true_idx = true_dir + 1  # -1→0, 0→1, 1→2
            pred_idx = int(np.argmax(direction_preds[i, pos, :]))
            if pred_idx == true_idx:
                correct[pos] += 1

    per_pos = {}
    for pos in range(n_pos):
        per_pos[f"P{pos}"] = float(correct[pos] / total) if total > 0 else 0.0

    return {
        "overall_acc": float(correct.sum() / (total * n_pos)) if total > 0 else 0.0,
        "per_pos": per_pos,
        "n_samples": total,
        "baseline": 1.0 / 3.0,
    }


def evaluate_range(candidate_sets: List[List[Set[int]]], data: LotteryData,
                   test_indices: np.ndarray) -> Dict:
    """Level 2：区间级评估

    Args:
        candidate_sets: candidates[i][pos] = {候选号码集合}
        data: LotteryData
        test_indices: 测试期索引

    Returns:
        {"avg_size": float, "survival": float, "reduction": float, "efficiency": float}
    """
    n_samples = len(test_indices)
    n_pos = data.red_count
    red_range = data.red_range

    sizes = []
    survivals = []

    for i in range(n_samples):
        t = int(test_indices[i])
        if t >= data.n_draws - 1:
            continue
        sample_survived = True
        sample_sizes = []
        for pos in range(n_pos):
            true_val = int(data.red_matrix[t + 1, pos])
            cands = candidate_sets[i][pos]
            sample_sizes.append(len(cands))
            if true_val not in cands:
                sample_survived = False
        sizes.append(np.mean(sample_sizes))
        survivals.append(1.0 if sample_survived else 0.0)

    avg_size = float(np.mean(sizes)) if sizes else red_range
    survival = float(np.mean(survivals)) if survivals else 0.0
    reduction = 1.0 - avg_size / red_range
    efficiency = reduction * survival

    return {
        "avg_candidate_size": avg_size,
        "survival": survival,
        "reduction": reduction,
        "efficiency": efficiency,
        "n_samples": len(sizes),
    }


def evaluate_weights(all_weights: List[List[Dict[int, float]]],
                     data: LotteryData, test_indices: np.ndarray) -> Dict:
    """Level 3：号码级评估（AUC + 排名）

    Args:
        all_weights: all_weights[i][pos] = {号码v: 权重w}
        data: LotteryData
        test_indices: 测试期索引

    Returns:
        {"overall_auc": float, "per_pos": {pos: auc}, "mean_rank": float}
    """
    n_pos = data.red_count
    red_range = data.red_range
    n_samples = len(test_indices)

    ranks = {pos: [] for pos in range(n_pos)}

    for i in range(n_samples):
        t = int(test_indices[i])
        if t >= data.n_draws - 1:
            continue
        for pos in range(n_pos):
            true_val = int(data.red_matrix[t + 1, pos])
            w_dict = all_weights[i][pos]
            sorted_nums = sorted(w_dict.items(), key=lambda x: x[1], reverse=True)
            found = False
            for r, (v, w) in enumerate(sorted_nums, 1):
                if v == true_val:
                    ranks[pos].append(r)
                    found = True
                    break
            if not found:
                ranks[pos].append(red_range)

    aucs = {}
    all_auc_vals = []
    all_ranks = []
    for pos in range(n_pos):
        r_arr = np.array(ranks[pos], dtype=float)
        auc = float(1.0 - (r_arr.mean() - 1) / (red_range - 1)) if red_range > 1 else 0.5
        aucs[f"P{pos}"] = auc
        all_auc_vals.append(auc)
        all_ranks.extend(ranks[pos])

    return {
        "overall_auc": float(np.mean(all_auc_vals)),
        "per_pos": aucs,
        "mean_rank": float(np.mean(all_ranks)),
        "n_samples": len(ranks[0]),
        "baseline_auc": 0.5,
    }


# ============================================================
# 权重转换工具
# ============================================================

def direction_to_weights(direction_preds: np.ndarray, data: LotteryData,
                         test_indices: np.ndarray) -> List[List[Dict[int, float]]]:
    """将方向概率转为号码权重（用均匀差分分布近似）

    Args:
        direction_preds: (n_samples, n_pos, 3) [P(D), P(E), P(U)]
        data: LotteryData
        test_indices: 测试期索引

    Returns:
        all_weights[i][pos] = {号码v: 权重w}
    """
    n_samples = len(test_indices)
    n_pos = data.red_count
    red_range = data.red_range
    all_weights = []

    for i in range(n_samples):
        t = int(test_indices[i])
        sample_weights = []
        for pos in range(n_pos):
            cur_val = int(data.red_matrix[t, pos])
            p_d, p_e, p_u = direction_preds[i, pos, :]
            weights = {}
            for v in range(1, red_range + 1):
                if v < cur_val:
                    weights[v] = p_d / max(cur_val - 1, 1)
                elif v == cur_val:
                    weights[v] = p_e
                else:
                    weights[v] = p_u / max(red_range - cur_val, 1)
            # 归一化
            total = sum(weights.values())
            if total > 0:
                weights = {v: w / total for v, w in weights.items()}
            sample_weights.append(weights)
        all_weights.append(sample_weights)

    return all_weights


def range_to_weights(candidate_sets: List[List[Set[int]]],
                     red_range: int) -> List[List[Dict[int, float]]]:
    """将候选区间转为号码权重（区间内均匀）"""
    all_weights = []
    for i in range(len(candidate_sets)):
        sample_weights = []
        for pos in range(len(candidate_sets[i])):
            cands = candidate_sets[i][pos]
            n_cands = len(cands) if cands else 1
            weights = {}
            for v in range(1, red_range + 1):
                weights[v] = 1.0 / n_cands if v in cands else 1e-10
            total = sum(weights.values())
            weights = {v: w / total for v, w in weights.items()}
            sample_weights.append(weights)
        all_weights.append(sample_weights)
    return all_weights


# ============================================================
# 信号融合
# ============================================================

def fuse_weights(weight_list: List[List[List[Dict[int, float]]]],
                 alphas: List[float],
                 red_range: int) -> List[List[Dict[int, float]]]:
    """多信号加权融合

    Args:
        weight_list: [signal_1_weights, signal_2_weights, ...]
                     每个 signal_x_weights[i][pos] = {v: w}
        alphas: 各信号权重
        red_range: 号码范围

    Returns:
        fused[i][pos] = {v: w}
    """
    n_signals = len(weight_list)
    n_samples = len(weight_list[0])
    n_pos = len(weight_list[0][0])

    fused = []
    for i in range(n_samples):
        sample = []
        for pos in range(n_pos):
            merged = {}
            for v in range(1, red_range + 1):
                w = 0.0
                for s in range(n_signals):
                    w += alphas[s] * weight_list[s][i][pos].get(v, 1e-10)
                merged[v] = w
            # 归一化
            total = sum(merged.values())
            if total > 0:
                merged = {v: w / total for v, w in merged.items()}
            sample.append(merged)
        fused.append(sample)
    return fused


# ============================================================
# 主入口
# ============================================================

def run_e5_sub(lottery_type: str, sub_name: str, run_func, **kwargs) -> Dict:
    """运行单个 E5 子实验并评估

    Args:
        lottery_type: 彩种
        sub_name: 子实验名称（如 'e5a', 'e5c'）
        run_func: 子实验函数，签名 (data, test_indices, max_train_idx, **kw)
                  返回 all_weights[i][pos] = {v: w}
        **kwargs: 传给 run_func 的额外参数

    Returns:
        评估结果字典
    """
    log(f"\n{'='*50}")
    log(f"  E5 子实验: {sub_name} [{lottery_type}]")
    log(f"{'='*50}")

    data = LotteryData(lottery_type)
    split = SPLIT_CONFIG[lottery_type]
    max_train_idx = split["train_end"]
    test_start = split["test_start"]
    test_indices = np.arange(test_start, data.n_draws - 1)

    with Timer(f"{sub_name} 运行"):
        all_weights = run_func(data, test_indices, max_train_idx, **kwargs)

    log(f"  输出样本数: {len(all_weights)}, 位置数: {len(all_weights[0])}")

    # Level 3 评估
    with Timer(f"{sub_name} 评估"):
        eval_result = evaluate_weights(all_weights, data, test_indices)

    log(f"  AUC: {eval_result['overall_auc']:.4f} "
        f"(基线 0.5, E3最优 ~0.89)")
    log(f"  平均排名: {eval_result['mean_rank']:.1f} / {data.red_range}")

    # 保存结果
    result = {
        "sub_experiment": sub_name,
        "lottery_type": lottery_type,
        "level3_eval": eval_result,
        "params": {k: str(v) for k, v in kwargs.items()},
    }
    save_json(result, E5_RESULTS_DIR / f"{sub_name}_{lottery_type}.json")

    return result


def run_e5_experiment(lottery_type: str, sub_experiments: List[str] = None):
    """E5 主入口"""
    setup_logging()
    log(f"\n{'#'*55}")
    log(f"  E5 数据驱动模式发现实验: {lottery_type}")
    log(f"{'#'*55}")

    all_results = {}

    # 动态导入子实验
    sub_map = {
        "e5c": ("research.experiment.e5c_sax", "run_e5c"),
        "e5a": ("research.experiment.e5a_matrix_profile", "run_e5a"),
        "e5f": ("research.experiment.e5f_dictionary", "run_e5f"),
        "e5b": ("research.experiment.e5b_autoencoder", "run_e5b"),
        "e5d": ("research.experiment.e5d_shapelet", "run_e5d"),
        "e5e": ("research.experiment.e5e_contrastive", "run_e5e"),
        "e5g": ("research.experiment.e5g_adaptive_features", "run_e5g"),
    }

    if sub_experiments is None:
        sub_experiments = list(sub_map.keys())

    for name in sub_experiments:
        if name not in sub_map:
            log(f"  跳过未知子实验: {name}")
            continue
        module_path, func_name = sub_map[name]
        try:
            import importlib
            mod = importlib.import_module(module_path)
            run_func = getattr(mod, func_name)
            result = run_e5_sub(lottery_type, name, run_func)
            all_results[name] = result
        except Exception as e:
            log(f"  {name} 失败: {e}")
            all_results[name] = {"error": str(e)}

    # 汇总
    summary = {
        "lottery_type": lottery_type,
        "sub_results": {
            k: v.get("level3_eval", {}).get("overall_auc", None)
            for k, v in all_results.items()
        },
    }
    save_json(summary, E5_RESULTS_DIR / f"e5_summary_{lottery_type}.json")
    log(f"\n  E5 汇总已保存")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E5 数据驱动模式发现")
    parser.add_argument("--lottery", default="daletou",
                        choices=["daletou", "shuangseqiu"])
    parser.add_argument("--sub", nargs="*", default=None,
                        help="指定子实验，如 e5c e5a")
    args = parser.parse_args()
    run_e5_experiment(args.lottery, args.sub)
