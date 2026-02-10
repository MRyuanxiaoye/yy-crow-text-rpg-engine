# -*- coding: utf-8 -*-
"""
E5c：SAX 符号化 + 频繁子串匹配

将数值序列转成符号串，用字符串算法找频繁模式，统计模式后续的号码分布。
主要评估级别：Level 1（方向级）+ Level 3（号码级）
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from research.data_loader import LotteryData
from research.experiment.utils import log, Timer
from research.experiment.e5_framework import (
    E5_RESULTS_DIR, SPLIT_CONFIG, evaluate_weights, evaluate_direction,
    direction_to_weights, save_json
)


# ============================================================
# SAX 符号化
# ============================================================

def sax_encode(series: np.ndarray, alphabet_size: int,
               breakpoints: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """将数值序列转为 SAX 符号序列

    Args:
        series: (n,) 数值序列
        alphabet_size: 字母表大小（3/5/7）
        breakpoints: 分位数边界，None 则自动计算

    Returns:
        symbols: (n,) 符号索引（0 ~ alphabet_size-1）
        breakpoints: 使用的分位数边界
    """
    if breakpoints is None:
        # 用等频分位数
        quantiles = np.linspace(0, 1, alphabet_size + 1)[1:-1]
        breakpoints = np.quantile(series, quantiles)

    symbols = np.digitize(series, breakpoints).astype(np.int8)
    return symbols, breakpoints


def multi_pos_sax(data: LotteryData, max_idx: int,
                  alphabet_size: int = 3) -> Tuple[Dict, Dict]:
    """对所有位置做 SAX 符号化

    Returns:
        pos_symbols: {pos: (n_draws,) 符号数组}
        pos_breakpoints: {pos: breakpoints}
    """
    pos_symbols = {}
    pos_breakpoints = {}

    for pos in range(data.red_count):
        series = data.position_series[pos].astype(np.float64)
        # 仅用训练期计算分位数
        train_series = series[:max_idx + 1]
        quantiles = np.linspace(0, 1, alphabet_size + 1)[1:-1]
        bp = np.quantile(train_series, quantiles)
        symbols = np.digitize(series, bp).astype(np.int8)
        pos_symbols[pos] = symbols
        pos_breakpoints[pos] = bp

    return pos_symbols, pos_breakpoints


# ============================================================
# 频繁子串挖掘
# ============================================================

def mine_frequent_substrings(symbols: np.ndarray, max_idx: int,
                             substr_len: int, min_freq: int = 5,
                             next_values: np.ndarray = None,
                             red_range: int = 35) -> Dict:
    """在训练期符号序列中挖掘频繁子串及其后续号码分布

    Args:
        symbols: (n,) 符号序列
        max_idx: 训练期最大索引
        substr_len: 子串长度 L
        min_freq: 最小频次
        next_values: (n,) 下一期号码值（用于统计后续分布）
        red_range: 号码范围

    Returns:
        pattern_db: {子串tuple: {"freq": int, "next_dist": {v: count}}}
    """
    pattern_db = defaultdict(lambda: {"freq": 0, "next_dist": defaultdict(int)})

    # 扫描训练期
    for t in range(substr_len - 1, max_idx):
        substr = tuple(symbols[t - substr_len + 1: t + 1])
        pattern_db[substr]["freq"] += 1
        # 记录下一期号码
        if next_values is not None and t + 1 < len(next_values):
            nv = int(next_values[t + 1])
            pattern_db[substr]["next_dist"][nv] += 1

    # 过滤低频
    filtered = {}
    for k, v in pattern_db.items():
        if v["freq"] >= min_freq:
            filtered[k] = {"freq": v["freq"], "next_dist": dict(v["next_dist"])}

    return filtered


# ============================================================
# SAX 预测
# ============================================================

def sax_predict_weights(symbols: np.ndarray, pattern_db: Dict,
                        test_idx: int, substr_len: int,
                        red_range: int, alphabet_size: int) -> Dict[int, float]:
    """用 SAX 模式匹配预测号码权重

    精确匹配 + 模糊匹配（允许1个符号不同）
    """
    query = tuple(symbols[test_idx - substr_len + 1: test_idx + 1])
    weights = defaultdict(float)

    # 精确匹配
    if query in pattern_db:
        entry = pattern_db[query]
        freq = entry["freq"]
        for v, cnt in entry["next_dist"].items():
            weights[v] += cnt * 2.0  # 精确匹配权重 x2

    # 模糊匹配：允许1个符号不同
    for offset in range(substr_len):
        for alt_sym in range(alphabet_size):
            if alt_sym == query[offset]:
                continue
            fuzzy = list(query)
            fuzzy[offset] = alt_sym
            fuzzy = tuple(fuzzy)
            if fuzzy in pattern_db:
                entry = pattern_db[fuzzy]
                for v, cnt in entry["next_dist"].items():
                    weights[v] += cnt * 0.5  # 模糊匹配权重 x0.5

    # 补全 + 归一化
    for v in range(1, red_range + 1):
        if v not in weights:
            weights[v] = 1e-10
    total = sum(weights.values())
    if total > 0:
        return {v: w / total for v, w in weights.items()}
    # 无匹配则均匀分布
    return {v: 1.0 / red_range for v in range(1, red_range + 1)}


# ============================================================
# 主运行函数
# ============================================================

def run_e5c(data: LotteryData, test_indices: np.ndarray,
            max_train_idx: int, **kwargs) -> List[List[Dict[int, float]]]:
    """E5c 主函数：多参数组合扫描，选最优"""

    n_pos = data.red_count
    red_range = data.red_range

    # 超参数候选
    alphabet_sizes = [3, 5, 7]
    substr_lens = [3, 5, 7, 10]
    min_freqs = [5, 10]

    best_auc = -1.0
    best_weights = None
    best_params = None

    for alpha in alphabet_sizes:
        # 符号化
        pos_symbols, _ = multi_pos_sax(data, max_train_idx, alpha)

        for L in substr_lens:
            for mf in min_freqs:
                # 对每个位置独立挖掘 + 预测
                all_weights = []
                for i, t in enumerate(test_indices):
                    t = int(t)
                    if t < L:
                        # 历史不够，均匀分布
                        all_weights.append([
                            {v: 1.0/red_range for v in range(1, red_range+1)}
                            for _ in range(n_pos)
                        ])
                        continue

                    sample_w = []
                    for pos in range(n_pos):
                        syms = pos_symbols[pos]
                        vals = data.position_series[pos]
                        # 挖掘模式库
                        pdb = mine_frequent_substrings(
                            syms, max_train_idx, L, mf, vals, red_range)
                        # 预测
                        w = sax_predict_weights(
                            syms, pdb, t, L, red_range, alpha)
                        sample_w.append(w)
                    all_weights.append(sample_w)

                # 快速评估 AUC
                auc = _quick_auc(all_weights, data, test_indices)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = all_weights
                    best_params = {"alpha": alpha, "L": L, "min_freq": mf}

    log(f"  E5c 最优参数: {best_params}, AUC={best_auc:.4f}")
    return best_weights


def _quick_auc(all_weights, data, test_indices):
    """快速计算整体 AUC（不保存中间结果）"""
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
            sorted_nums = sorted(w_dict.items(), key=lambda x: x[1], reverse=True)
            for r, (v, _) in enumerate(sorted_nums, 1):
                if v == true_val:
                    total_rank += r
                    count += 1
                    break

    if count == 0:
        return 0.5
    mean_rank = total_rank / count
    return 1.0 - (mean_rank - 1) / (red_range - 1)


if __name__ == "__main__":
    import argparse
    from research.experiment.utils import setup_logging
    from research.experiment.e5_framework import run_e5_sub

    parser = argparse.ArgumentParser()
    parser.add_argument("--lottery", default="daletou")
    args = parser.parse_args()

    setup_logging()
    run_e5_sub(args.lottery, "e5c", run_e5c)
