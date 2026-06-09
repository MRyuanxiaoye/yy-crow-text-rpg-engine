# -*- coding: utf-8 -*-
"""
E8：E5+E7 融合方法链端到端投注模拟

方法链：E5 权重筛选 → E7 约束传播 → 加权采样红球 → 蓝球预测 → 动态分配 → 奖金计算

滚动回测：每期完整重算（只用 [0, t-1] 数据），支持断点续跑。

用法:
    python -m research.experiment.e8_fusion_simulation daletou
    python -m research.experiment.e8_fusion_simulation shuangseqiu
"""

import sys
import argparse
import pickle
import time
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from math import comb as math_comb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_json, RESULTS_DIR
)
from research.experiment.e5_framework import (
    SPLIT_CONFIG, build_situation_vectors, knn_to_weights,
    evaluate_weights, fuse_weights, E5_RESULTS_DIR,
)
from research.experiment.e7_constraint_discovery import (
    enumerate_all_constraints, rank_and_filter_constraints,
    build_initial_candidates, ac3_propagate, apply_ordering_constraint,
    compute_context_features, classify_context, compute_volatility_thresholds,
    enumerate_conditioned_constraints,
)
from research.experiment.e0_step12_purchase_strategy import (
    PRIZE_TABLE, calculate_prize, TICKET_PRICE,
)

# === 路径 ===
E8_DIR = RESULTS_DIR / "e8_simulation"
E8_DIR.mkdir(parents=True, exist_ok=True)
E8_CACHE_DIR = E8_DIR / "cache"
E8_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === 常量 ===
MIN_TRAIN_PERIODS = 500   # 前 500 期不回测
BUDGET_BETS = 50          # 每期 50 注
BUDGET_YUAN = BUDGET_BETS * TICKET_PRICE  # 100 元
CHECKPOINT_INTERVAL = 50  # 每 50 期保存一次

# E5 累积概率截断阈值（越小候选越集中，越容易命中高等奖）
E5_CUMPROB_CUTOFF = 0.60

# E7 HDI 级别
E7_HDI_KEY = "hdi_90"


# ============================================================
#  E5 权重计算（预计算缓存 + KNN 回退）
# ============================================================

# 预计算缓存：{lottery_type: fused_weights}
_e5_cache = {}

# 最优融合 alpha（来自 E5 融合实验结果）
E5_BEST_ALPHAS = {
    "shuangseqiu": {
        "e5a": 0.119, "e5b": 0.077, "e5c": 0.428,
        "e5d": 0.107, "e5e": 0.080, "e5f": 0.190,
    },
    "daletou": {
        "e5a": 0.022, "e5b": 0.072, "e5c": 0.499,
        "e5d": 0.068, "e5e": 0.127, "e5f": 0.212,
    },
}
E5_SUB_NAMES = ["e5a", "e5b", "e5c", "e5d", "e5e", "e5f"]


def _load_e5_precomputed(lottery_type):
    """加载 E5 预计算权重缓存并融合，返回 fused[sample_idx][pos] = {v: w}"""
    if lottery_type in _e5_cache:
        return _e5_cache[lottery_type]

    cache_dir = E5_RESULTS_DIR / "weights_cache"
    alphas_dict = E5_BEST_ALPHAS[lottery_type]
    red_range = 33 if lottery_type == "shuangseqiu" else 35

    # 加载各子算法权重
    all_sub = []
    alphas = []
    for name in E5_SUB_NAMES:
        path = cache_dir / f"{name}_{lottery_type}_weights.pkl"
        if not path.exists():
            log(f"  [警告] E5 缓存不存在: {path.name}，将回退到 KNN")
            return None
        with open(path, "rb") as f:
            weights = pickle.load(f)
        all_sub.append(weights)
        alphas.append(alphas_dict[name])

    # 用 fuse_weights 融合
    fused = fuse_weights(all_sub, alphas, red_range)
    _e5_cache[lottery_type] = fused
    log(f"  E5 预计算缓存已加载: {lottery_type}, {len(fused)} 样本")
    return fused


# 子模型原始权重缓存
_e5_sub_cache = {}

def _load_e5_sub_weights(lottery_type):
    """加载 E5 各子模型的原始权重（融合前），用于诊断

    Returns:
        dict: {sub_name: weights_list}，weights_list[sample_idx][pos] = {v: w}
        如果缓存不存在返回 None
    """
    if lottery_type in _e5_sub_cache:
        return _e5_sub_cache[lottery_type]

    cache_dir = E5_RESULTS_DIR / "weights_cache"
    result = {}
    for name in E5_SUB_NAMES:
        path = cache_dir / f"{name}_{lottery_type}_weights.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            result[name] = pickle.load(f)

    _e5_sub_cache[lottery_type] = result
    return result


def _run_single_e5_knn(data, t, max_train_idx, window=10):
    """KNN 回退：运行单个窗口的 KNN，返回 pos_weights[pos] = {v: w}"""
    n_pos = data.red_count
    red_range = data.red_range
    uniform = [{v: 1.0 / red_range for v in range(1, red_range + 1)}
               for _ in range(n_pos)]

    vectors, valid_indices, (mean, std) = build_situation_vectors(
        data, max_train_idx, window=window)

    query_idx_in_valid = np.searchsorted(valid_indices, t)
    if query_idx_in_valid >= len(valid_indices) or valid_indices[query_idx_in_valid] != t:
        return uniform

    query_vec = vectors[query_idx_in_valid]
    train_mask = (valid_indices <= max_train_idx) & (valid_indices < t)
    train_vecs = vectors[train_mask]
    train_indices = valid_indices[train_mask]

    if len(train_vecs) < 10:
        return uniform

    pos_weights = []
    for pos in range(n_pos):
        next_vals = np.array([data.red_matrix[int(idx) + 1, pos]
                              for idx in train_indices])
        w = knn_to_weights(query_vec, train_vecs, next_vals,
                           k=20, red_range=red_range, metric='cosine')
        pos_weights.append(w)
    return pos_weights


def e5_filter_candidates(data, t):
    """E5 权重筛选：优先用预计算缓存，回退到 KNN

    Args:
        data: LotteryData
        t: 当前期索引（预测 t+1）

    Returns:
        candidates: list of set，每位置候选号码集
        weights: list of dict，每位置 {号码: 权重}
    """
    n_pos = data.red_count
    red_range = data.red_range
    lottery_type = data.lottery_type

    # 尝试从预计算缓存获取
    precomputed = _load_e5_precomputed(lottery_type)
    split = SPLIT_CONFIG[lottery_type]
    test_start = split["test_start"]

    if precomputed is not None:
        sample_idx = t - test_start + 1  # t 是用来预测 t+1 的，缓存索引对应 test_start 起
        if 0 <= sample_idx < len(precomputed):
            fused = precomputed[sample_idx]
        else:
            fused = _knn_fallback(data, t, n_pos, red_range)
    else:
        fused = _knn_fallback(data, t, n_pos, red_range)

    # 按累积概率截断
    candidates = []
    for pos in range(n_pos):
        sorted_nums = sorted(fused[pos].items(), key=lambda x: x[1], reverse=True)
        cum_prob = 0.0
        cand_set = set()
        for v, w in sorted_nums:
            cand_set.add(v)
            cum_prob += w
            if cum_prob >= E5_CUMPROB_CUTOFF:
                break
        if len(cand_set) < 3:
            for v, w in sorted_nums[:3]:
                cand_set.add(v)
        candidates.append(cand_set)

    return candidates, fused


def _knn_fallback(data, t, n_pos, red_range):
    """KNN 回退：用多窗口 KNN 等权融合"""
    windows = [5, 8, 10, 12, 15, 20]
    all_sub_weights = []
    for w in windows:
        pw = _run_single_e5_knn(data, t, t, window=w)
        all_sub_weights.append(pw)

    n_subs = len(all_sub_weights)
    fused = []
    for pos in range(n_pos):
        merged = {}
        for v in range(1, red_range + 1):
            w_sum = sum(sub[pos].get(v, 1e-10) for sub in all_sub_weights)
            merged[v] = w_sum / n_subs
        total = sum(merged.values())
        if total > 0:
            merged = {v: w / total for v, w in merged.items()}
        fused.append(merged)
    return fused


# ============================================================
#  E7 约束加载与位置过滤（拆分自 e7_constrain_candidates）
# ============================================================

# E7 约束缓存：避免每期重复枚举（训练期变化不大时复用）
_e7_cache = {"train_end": None, "selected": None}
E7_CACHE_REFRESH_INTERVAL = 50  # 每 50 期刷新一次约束


def e7_load_constraints(data, t):
    """加载/缓存 E7 约束（50 期刷新机制）

    Returns:
        selected: list of dict，筛选后的约束列表
    """
    global _e7_cache
    train_end = t + 1

    need_refresh = (
        _e7_cache["selected"] is None
        or _e7_cache["train_end"] is None
        or (train_end - _e7_cache["train_end"]) >= E7_CACHE_REFRESH_INTERVAL
    )

    if need_refresh:
        constraints = enumerate_all_constraints(data, train_end)
        selected, _ = rank_and_filter_constraints(
            constraints, data, train_end, train_end,
            hdi_key=E7_HDI_KEY, top_n=30)
        _e7_cache["train_end"] = train_end
        _e7_cache["selected"] = selected
    else:
        selected = _e7_cache["selected"]

    return selected


def e7_position_filter(data, t, e5_candidates):
    """仅用 A/C 类单位置约束收紧候选集

    Args:
        data: LotteryData
        t: 当前期索引
        e5_candidates: list of set，E5 输出的每位置候选集

    Returns:
        constrained: list of set，约束后的每位置候选集
    """
    selected = e7_load_constraints(data, t)
    n_pos = data.red_count
    red_range = data.red_range
    prev_vals = data.red_matrix[t]

    # 用 A 类 pos_diff 约束初始化
    e7_candidates = build_initial_candidates(
        prev_vals, selected, red_range, n_pos, hdi_key=E7_HDI_KEY)

    # 与 E5 候选取交集
    merged = []
    for pos in range(n_pos):
        intersection = e5_candidates[pos] & e7_candidates[pos]
        if len(intersection) >= 1:
            merged.append(intersection)
        else:
            merged.append(set(e5_candidates[pos]))

    # 排序约束
    merged = apply_ordering_constraint(merged, n_pos)

    # AC-3 传播
    merged = ac3_propagate(merged, selected, prev_vals, red_range,
                           n_pos, hdi_key=E7_HDI_KEY)

    # 再次排序约束
    merged = apply_ordering_constraint(merged, n_pos)

    # 安全检查
    for pos in range(n_pos):
        if len(merged[pos]) == 0:
            merged[pos] = set(e5_candidates[pos])

    return merged


# ============================================================
#  回溯枚举合法组合（E7 组合约束剪枝）
# ============================================================

def _extract_combo_constraints(selected_cons, hdi_key):
    """从约束列表中提取组合级约束参数

    Returns:
        gap_cons: {pos: (lo, hi)} — B 类间距约束
        triple_mid_cons: [(i, j, k, lo, hi)] — D 类三元组中间偏移约束
        sum_range: (lo, hi) — E 类和值约束
        span_range: (lo, hi) — E 类跨度约束
    """
    gap_cons = {}
    triple_mid_cons = []
    sum_range = (0, 9999)
    span_range = (0, 9999)

    for con in selected_cons:
        name = con["name"]
        hdi_lo, hdi_hi = con[hdi_key]

        if name.startswith("gap_P"):
            parts = name.split("_")
            g = int(parts[1][1])
            gap_cons[g] = (hdi_lo, hdi_hi)

        elif name.startswith("triple_mid_P"):
            parts = name.split("_")
            i = int(parts[2][1])
            j = int(parts[3][1])
            k = int(parts[4][1])
            triple_mid_cons.append((i, j, k, hdi_lo, hdi_hi))

        elif name == "agg_sum":
            sum_range = (hdi_lo, hdi_hi)

        elif name == "agg_span":
            span_range = (hdi_lo, hdi_hi)

    return gap_cons, triple_mid_cons, sum_range, span_range


def generate_valid_combos(candidates, e5_weights, selected_cons,
                          red_range, n_pos, hdi_key="hdi_90"):
    """回溯穷举所有满足组合约束的合法组合（无上限截断）

    剪枝策略：
    - 排序约束：v[pos] > v[pos-1]
    - B 类间距约束：gap_P*
    - D 类三元组约束：triple_mid_P*
    - E 类和值约束：累积和上下界
    - E 类跨度约束：v[-1] - v[0]

    Args:
        candidates: list of set，每位置候选号码集
        e5_weights: list of dict，每位置 {号码: 权重}
        selected_cons: E7 约束列表
        red_range: 号码范围上限
        n_pos: 位置数
        hdi_key: HDI 级别

    Returns:
        list of (combo_tuple, score)，按联合权重降序排列
    """
    cands = [sorted(c) for c in candidates]
    gap_cons, triple_mid_cons, sum_range, span_range = \
        _extract_combo_constraints(selected_cons, hdi_key)
    sum_lo, sum_hi = sum_range
    span_lo, span_hi = span_range

    # 预计算每位置的 log 权重（用于联合得分）
    log_weights = []
    for pos in range(n_pos):
        lw = {}
        for v in cands[pos]:
            w = e5_weights[pos].get(v, 1e-10)
            lw[v] = np.log(max(w, 1e-10))
        log_weights.append(lw)

    results = []

    def backtrack(pos, chosen, current_sum, current_log_score):
        if pos == n_pos:
            # 终止检查：跨度约束
            span = chosen[-1] - chosen[0]
            if not (span_lo <= span <= span_hi):
                return
            # 和值约束（精确检查）
            if not (sum_lo <= current_sum <= sum_hi):
                return
            results.append((tuple(chosen), current_log_score))
            return

        min_val = chosen[-1] + 1 if chosen else 1

        for v in cands[pos]:
            if v < min_val:
                continue

            # B 类间距剪枝
            if pos > 0 and (pos - 1) in gap_cons:
                gap = v - chosen[-1]
                g_lo, g_hi = gap_cons[pos - 1]
                if gap < g_lo or gap > g_hi:
                    continue

            # E 类和值上界剪枝
            remaining = n_pos - pos - 1
            min_rem = sum(v + k for k in range(1, remaining + 1))
            if current_sum + v + min_rem > sum_hi:
                continue

            # E 类和值下界剪枝
            max_rem = sum(red_range - k for k in range(remaining))
            if current_sum + v + max_rem < sum_lo:
                continue

            # D 类三元组中间偏移剪枝（当三个位置都已确定时检查）
            triple_ok = True
            for (ti, tj, tk, t_lo, t_hi) in triple_mid_cons:
                if tk == pos and tj < pos and ti < pos:
                    # i, j, k 都已确定
                    vi = chosen[ti]
                    vj = chosen[tj]
                    mid_offset = vj - (vi + v) / 2.0
                    if not (t_lo <= mid_offset <= t_hi):
                        triple_ok = False
                        break
                elif tj == pos and ti < pos and tk > pos:
                    # i 已确定，j=当前，k 未确定 — 无法完全检查，跳过
                    pass
            if not triple_ok:
                continue

            new_log_score = current_log_score + log_weights[pos].get(v, -23.0)
            backtrack(pos + 1, chosen + [v], current_sum + v, new_log_score)

    backtrack(0, [], 0, 0.0)

    # 按联合权重降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ============================================================
#  蓝球预测
# ============================================================

def predict_blue_topk(data, t, k):
    """蓝球预测：基于历史频率 + 方向惯性的简化预测

    Args:
        data: LotteryData
        t: 当前期索引
        k: 返回 top-K 个蓝球（组合）

    Returns:
        大乐透: list of tuple，每个 tuple 是 (b1, b2) 后区组合
        双色球: list of int，每个是蓝球号码
    """
    blue_range = data.blue_range
    blue_count = data.blue_count
    blue_matrix = data.blue_matrix

    # 用最近 100 期的频率作为基础权重
    lookback = min(100, t)
    recent = blue_matrix[t - lookback:t + 1]

    if blue_count == 1:
        # 双色球：1 个蓝球
        freq = np.zeros(blue_range + 1)
        for row in recent:
            freq[int(row[0])] += 1
        freq = freq[1:]  # 去掉 0 号

        # 加入方向惯性
        if t >= 1:
            last_blue = int(blue_matrix[t, 0])
            prev_blue = int(blue_matrix[t - 1, 0]) if t >= 1 else last_blue
            direction = last_blue - prev_blue
            # 惯性加权：同方向号码权重 +20%
            for v in range(1, blue_range + 1):
                if direction > 0 and v > last_blue:
                    freq[v - 1] *= 1.2
                elif direction < 0 and v < last_blue:
                    freq[v - 1] *= 1.2
                elif direction == 0 and v == last_blue:
                    freq[v - 1] *= 1.1

        # 归一化
        freq = freq / freq.sum() if freq.sum() > 0 else np.ones(blue_range) / blue_range

        # 返回 top-K
        sorted_idx = np.argsort(-freq)
        return [int(sorted_idx[i]) + 1 for i in range(min(k, blue_range))]

    else:
        # 大乐透：2 个蓝球，从 1-blue_range 中选 2
        freq = np.zeros(blue_range + 1)
        for row in recent:
            for b in range(blue_count):
                freq[int(row[b])] += 1
        freq = freq[1:]

        # 方向惯性
        if t >= 1:
            for b in range(blue_count):
                last_b = int(blue_matrix[t, b])
                prev_b = int(blue_matrix[t - 1, b]) if t >= 1 else last_b
                direction = last_b - prev_b
                for v in range(1, blue_range + 1):
                    if direction > 0 and v > last_b:
                        freq[v - 1] *= 1.15
                    elif direction < 0 and v < last_b:
                        freq[v - 1] *= 1.15

        freq = freq / freq.sum() if freq.sum() > 0 else np.ones(blue_range) / blue_range

        # 生成所有 C(blue_range, 2) 组合，按联合概率排序
        all_combos = []
        for b1, b2 in combinations(range(1, blue_range + 1), 2):
            prob = freq[b1 - 1] * freq[b2 - 1]
            all_combos.append(((b1, b2), prob))
        all_combos.sort(key=lambda x: x[1], reverse=True)

        return [c[0] for c in all_combos[:k]]


# ============================================================
#  动态注数分配
# ============================================================

def allocate_bets(red_combos, blue_preds, budget_bets, lottery_type):
    """将红球组合 × 蓝球预测分配到 budget_bets 注以内

    策略：红球按权重排序取 top，蓝球取 top-K，交叉组合不超过预算。

    Args:
        red_combos: list of tuple，红球组合（已按权重降序）
        blue_preds: 蓝球预测列表
        budget_bets: 最大注数
        lottery_type: 彩种

    Returns:
        tickets: list of dict，每注 {'red': tuple, 'blue': tuple/int}
    """
    tickets = []

    if lottery_type == "shuangseqiu":
        # 双色球：每注 = 1 红球组合 + 1 蓝球
        # 分配：top-N 红球 × top-M 蓝球，N*M <= budget
        n_blue = min(3, len(blue_preds))
        n_red = min(len(red_combos), budget_bets // max(n_blue, 1))
        for r in red_combos[:n_red]:
            for b in blue_preds[:n_blue]:
                if len(tickets) >= budget_bets:
                    break
                tickets.append({"red": r, "blue": (b,)})
            if len(tickets) >= budget_bets:
                break
    else:
        # 大乐透：每注 = 1 红球组合 + 1 蓝球组合(2个)
        n_blue = min(3, len(blue_preds))
        n_red = min(len(red_combos), budget_bets // max(n_blue, 1))
        for r in red_combos[:n_red]:
            for b in blue_preds[:n_blue]:
                if len(tickets) >= budget_bets:
                    break
                tickets.append({"red": r, "blue": b})
            if len(tickets) >= budget_bets:
                break

    return tickets


# ============================================================
#  单期模拟
# ============================================================

def simulate_one_period(data, t, budget_bets, lottery_type, rng):
    """对第 t 期执行完整方法链，返回投注和中奖结果

    方法链：E5 权重 → E7 位置过滤 → 回溯枚举合法组合 → 蓝球预测 → 分配 → 开奖

    Args:
        data: LotteryData
        t: 当前期索引（用 [0,t] 预测 t+1）
        budget_bets: 注数预算
        lottery_type: 彩种
        rng: numpy RandomState（保留接口兼容，不再用于采样）

    Returns:
        result: dict 包含投注、中奖、奖金等信息
    """
    n_pos = data.red_count
    red_range = data.red_range

    # --- Step 1: E5 权重（加载预计算缓存或回退 KNN）---
    e5_candidates, e5_weights = e5_filter_candidates(data, t)
    e5_sizes = [len(c) for c in e5_candidates]

    # --- Step 2: A 类单位置约束收紧候选集 ---
    constrained = e7_position_filter(data, t, e5_candidates)
    e7_sizes = [len(c) for c in constrained]

    # --- Step 3: 回溯枚举合法组合（B/D/E 类组合约束剪枝）---
    selected_cons = e7_load_constraints(data, t)
    valid_combos = generate_valid_combos(
        constrained, e5_weights, selected_cons,
        red_range, n_pos, hdi_key=E7_HDI_KEY)

    n_valid_combos = len(valid_combos)

    # --- Step 4: 按联合权重排序，取 top ---
    # valid_combos 已按 log_score 降序排列
    top_red = [c[0] for c in valid_combos[:budget_bets * 2]]

    # 如果合法组合不足，回退到逐位置独立组合
    if len(top_red) == 0:
        # 回退：从候选集中取笛卡尔积的 top 组合
        from itertools import product
        all_cands = [sorted(c) for c in constrained]
        fallback = []
        for combo in product(*all_cands):
            if all(combo[i] < combo[i+1] for i in range(len(combo)-1)):
                score = sum(np.log(max(e5_weights[p].get(v, 1e-10), 1e-10))
                            for p, v in enumerate(combo))
                fallback.append((combo, score))
            if len(fallback) >= budget_bets * 2:
                break
        fallback.sort(key=lambda x: x[1], reverse=True)
        top_red = [c[0] for c in fallback[:budget_bets * 2]]

    # --- Step 5: 蓝球预测 ---
    blue_preds = predict_blue_topk(data, t, k=5)

    # --- Step 6: 动态分配 ---
    tickets = allocate_bets(top_red, blue_preds, budget_bets, lottery_type)

    # --- Step 7: 开奖计算 ---
    true_red = set(data.red_matrix[t + 1].tolist())
    true_blue = set(data.blue_matrix[t + 1].tolist())

    total_prize = 0
    prize_details = defaultdict(int)
    best_prize = 0
    best_red_hit = 0
    best_blue_hit = 0
    hit_distribution = defaultdict(int)

    for ticket in tickets:
        pred_red = set(ticket["red"])
        pred_blue = set(ticket["blue"])
        prize = calculate_prize(true_red, true_blue, pred_red, pred_blue,
                                lottery_type)
        red_hit = len(true_red & pred_red)
        blue_hit = len(true_blue & pred_blue)
        hit_distribution[(red_hit, blue_hit)] += 1

        total_prize += prize
        if prize > 0:
            prize_details[prize] += 1
        if prize > best_prize:
            best_prize = prize
        if red_hit > best_red_hit or (red_hit == best_red_hit and blue_hit > best_blue_hit):
            best_red_hit = red_hit
            best_blue_hit = blue_hit

    cost = len(tickets) * TICKET_PRICE
    net = total_prize - cost

    return {
        "period_idx": t + 1,
        "n_tickets": len(tickets),
        "cost": cost,
        "total_prize": total_prize,
        "net": net,
        "best_prize": best_prize,
        "best_red_hit": best_red_hit,
        "best_blue_hit": best_blue_hit,
        "hit_distribution": {f"{k[0]}r{k[1]}b": v for k, v in hit_distribution.items()},
        "prize_details": dict(prize_details),
        "e5_cand_sizes": e5_sizes,
        "e7_cand_sizes": e7_sizes,
        "n_red_combos": len(top_red),
        "n_valid_combos": n_valid_combos,
        "true_red": sorted(true_red),
        "true_blue": sorted(true_blue),
    }


# ============================================================
#  诊断：追踪真实号码在各阶段的存活情况
# ============================================================

def diagnose_one_period(data, t, budget_bets, lottery_type):
    """诊断单期：逐步追踪真实中奖号码在每个子步骤的存活情况

    追踪阶段：
      E5-1: 各子模型单独排名
      E5-2: 融合后排名
      E5-3: 累积概率截断
      E7-1: A 类 build_initial_candidates
      E7-2: E5 ∩ E7 交集
      E7-3: apply_ordering_constraint（第一次）
      E7-4: ac3_propagate
      E7-5: apply_ordering_constraint（第二次）
      COMBO: B/D/E 类组合约束
      TOP: 排名截断

    Returns:
        diag: dict
    """
    n_pos = data.red_count
    red_range = data.red_range
    true_red = data.red_matrix[t + 1].tolist()
    true_blue = data.blue_matrix[t + 1].tolist()

    diag = {
        "period_idx": t + 1,
        "true_red": true_red,
        "true_blue": true_blue,
        "stages": {},
        "first_kill": {},  # {pos: stage_name} 每位置首次被排除的阶段
    }

    # 辅助：记录每位置存活状态
    alive = [True] * n_pos  # 当前是否存活

    def _check_alive(stage_name, cand_sets):
        """检查真实号码在 cand_sets 中的存活情况，更新 alive 和 first_kill"""
        detail = []
        for pos in range(n_pos):
            tv = true_red[pos]
            in_set = tv in cand_sets[pos]
            newly_killed = alive[pos] and not in_set
            if newly_killed:
                alive[pos] = False
                diag["first_kill"][pos] = stage_name
            detail.append({
                "pos": pos, "true_val": tv, "in_set": in_set,
                "cand_size": len(cand_sets[pos]),
                "newly_killed": newly_killed,
            })
        return detail

    # ================================================================
    # E5-1: 各子模型单独排名
    # ================================================================
    split = SPLIT_CONFIG[lottery_type]
    test_start = split["test_start"]
    sample_idx = t - test_start + 1

    sub_weights = _load_e5_sub_weights(lottery_type)
    stage_sub = {}
    if sub_weights is not None:
        for sub_name in E5_SUB_NAMES:
            sw = sub_weights[sub_name]
            if 0 <= sample_idx < len(sw):
                pos_ranks = []
                for pos in range(n_pos):
                    tv = true_red[pos]
                    sorted_nums = sorted(sw[sample_idx][pos].items(),
                                         key=lambda x: x[1], reverse=True)
                    rank = next((i + 1 for i, (v, _) in enumerate(sorted_nums)
                                 if v == tv), -1)
                    weight = sw[sample_idx][pos].get(tv, 0)
                    pos_ranks.append({"pos": pos, "true_val": tv,
                                      "rank": rank, "weight": round(weight, 6)})
                stage_sub[sub_name] = pos_ranks
    diag["stages"]["e5_1_sub_models"] = stage_sub

    # ================================================================
    # E5-2: 融合后排名（截断前）
    # ================================================================
    precomputed = _load_e5_precomputed(lottery_type)
    if precomputed is not None and 0 <= sample_idx < len(precomputed):
        fused = precomputed[sample_idx]
    else:
        fused = _knn_fallback(data, t, n_pos, red_range)

    stage_fused = []
    for pos in range(n_pos):
        tv = true_red[pos]
        sorted_nums = sorted(fused[pos].items(), key=lambda x: x[1], reverse=True)
        rank = next((i + 1 for i, (v, _) in enumerate(sorted_nums) if v == tv), -1)
        weight = fused[pos].get(tv, 0)
        # 计算截断位置（累积概率达到阈值时的排名）
        cum = 0.0
        cutoff_rank = 0
        for i, (v, w) in enumerate(sorted_nums):
            cum += w
            cutoff_rank = i + 1
            if cum >= E5_CUMPROB_CUTOFF:
                break
        stage_fused.append({
            "pos": pos, "true_val": tv,
            "fused_rank": rank, "fused_weight": round(weight, 6),
            "cutoff_rank": cutoff_rank,  # 截断保留了前几名
            "survived_cutoff": rank <= max(cutoff_rank, 3),
        })
    diag["stages"]["e5_2_fused_rank"] = stage_fused

    # ================================================================
    # E5-3: 累积概率截断
    # ================================================================
    e5_candidates, e5_weights = e5_filter_candidates(data, t)
    detail_e5_cut = _check_alive("e5_3_cutoff", e5_candidates)
    diag["stages"]["e5_3_cutoff"] = {
        "pos_detail": detail_e5_cut,
        "cand_sizes": [len(c) for c in e5_candidates],
    }

    # ================================================================
    # E7-1: A 类 build_initial_candidates
    # ================================================================
    selected_cons = e7_load_constraints(data, t)
    prev_vals = data.red_matrix[t]
    e7_initial = build_initial_candidates(
        prev_vals, selected_cons, red_range, n_pos, hdi_key=E7_HDI_KEY)
    detail_e7_init = _check_alive("e7_1_initial", e7_initial)
    diag["stages"]["e7_1_initial"] = {
        "pos_detail": detail_e7_init,
        "cand_sizes": [len(c) for c in e7_initial],
    }

    # ================================================================
    # E7-2: E5 ∩ E7 交集
    # ================================================================
    merged = []
    for pos in range(n_pos):
        intersection = e5_candidates[pos] & e7_initial[pos]
        if len(intersection) >= 1:
            merged.append(intersection)
        else:
            merged.append(set(e5_candidates[pos]))
    detail_intersect = _check_alive("e7_2_intersect", merged)
    diag["stages"]["e7_2_intersect"] = {
        "pos_detail": detail_intersect,
        "cand_sizes": [len(c) for c in merged],
    }

    # ================================================================
    # E7-3: apply_ordering_constraint（第一次）
    # ================================================================
    ordered1 = apply_ordering_constraint(list(set(s) for s in merged), n_pos)
    detail_ord1 = _check_alive("e7_3_ordering1", ordered1)
    diag["stages"]["e7_3_ordering1"] = {
        "pos_detail": detail_ord1,
        "cand_sizes": [len(c) for c in ordered1],
    }

    # ================================================================
    # E7-4: ac3_propagate
    # ================================================================
    ac3_result = ac3_propagate(
        list(set(s) for s in ordered1), selected_cons, prev_vals,
        red_range, n_pos, hdi_key=E7_HDI_KEY)
    detail_ac3 = _check_alive("e7_4_ac3", ac3_result)
    diag["stages"]["e7_4_ac3"] = {
        "pos_detail": detail_ac3,
        "cand_sizes": [len(c) for c in ac3_result],
    }

    # ================================================================
    # E7-5: apply_ordering_constraint（第二次）
    # ================================================================
    ordered2 = apply_ordering_constraint(list(set(s) for s in ac3_result), n_pos)
    # 安全检查
    for pos in range(n_pos):
        if len(ordered2[pos]) == 0:
            ordered2[pos] = set(e5_candidates[pos])
    detail_ord2 = _check_alive("e7_5_ordering2", ordered2)
    diag["stages"]["e7_5_ordering2"] = {
        "pos_detail": detail_ord2,
        "cand_sizes": [len(c) for c in ordered2],
    }

    # ================================================================
    # COMBO: 组合约束剪枝
    # ================================================================
    all_pos_survived = all(alive)

    if all_pos_survived:
        valid_combos = generate_valid_combos(
            ordered2, e5_weights, selected_cons,
            red_range, n_pos, hdi_key=E7_HDI_KEY)

        true_combo = tuple(true_red)
        combo_in_valid = any(c[0] == true_combo for c in valid_combos)
        combo_killed_by = None
        if not combo_in_valid:
            combo_killed_by = _diagnose_combo_constraints(
                true_red, selected_cons, E7_HDI_KEY)

        combo_rank = -1
        if combo_in_valid:
            for i, (c, s) in enumerate(valid_combos):
                if c == true_combo:
                    combo_rank = i + 1
                    break

        diag["stages"]["combo_constraint"] = {
            "combo_in_valid_set": combo_in_valid,
            "combo_rank": combo_rank,
            "total_valid_combos": len(valid_combos),
            "killed_by_constraint": combo_killed_by,
        }

        # TOP: 排名截断
        top_n = budget_bets * 2
        in_top = combo_rank != -1 and combo_rank <= top_n
        diag["stages"]["top_rank_cutoff"] = {
            "top_n": top_n,
            "combo_rank": combo_rank,
            "in_top": in_top,
        }
    else:
        diag["stages"]["combo_constraint"] = {"skipped": True,
                                               "reason": "位置级已排除"}
        diag["stages"]["top_rank_cutoff"] = {"skipped": True}
        combo_in_valid = False
        in_top = False
        combo_rank = -1

    # ================================================================
    # 最终判定
    # ================================================================
    if not all(alive[p] for p in range(n_pos)):
        killed_pos = {p: diag["first_kill"][p] for p in range(n_pos) if not alive[p]}
        diag["final_verdict"] = f"位置级排除: {killed_pos}"
    elif not combo_in_valid:
        diag["final_verdict"] = f"组合约束排除: {diag['stages']['combo_constraint'].get('killed_by_constraint')}"
    elif not in_top:
        diag["final_verdict"] = f"排名截断: rank={combo_rank}/{diag['stages']['combo_constraint']['total_valid_combos']}, top={budget_bets*2}"
    else:
        diag["final_verdict"] = f"存活! rank={combo_rank}"

    return diag


def _diagnose_combo_constraints(true_red, selected_cons, hdi_key):
    """检查真实组合违反了哪些组合约束"""
    violations = []
    for con in selected_cons:
        name = con["name"]
        hdi_lo, hdi_hi = con[hdi_key]

        if name.startswith("gap_P"):
            parts = name.split("_")
            g = int(parts[1][1])
            if g + 1 < len(true_red):
                gap = true_red[g + 1] - true_red[g]
                if gap < hdi_lo or gap > hdi_hi:
                    violations.append(f"{name}: gap={gap}, hdi=[{hdi_lo},{hdi_hi}]")

        elif name.startswith("triple_mid_P"):
            parts = name.split("_")
            i = int(parts[2][1])
            j = int(parts[3][1])
            k = int(parts[4][1])
            if max(i, j, k) < len(true_red):
                mid_offset = true_red[j] - (true_red[i] + true_red[k]) / 2.0
                if not (hdi_lo <= mid_offset <= hdi_hi):
                    violations.append(
                        f"{name}: mid_offset={mid_offset:.1f}, hdi=[{hdi_lo},{hdi_hi}]")

        elif name == "agg_sum":
            s = sum(true_red)
            if s < hdi_lo or s > hdi_hi:
                violations.append(f"agg_sum: sum={s}, hdi=[{hdi_lo},{hdi_hi}]")

        elif name == "agg_span":
            span = true_red[-1] - true_red[0]
            if span < hdi_lo or span > hdi_hi:
                violations.append(f"agg_span: span={span}, hdi=[{hdi_lo},{hdi_hi}]")

    return violations if violations else ["未知（可能是排序约束）"]


def run_diagnosis(lottery_type, n_periods=20, budget_bets=10):
    """对测试集前 n_periods 期运行细粒度诊断"""
    setup_logging()
    log(f"\n{'='*60}")
    log(f"  E8 诊断模式: {lottery_type}, {n_periods} 期")
    log(f"{'='*60}\n")

    data = LotteryData(lottery_type)
    split = SPLIT_CONFIG[lottery_type]
    start_t = split["test_start"] - 1
    end_t = min(start_t + n_periods - 1, data.n_draws - 2)
    n_pos = data.red_count

    all_diags = []
    # 各阶段首次排除计数（按位置）
    # stage_name -> {pos -> count}
    first_kill_by_stage = defaultdict(lambda: defaultdict(int))
    # 各阶段首次排除总计
    first_kill_stage_total = defaultdict(int)
    # 最终判定分布
    verdict_counts = defaultdict(int)

    # E5 子模型排名统计
    sub_rank_sums = defaultdict(lambda: defaultdict(list))  # {sub: {pos: [ranks]}}
    fused_rank_lists = defaultdict(list)  # {pos: [ranks]}

    for t in range(start_t, end_t + 1):
        diag = diagnose_one_period(data, t, budget_bets, lottery_type)
        all_diags.append(diag)

        # 收集 first_kill
        for pos, stage in diag["first_kill"].items():
            first_kill_by_stage[stage][pos] += 1
            first_kill_stage_total[stage] += 1

        # 最终判定
        v = diag["final_verdict"]
        if v.startswith("位置级"):
            verdict_counts["位置级排除"] += 1
        elif v.startswith("组合约束"):
            verdict_counts["组合约束排除"] += 1
        elif v.startswith("排名截断"):
            verdict_counts["排名截断"] += 1
        elif v.startswith("存活"):
            verdict_counts["存活"] += 1

        # E5 子模型排名
        for sub_name, pos_ranks in diag["stages"]["e5_1_sub_models"].items():
            for pr in pos_ranks:
                sub_rank_sums[sub_name][pr["pos"]].append(pr["rank"])

        # 融合排名
        for pf in diag["stages"]["e5_2_fused_rank"]:
            fused_rank_lists[pf["pos"]].append(pf["fused_rank"])

        # 逐期简要输出
        log(f"  期 {diag['period_idx']}: {v}")

    n = len(all_diags)

    # ============================================================
    # 汇总输出
    # ============================================================
    log(f"\n{'='*60}")
    log(f"  诊断汇总 ({n} 期, {n_pos} 位置)")
    log(f"{'='*60}")

    # 1. 最终判定分布
    log(f"\n  [最终判定分布]")
    for k in ["位置级排除", "组合约束排除", "排名截断", "存活"]:
        cnt = verdict_counts.get(k, 0)
        log(f"    {k}: {cnt} 期 ({cnt/n*100:.1f}%)")

    # 2. 首次排除阶段分布（按阶段汇总）
    stage_order = ["e5_3_cutoff", "e7_1_initial", "e7_2_intersect",
                   "e7_3_ordering1", "e7_4_ac3", "e7_5_ordering2"]
    stage_labels = {
        "e5_3_cutoff": "E5 累积概率截断",
        "e7_1_initial": "E7 A类约束(build_initial)",
        "e7_2_intersect": "E7 E5∩E7交集",
        "e7_3_ordering1": "E7 排序约束(第1次)",
        "e7_4_ac3": "E7 AC-3传播",
        "e7_5_ordering2": "E7 排序约束(第2次)",
    }
    log(f"\n  [首次排除阶段分布] (位置×期 总计={n * n_pos})")
    total_killed = sum(first_kill_stage_total.values())
    for stage in stage_order:
        cnt = first_kill_stage_total.get(stage, 0)
        if cnt > 0:
            pos_detail = ", ".join(f"P{p}={first_kill_by_stage[stage][p]}"
                                   for p in range(n_pos)
                                   if first_kill_by_stage[stage][p] > 0)
            log(f"    {stage_labels.get(stage, stage)}: {cnt} 次 "
                f"({cnt/total_killed*100:.1f}% of killed) [{pos_detail}]")

    # 3. E5 子模型排名对比
    log(f"\n  [E5 子模型 真实号码平均排名] (越小越好)")
    header = "    " + "".join(f"{'P'+str(p):>8}" for p in range(n_pos)) + "    平均"
    log(header)
    for sub_name in E5_SUB_NAMES:
        row = f"    {sub_name}"
        pos_avgs = []
        for pos in range(n_pos):
            ranks = sub_rank_sums[sub_name].get(pos, [])
            avg = np.mean(ranks) if ranks else 0
            pos_avgs.append(avg)
            row += f"{avg:8.1f}"
        row += f"{np.mean(pos_avgs):8.1f}"
        log(row)
    # 融合排名
    row = "    fused"
    pos_avgs = []
    for pos in range(n_pos):
        ranks = fused_rank_lists.get(pos, [])
        avg = np.mean(ranks) if ranks else 0
        pos_avgs.append(avg)
        row += f"{avg:8.1f}"
    row += f"{np.mean(pos_avgs):8.1f}"
    log(row)

    # 4. E5 融合排名 vs 截断位置
    log(f"\n  [E5 截断分析] 融合排名 vs 截断保留数")
    survived_cutoff = 0
    killed_cutoff = 0
    for diag in all_diags:
        for pf in diag["stages"]["e5_2_fused_rank"]:
            if pf["survived_cutoff"]:
                survived_cutoff += 1
            else:
                killed_cutoff += 1
    total_pos = n * n_pos
    log(f"    截断后存活: {survived_cutoff}/{total_pos} ({survived_cutoff/total_pos*100:.1f}%)")
    log(f"    截断后排除: {killed_cutoff}/{total_pos} ({killed_cutoff/total_pos*100:.1f}%)")
    # 被截断排除的号码的平均融合排名
    killed_ranks = []
    for diag in all_diags:
        for pf in diag["stages"]["e5_2_fused_rank"]:
            if not pf["survived_cutoff"]:
                killed_ranks.append(pf["fused_rank"])
    if killed_ranks:
        log(f"    被截断号码的平均融合排名: {np.mean(killed_ranks):.1f} "
            f"(中位数={np.median(killed_ranks):.0f})")

    # 5. 被截断位置的子模型溯源：融合帮了还是害了？
    log(f"\n  [子模型溯源：被E5截断的位置]")
    # 收集被截断位置的各子模型排名
    fusion_helped = 0  # 融合后排名比最佳子模型更好
    fusion_hurt = 0    # 融合后排名比最佳子模型更差
    fusion_same = 0
    # 各子模型在被截断位置的排名分布
    sub_killed_ranks = defaultdict(list)
    fused_killed_ranks = []
    # 被截断位置中，哪个子模型给了最好排名
    best_sub_counts = defaultdict(int)

    for diag in all_diags:
        sub_data = diag["stages"]["e5_1_sub_models"]
        fused_data = diag["stages"]["e5_2_fused_rank"]
        for pf in fused_data:
            if pf["survived_cutoff"]:
                continue
            pos = pf["pos"]
            fused_rank = pf["fused_rank"]
            fused_killed_ranks.append(fused_rank)

            # 各子模型在该位置的排名
            best_sub_rank = 999
            best_sub_name = None
            for sub_name, pos_ranks in sub_data.items():
                sr = pos_ranks[pos]["rank"]
                sub_killed_ranks[sub_name].append(sr)
                if sr < best_sub_rank:
                    best_sub_rank = sr
                    best_sub_name = sub_name

            if best_sub_name:
                best_sub_counts[best_sub_name] += 1

            if fused_rank < best_sub_rank:
                fusion_helped += 1
            elif fused_rank > best_sub_rank:
                fusion_hurt += 1
            else:
                fusion_same += 1

    total_killed = len(fused_killed_ranks)
    if total_killed > 0:
        log(f"    共 {total_killed} 个被截断的位置×期")
        log(f"    融合 vs 最佳子模型: 帮了={fusion_helped} "
            f"({fusion_helped/total_killed*100:.0f}%), "
            f"害了={fusion_hurt} ({fusion_hurt/total_killed*100:.0f}%), "
            f"持平={fusion_same}")
        log(f"\n    各子模型在被截断位置的平均排名:")
        for sub_name in E5_SUB_NAMES:
            ranks = sub_killed_ranks.get(sub_name, [])
            if ranks:
                avg = np.mean(ranks)
                # 该子模型能救回多少（排名 <= cutoff_rank 的比例）
                rescued = sum(1 for r in ranks if r <= 8) / len(ranks)
                log(f"      {sub_name}: 平均={avg:.1f}, "
                    f"排名<=8占比={rescued*100:.0f}%")
        log(f"      fused: 平均={np.mean(fused_killed_ranks):.1f}")
        log(f"\n    被截断位置中，最佳子模型分布:")
        for sub_name in E5_SUB_NAMES:
            cnt = best_sub_counts.get(sub_name, 0)
            if cnt > 0:
                log(f"      {sub_name}: {cnt} 次 ({cnt/total_killed*100:.0f}%)")

    log(f"\n{'='*60}\n")

    # 保存
    save_path = E8_DIR / f"e8_diagnosis_{lottery_type}.json"
    save_json(all_diags, save_path)
    log(f"  诊断详情已保存: {save_path.name}")

    return all_diags


# ============================================================
#  断点续跑：检查点管理
# ============================================================

def _checkpoint_path(lottery_type):
    return E8_CACHE_DIR / f"e8_checkpoint_{lottery_type}.pkl"


def save_checkpoint(lottery_type, results, last_t):
    """保存检查点"""
    path = _checkpoint_path(lottery_type)
    with open(path, "wb") as f:
        pickle.dump({"results": results, "last_t": last_t}, f)
    log(f"  检查点已保存: 已完成到 t={last_t}, 共 {len(results)} 期")


def load_checkpoint(lottery_type):
    """加载检查点，返回 (results, last_t) 或 ([], None)"""
    path = _checkpoint_path(lottery_type)
    if path.exists():
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        log(f"  检查点已加载: 从 t={ckpt['last_t']} 继续, 已有 {len(ckpt['results'])} 期")
        return ckpt["results"], ckpt["last_t"]
    return [], None


# ============================================================
#  滚动回测主循环
# ============================================================

def run_backtest(lottery_type, start_t=None, end_t=None, budget_bets=BUDGET_BETS,
                 resume=True, seed=42):
    """滚动回测主循环

    Args:
        lottery_type: 'daletou' or 'shuangseqiu'
        start_t: 起始期索引（默认 MIN_TRAIN_PERIODS）
        end_t: 结束期索引（默认 总期数-1）
        budget_bets: 每期注数
        resume: 是否断点续跑
        seed: 随机种子

    Returns:
        all_results: list of dict
        summary: dict
    """
    setup_logging()
    log(f"\n{'='*60}")
    log(f"  E8 融合模拟回测: {lottery_type}")
    log(f"{'='*60}\n")

    data = LotteryData(lottery_type)
    n_total = data.n_draws
    rng = np.random.RandomState(seed)

    if start_t is None:
        start_t = MIN_TRAIN_PERIODS
    if end_t is None:
        end_t = n_total - 2  # 最后一期用于开奖

    log(f"  数据: {n_total} 期, 回测范围: [{start_t}, {end_t}]")
    log(f"  每期预算: {budget_bets} 注 = {budget_bets * TICKET_PRICE} 元")

    # 断点续跑
    all_results = []
    resume_t = start_t
    if resume:
        all_results, last_t = load_checkpoint(lottery_type)
        if last_t is not None:
            resume_t = last_t + 1

    total_periods = end_t - resume_t + 1
    log(f"  待回测: {total_periods} 期 (从 t={resume_t})")

    t0 = time.time()
    for i, t in enumerate(range(resume_t, end_t + 1)):
        try:
            result = simulate_one_period(data, t, budget_bets, lottery_type, rng)
            all_results.append(result)

            # 进度日志：聚焦红球命中数
            if (i + 1) % 10 == 0 or result["best_red_hit"] >= 3:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                hit_str = f"最佳命中={result['best_red_hit']}r{result['best_blue_hit']}b"
                prize_str = f" 奖金={result['total_prize']}" if result['total_prize'] > 0 else ""
                log(f"  [{i+1}/{total_periods}] t={t} "
                    f"E7={result['e7_cand_sizes']} "
                    f"combos={result['n_valid_combos']} "
                    f"注={result['n_tickets']} {hit_str}{prize_str} "
                    f"({speed:.1f} 期/s)")

            # 检查点
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(lottery_type, all_results, t)

        except Exception as e:
            log(f"  [错误] t={t}: {e}")
            # 保存检查点后继续
            save_checkpoint(lottery_type, all_results, t - 1)
            continue

    # 最终保存
    save_checkpoint(lottery_type, all_results, end_t)

    # 汇总
    summary = compute_summary(all_results, lottery_type, budget_bets)
    return all_results, summary


# ============================================================
#  汇总报告
# ============================================================

def compute_summary(results, lottery_type, budget_bets):
    """计算回测汇总统计，聚焦一二等奖命中能力"""
    if not results:
        return {"error": "无回测结果"}

    n = len(results)
    total_cost = sum(r["cost"] for r in results)
    total_prize = sum(r["total_prize"] for r in results)
    total_net = total_prize - total_cost
    roi = total_net / total_cost if total_cost > 0 else 0

    # === 核心指标：每期最佳红球命中数分布 ===
    best_red_hits = [r["best_red_hit"] for r in results]
    red_hit_dist = defaultdict(int)
    for h in best_red_hits:
        red_hit_dist[h] += 1

    # 高命中期（红球 >= 4 个 = 接近一二等奖）
    high_hit_periods = [r for r in results if r["best_red_hit"] >= 4]
    mid_hit_periods = [r for r in results if r["best_red_hit"] == 3]

    # === 全注命中分布（所有注的命中情况汇总）===
    all_hit_dist = defaultdict(int)
    for r in results:
        for key, count in r["hit_distribution"].items():
            all_hit_dist[key] += count

    # 最大单期奖金
    max_prize = max(r["total_prize"] for r in results)
    max_prize_period = next(r["period_idx"] for r in results
                           if r["total_prize"] == max_prize)

    # 最佳命中
    max_red_hit = max(best_red_hits)
    max_hit_period = next(r["period_idx"] for r in results
                         if r["best_red_hit"] == max_red_hit)

    # E5/E7 候选集平均大小
    avg_e5 = [float(np.mean([r["e5_cand_sizes"][p] for r in results]))
              for p in range(len(results[0]["e5_cand_sizes"]))]
    avg_e7 = [float(np.mean([r["e7_cand_sizes"][p] for r in results]))
              for p in range(len(results[0]["e7_cand_sizes"]))]

    # 合法组合统计
    valid_combo_counts = [r.get("n_valid_combos", r.get("n_red_combos", 0))
                          for r in results]
    avg_valid_combos = float(np.mean(valid_combo_counts))
    median_valid_combos = float(np.median(valid_combo_counts))

    # 组合缩减率：相对于位置候选集笛卡尔积
    combo_reduction_rates = []
    for r in results:
        cartesian = 1
        for s in r["e7_cand_sizes"]:
            cartesian *= max(s, 1)
        n_vc = r.get("n_valid_combos", cartesian)
        rate = 1.0 - n_vc / cartesian if cartesian > 0 else 0
        combo_reduction_rates.append(max(0, rate))
    avg_combo_reduction = float(np.mean(combo_reduction_rates))

    # 滚动最佳命中（每 50 期）
    rolling_hits = []
    window = 50
    for i in range(0, n, window):
        chunk = results[i:i + window]
        chunk_max_red = max(r["best_red_hit"] for r in chunk)
        chunk_avg_red = np.mean([r["best_red_hit"] for r in chunk])
        chunk_prize = sum(r["total_prize"] for r in chunk)
        chunk_cost = sum(r["cost"] for r in chunk)
        rolling_hits.append({
            "start_idx": chunk[0]["period_idx"],
            "end_idx": chunk[-1]["period_idx"],
            "max_red_hit": chunk_max_red,
            "avg_red_hit": round(float(chunk_avg_red), 2),
            "prize": chunk_prize,
            "roi": round((chunk_prize - chunk_cost) / chunk_cost, 4) if chunk_cost > 0 else 0,
        })

    summary = {
        "lottery_type": lottery_type,
        "n_periods": n,
        "budget_bets": budget_bets,
        "total_cost": total_cost,
        "total_prize": total_prize,
        "total_net": total_net,
        "roi": round(roi, 4),
        # 核心：红球命中分布
        "best_red_hit_distribution": dict(sorted(red_hit_dist.items(), reverse=True)),
        "max_red_hit": max_red_hit,
        "max_red_hit_period": max_hit_period,
        "periods_with_4plus_red": len(high_hit_periods),
        "periods_with_3_red": len(mid_hit_periods),
        "avg_best_red_hit": round(float(np.mean(best_red_hits)), 2),
        # 全注命中分布
        "all_ticket_hit_distribution": dict(sorted(all_hit_dist.items(), reverse=True)),
        # 奖金
        "max_single_prize": max_prize,
        "max_prize_period": max_prize_period,
        # 候选集
        "avg_e5_cand_sizes": [round(x, 1) for x in avg_e5],
        "avg_e7_cand_sizes": [round(x, 1) for x in avg_e7],
        # 组合约束效果
        "avg_valid_combos": round(avg_valid_combos, 1),
        "median_valid_combos": round(median_valid_combos, 1),
        "combo_reduction_rate": round(avg_combo_reduction, 4),
        "rolling_hits": rolling_hits,
    }

    # 打印汇总（聚焦命中能力）
    log(f"\n{'='*60}")
    log(f"  E8 回测汇总: {lottery_type}")
    log(f"{'='*60}")
    log(f"  回测期数: {n}, 每期 {budget_bets} 注")
    log(f"  总投入: {total_cost:,} 元, 总奖金: {total_prize:,} 元, ROI: {roi*100:.2f}%")
    log(f"")
    log(f"  === 红球命中分布（每期最佳注）===")
    for hits in sorted(red_hit_dist.keys(), reverse=True):
        pct = red_hit_dist[hits] / n * 100
        bar = "█" * int(pct)
        log(f"    {hits} 个红球: {red_hit_dist[hits]:4d} 期 ({pct:5.1f}%) {bar}")
    log(f"  平均最佳红球命中: {np.mean(best_red_hits):.2f}")
    log(f"  最高命中: {max_red_hit} 个红球 (期 {max_hit_period})")
    if high_hit_periods:
        log(f"  红球>=4 的期数: {len(high_hit_periods)} ({len(high_hit_periods)/n*100:.1f}%)")
        for r in high_hit_periods[:10]:
            log(f"    期 {r['period_idx']}: {r['best_red_hit']}r{r['best_blue_hit']}b "
                f"奖金={r['total_prize']}")
    log(f"")
    log(f"  E5 平均候选: {[round(x,1) for x in avg_e5]}")
    log(f"  E7 平均候选: {[round(x,1) for x in avg_e7]}")
    log(f"  合法组合: 平均={avg_valid_combos:.0f}, 中位数={median_valid_combos:.0f}")
    log(f"  组合缩减率: {avg_combo_reduction*100:.1f}%")
    log(f"{'='*60}\n")

    # 保存
    save_json(summary, E8_DIR / f"e8_summary_{lottery_type}.json")
    save_json(results, E8_DIR / f"e8_details_{lottery_type}.json")

    return summary


# ============================================================
#  入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="E8 融合模拟回测")
    parser.add_argument("lottery_type", nargs="?", default="daletou",
                        choices=["daletou", "shuangseqiu"],
                        help="彩种 (默认 daletou)")
    parser.add_argument("--start", type=int, default=None, help="起始期索引")
    parser.add_argument("--end", type=int, default=None, help="结束期索引")
    parser.add_argument("--bets", type=int, default=BUDGET_BETS, help="每期注数")
    parser.add_argument("--no-resume", action="store_true", help="不断点续跑")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    results, summary = run_backtest(
        lottery_type=args.lottery_type,
        start_t=args.start,
        end_t=args.end,
        budget_bets=args.bets,
        resume=not args.no_resume,
        seed=args.seed,
    )

    return summary


if __name__ == "__main__":
    main()
