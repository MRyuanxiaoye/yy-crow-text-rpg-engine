# -*- coding: utf-8 -*-
"""
E7：数据驱动约束发现与传播
Phase 1-4 主文件

改进点（vs E6）：
- HDI 替代百分位数（抗异常值）
- 穷举 6 大类 ~75-114 种约束
- CSP AC-3 弧一致性传播（约束互相收紧）
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import deque
from math import comb as math_comb
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.data_loader import LotteryData
from research.experiment.utils import save_json, log, setup_logging, Timer

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "e7_discovery"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = {
    "daletou":     {"train_end": 2400, "total": 2833},
    "shuangseqiu": {"train_end": 2900, "total": 3413},
}


# ============================================================
#  HDI 计算
# ============================================================

def compute_hdi(samples, credible_mass=0.90):
    """用 KDE 计算 Highest Density Interval"""
    samples = np.asarray(samples, dtype=float)
    if len(samples) < 10:
        return float(np.min(samples)), float(np.max(samples))

    # 加微小抖动，防止 KDE 对纯整数数据退化
    jittered = samples + np.random.normal(0, 0.1, len(samples))
    try:
        kde = gaussian_kde(jittered, bw_method='scott')
    except Exception:
        return float(np.percentile(samples, (1 - credible_mass) / 2 * 100)), \
               float(np.percentile(samples, (1 + credible_mass) / 2 * 100))

    # 在数据范围内采样
    lo_bound = samples.min() - 2
    hi_bound = samples.max() + 2
    x_grid = np.linspace(lo_bound, hi_bound, 1000)
    density = kde(x_grid)

    # 按密度降序累积，找最小区间
    sorted_idx = np.argsort(-density)
    cumsum = np.cumsum(density[sorted_idx])
    cumsum /= cumsum[-1]
    cutoff_idx = np.searchsorted(cumsum, credible_mass)
    selected = sorted_idx[:cutoff_idx + 1]
    selected_x = x_grid[selected]

    hdi_lo = float(np.floor(selected_x.min()))
    hdi_hi = float(np.ceil(selected_x.max()))
    return hdi_lo, hdi_hi


# ============================================================
#  Phase 1：穷举约束枚举
# ============================================================

def enumerate_all_constraints(data: LotteryData, train_end: int):
    """枚举 6 大类约束，对训练期数据计算 HDI-70/80/90"""
    R = data.red_matrix[:train_end]  # (train_end, n_pos)
    n_pos = data.red_count
    red_range = data.red_range
    combo_stats = data.get_combo_stats_series()

    constraints = []

    def add_constraint(name, ctype, samples, positions):
        """为一种约束计算 3 级 HDI"""
        s = np.asarray(samples, dtype=float)
        hdis = {}
        for mass in [0.70, 0.80, 0.90]:
            lo, hi = compute_hdi(s, mass)
            hdis[f"hdi_{int(mass*100)}"] = [lo, hi]
        constraints.append({
            "name": name,
            "type": ctype,
            "positions": positions,
            "n_samples": len(s),
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
            **hdis,
        })

    # --- A. 单位置同期差分 ---
    diffs = R[1:] - R[:-1]
    for p in range(n_pos):
        add_constraint(f"pos_diff_P{p}", "A", diffs[:, p], [p])

    # --- B. 同期内两两约束 ---
    # B1: 相邻间距
    gaps = np.diff(R, axis=1)
    for g in range(n_pos - 1):
        add_constraint(f"gap_P{g}_P{g+1}", "B", gaps[:, g], [g, g + 1])
    # B2: 任意两位置差
    for i in range(n_pos):
        for j in range(i + 1, n_pos):
            if j == i + 1:
                continue  # 已在 gap 中覆盖
            add_constraint(f"pair_diff_P{i}_P{j}", "B",
                           R[:, j] - R[:, i], [i, j])

    # --- C. 跨期跨位置约束 ---
    for i in range(n_pos):
        for j in range(n_pos):
            cross = R[1:, i] - R[:-1, j]
            add_constraint(f"cross_P{i}_lag_P{j}", "C", cross, [i])

    # --- D. 三元组约束 ---
    for i, j, k in combinations(range(n_pos), 3):
        # 三位置跨度
        add_constraint(f"triple_span_P{i}_P{j}_P{k}", "D",
                       R[:, k] - R[:, i], [i, j, k])
        # 中间偏移
        mid_offset = R[:, j] - (R[:, i] + R[:, k]) / 2.0
        add_constraint(f"triple_mid_P{i}_P{j}_P{k}", "D",
                       mid_offset, [i, j, k])

    # --- E. 聚合统计约束 ---
    for stat_name in ["sum", "span", "odd_count", "big_count",
                       "ac_value", "consec_groups"]:
        s = combo_stats[stat_name][:train_end]
        add_constraint(f"agg_{stat_name}", "E", s, list(range(n_pos)))

    # --- F. 二阶差分约束 ---
    for p in range(n_pos):
        d1 = diffs[:, p]
        d2 = np.diff(d1)
        add_constraint(f"diff2_P{p}", "F", d2, [p])

    log(f"  Phase1: 共枚举 {len(constraints)} 种约束")
    return constraints


# ============================================================
#  Phase 2：约束质量排名
# ============================================================

def compute_constraint_value(con, data, train_end, test_end, hdi_key="hdi_90"):
    """对单个约束计算测试期的 reliability / tightness / usefulness"""
    name = con["name"]
    hdi_lo, hdi_hi = con[hdi_key]
    full_range = con["max"] - con["min"]
    if full_range == 0:
        full_range = 1.0

    # 在测试期重新计算该约束的值
    R = data.red_matrix
    n_pos = data.red_count
    combo_stats = data.get_combo_stats_series()
    test_vals = _extract_constraint_values(name, R, n_pos, combo_stats,
                                           train_end, test_end)
    if len(test_vals) == 0:
        return 0.0, 0.0, 0.0

    # reliability: 测试期落在 HDI 内的比例
    in_hdi = np.sum((test_vals >= hdi_lo) & (test_vals <= hdi_hi))
    reliability = in_hdi / len(test_vals)

    # tightness: 1 - HDI宽度/全范围
    hdi_width = hdi_hi - hdi_lo
    tightness = 1.0 - hdi_width / full_range
    tightness = max(0.0, tightness)

    usefulness = tightness * reliability
    return reliability, tightness, usefulness


def _extract_constraint_values(name, R, n_pos, combo_stats,
                                start_idx, end_idx):
    """从矩阵中提取指定约束在 [start_idx, end_idx) 的值序列"""
    parts = name.split("_")

    if name.startswith("pos_diff_P"):
        p = int(parts[2][1])
        diffs = R[1:, p] - R[:-1, p]
        # diffs[i] = R[i+1,p] - R[i,p]，对应期 i+1
        s, e = max(0, start_idx - 1), min(len(diffs), end_idx - 1)
        return diffs[s:e].astype(float)

    elif name.startswith("gap_P"):
        g = int(parts[1][1])
        gaps = R[:, g + 1] - R[:, g]
        return gaps[start_idx:end_idx].astype(float)

    elif name.startswith("pair_diff_P"):
        i = int(parts[2][1])
        j = int(parts[3][1])
        return (R[start_idx:end_idx, j] - R[start_idx:end_idx, i]).astype(float)

    elif name.startswith("cross_P"):
        # cross_P{i}_lag_P{j}: R[t, i] - R[t-1, j]
        i = int(parts[1][1])
        j = int(parts[3][1])
        cross = R[1:, i] - R[:-1, j]
        s, e = max(0, start_idx - 1), min(len(cross), end_idx - 1)
        return cross[s:e].astype(float)

    elif name.startswith("triple_span_P"):
        i = int(parts[2][1])
        k = int(parts[4][1])
        return (R[start_idx:end_idx, k] - R[start_idx:end_idx, i]).astype(float)

    elif name.startswith("triple_mid_P"):
        i = int(parts[2][1])
        j = int(parts[3][1])
        k = int(parts[4][1])
        mid = R[start_idx:end_idx, j] - \
              (R[start_idx:end_idx, i] + R[start_idx:end_idx, k]) / 2.0
        return mid.astype(float)

    elif name.startswith("agg_"):
        stat_name = name[4:]
        s = combo_stats[stat_name]
        return s[start_idx:end_idx].astype(float)

    elif name.startswith("diff2_P"):
        p = int(parts[1][1])
        d1 = R[1:, p] - R[:-1, p]
        d2 = np.diff(d1)
        # d2[i] 对应期 i+2
        s, e = max(0, start_idx - 2), min(len(d2), end_idx - 2)
        return d2[s:e].astype(float)

    return np.array([])


def rank_and_filter_constraints(constraints, data, train_end, test_end,
                                 hdi_key="hdi_90", top_n=30):
    """Phase 2: 排名 + 冗余检测 + 筛选 top-N"""
    R = data.red_matrix
    n_pos = data.red_count
    combo_stats = data.get_combo_stats_series()

    # 1. 计算每个约束的 usefulness
    for con in constraints:
        rel, tight, useful = compute_constraint_value(
            con, data, train_end, test_end, hdi_key)
        con["reliability"] = round(rel, 4)
        con["tightness"] = round(tight, 4)
        con["usefulness"] = round(useful, 4)

    # 2. 按 usefulness 降序排名
    ranked = sorted(constraints, key=lambda c: -c["usefulness"])

    # 3. 冗余检测：涉及相同位置且 HDI 被包含的约束标记冗余
    selected = []
    seen_pos_sets = []
    for con in ranked:
        pos_set = frozenset(con["positions"])
        hdi = con[hdi_key]

        redundant = False
        for sel, sel_hdi in seen_pos_sets:
            if pos_set == sel:
                # 同位置集合，检查 HDI 是否被包含
                if hdi[0] >= sel_hdi[0] and hdi[1] <= sel_hdi[1]:
                    redundant = True
                    break
        con["redundant"] = redundant
        if not redundant:
            selected.append(con)
            seen_pos_sets.append((pos_set, hdi))
        if len(selected) >= top_n:
            break

    log(f"  Phase2: 排名完成，筛选 {len(selected)}/{len(constraints)} 个约束")
    return selected, ranked


# ============================================================
#  Phase 3：AC-3 约束传播 + 组合计数
# ============================================================

def build_initial_candidates(prev_vals, selected_cons, red_range, n_pos,
                              hdi_key="hdi_90"):
    """用单位置约束（A类 pos_diff + F类 diff2）初始化每位置候选集"""
    candidates = [set(range(1, red_range + 1)) for _ in range(n_pos)]

    for con in selected_cons:
        name = con["name"]
        hdi_lo, hdi_hi = con[hdi_key]

        if name.startswith("pos_diff_P"):
            p = int(name.split("_")[2][1])
            v_min = max(1, int(prev_vals[p] + hdi_lo))
            v_max = min(red_range, int(prev_vals[p] + hdi_hi))
            candidates[p] &= set(range(v_min, v_max + 1))

    return candidates


def apply_ordering_constraint(candidates, n_pos):
    """应用排序约束 P0 < P1 < ... < P_{n-1}"""
    changed = True
    while changed:
        changed = False
        # 正向：P_{i+1} > min(P_i)
        for p in range(n_pos - 1):
            if not candidates[p]:
                continue
            min_val = min(candidates[p]) + 1
            before = len(candidates[p + 1])
            candidates[p + 1] = {v for v in candidates[p + 1] if v >= min_val}
            if len(candidates[p + 1]) < before:
                changed = True
        # 反向：P_i < max(P_{i+1})
        for p in range(n_pos - 1, 0, -1):
            if not candidates[p]:
                continue
            max_val = max(candidates[p]) - 1
            before = len(candidates[p - 1])
            candidates[p - 1] = {v for v in candidates[p - 1] if v <= max_val}
            if len(candidates[p - 1]) < before:
                changed = True
    return candidates


def ac3_propagate(candidates, selected_cons, prev_vals, red_range, n_pos,
                   hdi_key="hdi_90"):
    """AC-3 弧一致性传播，带空集检测和回退"""
    # 保存传播前的快照，用于空集回退
    snapshot = [set(c) for c in candidates]

    # 构建二元约束列表: (pi, pj, check_func)
    binary_cons = []

    for con in selected_cons:
        name = con["name"]
        hdi_lo, hdi_hi = con[hdi_key]

        if name.startswith("gap_P"):
            parts = name.split("_")
            pi = int(parts[1][1])
            pj = int(parts[2][1])
            lo, hi = hdi_lo, hdi_hi
            binary_cons.append((pi, pj,
                lambda vi, vj, lo=lo, hi=hi: lo <= (vj - vi) <= hi))

        elif name.startswith("pair_diff_P"):
            parts = name.split("_")
            pi = int(parts[2][1])
            pj = int(parts[3][1])
            lo, hi = hdi_lo, hdi_hi
            binary_cons.append((pi, pj,
                lambda vi, vj, lo=lo, hi=hi: lo <= (vj - vi) <= hi))

        elif name.startswith("cross_P") and prev_vals is not None:
            parts = name.split("_")
            pi = int(parts[1][1])
            pj_lag = int(parts[3][1])
            prev_v = prev_vals[pj_lag]
            lo, hi = hdi_lo, hdi_hi
            new_set = {v for v in candidates[pi]
                       if lo <= (v - prev_v) <= hi}
            if new_set:
                candidates[pi] = new_set
            # 如果 new_set 为空，保留原候选（跳过该约束）

        elif name.startswith("triple_span_P"):
            parts = name.split("_")
            pi = int(parts[2][1])
            pk = int(parts[4][1])
            lo, hi = hdi_lo, hdi_hi
            binary_cons.append((pi, pk,
                lambda vi, vk, lo=lo, hi=hi: lo <= (vk - vi) <= hi))

        elif name.startswith("triple_mid_P"):
            pass

    # AC-3 主循环
    queue = deque()
    for idx, (pi, pj, _) in enumerate(binary_cons):
        queue.append((pi, pj, idx))
        queue.append((pj, pi, idx))

    max_iterations = len(queue) * 5
    iterations = 0

    while queue and iterations < max_iterations:
        iterations += 1
        xi, xj, con_idx = queue.popleft()
        pi_orig, pj_orig, check = binary_cons[con_idx]

        if not candidates[xi] or not candidates[xj]:
            continue

        removed = set()
        for vi in list(candidates[xi]):
            has_support = False
            for vj in candidates[xj]:
                if xi == pi_orig:
                    ok = check(vi, vj)
                else:
                    ok = check(vj, vi)
                if ok:
                    has_support = True
                    break
            if not has_support:
                removed.add(vi)

        if removed:
            new_set = candidates[xi] - removed
            if not new_set:
                # 空集检测：回退到传播前快照，放弃本轮传播
                candidates = snapshot
                break
            candidates[xi] = new_set
            for idx2, (p1, p2, _) in enumerate(binary_cons):
                if idx2 == con_idx:
                    continue
                if p1 == xi or p2 == xi:
                    other = p2 if p1 == xi else p1
                    queue.append((other, xi, idx2))

    return candidates


def backtrack_count(candidates, selected_cons, red_range, n_pos,
                     hdi_key="hdi_90", max_count=500000):
    """回溯+剪枝计数满足所有约束的合法组合"""
    cands = [sorted(c) for c in candidates]

    # 提取聚合约束
    agg_cons = {}
    for con in selected_cons:
        if con["name"].startswith("agg_"):
            stat = con["name"][4:]
            agg_cons[stat] = con[hdi_key]

    sum_lo = agg_cons.get("sum", [0, 999])[0]
    sum_hi = agg_cons.get("sum", [0, 999])[1]
    span_lo = agg_cons.get("span", [0, 999])[0]
    span_hi = agg_cons.get("span", [0, 999])[1]

    # 提取间距约束
    gap_cons = {}
    for con in selected_cons:
        if con["name"].startswith("gap_P"):
            parts = con["name"].split("_")
            g = int(parts[1][1])
            gap_cons[g] = con[hdi_key]

    count = 0

    def backtrack(pos, chosen, current_sum):
        nonlocal count
        if count >= max_count:
            return

        if pos == n_pos:
            span = chosen[-1] - chosen[0]
            if span_lo <= span <= span_hi and sum_lo <= current_sum <= sum_hi:
                count += 1
            return

        min_val = chosen[-1] + 1 if chosen else 1
        for v in cands[pos]:
            if v < min_val:
                continue

            # 间距剪枝
            if pos > 0 and (pos - 1) in gap_cons:
                gap = v - chosen[-1]
                g_lo, g_hi = gap_cons[pos - 1]
                if gap < g_lo or gap > g_hi:
                    continue

            # 和值上界剪枝
            remaining = n_pos - pos - 1
            min_rem = sum(v + k for k in range(1, remaining + 1))
            if current_sum + v + min_rem > sum_hi:
                continue

            # 和值下界剪枝
            max_rem = sum(red_range - k for k in range(remaining))
            if current_sum + v + max_rem < sum_lo:
                continue

            backtrack(pos + 1, chosen + [v], current_sum + v)

    backtrack(0, [], 0)
    return count


def evaluate_propagation(data, selected_cons, train_end, test_end,
                          hdi_key="hdi_90"):
    """Phase 3: 对测试期逐期评估 AC-3 传播效果"""
    R = data.red_matrix
    n_pos = data.red_count
    red_range = data.red_range
    total_combos = math_comb(red_range, n_pos)

    test_indices = list(range(train_end + 1, test_end))
    results = []

    for idx in test_indices:
        prev = R[idx - 1]
        true = R[idx]

        # 1. 初始化候选
        cands = build_initial_candidates(prev, selected_cons, red_range,
                                          n_pos, hdi_key)
        # 2. 排序约束
        cands = apply_ordering_constraint(cands, n_pos)
        # 3. AC-3 传播
        cands = ac3_propagate(cands, selected_cons, prev, red_range,
                               n_pos, hdi_key)
        # 4. 再次排序约束（传播后可能需要）
        cands = apply_ordering_constraint(cands, n_pos)

        cand_sizes = [len(c) for c in cands]

        # 5. 检查位置存活
        pos_survived = all(true[p] in cands[p] for p in range(n_pos))

        # 6. 组合计数（候选集不太大时精确计数，否则用 C(max_cand, n_pos) 上界）
        max_cand = max(cand_sizes) if cand_sizes else 0
        total_cand = 1
        for s in cand_sizes:
            total_cand *= max(s, 1)

        if total_cand < 2_000_000:
            valid_count = backtrack_count(cands, selected_cons, red_range,
                                           n_pos, hdi_key)
        else:
            # 候选太多，用组合上界估算，但不超过全空间
            valid_count = min(math_comb(max_cand, n_pos), total_combos)

        # 7. 检查组合存活
        combo_survived = False
        if pos_survived:
            combo_survived = True
            # 检查聚合约束
            for con in selected_cons:
                if con["name"].startswith("agg_sum"):
                    lo, hi = con[hdi_key]
                    s = int(sum(true))
                    if not (lo <= s <= hi):
                        combo_survived = False
                        break
                elif con["name"].startswith("agg_span"):
                    lo, hi = con[hdi_key]
                    sp = true[-1] - true[0]
                    if not (lo <= sp <= hi):
                        combo_survived = False
                        break
            # 检查间距约束
            if combo_survived:
                for con in selected_cons:
                    if con["name"].startswith("gap_P"):
                        parts = con["name"].split("_")
                        g = int(parts[1][1])
                        gap = true[g + 1] - true[g]
                        lo, hi = con[hdi_key]
                        if not (lo <= gap <= hi):
                            combo_survived = False
                            break

        reduction = 1.0 - valid_count / total_combos if total_combos > 0 else 0

        results.append({
            "idx": idx,
            "cand_sizes": cand_sizes,
            "valid_combos": valid_count,
            "reduction": round(reduction, 4),
            "pos_survived": pos_survived,
            "combo_survived": combo_survived,
        })

        if len(results) % 50 == 0:
            log(f"    已评估 {len(results)}/{len(test_indices)} 期")

    # 汇总
    reductions = [r["reduction"] for r in results]
    combo_survivals = [r["combo_survived"] for r in results]
    avg_reduction = float(np.mean(reductions))
    survival_rate = sum(combo_survivals) / len(combo_survivals)
    efficiency = avg_reduction * survival_rate

    summary = {
        "hdi_level": hdi_key,
        "n_constraints": len(selected_cons),
        "n_test": len(results),
        "avg_reduction": round(avg_reduction, 4),
        "combo_survival": round(survival_rate, 4),
        "efficiency": round(efficiency, 4),
        "avg_valid_combos": int(np.mean([r["valid_combos"] for r in results])),
        "median_valid_combos": int(np.median([r["valid_combos"]
                                               for r in results])),
    }
    return summary, results


# ============================================================
#  Phase 4：对比 E6
# ============================================================

def load_e6_results(lottery_type):
    """加载 E6 结果用于对比"""
    e6_dir = Path(__file__).resolve().parents[1] / "results" / "e6_constraint"
    path = e6_dir / f"e6_results_{lottery_type}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


# ============================================================
#  主流程
# ============================================================

def run_e7(lottery_type="daletou"):
    """运行 E7 全流程（Phase 1-4）"""
    setup_logging()
    log(f"\n{'='*60}")
    log(f"  E7 数据驱动约束发现与传播: {lottery_type}")
    log(f"{'='*60}\n")

    data = LotteryData(lottery_type)
    cfg = SPLIT[lottery_type]
    train_end = cfg["train_end"]
    test_end = cfg["total"]
    red_range = data.red_range
    n_pos = data.red_count
    total_combos = math_comb(red_range, n_pos)

    log(f"全组合空间: C({red_range},{n_pos}) = {total_combos:,}")

    # === Phase 1 ===
    with Timer("Phase1 约束枚举"):
        constraints = enumerate_all_constraints(data, train_end)

    # 保存 Phase1
    save_json({"lottery_type": lottery_type,
               "n_constraints": len(constraints),
               "constraints": constraints},
              RESULTS_DIR / f"e7_phase1_constraints_{lottery_type}.json")

    # === Phase 2 ===
    with Timer("Phase2 约束排名"):
        selected, ranked = rank_and_filter_constraints(
            constraints, data, train_end, test_end,
            hdi_key="hdi_90", top_n=30)

    # 保存 Phase2
    ranking_info = [{
        "rank": i + 1,
        "name": c["name"],
        "type": c["type"],
        "reliability": c["reliability"],
        "tightness": c["tightness"],
        "usefulness": c["usefulness"],
        "hdi_90": c["hdi_90"],
    } for i, c in enumerate(selected)]

    save_json({"lottery_type": lottery_type,
               "top_constraints": ranking_info,
               "all_ranked": [{
                   "name": c["name"], "type": c["type"],
                   "usefulness": c.get("usefulness", 0),
                   "redundant": c.get("redundant", False),
               } for c in ranked[:50]]},
              RESULTS_DIR / f"e7_phase2_ranking_{lottery_type}.json")

    log(f"\n  Top-10 约束:")
    for i, c in enumerate(selected[:10]):
        log(f"    {i+1}. {c['name']:30s} useful={c['usefulness']:.4f} "
            f"rel={c['reliability']:.3f} tight={c['tightness']:.3f}")

    # === Phase 3: 对 HDI-70/80/90 各跑一轮 ===
    phase3_results = {}
    for hdi_key in ["hdi_70", "hdi_80", "hdi_90"]:
        log(f"\n  Phase3 评估 [{hdi_key}]...")
        with Timer(f"Phase3 {hdi_key}"):
            summary, details = evaluate_propagation(
                data, selected, train_end, test_end, hdi_key)
        phase3_results[hdi_key] = summary
        log(f"    缩减率={summary['avg_reduction']*100:.1f}%  "
            f"存活率={summary['combo_survival']*100:.1f}%  "
            f"效率={summary['efficiency']:.4f}")

    # === Phase 4: 对比 E6 ===
    e6 = load_e6_results(lottery_type)
    comparison = None
    if e6:
        comparison = {
            "e6_reduction": e6.get("avg_reduction_rate", 0),
            "e6_survival": e6.get("combo_survival_rate", 0),
            "e6_efficiency": e6.get("efficiency", 0),
        }
        for hdi_key, res in phase3_results.items():
            comparison[f"e7_{hdi_key}_reduction"] = res["avg_reduction"]
            comparison[f"e7_{hdi_key}_survival"] = res["combo_survival"]
            comparison[f"e7_{hdi_key}_efficiency"] = res["efficiency"]

        log(f"\n  Phase4 对比 E6:")
        log(f"    E6: 缩减={comparison['e6_reduction']*100:.1f}%  "
            f"存活={comparison['e6_survival']*100:.1f}%  "
            f"效率={comparison['e6_efficiency']:.4f}")
        for hdi_key in ["hdi_70", "hdi_80", "hdi_90"]:
            r = phase3_results[hdi_key]
            log(f"    E7 {hdi_key}: 缩减={r['avg_reduction']*100:.1f}%  "
                f"存活={r['combo_survival']*100:.1f}%  "
                f"效率={r['efficiency']:.4f}")

    # 保存 Phase3+4
    save_json({
        "lottery_type": lottery_type,
        "total_combos": total_combos,
        "phase3": phase3_results,
        "e6_comparison": comparison,
    }, RESULTS_DIR / f"e7_phase3_propagation_{lottery_type}.json")

    # 汇总
    best_hdi = max(phase3_results.keys(),
                   key=lambda k: phase3_results[k]["efficiency"])
    best = phase3_results[best_hdi]

    summary_all = {
        "lottery_type": lottery_type,
        "n_constraints_total": len(constraints),
        "n_constraints_selected": len(selected),
        "best_hdi_level": best_hdi,
        "best_efficiency": best["efficiency"],
        "best_reduction": best["avg_reduction"],
        "best_survival": best["combo_survival"],
        "e6_efficiency": comparison["e6_efficiency"] if comparison else None,
        "improvement_vs_e6": round(best["efficiency"] -
                                    (comparison["e6_efficiency"]
                                     if comparison else 0), 4),
    }
    save_json(summary_all,
              RESULTS_DIR / f"e7_summary_{lottery_type}.json")

    log(f"\n{'='*60}")
    log(f"  E7 完成: {lottery_type}")
    log(f"  最优 HDI: {best_hdi}")
    log(f"  效率: {best['efficiency']:.4f} "
        f"(E6: {comparison['e6_efficiency']:.4f})" if comparison else "")
    log(f"{'='*60}\n")

    return summary_all


# ============================================================
#  Phase 5：局面条件化约束
# ============================================================

def compute_context_features(R, idx):
    """计算第 idx 期的局面特征向量 [volatility, trend, dispersion]"""
    if idx < 3:
        return np.array([0.0, 0.0, 0.0])

    n_pos = R.shape[1]
    # 波动率：最近 3 期差分的绝对值均值
    diffs = np.abs(R[idx] - R[idx - 1]).astype(float)
    diffs_prev = np.abs(R[idx - 1] - R[idx - 2]).astype(float)
    volatility = float(np.mean(np.concatenate([diffs, diffs_prev])))

    # 趋势：最近 3 期的平均变化方向（正=上升，负=下降）
    trend = float(np.mean(R[idx] - R[idx - 2]))

    # 离散度：当期号码的标准差 / 均值
    vals = R[idx].astype(float)
    dispersion = float(np.std(vals) / max(np.mean(vals), 1))

    return np.array([volatility, trend, dispersion])


def classify_context(features, thresholds):
    """根据波动率将局面分为 3 类: low / mid / high"""
    vol = features[0]
    if vol <= thresholds[0]:
        return "low"
    elif vol <= thresholds[1]:
        return "mid"
    else:
        return "high"


def compute_volatility_thresholds(R, train_end):
    """用训练期数据计算波动率的 33/67 百分位作为分类阈值"""
    vols = []
    for i in range(3, train_end):
        feat = compute_context_features(R, i)
        vols.append(feat[0])
    vols = np.array(vols)
    return [float(np.percentile(vols, 33)), float(np.percentile(vols, 67))]


def enumerate_conditioned_constraints(data, train_end, base_constraints,
                                       top_n=30):
    """Phase 5: 对每类局面独立计算 HDI，返回条件化约束"""
    R = data.red_matrix
    n_pos = data.red_count
    combo_stats = data.get_combo_stats_series()

    # 1. 计算波动率阈值
    thresholds = compute_volatility_thresholds(R, train_end)
    log(f"  Phase5: 波动率阈值 = {thresholds}")

    # 2. 按局面分组训练期索引
    context_groups = {"low": [], "mid": [], "high": []}
    for i in range(3, train_end):
        feat = compute_context_features(R, i)
        ctx = classify_context(feat, thresholds)
        context_groups[ctx].append(i)

    for ctx, indices in context_groups.items():
        log(f"    {ctx}: {len(indices)} 期")

    # 3. 对 top-N 约束，在每类局面下重新计算 HDI
    conditioned = []
    for con in base_constraints[:top_n]:
        name = con["name"]
        cond_hdis = {}

        for ctx, indices in context_groups.items():
            if len(indices) < 20:
                # 样本太少，用全局 HDI
                cond_hdis[ctx] = {
                    "hdi_70": con["hdi_70"],
                    "hdi_80": con["hdi_80"],
                    "hdi_90": con["hdi_90"],
                }
                continue

            # 提取该局面下的约束值
            vals = _extract_constraint_values(
                name, R, n_pos, combo_stats, 0, train_end)

            # 只保留属于该局面的样本
            # 需要对齐索引：vals 的索引取决于约束类型
            if name.startswith("pos_diff_") or name.startswith("cross_"):
                # 这些约束的值对应期 i+1，所以 vals[k] 对应期 k+1
                ctx_vals = []
                for i in indices:
                    k = i - 1  # vals 的索引
                    if 0 <= k < len(vals):
                        ctx_vals.append(vals[k])
            elif name.startswith("diff2_"):
                # d2[k] 对应期 k+2
                ctx_vals = []
                for i in indices:
                    k = i - 2
                    if 0 <= k < len(vals):
                        ctx_vals.append(vals[k])
            else:
                # 其他约束直接按期索引
                ctx_vals = []
                for i in indices:
                    if i < len(vals):
                        ctx_vals.append(vals[i])

            if len(ctx_vals) < 10:
                cond_hdis[ctx] = {
                    "hdi_70": con["hdi_70"],
                    "hdi_80": con["hdi_80"],
                    "hdi_90": con["hdi_90"],
                }
                continue

            ctx_vals = np.array(ctx_vals)
            hdis = {}
            for mass in [0.70, 0.80, 0.90]:
                lo, hi = compute_hdi(ctx_vals, mass)
                hdis[f"hdi_{int(mass*100)}"] = [lo, hi]
            cond_hdis[ctx] = hdis

        cond_con = dict(con)
        cond_con["context_hdis"] = cond_hdis
        conditioned.append(cond_con)

    return conditioned, thresholds, context_groups


def evaluate_conditioned_propagation(data, cond_constraints, thresholds,
                                      train_end, test_end, hdi_key="hdi_90",
                                      adapt_contexts=None):
    """Phase 5: 用条件化 HDI 评估传播效果

    adapt_contexts: 只对这些局面使用条件化 HDI，其余用全局。
                    None 表示全部条件化。例如 ["high"] 只对高波动局面条件化。
    """
    R = data.red_matrix
    n_pos = data.red_count
    red_range = data.red_range
    total_combos = math_comb(red_range, n_pos)

    test_indices = list(range(train_end + 1, test_end))
    results = []
    ctx_counts = {"low": 0, "mid": 0, "high": 0}

    for idx in test_indices:
        prev = R[idx - 1]
        true = R[idx]

        # 判断上一期局面
        feat = compute_context_features(R, idx - 1)
        ctx = classify_context(feat, thresholds)
        ctx_counts[ctx] += 1

        # 决定是否使用条件化 HDI
        use_conditioned = (adapt_contexts is None) or (ctx in adapt_contexts)

        # 为每个约束选择对应局面的 HDI
        adapted_cons = []
        for con in cond_constraints:
            adapted = dict(con)
            if use_conditioned:
                ctx_hdis = con["context_hdis"].get(ctx, {})
                for k in ["hdi_70", "hdi_80", "hdi_90"]:
                    if k in ctx_hdis:
                        adapted[k] = ctx_hdis[k]
            # else: 保留全局 HDI
            adapted_cons.append(adapted)

        # 1-4 同 Phase 3
        cands = build_initial_candidates(prev, adapted_cons, red_range,
                                          n_pos, hdi_key)
        cands = apply_ordering_constraint(cands, n_pos)
        cands = ac3_propagate(cands, adapted_cons, prev, red_range,
                               n_pos, hdi_key)
        cands = apply_ordering_constraint(cands, n_pos)

        cand_sizes = [len(c) for c in cands]
        pos_survived = all(true[p] in cands[p] for p in range(n_pos))

        max_cand = max(cand_sizes) if cand_sizes else 0
        total_cand = 1
        for s in cand_sizes:
            total_cand *= max(s, 1)

        if total_cand < 2_000_000:
            valid_count = backtrack_count(cands, adapted_cons, red_range,
                                           n_pos, hdi_key)
        else:
            valid_count = min(math_comb(max_cand, n_pos), total_combos)

        combo_survived = False
        if pos_survived:
            combo_survived = True
            for con in adapted_cons:
                if con["name"].startswith("agg_sum"):
                    lo, hi = con[hdi_key]
                    s = int(sum(true))
                    if not (lo <= s <= hi):
                        combo_survived = False
                        break
                elif con["name"].startswith("agg_span"):
                    lo, hi = con[hdi_key]
                    sp = true[-1] - true[0]
                    if not (lo <= sp <= hi):
                        combo_survived = False
                        break
            if combo_survived:
                for con in adapted_cons:
                    if con["name"].startswith("gap_P"):
                        parts = con["name"].split("_")
                        g = int(parts[1][1])
                        gap = true[g + 1] - true[g]
                        lo, hi = con[hdi_key]
                        if not (lo <= gap <= hi):
                            combo_survived = False
                            break

        reduction = 1.0 - valid_count / total_combos if total_combos > 0 else 0

        results.append({
            "idx": idx,
            "context": ctx,
            "cand_sizes": cand_sizes,
            "valid_combos": valid_count,
            "reduction": round(reduction, 4),
            "pos_survived": pos_survived,
            "combo_survived": combo_survived,
        })

        if len(results) % 50 == 0:
            log(f"    已评估 {len(results)}/{len(test_indices)} 期")

    # 汇总
    reductions = [r["reduction"] for r in results]
    combo_survivals = [r["combo_survived"] for r in results]
    avg_reduction = float(np.mean(reductions))
    survival_rate = sum(combo_survivals) / len(combo_survivals)
    efficiency = avg_reduction * survival_rate

    # 按局面分组统计
    ctx_stats = {}
    for ctx in ["low", "mid", "high"]:
        ctx_results = [r for r in results if r["context"] == ctx]
        if ctx_results:
            ctx_red = np.mean([r["reduction"] for r in ctx_results])
            ctx_surv = sum(r["combo_survived"] for r in ctx_results) / len(ctx_results)
            ctx_stats[ctx] = {
                "n": len(ctx_results),
                "avg_reduction": round(float(ctx_red), 4),
                "survival": round(float(ctx_surv), 4),
                "efficiency": round(float(ctx_red * ctx_surv), 4),
            }

    summary = {
        "hdi_level": hdi_key,
        "conditioned": True,
        "n_constraints": len(cond_constraints),
        "n_test": len(results),
        "avg_reduction": round(avg_reduction, 4),
        "combo_survival": round(survival_rate, 4),
        "efficiency": round(efficiency, 4),
        "context_stats": ctx_stats,
        "context_counts": ctx_counts,
    }
    return summary, results


def run_e7_phase5(lottery_type="daletou"):
    """运行 E7 Phase 5"""
    setup_logging()
    log(f"\n{'='*60}")
    log(f"  E7 Phase 5 局面条件化约束: {lottery_type}")
    log(f"{'='*60}\n")

    data = LotteryData(lottery_type)
    cfg = SPLIT[lottery_type]
    train_end = cfg["train_end"]
    test_end = cfg["total"]

    # 先跑 Phase 1-2 获取基础约束
    with Timer("Phase1"):
        constraints = enumerate_all_constraints(data, train_end)
    with Timer("Phase2"):
        selected, _ = rank_and_filter_constraints(
            constraints, data, train_end, test_end,
            hdi_key="hdi_90", top_n=30)

    # Phase 5: 条件化
    with Timer("Phase5 条件化约束"):
        cond_cons, thresholds, ctx_groups = enumerate_conditioned_constraints(
            data, train_end, selected, top_n=30)

    # 加载 Phase 3 结果对比
    p3_path = RESULTS_DIR / f"e7_phase3_propagation_{lottery_type}.json"
    p3_data = None
    if p3_path.exists():
        with open(p3_path, "r") as f:
            p3_data = json.load(f)

    # 策略对比：全条件化 vs 仅高波动条件化 vs 高+中波动条件化
    strategies = {
        "all_conditioned": None,                    # 全部条件化
        "high_only": ["high"],                      # 仅高波动
        "high_mid": ["high", "mid"],                # 高+中波动
    }

    all_results = {}
    for strat_name, adapt_ctx in strategies.items():
        log(f"\n  === 策略: {strat_name} ===")
        strat_results = {}
        for hdi_key in ["hdi_80", "hdi_90"]:
            log(f"  Phase5 评估 [{hdi_key}] ({strat_name})...")
            with Timer(f"Phase5 {hdi_key} {strat_name}"):
                summary, details = evaluate_conditioned_propagation(
                    data, cond_cons, thresholds, train_end, test_end,
                    hdi_key, adapt_contexts=adapt_ctx)
            strat_results[hdi_key] = summary
            log(f"    缩减率={summary['avg_reduction']*100:.1f}%  "
                f"存活率={summary['combo_survival']*100:.1f}%  "
                f"效率={summary['efficiency']:.4f}")
            for ctx, st in summary["context_stats"].items():
                log(f"      {ctx}: n={st['n']} 缩减={st['avg_reduction']*100:.1f}% "
                    f"存活={st['survival']*100:.1f}% 效率={st['efficiency']:.4f}")
        all_results[strat_name] = strat_results

    # 汇总对比
    log(f"\n  {'='*60}")
    log(f"  Phase5 策略对比 (HDI-90):")
    p3_eff = p3_data["phase3"]["hdi_90"]["efficiency"] if p3_data else 0
    log(f"    Phase3 全局:        效率={p3_eff:.4f}")
    for strat_name, strat_res in all_results.items():
        r = strat_res["hdi_90"]
        delta = r["efficiency"] - p3_eff
        log(f"    Phase5 {strat_name:20s}: 效率={r['efficiency']:.4f} "
            f"(Δ={delta:+.4f}) 缩减={r['avg_reduction']*100:.1f}% "
            f"存活={r['combo_survival']*100:.1f}%")

    # 选最优策略
    best_strat = max(all_results.keys(),
                     key=lambda s: max(v["efficiency"]
                                       for v in all_results[s].values()))
    best_hdi = max(all_results[best_strat].keys(),
                   key=lambda k: all_results[best_strat][k]["efficiency"])
    best = all_results[best_strat][best_hdi]

    # 保存
    save_json({
        "lottery_type": lottery_type,
        "thresholds": thresholds,
        "strategies": {s: r for s, r in all_results.items()},
        "best_strategy": best_strat,
        "best_hdi": best_hdi,
        "best_efficiency": best["efficiency"],
        "phase3_efficiency": p3_eff,
    }, RESULTS_DIR / f"e7_phase5_conditioned_{lottery_type}.json")

    log(f"\n{'='*60}")
    log(f"  E7 Phase5 完成: {lottery_type}")
    log(f"  最优策略: {best_strat} {best_hdi} 效率={best['efficiency']:.4f}")
    log(f"{'='*60}\n")

    return all_results


if __name__ == "__main__":
    lt = sys.argv[1] if len(sys.argv) > 1 else "daletou"
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    if mode == "phase5":
        run_e7_phase5(lt)
    else:
        run_e7(lt)
