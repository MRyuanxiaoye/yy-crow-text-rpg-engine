"""E1.5：多维权重化评估方案

将硬阈值排除改为权重化方案，建立科学评估基线。
包含三个步骤：
  Step A: 权重化替代硬排除（差分分布 + 阈值敏感性 + AUC + E1对比）
  Step B: 组合层约束（蒙特卡洛采样评估）
  Step C: 跨位置关联分析（条件方向概率 + KL散度 + 量化改进）

用法: python3 -m src.research.experiment.e1_5_weighted_evaluation
"""

import sys
import time
import traceback
from bisect import bisect_left
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import research.experiment.utils as exp_utils
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_npz,
)
from research.experiment.e1_combo_evaluation import (
    get_best_probs, count_ordered_combos, check_combo_survival,
)
from research.data_loader import LotteryData

# === 路径常量 ===
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_EXPERIMENT_DIR = RESULTS_DIR / "experiment_strict"


# ============================================================
#  Step A: 权重化替代硬排除
# ============================================================

def compute_diff_distributions(data, max_train_idx):
    """从训练期统计每个位置的差分条件分布

    返回 diff_dist[pos][dir_label] = Counter({diff_val: count})
    dir_label: 'U', 'D', 'E'
    """
    n_pos = data.red_count
    diff_dist = {}

    for pos in range(n_pos):
        diff_dist[pos] = {'U': Counter(), 'D': Counter(), 'E': Counter()}
        series = data.position_series[pos]
        # 差分从第1期开始（series[t] - series[t-1]）
        for t in range(1, max_train_idx):
            diff_val = int(series[t]) - int(series[t - 1])
            if diff_val > 0:
                diff_dist[pos]['U'][diff_val] += 1
            elif diff_val < 0:
                diff_dist[pos]['D'][diff_val] += 1
            else:
                diff_dist[pos]['E'][0] += 1

    return diff_dist


def compute_number_weights(probs, data, test_indices, diff_dist):
    """基于方向概率和差分分布计算每个号码的权重

    核心公式: W(v) = P(U)*P(diff=v-cur|U) + P(D)*P(diff=v-cur|D) + P(E)*P(diff=0|E)*I(v==cur)

    返回 all_weights: list[list[dict]]  # [sample_idx][pos] = {v: weight}
    """
    n_samples, n_pos, _ = probs.shape
    red_range = data.red_range
    all_weights = []

    for i in range(n_samples):
        t = int(test_indices[i])
        sample_weights = []

        for pos in range(n_pos):
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2)
            p_down, p_equal, p_up = (
                float(probs[i, pos, 0]),
                float(probs[i, pos, 1]),
                float(probs[i, pos, 2]),
            )

            weights = {}
            dist_u = diff_dist[pos]['U']
            dist_d = diff_dist[pos]['D']
            dist_e = diff_dist[pos]['E']

            # 各方向的总样本数（用于拉普拉斯平滑）
            total_u = sum(dist_u.values()) + red_range  # 拉普拉斯
            total_d = sum(dist_d.values()) + red_range
            total_e = sum(dist_e.values()) + 1  # E 只有 diff=0

            for v in range(1, red_range + 1):
                diff_val = v - current_val
                w = 0.0

                if diff_val > 0:
                    # 上升方向
                    p_diff_given_u = (dist_u.get(diff_val, 0) + 1) / total_u
                    w += p_up * p_diff_given_u
                elif diff_val < 0:
                    # 下降方向
                    p_diff_given_d = (dist_d.get(diff_val, 0) + 1) / total_d
                    w += p_down * p_diff_given_d
                else:
                    # 持平
                    p_diff_given_e = (dist_e.get(0, 0) + 1) / total_e
                    w += p_equal * p_diff_given_e

                weights[v] = w

            # 归一化
            total_w = sum(weights.values())
            if total_w > 0:
                for v in weights:
                    weights[v] /= total_w

            sample_weights.append(weights)
        all_weights.append(sample_weights)

    return all_weights


def step_a1_weights(probs, data, test_indices):
    """Step A1: 计算号码权重"""
    log("\n--- Step A1: 基于差分分布的号码权重 ---")

    max_train_idx = int(test_indices[0])
    log(f"  训练期: [0, {max_train_idx}), 测试期: [{max_train_idx}, {data.n_draws})")

    with Timer("统计差分分布"):
        diff_dist = compute_diff_distributions(data, max_train_idx)

    # 打印差分分布摘要
    for pos in range(data.red_count):
        n_u = sum(diff_dist[pos]['U'].values())
        n_d = sum(diff_dist[pos]['D'].values())
        n_e = sum(diff_dist[pos]['E'].values())
        log(f"  P{pos}: U={n_u}, D={n_d}, E={n_e}")

    with Timer("计算号码权重"):
        all_weights = compute_number_weights(probs, data, test_indices, diff_dist)

    # 验证归一化
    for i in range(min(5, len(all_weights))):
        for pos in range(data.red_count):
            total = sum(all_weights[i][pos].values())
            assert abs(total - 1.0) < 1e-6, f"权重未归一化: sample={i}, pos={pos}, sum={total}"
    log("  权重归一化验证通过")

    return all_weights, diff_dist


def step_a2_threshold_sensitivity(all_weights, data, test_indices):
    """Step A2: 阈值敏感性分析

    对每位置按权重降序排列，扫描 K=1..red_range，计算 survival_rate(K)。
    找 survival >= 0.95/0.90/0.85 的最小 K。
    """
    log("\n--- Step A2: 阈值敏感性分析 ---")

    n_samples = len(all_weights)
    n_pos = data.red_count
    red_range = data.red_range

    # 对每个 K 值，统计各位置的存活率
    # survival_by_k[pos][k] = 存活样本数
    survival_by_k = {pos: np.zeros(red_range + 1) for pos in range(n_pos)}

    for i in range(n_samples):
        t = int(test_indices[i])
        for pos in range(n_pos):
            true_val = int(data.red_matrix[t, pos])
            # 按权重降序排列
            sorted_nums = sorted(all_weights[i][pos].items(),
                                 key=lambda x: x[1], reverse=True)
            # 找到正确号码的排名（1-based）
            rank = -1
            for r, (v, w) in enumerate(sorted_nums, 1):
                if v == true_val:
                    rank = r
                    break
            # K >= rank 时存活
            if rank > 0:
                for k in range(rank, red_range + 1):
                    survival_by_k[pos][k] += 1

    # 计算存活率
    results = {}
    for pos in range(n_pos):
        survival_rates = survival_by_k[pos] / n_samples
        # 找各档位的最优 K
        optimal = {}
        for target, label in [(0.95, 'k_95'), (0.90, 'k_90'), (0.85, 'k_85')]:
            best_k = red_range
            for k in range(1, red_range + 1):
                if survival_rates[k] >= target:
                    best_k = k
                    break
            optimal[label] = best_k
            optimal[f'{label}_survival'] = float(survival_rates[best_k])
            optimal[f'{label}_reduction'] = 1.0 - best_k / red_range

        results[f'P{pos}'] = {
            'optimal': optimal,
            'survival_curve': [float(survival_rates[k]) for k in range(1, red_range + 1)],
        }
        log(f"  P{pos}: K95={optimal['k_95']}({optimal['k_95_reduction']:.1%}缩减), "
            f"K90={optimal['k_90']}({optimal['k_90_reduction']:.1%}), "
            f"K85={optimal['k_85']}({optimal['k_85_reduction']:.1%})")

    return results


def step_a3_auc(all_weights, data, test_indices):
    """Step A3: ROC-AUC 排序能力评估

    计算正确号码在权重排序中的排名，AUC = 1 - (rank-1)/(red_range-1)
    """
    log("\n--- Step A3: AUC 排序能力评估 ---")

    n_samples = len(all_weights)
    n_pos = data.red_count
    red_range = data.red_range

    ranks = {pos: [] for pos in range(n_pos)}

    for i in range(n_samples):
        t = int(test_indices[i])
        for pos in range(n_pos):
            true_val = int(data.red_matrix[t, pos])
            sorted_nums = sorted(all_weights[i][pos].items(),
                                 key=lambda x: x[1], reverse=True)
            for r, (v, w) in enumerate(sorted_nums, 1):
                if v == true_val:
                    ranks[pos].append(r)
                    break

    results = {}
    all_aucs = []
    for pos in range(n_pos):
        r_arr = np.array(ranks[pos], dtype=float)
        auc = float(1.0 - (r_arr.mean() - 1) / (red_range - 1))
        results[f'P{pos}'] = {
            'mean_rank': float(r_arr.mean()),
            'median_rank': float(np.median(r_arr)),
            'std_rank': float(r_arr.std()),
            'auc': auc,
            'rank_percentiles': {
                'p10': float(np.percentile(r_arr, 10)),
                'p25': float(np.percentile(r_arr, 25)),
                'p50': float(np.percentile(r_arr, 50)),
                'p75': float(np.percentile(r_arr, 75)),
                'p90': float(np.percentile(r_arr, 90)),
            },
        }
        all_aucs.append(auc)
        log(f"  P{pos}: AUC={auc:.4f}, 平均排名={r_arr.mean():.1f}/{red_range}")

    results['overall_auc'] = float(np.mean(all_aucs))
    log(f"  整体 AUC: {results['overall_auc']:.4f} (随机基线=0.5)")

    return results


def step_a4_compare_e1(all_weights, threshold_results, probs, data, test_indices):
    """Step A4: 与 E1 硬阈值方案对比

    用 optimal_k(95%) 生成候选集，计算组合数并与 E1 对比。
    """
    log("\n--- Step A4: 与 E1 硬阈值方案对比 ---")

    n_samples = len(all_weights)
    n_pos = data.red_count
    red_range = data.red_range
    baseline_combos = comb(red_range, n_pos)

    # 获取每位置的 optimal_k (95% 存活率)
    optimal_ks = []
    for pos in range(n_pos):
        k = threshold_results[f'P{pos}']['optimal']['k_95']
        optimal_ks.append(k)
    log(f"  各位置 K95: {optimal_ks}")

    # 权重化方案：按权重 top-K 生成候选集
    weighted_combo_counts = []
    weighted_survivals = []

    for i in range(n_samples):
        t = int(test_indices[i])
        true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
        cand_sets = []

        for pos in range(n_pos):
            sorted_nums = sorted(all_weights[i][pos].items(),
                                 key=lambda x: x[1], reverse=True)
            top_k = set(v for v, w in sorted_nums[:optimal_ks[pos]])
            cand_sets.append(top_k)

        n_combos = count_ordered_combos(cand_sets)
        weighted_combo_counts.append(n_combos)
        survived = check_combo_survival(cand_sets, true_vals)
        weighted_survivals.append(1.0 if survived else 0.0)

    weighted_combo_counts = np.array(weighted_combo_counts, dtype=float)
    weighted_survivals = np.array(weighted_survivals)

    # E1 硬阈值方案（复现）
    from research.experiment.e1_combo_evaluation import generate_candidate_sets
    e1_candidates = generate_candidate_sets(probs, data, test_indices, threshold=0.15)
    e1_combo_counts = []
    e1_survivals = []

    for i in range(n_samples):
        t = int(test_indices[i])
        true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
        n_combos = count_ordered_combos(e1_candidates[i])
        e1_combo_counts.append(n_combos)
        survived = check_combo_survival(e1_candidates[i], true_vals)
        e1_survivals.append(1.0 if survived else 0.0)

    e1_combo_counts = np.array(e1_combo_counts, dtype=float)
    e1_survivals = np.array(e1_survivals)

    results = {
        'weighted_k95': {
            'optimal_ks': optimal_ks,
            'avg_combo_count': float(np.mean(weighted_combo_counts)),
            'combo_reduction': float(1.0 - np.mean(weighted_combo_counts) / baseline_combos),
            'combo_survival': float(np.mean(weighted_survivals)),
            'combo_efficiency': float(
                (1.0 - np.mean(weighted_combo_counts) / baseline_combos) * np.mean(weighted_survivals)
            ),
        },
        'e1_hard_threshold': {
            'threshold': 0.15,
            'avg_combo_count': float(np.mean(e1_combo_counts)),
            'combo_reduction': float(1.0 - np.mean(e1_combo_counts) / baseline_combos),
            'combo_survival': float(np.mean(e1_survivals)),
            'combo_efficiency': float(
                (1.0 - np.mean(e1_combo_counts) / baseline_combos) * np.mean(e1_survivals)
            ),
        },
    }

    log(f"\n  权重化 K95:")
    log(f"    平均组合数: {results['weighted_k95']['avg_combo_count']:,.0f}")
    log(f"    组合缩减率: {results['weighted_k95']['combo_reduction']:.2%}")
    log(f"    组合存活率: {results['weighted_k95']['combo_survival']:.2%}")
    log(f"    组合效率:   {results['weighted_k95']['combo_efficiency']:.4f}")
    log(f"\n  E1 硬阈值 (0.15):")
    log(f"    平均组合数: {results['e1_hard_threshold']['avg_combo_count']:,.0f}")
    log(f"    组合缩减率: {results['e1_hard_threshold']['combo_reduction']:.2%}")
    log(f"    组合存活率: {results['e1_hard_threshold']['combo_survival']:.2%}")
    log(f"    组合效率:   {results['e1_hard_threshold']['combo_efficiency']:.4f}")

    return results


# ============================================================
#  Step B: 组合层约束
# ============================================================

def compute_combo_stats_ranges(data, max_train_idx):
    """B0: 从训练期统计组合统计量的百分位范围"""
    log("\n--- Step B0: 训练期组合统计量分布 ---")

    combo_stats = data.get_combo_stats_series()
    stat_names = ['sum', 'span', 'odd_count', 'big_count', 'ac_value', 'consec_groups']
    ranges = {}

    for sn in stat_names:
        vals = combo_stats[sn][:max_train_idx]
        ranges[sn] = {
            'p1': float(np.percentile(vals, 1)),
            'p5': float(np.percentile(vals, 5)),
            'p25': float(np.percentile(vals, 25)),
            'p50': float(np.percentile(vals, 50)),
            'p75': float(np.percentile(vals, 75)),
            'p95': float(np.percentile(vals, 95)),
            'p99': float(np.percentile(vals, 99)),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
        }
        log(f"  {sn}: P5={ranges[sn]['p5']:.1f}, P50={ranges[sn]['p50']:.1f}, "
            f"P95={ranges[sn]['p95']:.1f}")

    # 相邻位置间距分布
    gap_ranges = {}
    for pos in range(data.red_count - 1):
        gaps = data.red_matrix[:max_train_idx, pos + 1] - data.red_matrix[:max_train_idx, pos]
        gap_key = f'gap_P{pos}_P{pos+1}'
        gap_ranges[gap_key] = {
            'p1': float(np.percentile(gaps, 1)),
            'p5': float(np.percentile(gaps, 5)),
            'p50': float(np.percentile(gaps, 50)),
            'p95': float(np.percentile(gaps, 95)),
            'p99': float(np.percentile(gaps, 99)),
            'mean': float(np.mean(gaps)),
        }
        log(f"  {gap_key}: P5={gap_ranges[gap_key]['p5']:.0f}, "
            f"P50={gap_ranges[gap_key]['p50']:.0f}, P95={gap_ranges[gap_key]['p95']:.0f}")

    ranges['gaps'] = gap_ranges
    return ranges, combo_stats


def sample_combo_weighted(weights_per_pos, n_pos, max_retries=50):
    """按权重条件采样一个满足排序约束的组合

    逐位置采样，每个位置只从大于前一位置值的号码中按权重采样。
    返回 combo (tuple) 或 None（采样失败）。
    """
    combo = []
    prev_val = 0

    for pos in range(n_pos):
        candidates = {v: w for v, w in weights_per_pos[pos].items() if v > prev_val}
        if not candidates:
            return None

        nums = list(candidates.keys())
        ws = np.array([candidates[v] for v in nums], dtype=float)
        ws_sum = ws.sum()
        if ws_sum <= 0:
            return None
        ws /= ws_sum

        chosen = np.random.choice(nums, p=ws)
        combo.append(int(chosen))
        prev_val = int(chosen)

    return tuple(combo)


def compute_combo_stat(combo, data):
    """计算单个组合的统计量"""
    arr = np.array(combo)
    mid = (1 + data.red_range) / 2

    stat_sum = int(np.sum(arr))
    stat_span = int(np.max(arr) - np.min(arr))
    odd_count = int(np.sum(arr % 2 == 1))
    big_count = int(np.sum(arr > mid))

    # AC值
    diffs = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            diffs.add(abs(int(arr[i]) - int(arr[j])))
    ac_value = len(diffs) - (len(arr) - 1)

    # 相邻间距
    gaps = [int(arr[i + 1]) - int(arr[i]) for i in range(len(arr) - 1)]

    return {
        'sum': stat_sum,
        'span': stat_span,
        'odd_count': odd_count,
        'big_count': big_count,
        'ac_value': ac_value,
        'gaps': gaps,
    }


def step_b_combo_constraints(all_weights, data, test_indices, n_mc=10000):
    """Step B: 蒙特卡洛采样评估组合层约束"""
    log("\n--- Step B: 组合层约束评估 ---")

    max_train_idx = int(test_indices[0])
    n_samples = len(all_weights)
    n_pos = data.red_count

    # B0: 统计训练期分布
    stat_ranges, combo_stats = compute_combo_stats_ranges(data, max_train_idx)

    # 定义约束（使用 P5/P95 范围）
    constraints = {
        'sum': (stat_ranges['sum']['p5'], stat_ranges['sum']['p95']),
        'odd_count': (stat_ranges['odd_count']['p5'], stat_ranges['odd_count']['p95']),
        'big_count': (stat_ranges['big_count']['p5'], stat_ranges['big_count']['p95']),
        'ac_value': (stat_ranges['ac_value']['p5'], stat_ranges['ac_value']['p95']),
        'span': (stat_ranges['span']['p5'], stat_ranges['span']['p95']),
    }
    log(f"\n  约束范围 (P5-P95):")
    for name, (lo, hi) in constraints.items():
        log(f"    {name}: [{lo:.0f}, {hi:.0f}]")

    # 间距约束
    gap_constraints = {}
    for gap_key, gr in stat_ranges.get('gaps', {}).items():
        gap_constraints[gap_key] = (gr['p5'], gr['p95'])

    # 逐期蒙特卡洛采样
    log(f"\n  蒙特卡洛采样: 每期 {n_mc} 个组合...")

    constraint_names = list(constraints.keys()) + ['gaps_all']
    filter_counts = {c: 0 for c in constraint_names}
    filter_counts['all_combined'] = 0
    total_sampled = 0
    sample_fail_count = 0

    # 对正确组合检查约束是否成立
    true_pass = {c: 0 for c in constraint_names}
    true_pass['all_combined'] = 0
    true_total = 0

    # 抽样部分期数评估
    eval_indices = np.linspace(0, n_samples - 1, min(100, n_samples), dtype=int)

    for idx_i, i in enumerate(eval_indices):
        t = int(test_indices[i])
        true_vals = tuple(int(data.red_matrix[t, pos]) for pos in range(n_pos))
        true_stat = compute_combo_stat(true_vals, data)

        # 检查正确组合是否通过各约束
        true_total += 1
        all_pass_true = True
        for cname, (lo, hi) in constraints.items():
            if lo <= true_stat[cname] <= hi:
                true_pass[cname] += 1
            else:
                all_pass_true = False

        gaps_pass_true = True
        for g_idx, (gap_key, (glo, ghi)) in enumerate(gap_constraints.items()):
            if not (glo <= true_stat['gaps'][g_idx] <= ghi):
                gaps_pass_true = False
        if gaps_pass_true:
            true_pass['gaps_all'] += 1
        else:
            all_pass_true = False

        if all_pass_true:
            true_pass['all_combined'] += 1

        # 蒙特卡洛采样
        sampled = 0
        for _ in range(n_mc):
            combo = sample_combo_weighted(all_weights[i], n_pos)
            if combo is None:
                sample_fail_count += 1
                continue
            sampled += 1
            stat = compute_combo_stat(combo, data)

            all_pass = True
            for cname, (lo, hi) in constraints.items():
                if not (lo <= stat[cname] <= hi):
                    filter_counts[cname] += 1
                    all_pass = False

            gaps_pass = True
            for g_idx, (gap_key, (glo, ghi)) in enumerate(gap_constraints.items()):
                if not (glo <= stat['gaps'][g_idx] <= ghi):
                    gaps_pass = False
            if not gaps_pass:
                filter_counts['gaps_all'] += 1
                all_pass = False

            if not all_pass:
                filter_counts['all_combined'] += 1

        total_sampled += sampled

        if (idx_i + 1) % 20 == 0:
            log(f"    进度: {idx_i+1}/{len(eval_indices)}")

    # 汇总
    results = {
        'stat_ranges': stat_ranges,
        'constraints_p5_p95': {k: {'lo': v[0], 'hi': v[1]} for k, v in constraints.items()},
        'gap_constraints': {k: {'lo': v[0], 'hi': v[1]} for k, v in gap_constraints.items()},
        'mc_samples_per_period': n_mc,
        'n_eval_periods': len(eval_indices),
        'total_sampled': total_sampled,
        'sample_fail_rate': float(
            sample_fail_count / (len(eval_indices) * n_mc)) if n_mc > 0 else 0,
    }

    log(f"\n  采样成功率: {1.0 - results['sample_fail_rate']:.2%}")
    log(f"  总有效样本: {total_sampled}")

    constraint_results = {}
    for cname in constraint_names + ['all_combined']:
        filter_rate = filter_counts[cname] / total_sampled if total_sampled > 0 else 0
        survival_impact = true_pass.get(cname, 0) / true_total if true_total > 0 else 0

        constraint_results[cname] = {
            'filter_rate': float(filter_rate),
            'true_survival': float(survival_impact),
            'net_value': float(filter_rate * survival_impact),
        }
        log(f"  {cname}: 过滤率={filter_rate:.2%}, 正确组合通过率={survival_impact:.2%}")

    results['constraint_effects'] = constraint_results
    return results


# ============================================================
#  Step C: 跨位置关联分析
# ============================================================

def step_c_cross_position(data, test_indices):
    """Step C: 跨位置关联分析"""
    log("\n--- Step C: 跨位置关联分析 ---")

    max_train_idx = int(test_indices[0])
    n_pos = data.red_count
    dir_series = data.direction_series  # {pos: array}, 值为 -1/0/1

    # C1: 相邻位置条件方向概率
    log("\n  C1: 位置间条件方向概率")
    dir_labels = ['D', 'E', 'U']  # -1, 0, 1
    dir_map = {-1: 0, 0: 1, 1: 2}

    cond_probs = {}
    for pos in range(n_pos - 1):
        # 统计 P(dir_{pos+1} | dir_pos) 的 3x3 矩阵
        joint = np.zeros((3, 3), dtype=int)
        # dir_series[pos] 长度 = n_draws - 1
        max_dir_idx = min(max_train_idx - 1, len(dir_series[pos]), len(dir_series[pos + 1]))
        for t in range(max_dir_idx):
            d_cur = dir_map[int(dir_series[pos][t])]
            d_next = dir_map[int(dir_series[pos + 1][t])]
            joint[d_cur, d_next] += 1

        # 条件概率
        row_sums = joint.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)  # 避免除零
        cond_prob = joint / row_sums

        pair_key = f'P{pos}_P{pos+1}'
        cond_probs[pair_key] = {
            'joint_counts': joint.tolist(),
            'cond_prob': cond_prob.tolist(),
            'row_labels': dir_labels,
            'col_labels': dir_labels,
        }
        log(f"  {pair_key} 条件概率矩阵:")
        for r in range(3):
            row_str = "    " + " ".join(f"{cond_prob[r, c]:.3f}" for c in range(3))
            log(f"    {dir_labels[r]}: {row_str}")

    # C2: 联合方向分布 vs 独立假设
    log("\n  C2: 联合方向分布 vs 独立假设")

    # 只分析相邻位置对（避免样本量不足）
    kl_results = {}
    for pos in range(n_pos - 1):
        pair_key = f'P{pos}_P{pos+1}'
        max_dir_idx = min(max_train_idx - 1, len(dir_series[pos]), len(dir_series[pos + 1]))

        # 联合分布
        joint = np.zeros((3, 3), dtype=float)
        for t in range(max_dir_idx):
            d_a = dir_map[int(dir_series[pos][t])]
            d_b = dir_map[int(dir_series[pos + 1][t])]
            joint[d_a, d_b] += 1

        n_total = joint.sum()
        if n_total == 0:
            continue

        joint_prob = joint / n_total

        # 边缘分布
        margin_a = joint_prob.sum(axis=1)
        margin_b = joint_prob.sum(axis=0)

        # 独立假设下的联合分布
        indep_prob = np.outer(margin_a, margin_b)

        # KL 散度: KL(joint || indep)
        kl_div = 0.0
        for a in range(3):
            for b in range(3):
                if joint_prob[a, b] > 0 and indep_prob[a, b] > 0:
                    kl_div += joint_prob[a, b] * np.log(
                        joint_prob[a, b] / indep_prob[a, b])

        # 卡方检验
        from scipy import stats as sp_stats
        chi2_stat, chi2_p, chi2_dof, _ = sp_stats.chi2_contingency(
            np.maximum(joint, 0.5))  # 加0.5避免零单元格

        kl_results[pair_key] = {
            'kl_divergence': float(kl_div),
            'chi2_statistic': float(chi2_stat),
            'chi2_p_value': float(chi2_p),
            'chi2_dof': int(chi2_dof),
            'n_samples': int(n_total),
            'margin_a': margin_a.tolist(),
            'margin_b': margin_b.tolist(),
        }
        sig = "显著" if chi2_p < 0.05 else "不显著"
        log(f"  {pair_key}: KL={kl_div:.6f}, χ²={chi2_stat:.2f}, "
            f"p={chi2_p:.4f} ({sig})")

    # C3: 量化独立假设的损失
    log("\n  C3: 条件预测 vs 独立预测")

    # 在测试期上对比：
    # 独立预测：每位置用边缘方向概率
    # 条件预测：后续位置用条件概率修正
    n_test = len(test_indices)
    indep_correct = 0
    cond_correct = 0
    total_predictions = 0

    for i in range(n_test):
        t = int(test_indices[i])
        if t < 1 or t >= data.n_draws:
            continue

        for pos in range(1, n_pos):
            dir_idx = t - 1
            if dir_idx >= len(dir_series[pos]) or dir_idx >= len(dir_series[pos - 1]):
                continue

            true_dir = dir_map[int(dir_series[pos][dir_idx])]
            prev_dir = dir_map[int(dir_series[pos - 1][dir_idx])]

            pair_key = f'P{pos-1}_P{pos}'
            if pair_key not in kl_results:
                continue

            # 独立预测：用边缘分布
            margin = np.array(kl_results[pair_key]['margin_b'])
            indep_pred = int(np.argmax(margin))

            # 条件预测：用条件概率
            cond_prob_matrix = np.array(cond_probs[pair_key]['cond_prob'])
            cond_pred = int(np.argmax(cond_prob_matrix[prev_dir]))

            if indep_pred == true_dir:
                indep_correct += 1
            if cond_pred == true_dir:
                cond_correct += 1
            total_predictions += 1

    indep_acc = indep_correct / total_predictions if total_predictions > 0 else 0
    cond_acc = cond_correct / total_predictions if total_predictions > 0 else 0
    improvement = cond_acc - indep_acc

    c3_results = {
        'total_predictions': total_predictions,
        'independent_accuracy': float(indep_acc),
        'conditional_accuracy': float(cond_acc),
        'improvement': float(improvement),
        'relative_improvement': float(improvement / indep_acc) if indep_acc > 0 else 0,
    }
    log(f"  独立预测准确率: {indep_acc:.4f}")
    log(f"  条件预测准确率: {cond_acc:.4f}")
    log(f"  改进幅度: {improvement:+.4f} ({c3_results['relative_improvement']:+.2%})")

    return {
        'c1_conditional_probs': cond_probs,
        'c2_independence_test': kl_results,
        'c3_conditional_vs_independent': c3_results,
    }


# ============================================================
#  主评估流程
# ============================================================

def evaluate_e1_5(lottery_type):
    """对单个彩种执行 E1.5 全流程评估"""
    log(f"\n{'═'*55}")
    log(f"  E1.5 多维权重化评估: {lottery_type}")
    log(f"{'═'*55}")

    # 获取 probs
    with Timer("获取方向概率"):
        probs, test_indices, data, best_method = get_best_probs(lottery_type)

    n_samples, n_pos = probs.shape[0], probs.shape[1]
    log(f"  测试样本数: {n_samples}, 位置数: {n_pos}")
    log(f"  最优方案: {best_method}")
    log(f"  红球范围: 1-{data.red_range}")

    # === Step A ===
    log(f"\n{'─'*40}")
    log(f"  Step A: 权重化替代硬排除")
    log(f"{'─'*40}")

    all_weights, diff_dist = step_a1_weights(probs, data, test_indices)
    threshold_results = step_a2_threshold_sensitivity(all_weights, data, test_indices)
    auc_results = step_a3_auc(all_weights, data, test_indices)
    compare_results = step_a4_compare_e1(
        all_weights, threshold_results, probs, data, test_indices)

    step_a = {
        'threshold_sensitivity': threshold_results,
        'auc': auc_results,
        'e1_comparison': compare_results,
    }
    save_json(step_a, STRICT_EXPERIMENT_DIR / f"e1_5_step_a_{lottery_type}.json")

    # === Step B ===
    log(f"\n{'─'*40}")
    log(f"  Step B: 组合层约束")
    log(f"{'─'*40}")

    step_b = step_b_combo_constraints(all_weights, data, test_indices)
    save_json(step_b, STRICT_EXPERIMENT_DIR / f"e1_5_step_b_{lottery_type}.json")

    # === Step C ===
    log(f"\n{'─'*40}")
    log(f"  Step C: 跨位置关联分析")
    log(f"{'─'*40}")

    step_c = step_c_cross_position(data, test_indices)
    save_json(step_c, STRICT_EXPERIMENT_DIR / f"e1_5_step_c_{lottery_type}.json")

    return {
        'lottery_type': lottery_type,
        'best_method': best_method,
        'n_test_samples': n_samples,
        'step_a_summary': {
            'overall_auc': auc_results['overall_auc'],
            'weighted_k95_efficiency': compare_results['weighted_k95']['combo_efficiency'],
            'e1_efficiency': compare_results['e1_hard_threshold']['combo_efficiency'],
            'improvement': (
                compare_results['weighted_k95']['combo_efficiency']
                - compare_results['e1_hard_threshold']['combo_efficiency']
            ),
        },
        'step_b_summary': {
            'all_combined_filter_rate': step_b['constraint_effects']['all_combined']['filter_rate'],
            'all_combined_true_survival': step_b['constraint_effects']['all_combined']['true_survival'],
        },
        'step_c_summary': {
            'conditional_improvement': step_c['c3_conditional_vs_independent']['improvement'],
            'any_significant_pair': any(
                v['chi2_p_value'] < 0.05
                for v in step_c['c2_independence_test'].values()
            ),
        },
    }


def main():
    STRICT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # 配置日志
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STRICT_EXPERIMENT_DIR
    setup_logging()
    exp_utils.EXPERIMENT_DIR = original_exp_dir

    start_time = time.time()

    log("=" * 60)
    log("  E1.5：多维权重化评估方案")
    log("=" * 60)

    all_results = {}

    for lottery_type in ["daletou", "shuangseqiu"]:
        try:
            results = evaluate_e1_5(lottery_type)
            all_results[lottery_type] = results
        except Exception as e:
            log(f"\n[错误] {lottery_type} 评估失败: {e}")
            log(traceback.format_exc())
            all_results[lottery_type] = {"error": str(e)}

    # 汇总
    total_time = time.time() - start_time
    summary = {
        'total_time_seconds': round(total_time, 1),
        'total_time_minutes': round(total_time / 60, 1),
        'results': all_results,
    }
    save_json(summary, STRICT_EXPERIMENT_DIR / "e1_5_summary.json")

    log(f"\n{'═'*60}")
    log(f"  E1.5 评估完成！总耗时: {summary['total_time_minutes']:.1f} 分钟")
    log(f"  结果目录: {STRICT_EXPERIMENT_DIR}")
    log(f"{'═'*60}")


if __name__ == "__main__":
    main()
