"""E3：规则灵活运用框架（四层递进架构）

Layer 0: 硬排除层（数据驱动的确定性排除）
Layer 1: 号码权重层（多信号融合的连续权重）
Layer 2: 组合构造层（约束感知的 Beam Search）
Layer 3: 组合排序层（全局最优选择）

用法: python3 -m src.research.experiment.e3_rule_framework
"""

import sys
import time
import traceback
from collections import Counter, defaultdict
from math import comb, log as math_log
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import research.experiment.utils as exp_utils
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_json, load_npz,
    parse_target, EXPERIMENT_DIR,
)
from research.experiment.e1_combo_evaluation import (
    get_best_probs, count_ordered_combos, check_combo_survival,
)
from research.experiment.e1_5_weighted_evaluation import (
    compute_diff_distributions, compute_number_weights,
    compute_combo_stat, compute_combo_stats_ranges,
    sample_combo_weighted,
)
from research.experiment.e2a_conditional_correction import (
    build_cond_prob_matrix, DIR_MAP,
)
from research.experiment.phase2_replay import (
    check_rule_conditions, check_direction_pattern,
    load_a3_rules, load_a4_rules,
    precompute_a3_discrete, precompute_a4_discrete,
    select_a3_top_patterns, select_a4_top_patterns,
    check_a3_match, check_a4_match,
    A3_LABEL_TO_CODE,
)
from research.data_loader import LotteryData

# === 路径常量 ===
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_EXPERIMENT_DIR = RESULTS_DIR / "experiment_strict"


# ============================================================
#  E3a: Layer 0 硬排除边界验证
# ============================================================

def e3a_hard_exclusion(lottery_type):
    """E3a: 验证哪些规则可以安全硬排除（误排除率 < 1%）"""
    log(f"\n{'═'*55}")
    log(f"  E3a 硬排除边界验证: {lottery_type}")
    log(f"{'═'*55}")

    probs, test_indices, data, best_method = get_best_probs(lottery_type)
    n_samples, n_pos = probs.shape[0], probs.shape[1]
    red_range = data.red_range
    max_train_idx = int(test_indices[0])

    log(f"  测试样本: {n_samples}, 位置数: {n_pos}, 红球范围: 1-{red_range}")

    # 预计算差分分布
    with Timer("统计差分分布"):
        diff_dist = compute_diff_distributions(data, max_train_idx)

    results = {"lottery_type": lottery_type, "rules": {}}

    # === 规则 1：排序约束值域传播 ===
    log("\n  --- 规则 1：排序约束值域传播 ---")
    rule1_errors = 0
    rule1_total = 0
    rule1_excluded_counts = []

    for i in range(n_samples):
        t = int(test_indices[i])
        true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
        excluded_count = 0

        for pos in range(n_pos):
            # P_i ∈ [i+1, red_range - (n_pos - 1 - i)]
            lo = pos + 1
            hi = red_range - (n_pos - 1 - pos)
            true_v = true_vals[pos]
            # 排除范围外的号码数
            excluded_count += (lo - 1) + (red_range - hi)
            if true_v < lo or true_v > hi:
                rule1_errors += 1

        rule1_total += n_pos
        rule1_excluded_counts.append(excluded_count)

    rule1_error_rate = rule1_errors / rule1_total if rule1_total > 0 else 0
    rule1_avg_excluded = float(np.mean(rule1_excluded_counts)) / n_pos  # 每位置平均排除数
    log(f"  误排除率: {rule1_error_rate:.4%} ({rule1_errors}/{rule1_total})")
    log(f"  每位置平均排除: {rule1_avg_excluded:.1f}/{red_range}")

    results["rules"]["rule1_ordering_constraint"] = {
        "error_rate": rule1_error_rate,
        "errors": rule1_errors,
        "total": rule1_total,
        "avg_excluded_per_pos": rule1_avg_excluded,
        "go": rule1_error_rate < 0.01,
    }

    # === 规则 2：方向极端概率排除 ===
    log("\n  --- 规则 2：方向极端概率排除 ---")
    thresholds = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    rule2_results = {}

    for theta in thresholds:
        errors = 0
        total_excluded = 0
        total_checks = 0

        for i in range(n_samples):
            t = int(test_indices[i])
            for pos in range(n_pos):
                current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2)
                true_val = int(data.red_matrix[t, pos])
                p_d, p_e, p_u = float(probs[i, pos, 0]), float(probs[i, pos, 1]), float(probs[i, pos, 2])

                excluded = set()
                # P(U) > theta → 排除 v <= current_val
                if p_u > theta:
                    excluded.update(v for v in range(1, current_val + 1))
                # P(D) > theta → 排除 v >= current_val
                if p_d > theta:
                    excluded.update(v for v in range(current_val, red_range + 1))

                if excluded:
                    total_checks += 1
                    total_excluded += len(excluded)
                    if true_val in excluded:
                        errors += 1

        error_rate = errors / total_checks if total_checks > 0 else 0
        avg_excluded = total_excluded / total_checks if total_checks > 0 else 0
        trigger_rate = total_checks / (n_samples * n_pos)

        rule2_results[f"theta_{theta}"] = {
            "theta": theta,
            "error_rate": error_rate,
            "errors": errors,
            "total_checks": total_checks,
            "trigger_rate": trigger_rate,
            "avg_excluded_when_triggered": avg_excluded,
            "go": error_rate < 0.01,
        }
        status = "GO" if error_rate < 0.01 else "NO-GO"
        log(f"  θ={theta:.2f}: 误排除={error_rate:.4%}, 触发率={trigger_rate:.2%}, "
            f"平均排除={avg_excluded:.1f} [{status}]")

    # 选误排除率 < 1% 的最低阈值
    best_theta = None
    for theta in thresholds:
        key = f"theta_{theta}"
        if rule2_results[key]["go"] and rule2_results[key]["total_checks"] > 0:
            best_theta = theta
            break

    results["rules"]["rule2_direction_extreme"] = {
        "scan_results": rule2_results,
        "best_theta": best_theta,
        "go": best_theta is not None,
    }
    if best_theta:
        log(f"  最优阈值: θ={best_theta}")
    else:
        log(f"  无满足条件的阈值")

    # === 规则 3：差分幅度极端值排除 ===
    log("\n  --- 规则 3：差分幅度极端值排除 ---")
    # 用训练期差分分布的 1%-99% 百分位做硬排除
    rule3_errors = 0
    rule3_total = 0
    rule3_excluded_counts = []

    # 预计算每位置每方向的差分百分位
    diff_percentiles = {}
    for pos in range(n_pos):
        diff_percentiles[pos] = {}
        for dir_label in ['U', 'D']:
            diffs = list(diff_dist[pos][dir_label].elements())
            if len(diffs) >= 10:
                diff_percentiles[pos][dir_label] = {
                    'p1': float(np.percentile(diffs, 1)),
                    'p99': float(np.percentile(diffs, 99)),
                }
            else:
                # 样本太少，不做排除
                diff_percentiles[pos][dir_label] = None

    for i in range(n_samples):
        t = int(test_indices[i])
        excluded_count = 0

        for pos in range(n_pos):
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2)
            true_val = int(data.red_matrix[t, pos])
            true_diff = true_val - current_val

            excluded = set()
            # 上升方向的差分范围
            up_pct = diff_percentiles[pos].get('U')
            if up_pct is not None:
                for v in range(current_val + 1, red_range + 1):
                    diff = v - current_val
                    if diff < up_pct['p1'] or diff > up_pct['p99']:
                        excluded.add(v)

            # 下降方向的差分范围
            down_pct = diff_percentiles[pos].get('D')
            if down_pct is not None:
                for v in range(1, current_val):
                    diff = v - current_val  # 负数
                    if diff < down_pct['p1'] or diff > down_pct['p99']:
                        excluded.add(v)

            excluded_count += len(excluded)
            if true_val in excluded:
                rule3_errors += 1
            rule3_total += 1

        rule3_excluded_counts.append(excluded_count)

    rule3_error_rate = rule3_errors / rule3_total if rule3_total > 0 else 0
    rule3_avg_excluded = float(np.mean(rule3_excluded_counts)) / n_pos
    log(f"  误排除率: {rule3_error_rate:.4%} ({rule3_errors}/{rule3_total})")
    log(f"  每位置平均排除: {rule3_avg_excluded:.1f}/{red_range}")

    results["rules"]["rule3_diff_extreme"] = {
        "error_rate": rule3_error_rate,
        "errors": rule3_errors,
        "total": rule3_total,
        "avg_excluded_per_pos": rule3_avg_excluded,
        "diff_percentiles": {
            f"P{pos}": {
                d: diff_percentiles[pos][d] for d in ['U', 'D']
                if diff_percentiles[pos].get(d) is not None
            } for pos in range(n_pos)
        },
        "go": rule3_error_rate < 0.01,
    }

    # === 汇总 ===
    go_rules = [k for k, v in results["rules"].items() if v.get("go")]
    results["summary"] = {
        "go_rules": go_rules,
        "n_go_rules": len(go_rules),
        "overall_go": len(go_rules) >= 1,
    }
    log(f"\n  E3a 汇总: {len(go_rules)} 条规则通过 ({go_rules})")

    save_json(results, STRICT_EXPERIMENT_DIR / f"e3a_hard_exclusion_{lottery_type}.json")
    return results


def apply_hard_exclusion(probs_i, pos, current_val, red_range, n_pos,
                         e3a_results, diff_percentiles_cache):
    """对单个样本单个位置应用硬排除，返回候选号码集合"""
    candidates = set(range(1, red_range + 1))

    rules = e3a_results.get("rules", {})

    # 规则 1：排序约束
    if rules.get("rule1_ordering_constraint", {}).get("go"):
        lo = pos + 1
        hi = red_range - (n_pos - 1 - pos)
        candidates = {v for v in candidates if lo <= v <= hi}

    # 规则 2：方向极端概率
    rule2 = rules.get("rule2_direction_extreme", {})
    if rule2.get("go"):
        theta = rule2["best_theta"]
        p_d, p_e, p_u = float(probs_i[0]), float(probs_i[1]), float(probs_i[2])
        if p_u > theta:
            candidates = {v for v in candidates if v > current_val}
        if p_d > theta:
            candidates = {v for v in candidates if v < current_val}

    # 规则 3：差分极端值
    if rules.get("rule3_diff_extreme", {}).get("go"):
        pcts = diff_percentiles_cache.get(pos, {})
        up_pct = pcts.get('U')
        down_pct = pcts.get('D')
        to_remove = set()
        for v in candidates:
            diff = v - current_val
            if diff > 0 and up_pct is not None:
                if diff < up_pct['p1'] or diff > up_pct['p99']:
                    to_remove.add(v)
            elif diff < 0 and down_pct is not None:
                if diff < down_pct['p1'] or diff > down_pct['p99']:
                    to_remove.add(v)
        candidates -= to_remove

    # 至少保留 1 个候选
    if not candidates:
        candidates = set(range(1, red_range + 1))

    return candidates


# ============================================================
#  E3b: Layer 1 多信号融合验证
# ============================================================

def signal_a_baseline(probs, data, test_indices, diff_dist):
    """信号源 A：E1.5 基线权重（方向概率 × 差分分布）"""
    return compute_number_weights(probs, data, test_indices, diff_dist)


def signal_b_rule_cluster(probs, data, test_indices, diff_dist, max_train_idx):
    """信号源 B：规则簇条件权重

    对每个位置，统计训练期中各号码在规则簇触发/未触发时的出现频率差异，
    作为额外的权重信号。
    """
    n_samples, n_pos, _ = probs.shape
    red_range = data.red_range

    # 加载规则簇
    rules_dir = RESULTS_DIR / "rules_strict"
    cluster_path = rules_dir / f"a2_clusters_{data.lottery_type}.json"
    if not cluster_path.exists():
        # fallback: experiment 目录下的 phase1 聚类结果
        cluster_path = RESULTS_DIR / "experiment" / f"phase1_clusters_{data.lottery_type}.json"
    if not cluster_path.exists():
        log(f"  [信号B] 规则簇文件不存在，跳过")
        return None

    raw = load_json(cluster_path)
    if isinstance(raw, dict) and 'clusters' in raw:
        clusters = raw['clusters']
    else:
        clusters = raw
    if not clusters:
        log(f"  [信号B] 规则簇为空，跳过")
        return None

    log(f"  [信号B] 加载 {len(clusters)} 个规则簇")

    # 预计算差分序列和组合统计量
    diff_series = {}
    for pos in range(n_pos):
        diff_series[pos] = data.get_diff_series(pos, 1)
    combo_stats = data.get_combo_stats_series()

    # 统计训练期中每个规则簇触发时各位置各号码的出现频率
    # cluster_freq[cid][pos][v] = 触发时 v 出现的次数
    cluster_freq = {}
    cluster_trigger_count = {}

    for cid, cinfo in clusters.items():
        rule = cinfo['representative']
        cluster_freq[cid] = {pos: Counter() for pos in range(n_pos)}
        trigger_count = 0

        for t in range(20, max_train_idx):
            triggered = check_rule_conditions(
                rule['conditions'], data.red_matrix, data.direction_series,
                diff_series, combo_stats, t, data.red_range)
            if triggered:
                trigger_count += 1
                for pos in range(n_pos):
                    v = int(data.red_matrix[t, pos])
                    cluster_freq[cid][pos][v] += 1

        cluster_trigger_count[cid] = trigger_count

    # 计算全局频率（训练期）
    global_freq = {pos: Counter() for pos in range(n_pos)}
    for t in range(20, max_train_idx):
        for pos in range(n_pos):
            v = int(data.red_matrix[t, pos])
            global_freq[pos][v] += 1
    n_train = max_train_idx - 20

    # 对测试期每个样本，检查哪些规则簇触发，计算条件权重
    all_weights_b = []
    for i in range(n_samples):
        t = int(test_indices[i])
        sample_weights = []

        # 检查哪些规则簇在 t 期触发
        triggered_cids = []
        for cid, cinfo in clusters.items():
            rule = cinfo['representative']
            if check_rule_conditions(
                    rule['conditions'], data.red_matrix, data.direction_series,
                    diff_series, combo_stats, t, data.red_range):
                triggered_cids.append(cid)

        for pos in range(n_pos):
            weights = {}
            if triggered_cids:
                # 用触发规则簇的条件频率加权
                for v in range(1, red_range + 1):
                    w = 0.0
                    total_triggers = 0
                    for cid in triggered_cids:
                        tc = cluster_trigger_count[cid]
                        if tc > 0:
                            # 条件频率（拉普拉斯平滑）
                            freq = (cluster_freq[cid][pos].get(v, 0) + 1) / (tc + red_range)
                            w += freq * cinfo.get('avg_confidence', 0.5)
                            total_triggers += 1
                    if total_triggers > 0:
                        w /= total_triggers
                    else:
                        w = 1.0 / red_range
                    weights[v] = w
            else:
                # 无触发，用全局频率
                for v in range(1, red_range + 1):
                    weights[v] = (global_freq[pos].get(v, 0) + 1) / (n_train + red_range)

            # 归一化
            total_w = sum(weights.values())
            if total_w > 0:
                for v in weights:
                    weights[v] /= total_w
            sample_weights.append(weights)
        all_weights_b.append(sample_weights)

    return all_weights_b


def signal_c_cross_position(probs, data, test_indices, cond_matrices):
    """信号源 C：跨位置一致性奖惩

    利用相邻位置的条件方向概率矩阵，对号码权重进行一致性调整。
    如果某号码对应的方向与相邻位置的条件预测一致，则奖励；否则惩罚。
    """
    n_samples, n_pos, _ = probs.shape
    red_range = data.red_range

    if cond_matrices is None:
        log(f"  [信号C] 条件概率矩阵不可用，跳过")
        return None

    all_weights_c = []
    for i in range(n_samples):
        t = int(test_indices[i])
        current_vals = []
        for pos in range(n_pos):
            current_vals.append(
                int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2))

        sample_weights = []
        for pos in range(n_pos):
            weights = {}
            for v in range(1, red_range + 1):
                diff = v - current_vals[pos]
                if diff > 0:
                    v_dir = 2  # U
                elif diff < 0:
                    v_dir = 0  # D
                else:
                    v_dir = 1  # E

                # 一致性得分：与相邻位置的条件预测对齐程度
                consistency = 0.0
                n_neighbors = 0

                # 前一位置 → 当前位置
                if pos > 0:
                    fwd_key = (pos - 1, pos)
                    if fwd_key in cond_matrices:
                        # 前一位置的预测方向
                        prev_dir_probs = probs[i, pos - 1, :]
                        # 条件概率：P(v_dir | prev_dir)
                        cond_mat = cond_matrices[fwd_key]
                        p_cond = float(prev_dir_probs @ cond_mat[:, v_dir])
                        consistency += p_cond
                        n_neighbors += 1

                # 当前位置 → 后一位置（反向一致性）
                if pos < n_pos - 1:
                    bwd_key = (pos, pos + 1)
                    if bwd_key in cond_matrices:
                        next_dir_probs = probs[i, pos + 1, :]
                        cond_mat = cond_matrices[bwd_key]
                        # P(next_dir | v_dir) 与 next_dir_probs 的对齐
                        p_given_v = cond_mat[v_dir, :]  # (3,)
                        alignment = float(np.dot(p_given_v, next_dir_probs))
                        consistency += alignment
                        n_neighbors += 1

                if n_neighbors > 0:
                    consistency /= n_neighbors

                weights[v] = max(consistency, 1e-10)

            # 归一化
            total_w = sum(weights.values())
            if total_w > 0:
                for v in weights:
                    weights[v] /= total_w
            sample_weights.append(weights)
        all_weights_c.append(sample_weights)

    return all_weights_c


def signal_d_a3_diff(probs, data, test_indices, max_train_idx):
    """信号源 D：A3 差分模式条件权重

    对每个位置，匹配当前差分序列的 A3 模式，用 prediction 方向
    对号码权重进行调整：预测方向一致的号码获得更高权重。
    """
    n_samples, n_pos, _ = probs.shape
    red_range = data.red_range
    lt = data.lottery_type
    rules_dir = RESULTS_DIR / "rules_strict"

    # 加载 A3 规则
    a3_rules = load_a3_rules(rules_dir, lt)
    if not a3_rules:
        log(f"  [信号D] A3 规则为空，跳过")
        return None

    # 筛选高质量规则
    a3_rules = [r for r in a3_rules if r["chi2"] >= 10.83 and r.get("prediction_confidence", 0) >= 0.3]
    if not a3_rules:
        log(f"  [信号D] 筛选后 A3 规则为空，跳过")
        return None

    a3_grouped = select_a3_top_patterns(a3_rules, n_pos, top_n=20)
    log(f"  [信号D] A3 规则: {sum(len(v) for v in a3_grouped.values())} 条 (分 {len(a3_grouped)} 组)")

    # 预计算离散化序列
    a3_discrete = precompute_a3_discrete(data, max_train_idx)

    # A3 方向预测 → 号码权重调整
    # prediction 是文本（大升/大降/小升/小降/平），表示下一期差分的方向
    # 大升: diff > p80, 小升: p60 < diff <= p80, 平: p40 < diff <= p60
    # 小降: p20 < diff <= p40, 大降: diff <= p20

    # 预计算每个位置的差分阈值（用于将 prediction 映射到号码范围）
    diff_thresholds = {}
    for pos in range(n_pos):
        diff = data.get_diff_series(pos, 1)
        fit_data = diff[:max(10, max_train_idx - 1)]
        diff_thresholds[pos] = np.percentile(fit_data, [20, 40, 60, 80])

    all_weights_d = []
    for i in range(n_samples):
        t = int(test_indices[i])
        sample_weights = []

        for pos in range(n_pos):
            weights = {}
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2)

            # 收集所有匹配模式的预测
            predictions = []  # [(prediction_label, confidence)]
            for (p_pos, p_order), pats in a3_grouped.items():
                if p_pos != pos:
                    continue
                disc_key = (pos, p_order)
                if disc_key not in a3_discrete:
                    continue
                disc_series, _ = a3_discrete[disc_key]
                for pat in pats:
                    if check_a3_match(pat["pattern_codes"], pat["window"], disc_series, t):
                        predictions.append((pat["prediction"], pat["prediction_confidence"]))

            if predictions:
                # 将预测方向转换为号码权重
                thresholds = diff_thresholds[pos]
                for v in range(1, red_range + 1):
                    diff_val = v - current_val
                    w = 0.0
                    for pred_label, conf in predictions:
                        # 检查 diff_val 是否落在预测方向对应的区间
                        match = False
                        if pred_label == "大降":
                            match = diff_val <= thresholds[0]
                        elif pred_label == "小降":
                            match = thresholds[0] < diff_val <= thresholds[1]
                        elif pred_label == "平":
                            match = thresholds[1] < diff_val <= thresholds[2]
                        elif pred_label == "小升":
                            match = thresholds[2] < diff_val <= thresholds[3]
                        elif pred_label == "大升":
                            match = diff_val > thresholds[3]
                        if match:
                            w += conf
                        else:
                            w += (1.0 - conf) / 4.0  # 非预测方向均分剩余概率
                    weights[v] = max(w, 1e-10)
            else:
                # 无匹配模式，均匀分布
                for v in range(1, red_range + 1):
                    weights[v] = 1.0 / red_range

            # 归一化
            total_w = sum(weights.values())
            if total_w > 0:
                for v in weights:
                    weights[v] /= total_w
            sample_weights.append(weights)
        all_weights_d.append(sample_weights)

    return all_weights_d


def signal_e_a4_stat(probs, data, test_indices, max_train_idx):
    """信号源 E：A4 统计量模式条件权重

    匹配当前统计量序列的 A4 模式，用 prediction 对组合统计量进行约束，
    转换为对各位置号码的权重调整。
    """
    n_samples, n_pos, _ = probs.shape
    red_range = data.red_range
    lt = data.lottery_type
    rules_dir = RESULTS_DIR / "rules_strict"

    # 加载 A4 规则
    a4_rules = load_a4_rules(rules_dir, lt)
    if not a4_rules:
        log(f"  [信号E] A4 规则为空，跳过")
        return None

    # 筛选高质量规则
    a4_rules = [r for r in a4_rules if r["chi2"] >= 10.83 and r.get("prediction_confidence", 0) >= 0.3]
    if not a4_rules:
        log(f"  [信号E] 筛选后 A4 规则为空，跳过")
        return None

    a4_grouped = select_a4_top_patterns(a4_rules, top_n_single=20, top_n_joint=10)
    log(f"  [信号E] A4 规则: {sum(len(v) for v in a4_grouped.values())} 条 (分 {len(a4_grouped)} 组)")

    # 预计算离散化序列
    combo_stats = data.get_combo_stats_series()
    a4_discrete = precompute_a4_discrete(combo_stats, max_train_idx)

    # 预计算各统计量的分箱边界（用于将 prediction 映射回数值范围）
    stat_edges = {}
    single_stats = ['sum', 'span', 'odd_count', 'big_count', 'ac_value', 'consec_groups']
    for sn in single_stats:
        if sn in combo_stats:
            fit_data = combo_stats[sn][:max_train_idx].astype(float)
            edges = np.percentile(fit_data, np.linspace(0, 100, 4))  # n_levels=3
            stat_edges[sn] = np.unique(edges)

    all_weights_e = []
    for i in range(n_samples):
        t = int(test_indices[i])
        sample_weights = []

        # 收集所有匹配的 A4 模式预测
        stat_predictions = {}  # {stat_name: [(prediction_bin, confidence)]}
        for stat_name, pats in a4_grouped.items():
            if stat_name not in a4_discrete:
                continue
            disc_series = a4_discrete[stat_name]
            for pat in pats:
                if check_a4_match(pat["pattern_values"], pat["window"], disc_series, t):
                    if stat_name not in stat_predictions:
                        stat_predictions[stat_name] = []
                    stat_predictions[stat_name].append(
                        (pat["prediction"], pat["prediction_confidence"]))

        for pos in range(n_pos):
            weights = {}
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2)

            if stat_predictions:
                # 对每个候选号码，计算其对组合统计量的贡献是否与预测一致
                for v in range(1, red_range + 1):
                    w = 1.0
                    # sum 约束：v 对 sum 的贡献
                    if 'sum' in stat_predictions and 'sum' in stat_edges:
                        edges = stat_edges['sum']
                        # 粗略估计：v 替换当前位置后 sum 的变化方向
                        delta = v - current_val
                        for pred_bin, conf in stat_predictions['sum']:
                            # pred_bin 越大表示 sum 越大
                            if (pred_bin >= len(edges) - 1 and delta > 0) or \
                               (pred_bin == 0 and delta < 0) or \
                               (0 < pred_bin < len(edges) - 1):
                                w *= (1.0 + conf * 0.5)
                            else:
                                w *= (1.0 - conf * 0.3)

                    # odd_count 约束
                    if 'odd_count' in stat_predictions:
                        is_odd = v % 2 == 1
                        for pred_bin, conf in stat_predictions['odd_count']:
                            # pred_bin 大 → 奇数多
                            if (pred_bin >= 2 and is_odd) or (pred_bin == 0 and not is_odd):
                                w *= (1.0 + conf * 0.3)

                    # big_count 约束
                    if 'big_count' in stat_predictions:
                        mid = (1 + red_range) / 2
                        is_big = v > mid
                        for pred_bin, conf in stat_predictions['big_count']:
                            if (pred_bin >= 2 and is_big) or (pred_bin == 0 and not is_big):
                                w *= (1.0 + conf * 0.3)

                    weights[v] = max(w, 1e-10)
            else:
                for v in range(1, red_range + 1):
                    weights[v] = 1.0 / red_range

            # 归一化
            total_w = sum(weights.values())
            if total_w > 0:
                for v in weights:
                    weights[v] /= total_w
            sample_weights.append(weights)
        all_weights_e.append(sample_weights)

    return all_weights_e


def _load_e1_rules_by_position(lottery_type):
    """加载 E1 规律库并按位置分组

    返回 {pos_int: [rule, ...]}，每条 rule 包含原始字段。
    """
    lib_dir = RESULTS_DIR / "e1_search" / "library"
    lt_map = {"shuangseqiu": "shuangseqiu", "daletou": "daletou"}
    filename = f"{lt_map.get(lottery_type, lottery_type)}_rules.json"
    path = lib_dir / filename
    if not path.exists():
        log(f"  [信号F] E1 规律库不存在: {path}")
        return {}

    raw = load_json(str(path))
    rules = raw.get("rules", []) if isinstance(raw, dict) else raw

    by_pos = {}
    for rule in rules:
        rid = rule.get("rule_id", "")
        # rule_id 格式: dlt_P4_0000 或 ssq_P2_0001
        parts = rid.split("_")
        pos_str = None
        for p in parts:
            if p.startswith("P") and len(p) == 2 and p[1].isdigit():
                pos_str = p
                break
        if pos_str is None:
            continue
        pos = int(pos_str[1])
        if pos not in by_pos:
            by_pos[pos] = []
        by_pos[pos].append(rule)

    return by_pos


def _rebuild_e1_condition(rule, data, pos, t, edges_cache):
    """根据 E1 规则的 dimension_detail 和 condition，重建当前期的条件编码值

    返回重建的条件元组（与 rule['condition'] 比较），匹配则返回 True。
    """
    from research.experiment.e1_search_engine import (
        encode_direction_seq, encode_amplitude_bin, encode_value_zone,
        encode_diff_sign_amp, encode_diff_change_pattern,
        encode_miss_zone, compute_miss_periods,
        encode_cross_pos_direction, encode_combo_stat_zone,
        encode_odd_big_pattern, encode_long_trend,
        encode_consec_status, encode_hot_cold,
    )

    dim_detail = rule.get("dimension_detail", "")
    cond_str = rule.get("condition", "")

    # 获取预计算边界
    edges = edges_cache.get(pos)
    if edges is None:
        return False

    series = data.position_series[pos]
    diff_series = edges["diff_series"]
    dir_series = data.direction_series[pos]
    combo_stats = edges["combo_stats"]

    # 解析 dimension_detail 中的各维度
    # 格式如: "D1_n1_x_D3_x_D7", "D7_x_D3", "D3", "D8_sum"
    dim_parts = dim_detail.split("_x_")

    encoded_parts = []
    for dp in dim_parts:
        dp = dp.strip()
        if dp.startswith("D1_n"):
            n = int(dp[4:])
            # D1 编码基于 direction_series，索引 t-1（因为 diff 对齐）
            val = encode_direction_seq(dir_series, t - 1, n)
        elif dp.startswith("D2_n"):
            n = int(dp[4:])
            val = encode_amplitude_bin(diff_series, t - 1, n, edges["amp_bin_edges"])
        elif dp == "D3":
            val = encode_value_zone(series[t], edges["zone_edges"])
        elif dp == "D4":
            if t < 2:
                val = None
            else:
                val = encode_diff_sign_amp(diff_series[t - 2], edges["amp_5_edges"])
        elif dp == "D5":
            val = encode_diff_change_pattern(diff_series, t - 1)
        elif dp == "D6":
            miss = compute_miss_periods(series, t, series[t])
            val = encode_miss_zone(miss)
        elif dp == "D7":
            val = encode_cross_pos_direction(data.direction_series, pos, t - 1)
        elif dp.startswith("D8_"):
            stat_name = dp[3:]
            stat_edges = edges["combo_edges"].get(stat_name)
            if stat_edges is not None:
                val = encode_combo_stat_zone(combo_stats, stat_name, t, stat_edges)
            else:
                val = None
        elif dp == "D8":
            # 无后缀的 D8 不应出现，跳过
            val = None
        elif dp == "D9":
            val = encode_odd_big_pattern(data.red_matrix, t, data.red_count, data.red_range)
        elif dp.startswith("D10_w"):
            window = int(dp[5:])
            val = encode_long_trend(series, t, window)
        elif dp == "D11":
            val = encode_consec_status(data.red_matrix, t, pos)
        elif dp == "D12":
            val = encode_hot_cold(series, t, series[t], window=10)
        else:
            val = None

        if val is None:
            return False
        encoded_parts.append(val)

    # 构造条件元组
    if len(encoded_parts) == 1:
        rebuilt = str(encoded_parts[0])
    else:
        rebuilt = str(tuple(encoded_parts))

    # 比较（需要处理 numpy 类型差异，如 np.int64(2) vs 2）
    # 标准化比较：去掉 np.int64/np.float64 包装
    import re
    norm_cond = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', cond_str)
    norm_rebuilt = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', rebuilt)

    return norm_rebuilt == norm_cond


def _precompute_e1_edges(data, pos, max_train_idx):
    """为指定位置预计算 E1 编码所需的边界值"""
    series = data.position_series[pos]
    diff_series = np.diff(series.astype(np.float64))
    train_abs = np.abs(diff_series[:max(1, max_train_idx - 1)])

    combo_stats = data.get_combo_stats_series()
    combo_edges = {}
    for stat_name, arr in combo_stats.items():
        train_arr = arr[:max_train_idx]
        combo_edges[stat_name] = np.percentile(train_arr, np.linspace(0, 100, 6))

    return {
        "diff_series": diff_series,
        "amp_bin_edges": np.percentile(train_abs, np.linspace(0, 100, 11)),
        "amp_5_edges": np.percentile(train_abs, np.linspace(0, 100, 6)),
        "zone_edges": np.linspace(1, data.red_range, 6),
        "combo_edges": combo_edges,
        "combo_stats": combo_stats,
    }


def signal_f_e1_amplitude(probs, data, test_indices, max_train_idx):
    """信号源 F：E1 规律库幅度分布条件权重

    对每个测试样本的每个位置：
    1. 用 E1 编码函数重建当前期的条件
    2. 匹配 E1 规律库中的规则
    3. 用匹配规则的 amplitude_distribution（高斯拟合）生成号码权重
       - 幅度 = |v - current_val|
       - 权重 ∝ Σ quality_i × N(amplitude; mean_i, iqr_i/1.35)
    4. 无匹配规则时回退到均匀分布
    """
    n_samples, n_pos, _ = probs.shape
    red_range = data.red_range
    lt = data.lottery_type

    # 加载规律库
    rules_by_pos = _load_e1_rules_by_position(lt)
    if not rules_by_pos:
        log(f"  [信号F] 无可用 E1 规则，跳过")
        return None

    total_rules = sum(len(v) for v in rules_by_pos.values())
    log(f"  [信号F] 加载 E1 规律库: {total_rules} 条规则, "
        f"覆盖 {len(rules_by_pos)} 个位置")

    # 预计算各位置的编码边界
    edges_cache = {}
    for pos in range(n_pos):
        edges_cache[pos] = _precompute_e1_edges(data, pos, max_train_idx)

    # 对每个样本每个位置计算权重
    all_weights_f = []
    match_counts = []

    for i in range(n_samples):
        t = int(test_indices[i])
        sample_weights = []

        for pos in range(n_pos):
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2)
            pos_rules = rules_by_pos.get(pos, [])

            # 匹配规则
            matched = []
            for rule in pos_rules:
                try:
                    if _rebuild_e1_condition(rule, data, pos, t, edges_cache):
                        matched.append(rule)
                except Exception:
                    continue

            match_counts.append(len(matched))

            if matched:
                # 用匹配规则的幅度分布生成权重
                weights = {}
                for v in range(1, red_range + 1):
                    amplitude = abs(v - current_val)
                    w = 0.0
                    for rule in matched:
                        amp_dist = rule.get("amplitude_distribution", {})
                        mean_amp = amp_dist.get("mean", red_range / 2)
                        iqr = amp_dist.get("iqr", red_range / 4)
                        quality = rule.get("quality_score", 0.5)
                        # 用 IQR 估计标准差: σ ≈ IQR / 1.35
                        sigma = max(iqr / 1.35, 1.0)
                        # 高斯权重
                        z = (amplitude - mean_amp) / sigma
                        gauss_w = np.exp(-0.5 * z * z)
                        w += quality * gauss_w
                    weights[v] = max(w, 1e-10)
            else:
                # 无匹配，均匀分布
                weights = {v: 1.0 / red_range for v in range(1, red_range + 1)}

            # 归一化
            total_w = sum(weights.values())
            if total_w > 0:
                for v in weights:
                    weights[v] /= total_w
            sample_weights.append(weights)
        all_weights_f.append(sample_weights)

    avg_matches = float(np.mean(match_counts)) if match_counts else 0
    log(f"  [信号F] 平均每位置匹配规则数: {avg_matches:.1f}")

    return all_weights_f


def fuse_signals(weights_list, alphas, candidates_per_sample=None):
    """融合多个信号源的权重

    weights_list: [weights_A, weights_B, weights_C, ...]
                  每个 weights_X 是 list[list[dict]]  [sample][pos] = {v: w}
    alphas: [alpha_A, alpha_B, alpha_C, ...]  融合权重
    candidates_per_sample: 可选，list[list[set]]  硬排除后的候选集

    返回 fused_weights: list[list[dict]]
    """
    # 过滤掉 None 的信号源
    valid = [(w, a) for w, a in zip(weights_list, alphas) if w is not None]
    if not valid:
        return weights_list[0]  # 回退到第一个

    n_samples = len(valid[0][0])
    n_pos = len(valid[0][0][0])
    fused = []

    for i in range(n_samples):
        sample_fused = []
        for pos in range(n_pos):
            weights = {}
            # 收集所有号码
            all_nums = set()
            for w, a in valid:
                all_nums.update(w[i][pos].keys())

            # 如果有硬排除候选集，只保留候选号码
            if candidates_per_sample is not None:
                all_nums &= candidates_per_sample[i][pos]

            for v in all_nums:
                fused_w = 0.0
                for w, a in valid:
                    fused_w += a * w[i][pos].get(v, 0.0)
                weights[v] = fused_w

            # 归一化
            total_w = sum(weights.values())
            if total_w > 0:
                for v in weights:
                    weights[v] /= total_w
            elif all_nums:
                # 均匀分布
                uniform = 1.0 / len(all_nums)
                for v in all_nums:
                    weights[v] = uniform

            sample_fused.append(weights)
        fused.append(sample_fused)

    return fused


def compute_auc_and_survival(all_weights, data, test_indices, top_k_per_pos=None):
    """计算 AUC 和 top-K 存活率"""
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
            else:
                ranks[pos].append(len(sorted_nums) + 1)

    aucs = {}
    all_auc_vals = []
    for pos in range(n_pos):
        r_arr = np.array(ranks[pos], dtype=float)
        n_candidates = red_range  # 最大可能排名
        auc = float(1.0 - (r_arr.mean() - 1) / (n_candidates - 1)) if n_candidates > 1 else 0.5
        aucs[f"P{pos}"] = {"auc": auc, "mean_rank": float(r_arr.mean())}
        all_auc_vals.append(auc)

    aucs["overall_auc"] = float(np.mean(all_auc_vals))

    # top-K 存活率
    if top_k_per_pos is not None:
        survivals = []
        combo_counts = []
        baseline_combos = comb(red_range, n_pos)

        for i in range(n_samples):
            t = int(test_indices[i])
            true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
            cand_sets = []
            for pos in range(n_pos):
                sorted_nums = sorted(all_weights[i][pos].items(),
                                     key=lambda x: x[1], reverse=True)
                k = top_k_per_pos[pos] if isinstance(top_k_per_pos, list) else top_k_per_pos
                top_set = set(v for v, w in sorted_nums[:k])
                cand_sets.append(top_set)

            n_combos = count_ordered_combos(cand_sets)
            combo_counts.append(n_combos)
            survived = check_combo_survival(cand_sets, true_vals)
            survivals.append(1.0 if survived else 0.0)

        cc = np.array(combo_counts, dtype=float)
        sv = np.array(survivals)
        aucs["combo_stats"] = {
            "avg_combo_count": float(np.mean(cc)),
            "combo_reduction": float(1.0 - np.mean(cc) / baseline_combos),
            "combo_survival": float(np.mean(sv)),
            "combo_efficiency": float(
                (1.0 - np.mean(cc) / baseline_combos) * np.mean(sv)),
        }

    return aucs


def e3b_signal_fusion(lottery_type, e3a_results=None):
    """E3b: 多信号融合验证（全量信号 A/B/C/D/E）"""
    log(f"\n{'═'*55}")
    log(f"  E3b 多信号融合验证: {lottery_type}")
    log(f"{'═'*55}")

    probs, test_indices, data, best_method = get_best_probs(lottery_type)
    n_samples, n_pos = probs.shape[0], probs.shape[1]
    red_range = data.red_range
    max_train_idx = int(test_indices[0])

    # 预计算
    with Timer("预计算差分分布"):
        diff_dist = compute_diff_distributions(data, max_train_idx)

    # 条件概率矩阵
    with Timer("构建条件概率矩阵"):
        max_train_dir_idx = max_train_idx - 1
        cond_matrices = build_cond_prob_matrix(data, 0, max_train_dir_idx)

    # === 信号源 A：基线权重 ===
    with Timer("信号源 A（基线权重）"):
        weights_a = signal_a_baseline(probs, data, test_indices, diff_dist)
    log(f"  信号 A: {len(weights_a)} 样本")

    # === 信号源 B：规则簇条件权重 ===
    with Timer("信号源 B（规则簇条件权重）"):
        weights_b = signal_b_rule_cluster(probs, data, test_indices, diff_dist, max_train_idx)
    if weights_b:
        log(f"  信号 B: {len(weights_b)} 样本")
    else:
        log(f"  信号 B: 不可用")

    # === 信号源 C：跨位置一致性 ===
    with Timer("信号源 C（跨位置一致性）"):
        weights_c = signal_c_cross_position(probs, data, test_indices, cond_matrices)
    if weights_c:
        log(f"  信号 C: {len(weights_c)} 样本")
    else:
        log(f"  信号 C: 不可用")

    # === 信号源 D：A3 差分模式 ===
    with Timer("信号源 D（A3 差分模式）"):
        weights_d = signal_d_a3_diff(probs, data, test_indices, max_train_idx)
    if weights_d:
        log(f"  信号 D: {len(weights_d)} 样本")
    else:
        log(f"  信号 D: 不可用")

    # === 信号源 E：A4 统计量模式 ===
    with Timer("信号源 E（A4 统计量模式）"):
        weights_e = signal_e_a4_stat(probs, data, test_indices, max_train_idx)
    if weights_e:
        log(f"  信号 E: {len(weights_e)} 样本")
    else:
        log(f"  信号 E: 不可用")

    # === 信号源 F：E1 规律库幅度分布 ===
    with Timer("信号源 F（E1 规律库幅度分布）"):
        weights_f = signal_f_e1_amplitude(probs, data, test_indices, max_train_idx)
    if weights_f:
        log(f"  信号 F: {len(weights_f)} 样本")
    else:
        log(f"  信号 F: 不可用")

    # === 网格搜索最优融合权重 ===
    log("\n  --- 网格搜索融合权重 ---")

    # 基线 AUC
    base_aucs = compute_auc_and_survival(weights_a, data, test_indices)
    base_auc = base_aucs["overall_auc"]
    log(f"  基线 AUC (A only): {base_auc:.4f}")

    # 收集所有可用信号
    all_signals = [("A", weights_a)]
    if weights_b is not None:
        all_signals.append(("B", weights_b))
    if weights_c is not None:
        all_signals.append(("C", weights_c))
    if weights_d is not None:
        all_signals.append(("D", weights_d))
    if weights_e is not None:
        all_signals.append(("E", weights_e))
    if weights_f is not None:
        all_signals.append(("F", weights_f))

    n_signals = len(all_signals)
    signal_names = [s[0] for s in all_signals]
    signal_weights_list = [s[1] for s in all_signals]
    log(f"  可用信号: {signal_names}")

    # 两阶段搜索：
    # 阶段1：粗搜索（步长 0.2）
    # 阶段2：在最优附近细搜索（步长 0.05）
    grid_results = []
    best_config = {"alphas": {s: 0.0 for s in signal_names}, "auc": base_auc}
    best_config["alphas"]["A"] = 1.0

    def _generate_alpha_grid(n, step=0.2, min_a=0.2):
        """生成 n 个信号的 alpha 网格（和为 1，A >= min_a）"""
        import itertools
        levels = [round(x, 2) for x in np.arange(0.0, 1.01, step)]
        for combo in itertools.product(levels, repeat=n):
            if abs(sum(combo) - 1.0) > 0.01:
                continue
            if combo[0] < min_a - 0.01:  # A 的最小权重
                continue
            yield tuple(round(c, 2) for c in combo)

    # 阶段1：粗搜索
    for alpha_tuple in _generate_alpha_grid(n_signals, step=0.2, min_a=0.2):
        alphas_list = list(alpha_tuple)
        # 过滤掉权重为 0 的信号
        active_signals = []
        active_alphas = []
        for j, a in enumerate(alphas_list):
            if a > 0.001:
                active_signals.append(signal_weights_list[j])
                active_alphas.append(a)
        if not active_signals:
            continue
        fused = fuse_signals(active_signals, active_alphas)
        aucs = compute_auc_and_survival(fused, data, test_indices)
        auc = aucs["overall_auc"]
        alpha_dict = {signal_names[j]: alphas_list[j] for j in range(n_signals)}
        grid_results.append({"alphas": alpha_dict, "auc": auc})
        if auc > best_config["auc"]:
            best_config = {"alphas": alpha_dict, "auc": auc}

    log(f"  阶段1 粗搜索: {len(grid_results)} 种配置, 最优 AUC={best_config['auc']:.4f}")

    # 阶段2：在最优附近细搜索（±0.1，步长 0.05）
    if n_signals >= 2:
        base_alphas = best_config["alphas"]
        fine_results = []
        fine_step = 0.05
        fine_range = 0.15

        def _fine_grid(base, names, step, rng):
            """在 base 附近生成细搜索网格"""
            import itertools
            per_signal = {}
            for name in names:
                center = base[name]
                lo = max(0.0, center - rng)
                hi = min(1.0, center + rng)
                per_signal[name] = [round(x, 2) for x in np.arange(lo, hi + step / 2, step)]
            for combo in itertools.product(*[per_signal[n] for n in names]):
                if abs(sum(combo) - 1.0) > 0.01:
                    continue
                if combo[0] < 0.15:  # A 最小权重
                    continue
                yield {names[i]: round(combo[i], 2) for i in range(len(names))}

        for alpha_dict in _fine_grid(base_alphas, signal_names, fine_step, fine_range):
            active_signals = []
            active_alphas = []
            for j, name in enumerate(signal_names):
                a = alpha_dict[name]
                if a > 0.001:
                    active_signals.append(signal_weights_list[j])
                    active_alphas.append(a)
            if not active_signals:
                continue
            fused = fuse_signals(active_signals, active_alphas)
            aucs = compute_auc_and_survival(fused, data, test_indices)
            auc = aucs["overall_auc"]
            fine_results.append({"alphas": alpha_dict, "auc": auc})
            if auc > best_config["auc"]:
                best_config = {"alphas": alpha_dict, "auc": auc}

        grid_results.extend(fine_results)
        log(f"  阶段2 细搜索: {len(fine_results)} 种配置, 最优 AUC={best_config['auc']:.4f}")

    log(f"  总搜索: {len(grid_results)} 种配置")
    log(f"  最优配置: {best_config['alphas']}, AUC={best_config['auc']:.4f}")
    log(f"  AUC 提升: {best_config['auc'] - base_auc:+.4f}")

    # === 用最优配置计算完整指标 ===
    best_alphas = best_config["alphas"]
    active_signals = []
    active_alphas = []
    for j, name in enumerate(signal_names):
        a = best_alphas.get(name, 0.0)
        if a > 0.001:
            active_signals.append(signal_weights_list[j])
            active_alphas.append(a)

    fused_best = fuse_signals(active_signals, active_alphas)

    # 找每位置的 K95
    log("\n  --- 最优配置的 K95 评估 ---")
    optimal_ks = []
    for pos in range(n_pos):
        survival_by_k = np.zeros(red_range + 1)
        for i in range(n_samples):
            t = int(test_indices[i])
            true_val = int(data.red_matrix[t, pos])
            sorted_nums = sorted(fused_best[i][pos].items(),
                                 key=lambda x: x[1], reverse=True)
            for r, (v, w) in enumerate(sorted_nums, 1):
                if v == true_val:
                    for k in range(r, red_range + 1):
                        survival_by_k[k] += 1
                    break
        surv_rates = survival_by_k / n_samples
        best_k = red_range
        for k in range(1, red_range + 1):
            if surv_rates[k] >= 0.95:
                best_k = k
                break
        optimal_ks.append(best_k)

    log(f"  K95: {optimal_ks}")
    full_aucs = compute_auc_and_survival(fused_best, data, test_indices, top_k_per_pos=optimal_ks)
    log(f"  整体 AUC: {full_aucs['overall_auc']:.4f}")
    if "combo_stats" in full_aucs:
        cs = full_aucs["combo_stats"]
        log(f"  组合缩减率: {cs['combo_reduction']:.2%}")
        log(f"  组合存活率: {cs['combo_survival']:.2%}")
        log(f"  组合效率: {cs['combo_efficiency']:.4f}")

    # === 各信号边际贡献分析 ===
    log("\n  --- 各信号边际贡献 ---")
    marginal = {}
    for j, name in enumerate(signal_names):
        if name == "A":
            continue
        # 去掉该信号后的 AUC
        ablation_signals = []
        ablation_alphas = []
        for k, n2 in enumerate(signal_names):
            if n2 == name:
                continue
            a = best_alphas.get(n2, 0.0)
            if a > 0.001:
                ablation_signals.append(signal_weights_list[k])
                ablation_alphas.append(a)
        if ablation_signals:
            # 重新归一化
            total_a = sum(ablation_alphas)
            if total_a > 0:
                ablation_alphas = [a / total_a for a in ablation_alphas]
            fused_abl = fuse_signals(ablation_signals, ablation_alphas)
            abl_aucs = compute_auc_and_survival(fused_abl, data, test_indices)
            abl_auc = abl_aucs["overall_auc"]
            delta = best_config["auc"] - abl_auc
            marginal[name] = {"auc_without": abl_auc, "marginal_delta": delta}
            log(f"  去掉 {name}: AUC={abl_auc:.4f}, 边际贡献={delta:+.4f}")

    results = {
        "lottery_type": lottery_type,
        "signals_available": {name: True for name in signal_names},
        "baseline_auc": base_auc,
        "best_config": best_config,
        "auc_improvement": best_config["auc"] - base_auc,
        "optimal_ks": optimal_ks,
        "full_evaluation": full_aucs,
        "marginal_contribution": marginal,
        "grid_search": {
            "n_configs": len(grid_results),
            "top5": sorted(grid_results, key=lambda x: x["auc"], reverse=True)[:5],
        },
    }

    save_json(results, STRICT_EXPERIMENT_DIR / f"e3b_signal_fusion_{lottery_type}.json")
    return results, fused_best, optimal_ks


# ============================================================
#  E3c: Layer 2 组合构造验证（数据驱动的规则评分）
# ============================================================

class RuleScorer:
    """数据驱动的组合评分器

    从 phase1 挖掘的 A2 规则簇和 A1 方向模式中提取约束，
    用规则触发状态对候选组合评分。所有约束都来自算力挖掘，
    不包含任何人工拍脑袋的特征。
    """

    def __init__(self, data, max_train_idx):
        self.data = data
        self.red_range = data.red_range
        self.n_pos = data.red_count
        self.max_train_idx = max_train_idx

        # 预计算差分序列和组合统计量（用于规则条件检查）
        self.diff_series = {}
        for pos in range(self.n_pos):
            self.diff_series[pos] = data.get_diff_series(pos, 1)
        self.combo_stats = data.get_combo_stats_series()

        # 加载规则
        self.combo_rules = []      # 包含 combo_* 条件的规则
        self.position_rules = []   # 纯位置条件的规则（target 含方向预测）
        self.a1_patterns = []      # A1 方向模式
        self.a3_grouped = {}       # A3 差分模式（按 pos×order 分组）
        self.a4_grouped = {}       # A4 统计量模式（按 stat_name 分组）
        self.a3_discrete = {}      # A3 离散化序列
        self.a4_discrete = {}      # A4 离散化序列
        self.diff_thresholds = {}  # 各位置差分阈值（A3 用）
        self._load_rules()

    def _load_rules(self):
        """加载并筛选规则簇"""
        lt = self.data.lottery_type
        rules_dir = RESULTS_DIR / "rules_strict"

        # 加载 A2 规则簇
        cluster_path = rules_dir / f"a2_clusters_{lt}.json"
        if not cluster_path.exists():
            # 尝试加载原始排除规则（取 top 规则）
            excl_path = rules_dir / f"a2_exclusion_rules_{lt}.json"
            if excl_path.exists():
                log(f"  [RuleScorer] 加载原始 A2 规则: {excl_path.name}")
                raw = load_json(str(excl_path))
                # 兼容 dict 格式（含 rules 字段）和 list 格式
                if isinstance(raw, dict) and 'rules' in raw:
                    all_rules = raw['rules']
                elif isinstance(raw, list):
                    all_rules = raw
                else:
                    all_rules = []
                if all_rules:
                    # 按 lift * confidence 排序，取 top 规则
                    all_rules.sort(
                        key=lambda r: r.get('lift', 0) * r.get('confidence', 0),
                        reverse=True)
                    all_rules = all_rules[:500]
                    self._classify_rules(all_rules)
            else:
                log(f"  [RuleScorer] 无 A2 规则文件")
        else:
            log(f"  [RuleScorer] 加载 A2 规则簇: {cluster_path.name}")
            clusters = load_json(str(cluster_path))
            rules = [c['representative'] for c in clusters.values()]
            self._classify_rules(rules)

        # 加载 A1 方向模式
        a1_path = rules_dir / f"a1_direction_patterns_{lt}.json"
        if a1_path.exists():
            raw = load_json(str(a1_path))
            # 兼容 dict 格式（含 patterns 字段）和 list 格式
            if isinstance(raw, dict) and 'patterns' in raw:
                patterns = raw['patterns']
            elif isinstance(raw, list):
                patterns = raw
            else:
                patterns = []
            if isinstance(patterns, list):
                # 筛选高置信度模式
                self.a1_patterns = [
                    p for p in patterns
                    if p.get('prediction_confidence', 0) >= 0.4
                       and p.get('observed', 0) >= 3
                ]
                log(f"  [RuleScorer] A1 方向模式: {len(self.a1_patterns)} 条")
        else:
            log(f"  [RuleScorer] 无 A1 方向模式文件")

        log(f"  [RuleScorer] 组合级规则: {len(self.combo_rules)} 条, "
            f"位置级规则: {len(self.position_rules)} 条")

        # 加载 A3 差分模式
        a3_rules = load_a3_rules(rules_dir, lt)
        a3_rules = [r for r in a3_rules
                     if r["chi2"] >= 10.83 and r.get("prediction_confidence", 0) >= 0.3]
        if a3_rules:
            self.a3_grouped = select_a3_top_patterns(a3_rules, self.n_pos, top_n=20)
            self.a3_discrete = precompute_a3_discrete(self.data, self.max_train_idx)
            # 预计算差分阈值
            for pos in range(self.n_pos):
                diff = self.data.get_diff_series(pos, 1)
                fit_data = diff[:max(10, self.max_train_idx - 1)]
                self.diff_thresholds[pos] = np.percentile(fit_data, [20, 40, 60, 80])
            n_a3 = sum(len(v) for v in self.a3_grouped.values())
            log(f"  [RuleScorer] A3 差分模式: {n_a3} 条 ({len(self.a3_grouped)} 组)")
        else:
            log(f"  [RuleScorer] 无可用 A3 规则")

        # 加载 A4 统计量模式
        a4_rules = load_a4_rules(rules_dir, lt)
        a4_rules = [r for r in a4_rules
                     if r["chi2"] >= 10.83 and r.get("prediction_confidence", 0) >= 0.3]
        if a4_rules:
            self.a4_grouped = select_a4_top_patterns(a4_rules, top_n_single=20, top_n_joint=10)
            self.a4_discrete = precompute_a4_discrete(self.combo_stats, self.max_train_idx)
            n_a4 = sum(len(v) for v in self.a4_grouped.values())
            log(f"  [RuleScorer] A4 统计量模式: {n_a4} 条 ({len(self.a4_grouped)} 组)")
        else:
            log(f"  [RuleScorer] 无可用 A4 规则")

    def _classify_rules(self, rules):
        """将规则分为组合级和位置级"""
        for rule in rules:
            conditions = rule.get('conditions', [])
            has_combo = any(c.startswith('combo_') for c in conditions)
            if has_combo:
                self.combo_rules.append(rule)
            else:
                self.position_rules.append(rule)

    def prefilter_for_period(self, t):
        """每期调用一次，预过滤 A1/A3/A4 模式（缓存匹配结果）

        A1 模式匹配只依赖历史方向序列（与候选组合无关），
        A3/A4 模式匹配只依赖历史差分/统计量序列，
        所以可以每期只做一次，避免在 beam search 内重复检查。
        """
        self._cached_t = t
        self._cached_a1_matched = []
        self._cached_a3_predictions = {}  # {pos: [(prediction_label, confidence)]}
        self._cached_a4_predictions = {}  # {stat_name: [(prediction_bin, confidence)]}

        if self.a1_patterns and t >= 1:
            for pattern in self.a1_patterns:
                try:
                    if check_direction_pattern(pattern, self.data.direction_series, t - 1):
                        self._cached_a1_matched.append(pattern)
                except (IndexError, KeyError):
                    continue

        # A3 预匹配
        if self.a3_grouped:
            for (p_pos, p_order), pats in self.a3_grouped.items():
                disc_key = (p_pos, p_order)
                if disc_key not in self.a3_discrete:
                    continue
                disc_series, _ = self.a3_discrete[disc_key]
                for pat in pats:
                    try:
                        if check_a3_match(pat["pattern_codes"], pat["window"], disc_series, t):
                            if p_pos not in self._cached_a3_predictions:
                                self._cached_a3_predictions[p_pos] = []
                            self._cached_a3_predictions[p_pos].append(
                                (pat["prediction"], pat["prediction_confidence"]))
                    except (IndexError, KeyError):
                        continue

        # A4 预匹配
        if self.a4_grouped:
            for stat_name, pats in self.a4_grouped.items():
                if stat_name not in self.a4_discrete:
                    continue
                disc_series = self.a4_discrete[stat_name]
                for pat in pats:
                    try:
                        if check_a4_match(pat["pattern_values"], pat["window"], disc_series, t):
                            if stat_name not in self._cached_a4_predictions:
                                self._cached_a4_predictions[stat_name] = []
                            self._cached_a4_predictions[stat_name].append(
                                (pat["prediction"], pat["prediction_confidence"]))
                    except (IndexError, KeyError):
                        continue

    def _compute_combo_stats_for_candidate(self, combo):
        """对候选组合计算统计量（与 data_loader 中的逻辑一致）"""
        n = len(combo)
        s = sum(combo)
        span = combo[-1] - combo[0]
        odd_count = sum(1 for v in combo if v % 2 == 1)
        mid = (1 + self.red_range) / 2
        big_count = sum(1 for v in combo if v > mid)

        # AC 值
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(abs(combo[i] - combo[j]))
        ac_value = len(diffs) - (n - 1)

        # 连号组数
        groups = 0
        in_group = False
        for k in range(1, n):
            if combo[k] - combo[k - 1] == 1:
                if not in_group:
                    groups += 1
                    in_group = True
            else:
                in_group = False

        return {
            'sum': s,
            'span': span,
            'odd_count': odd_count,
            'big_count': big_count,
            'ac_value': ac_value,
            'consec_groups': groups,
        }

    def _check_combo_condition(self, cond, cand_stats):
        """检查单个 combo_* 条件是否对候选组合成立"""
        parts = cond.split('_')
        if parts[0] != 'combo':
            return True  # 非组合条件，跳过
        lo, hi = int(parts[-2]), int(parts[-1])
        stat_name = '_'.join(parts[1:-2])
        val = cand_stats.get(stat_name)
        if val is None:
            return False
        return lo <= val <= hi

    def _check_position_condition(self, cond, combo, t):
        """检查单个位置级条件是否对候选组合+当期局面成立"""
        parts = cond.split('_')
        if not (parts[0].startswith('P') and len(parts[0]) == 2):
            return True
        pos = int(parts[0][1])
        if pos >= len(combo):
            return False
        val = combo[pos]

        if parts[1] == 'val':
            lo, hi = int(parts[2]), int(parts[3])
            return lo <= val <= hi
        elif parts[1] == 'dir':
            # 方向需要对比前一期
            if t < 1:
                return False
            prev_val = int(self.data.red_matrix[t - 1, pos])
            dir_map = {'U': 1, 'D': -1, 'E': 0}
            target_dir = dir_map.get(parts[2], 0)
            actual_diff = val - prev_val
            if target_dir == 1:
                return actual_diff > 0
            elif target_dir == -1:
                return actual_diff < 0
            else:
                return actual_diff == 0
        elif parts[1] == 'odd':
            return val % 2 == 1
        elif parts[1] == 'even':
            return val % 2 == 0
        elif parts[1] == 'big':
            return val > self.red_range / 2
        elif parts[1] == 'small':
            return val <= self.red_range / 2
        elif parts[1] == 'diff':
            if t < 1:
                return False
            prev_val = int(self.data.red_matrix[t - 1, pos])
            diff_val = val - prev_val
            lo, hi = int(parts[2]), int(parts[3])
            return lo <= diff_val <= hi
        return True

    def score_combo(self, combo, t):
        """用数据挖掘的规则对候选组合评分

        评分 = Σ (confidence_i × lift_i) / n_rules  对所有触发的规则

        Args:
            combo: list[int]  候选组合（已排序）
            t: int  当前期索引（用于检查位置级条件中的方向/差分）

        Returns:
            float  规则评分（越高越好）
        """
        cand_stats = self._compute_combo_stats_for_candidate(combo)

        total_score = 0.0
        n_triggered = 0

        # 检查组合级规则
        for rule in self.combo_rules:
            conditions = rule.get('conditions', [])
            all_match = True
            for cond in conditions:
                if cond.startswith('combo_'):
                    if not self._check_combo_condition(cond, cand_stats):
                        all_match = False
                        break
                else:
                    if not self._check_position_condition(cond, combo, t):
                        all_match = False
                        break
            if all_match:
                # 规则触发，用 confidence × lift 作为权重
                conf = rule.get('confidence', 0.5)
                lift = rule.get('lift', 1.0)
                total_score += conf * lift
                n_triggered += 1

        # 检查位置级规则
        for rule in self.position_rules:
            conditions = rule.get('conditions', [])
            all_match = True
            for cond in conditions:
                if not self._check_position_condition(cond, combo, t):
                    all_match = False
                    break
            if all_match:
                # 检查 target 方向是否与候选组合一致
                target = rule.get('target', '')
                target_parts = target.split('_')
                if len(target_parts) >= 2 and target_parts[0].startswith('next'):
                    # 格式: next_P2_U
                    tgt_pos_str = target_parts[1] if len(target_parts) == 3 else target_parts[0]
                    tgt_dir_str = target_parts[-1]
                    # 解析位置
                    if tgt_pos_str.startswith('P') and len(tgt_pos_str) == 2:
                        tgt_pos = int(tgt_pos_str[1])
                        if tgt_pos < len(combo) and t > 0:
                            prev_val = int(self.data.red_matrix[t - 1, tgt_pos])
                            cand_val = combo[tgt_pos]
                            cand_diff = cand_val - prev_val
                            dir_match = (
                                (tgt_dir_str == 'U' and cand_diff > 0) or
                                (tgt_dir_str == 'D' and cand_diff < 0) or
                                (tgt_dir_str == 'E' and cand_diff == 0)
                            )
                            if dir_match:
                                conf = rule.get('confidence', 0.5)
                                lift = rule.get('lift', 1.0)
                                total_score += conf * lift
                                n_triggered += 1

        # 归一化：避免规则数量差异导致的尺度问题
        n_total_rules = len(self.combo_rules) + len(self.position_rules)
        if n_total_rules > 0 and n_triggered > 0:
            # 触发率加权：触发越多规则得分越高
            score = total_score / n_total_rules
        else:
            score = 0.0

        return score

    def get_a1_direction_bonus(self, combo, t):
        """用 A1 方向模式对候选组合的方向一致性打分

        使用 prefilter_for_period(t) 缓存的匹配结果，
        避免重复检查模式匹配。
        """
        if t < 1:
            return 0.0

        # 使用预过滤缓存
        if hasattr(self, '_cached_t') and self._cached_t == t:
            matched_patterns = self._cached_a1_matched
        else:
            # 未预过滤，回退到全量检查
            self.prefilter_for_period(t)
            matched_patterns = self._cached_a1_matched

        if not matched_patterns:
            return 0.0

        bonus = 0.0
        n_matched = 0

        for pattern in matched_patterns:
            n_matched += 1
            next_dist = pattern.get('next_distribution', {})
            if not next_dist:
                continue

            # 找最高概率的方向预测
            best_dir_key = max(next_dist, key=lambda k: next_dist[k].get('prob', 0))
            best_prob = next_dist[best_dir_key].get('prob', 0)

            # 解析方向元组，如 "(1, -1, -1, -1, 1)"
            try:
                dir_tuple = eval(best_dir_key)
            except Exception:
                continue

            # 检查候选组合的方向是否与预测一致
            positions = pattern.get('positions', [])
            match_count = 0
            total_count = 0
            for k, pos in enumerate(positions):
                if pos >= len(combo) or k >= len(dir_tuple):
                    continue
                prev_val = int(self.data.red_matrix[t - 1, pos])
                cand_diff = combo[pos] - prev_val
                predicted_dir = dir_tuple[k]
                total_count += 1
                if ((predicted_dir == 1 and cand_diff > 0) or
                        (predicted_dir == -1 and cand_diff < 0) or
                        (predicted_dir == 0 and cand_diff == 0)):
                    match_count += 1

            if total_count > 0:
                alignment = match_count / total_count
                bonus += alignment * best_prob

        if n_matched > 0:
            bonus /= n_matched

        return bonus

    def get_a3_diff_bonus(self, combo, t):
        """用 A3 差分模式对候选组合的差分方向一致性打分

        使用 prefilter_for_period(t) 缓存的匹配结果。
        对每个位置，检查候选号码的差分是否落在预测方向区间内。
        """
        if t < 1 or not self._cached_a3_predictions:
            return 0.0

        bonus = 0.0
        n_checked = 0

        for pos in range(len(combo)):
            predictions = self._cached_a3_predictions.get(pos, [])
            if not predictions:
                continue

            prev_val = int(self.data.red_matrix[t - 1, pos])
            diff_val = combo[pos] - prev_val
            thresholds = self.diff_thresholds.get(pos)
            if thresholds is None:
                continue

            for pred_label, conf in predictions:
                match = False
                if pred_label == "大降":
                    match = diff_val <= thresholds[0]
                elif pred_label == "小降":
                    match = thresholds[0] < diff_val <= thresholds[1]
                elif pred_label == "平":
                    match = thresholds[1] < diff_val <= thresholds[2]
                elif pred_label == "小升":
                    match = thresholds[2] < diff_val <= thresholds[3]
                elif pred_label == "大升":
                    match = diff_val > thresholds[3]

                if match:
                    bonus += conf
                n_checked += 1

        return bonus / n_checked if n_checked > 0 else 0.0

    def get_a4_stat_bonus(self, combo, t):
        """用 A4 统计量模式对候选组合的统计量一致性打分

        使用 prefilter_for_period(t) 缓存的匹配结果。
        检查候选组合的统计量离散值是否与预测一致。
        """
        if not self._cached_a4_predictions:
            return 0.0

        cand_stats = self._compute_combo_stats_for_candidate(combo)

        bonus = 0.0
        n_checked = 0

        # 预计算候选组合统计量的离散值
        single_stats = ['sum', 'span', 'odd_count', 'big_count', 'ac_value', 'consec_groups']
        cand_discrete = {}
        for sn in single_stats:
            if sn in cand_stats and sn in self.a4_discrete:
                # 用训练期拟合的分箱边界离散化
                series = self.combo_stats.get(sn)
                if series is None:
                    continue
                fit_data = series[:self.max_train_idx].astype(float)
                edges = np.percentile(fit_data, np.linspace(0, 100, 4))
                edges = np.unique(edges)
                val = cand_stats[sn]
                disc = 0
                for i in range(len(edges) - 1):
                    if val >= edges[i]:
                        disc = min(i, len(edges) - 2)
                cand_discrete[sn] = disc

        for stat_name, predictions in self._cached_a4_predictions.items():
            if "+" in stat_name:
                # 联合统计量
                parts = stat_name.split("+")
                if len(parts) == 2 and parts[0] in cand_discrete and parts[1] in cand_discrete:
                    cand_val = cand_discrete[parts[0]] * 3 + cand_discrete[parts[1]]
                else:
                    continue
            else:
                if stat_name not in cand_discrete:
                    continue
                cand_val = cand_discrete[stat_name]

            for pred_bin, conf in predictions:
                if cand_val == pred_bin:
                    bonus += conf
                n_checked += 1

        return bonus / n_checked if n_checked > 0 else 0.0

    def full_score(self, combo, t, a1_weight=0.3, a3_weight=0.15, a4_weight=0.15):
        """综合评分 = A2 规则评分 + A1 方向奖励 + A3 差分奖励 + A4 统计量奖励"""
        rule_score = self.score_combo(combo, t)
        a1_bonus = self.get_a1_direction_bonus(combo, t)
        a3_bonus = self.get_a3_diff_bonus(combo, t)
        a4_bonus = self.get_a4_stat_bonus(combo, t)
        return rule_score + a1_weight * a1_bonus + a3_weight * a3_bonus + a4_weight * a4_bonus


def beam_search_combos(fused_weights_i, n_pos, red_range, beam_width=200,
                       rule_weight=0.3, candidates_sets=None,
                       rule_scorer=None, t=None):
    """对单个样本用 Beam Search 构造组合

    Args:
        fused_weights_i: list[dict]  每位置的 {号码: 权重}
        n_pos: 位置数
        red_range: 红球范围
        beam_width: beam 宽度
        rule_weight: 规则评分的权重（vs 号码权重）
        candidates_sets: 可选，list[set] 每位置的候选集（硬排除后）
        rule_scorer: 可选，RuleScorer 实例（数据驱动的规则评分器）
        t: 可选，当前期索引（rule_scorer 需要）

    Returns:
        combos: list[tuple]  排序后的组合列表（得分从高到低）
        scores: list[float]  对应得分
    """
    # beam: list of (partial_combo, cumulative_score)
    beam = [([], 0.0)]

    for pos in range(n_pos):
        new_beam = []
        weights = fused_weights_i[pos]

        # 候选号码
        if candidates_sets is not None and pos < len(candidates_sets):
            candidates = candidates_sets[pos]
        else:
            candidates = set(weights.keys())

        for partial, cum_score in beam:
            # 排序约束：当前号码必须大于前一个
            min_val = partial[-1] + 1 if partial else 1
            # 还需要为后续位置留空间
            max_val = red_range - (n_pos - 1 - pos)

            for v in candidates:
                if v < min_val or v > max_val:
                    continue
                w = weights.get(v, 0.0)
                # 号码权重得分（取 log 避免连乘下溢）
                w_score = math_log(max(w, 1e-15))
                new_score = cum_score + w_score
                new_beam.append((partial + [v], new_score))

        # 剪枝：保留 top beam_width
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

        if not beam:
            break

    if not beam:
        return [], []

    # 对完整组合用数据挖掘的规则评分
    scored_combos = []
    for combo, w_score in beam:
        if len(combo) == n_pos:
            if rule_scorer is not None and t is not None:
                r_score = rule_scorer.full_score(combo, t)
                # 综合得分 = (1 - λ) * 号码权重得分 + λ * 规则得分（归一化到同量级）
                final_score = (1 - rule_weight) * w_score + rule_weight * r_score * n_pos
            else:
                final_score = w_score
            scored_combos.append((tuple(combo), final_score))

    scored_combos.sort(key=lambda x: x[1], reverse=True)
    combos = [c for c, s in scored_combos]
    scores = [s for c, s in scored_combos]
    return combos, scores


def independent_topk_combos(fused_weights_i, n_pos, red_range, top_k_per_pos):
    """基线方法：逐位置独立 top-K，然后枚举合法组合"""
    cand_sets = []
    for pos in range(n_pos):
        sorted_nums = sorted(fused_weights_i[pos].items(),
                             key=lambda x: x[1], reverse=True)
        k = top_k_per_pos[pos] if isinstance(top_k_per_pos, list) else top_k_per_pos
        top_set = sorted([v for v, w in sorted_nums[:k]])
        cand_sets.append(top_set)

    # 枚举满足排序约束的组合（用 DFS）
    combos = []

    def dfs(pos, prev_val, current):
        if pos == n_pos:
            combos.append(tuple(current))
            return
        for v in cand_sets[pos]:
            if v > prev_val:
                current.append(v)
                dfs(pos + 1, v, current)
                current.pop()

    dfs(0, 0, [])
    return combos


def e3c_combo_construction(lottery_type, fused_weights=None, optimal_ks=None):
    """E3c: 组合构造验证（数据驱动的规则评分 vs 纯权重基线）"""
    log(f"\n{'═'*55}")
    log(f"  E3c 组合构造验证（规则驱动）: {lottery_type}")
    log(f"{'═'*55}")

    probs, test_indices, data, best_method = get_best_probs(lottery_type)
    n_samples, n_pos = probs.shape[0], probs.shape[1]
    red_range = data.red_range
    max_train_idx = int(test_indices[0])

    # 如果没有传入融合权重，用基线权重
    if fused_weights is None:
        diff_dist = compute_diff_distributions(data, max_train_idx)
        fused_weights = signal_a_baseline(probs, data, test_indices, diff_dist)

    if optimal_ks is None:
        optimal_ks = [red_range // 2] * n_pos

    # 构建规则评分器
    with Timer("构建 RuleScorer"):
        scorer = RuleScorer(data, max_train_idx)

    n_rules_total = len(scorer.combo_rules) + len(scorer.position_rules)
    n_a1 = len(scorer.a1_patterns)
    log(f"  规则评分器: {n_rules_total} 条 A2 规则, {n_a1} 条 A1 模式")

    # === Beam Search 参数扫描 ===
    log("\n  --- Beam Search 参数扫描 ---")
    beam_configs = [
        {"beam_width": 200, "rule_weight": 0.0},   # 纯权重基线（无规则）
        {"beam_width": 200, "rule_weight": 0.1},
        {"beam_width": 200, "rule_weight": 0.2},
        {"beam_width": 200, "rule_weight": 0.3},
        {"beam_width": 200, "rule_weight": 0.5},
        {"beam_width": 500, "rule_weight": 0.0},
        {"beam_width": 500, "rule_weight": 0.2},
        {"beam_width": 500, "rule_weight": 0.3},
    ]

    # 基线：独立 top-K
    log("\n  基线（独立 top-K）:")
    baseline_survivals = 0
    baseline_combo_counts = []
    baseline_combos_all = comb(red_range, n_pos)

    for i in range(n_samples):
        t = int(test_indices[i])
        true_vals = tuple(int(data.red_matrix[t, pos]) for pos in range(n_pos))
        combos = independent_topk_combos(fused_weights[i], n_pos, red_range, optimal_ks)
        baseline_combo_counts.append(len(combos))
        if true_vals in set(combos):
            baseline_survivals += 1

    bl_surv = baseline_survivals / n_samples
    bl_avg_combos = float(np.mean(baseline_combo_counts))
    bl_reduction = 1.0 - bl_avg_combos / baseline_combos_all
    log(f"  存活率: {bl_surv:.2%}, 平均组合数: {bl_avg_combos:.0f}, "
        f"缩减率: {bl_reduction:.2%}")

    # Beam Search 各配置
    beam_results = {}
    best_beam_config = None
    best_beam_efficiency = -1

    for cfg in beam_configs:
        bw = cfg["beam_width"]
        rw = cfg["rule_weight"]
        key = f"bw{bw}_rw{rw}"
        use_rules = rw > 0 and n_rules_total > 0
        log(f"\n  Beam(width={bw}, rule_weight={rw}, rules={'ON' if use_rules else 'OFF'}):")

        survivals = 0
        combo_counts = []
        rule_scores_hit = []
        rule_scores_miss = []

        for i in range(n_samples):
            t = int(test_indices[i])
            true_vals = tuple(int(data.red_matrix[t, pos]) for pos in range(n_pos))

            # 每期预过滤 A1 模式（避免 beam search 内重复检查）
            if scorer is not None and use_rules:
                scorer.prefilter_for_period(t)

            combos, scores = beam_search_combos(
                fused_weights[i], n_pos, red_range,
                beam_width=bw, rule_weight=rw,
                rule_scorer=scorer if use_rules else None,
                t=t if use_rules else None)

            combo_counts.append(len(combos))
            if true_vals in set(combos):
                survivals += 1
                rule_scores_hit.append(
                    scorer.full_score(list(true_vals), t))
            else:
                rule_scores_miss.append(
                    scorer.full_score(list(true_vals), t))

        surv = survivals / n_samples
        avg_combos = float(np.mean(combo_counts))
        reduction = 1.0 - avg_combos / baseline_combos_all
        efficiency = reduction * surv

        log(f"  存活率: {surv:.2%}, 平均组合数: {avg_combos:.0f}, "
            f"缩减率: {reduction:.2%}, 效率: {efficiency:.4f}")

        if rule_scores_hit:
            log(f"  命中组合规则得分: {np.mean(rule_scores_hit):.4f}")
        if rule_scores_miss:
            log(f"  未命中组合规则得分: {np.mean(rule_scores_miss):.4f}")

        beam_results[key] = {
            "beam_width": bw,
            "rule_weight": rw,
            "rules_enabled": use_rules,
            "survival_rate": surv,
            "avg_combo_count": avg_combos,
            "combo_reduction": reduction,
            "efficiency": efficiency,
            "avg_rule_score_hit": float(np.mean(rule_scores_hit)) if rule_scores_hit else None,
            "avg_rule_score_miss": float(np.mean(rule_scores_miss)) if rule_scores_miss else None,
        }

        if efficiency > best_beam_efficiency:
            best_beam_efficiency = efficiency
            best_beam_config = key

    # 对比有规则 vs 无规则
    no_rule_key = "bw200_rw0.0"
    no_rule_eff = beam_results.get(no_rule_key, {}).get("efficiency", 0)
    log(f"\n  最优 Beam 配置: {best_beam_config}, 效率: {best_beam_efficiency:.4f}")
    log(f"  无规则基线 (bw200_rw0.0) 效率: {no_rule_eff:.4f}")
    log(f"  规则带来的效率提升: {best_beam_efficiency - no_rule_eff:+.4f}")

    results = {
        "lottery_type": lottery_type,
        "rule_scorer_info": {
            "n_combo_rules": len(scorer.combo_rules),
            "n_position_rules": len(scorer.position_rules),
            "n_a1_patterns": len(scorer.a1_patterns),
        },
        "baseline_topk": {
            "survival_rate": bl_surv,
            "avg_combo_count": bl_avg_combos,
            "combo_reduction": bl_reduction,
            "efficiency": bl_reduction * bl_surv,
        },
        "beam_search": beam_results,
        "best_beam_config": best_beam_config,
        "best_beam_efficiency": best_beam_efficiency,
        "rule_improvement": best_beam_efficiency - no_rule_eff,
        "improvement_over_topk": best_beam_efficiency - bl_reduction * bl_surv,
    }

    save_json(results, STRICT_EXPERIMENT_DIR / f"e3c_combo_construction_{lottery_type}.json")
    return results


# ============================================================
#  E3d: 端到端评估（Layer 0-3 串联）
# ============================================================

def e3d_end_to_end(lottery_type):
    """E3d: 串联四层，端到端评估完整框架"""
    log(f"\n{'═'*55}")
    log(f"  E3d 端到端评估: {lottery_type}")
    log(f"{'═'*55}")

    probs, test_indices, data, best_method = get_best_probs(lottery_type)
    n_samples, n_pos = probs.shape[0], probs.shape[1]
    red_range = data.red_range
    max_train_idx = int(test_indices[0])
    baseline_combos_all = comb(red_range, n_pos)

    # === Layer 0: 硬排除 ===
    log("\n  [Layer 0] 硬排除...")
    e3a_path = STRICT_EXPERIMENT_DIR / f"e3a_hard_exclusion_{lottery_type}.json"
    if e3a_path.exists():
        e3a_results = load_json(str(e3a_path))
    else:
        e3a_results = e3a_hard_exclusion(lottery_type)

    # 预计算差分百分位缓存
    diff_dist = compute_diff_distributions(data, max_train_idx)
    diff_pct_cache = {}
    for pos in range(n_pos):
        diff_pct_cache[pos] = {}
        for dir_label in ['U', 'D']:
            diffs = list(diff_dist[pos][dir_label].elements())
            if len(diffs) >= 10:
                diff_pct_cache[pos][dir_label] = {
                    'p1': float(np.percentile(diffs, 1)),
                    'p99': float(np.percentile(diffs, 99)),
                }

    # 对每个样本每个位置计算候选集
    candidates_per_sample = []
    layer0_stats = {"avg_candidates": [], "exclusion_errors": 0}

    for i in range(n_samples):
        t = int(test_indices[i])
        sample_cands = []
        for pos in range(n_pos):
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2)
            true_val = int(data.red_matrix[t, pos])
            cands = apply_hard_exclusion(
                probs[i, pos, :], pos, current_val, red_range, n_pos,
                e3a_results, diff_pct_cache)
            sample_cands.append(cands)
            layer0_stats["avg_candidates"].append(len(cands))
            if true_val not in cands:
                layer0_stats["exclusion_errors"] += 1
        candidates_per_sample.append(sample_cands)

    avg_cands = float(np.mean(layer0_stats["avg_candidates"]))
    err_rate = layer0_stats["exclusion_errors"] / (n_samples * n_pos)
    log(f"  平均候选数: {avg_cands:.1f}/{red_range}, 误排除率: {err_rate:.4%}")

    # === Layer 1: 多信号融合 ===
    log("\n  [Layer 1] 多信号融合...")
    e3b_path = STRICT_EXPERIMENT_DIR / f"e3b_signal_fusion_{lottery_type}.json"
    if e3b_path.exists():
        e3b_saved = load_json(str(e3b_path))
        best_alphas = e3b_saved.get("best_config", {}).get("alphas", {"A": 1.0})
        optimal_ks = e3b_saved.get("optimal_ks")
        # 兼容旧格式（list → dict）
        if isinstance(best_alphas, list):
            names = ["A", "B", "C"]
            best_alphas = {names[i]: best_alphas[i] for i in range(min(len(best_alphas), len(names)))}
    else:
        best_alphas = {"A": 1.0}
        optimal_ks = None

    # 重新计算融合权重（应用硬排除候选集）
    cond_matrices = build_cond_prob_matrix(data, 0, max_train_idx - 1)

    weights_a = signal_a_baseline(probs, data, test_indices, diff_dist)
    weights_b = signal_b_rule_cluster(probs, data, test_indices, diff_dist, max_train_idx)
    weights_c = signal_c_cross_position(probs, data, test_indices, cond_matrices)
    weights_d = signal_d_a3_diff(probs, data, test_indices, max_train_idx)
    weights_e = signal_e_a4_stat(probs, data, test_indices, max_train_idx)
    weights_f = signal_f_e1_amplitude(probs, data, test_indices, max_train_idx)

    # 按 best_alphas 构建信号列表
    signal_map = {
        "A": weights_a,
        "B": weights_b,
        "C": weights_c,
        "D": weights_d,
        "E": weights_e,
        "F": weights_f,
    }
    signal_list = []
    alpha_list = []
    for name in ["A", "B", "C", "D", "E", "F"]:
        w = signal_map.get(name)
        a = best_alphas.get(name, 0.0)
        if w is not None and a > 0.001:
            signal_list.append(w)
            alpha_list.append(a)

    fused_weights = fuse_signals(signal_list, alpha_list, candidates_per_sample)
    log(f"  融合权重: alphas={best_alphas}")

    # 如果没有 optimal_ks，重新计算
    if optimal_ks is None:
        optimal_ks = []
        for pos in range(n_pos):
            survival_by_k = np.zeros(red_range + 1)
            for i in range(n_samples):
                t = int(test_indices[i])
                true_val = int(data.red_matrix[t, pos])
                sorted_nums = sorted(fused_weights[i][pos].items(),
                                     key=lambda x: x[1], reverse=True)
                for r, (v, w) in enumerate(sorted_nums, 1):
                    if v == true_val:
                        for k in range(r, red_range + 1):
                            survival_by_k[k] += 1
                        break
            surv_rates = survival_by_k / n_samples
            best_k = red_range
            for k in range(1, red_range + 1):
                if surv_rates[k] >= 0.95:
                    best_k = k
                    break
            optimal_ks.append(best_k)

    log(f"  K95: {optimal_ks}")

    # === Layer 2: Beam Search 组合构造（规则驱动） ===
    log("\n  [Layer 2] Beam Search 组合构造（规则驱动）...")

    # 构建规则评分器
    with Timer("构建 RuleScorer"):
        scorer = RuleScorer(data, max_train_idx)
    n_rules = len(scorer.combo_rules) + len(scorer.position_rules)
    n_a3 = sum(len(v) for v in scorer.a3_grouped.values())
    n_a4 = sum(len(v) for v in scorer.a4_grouped.values())
    log(f"  规则评分器: {n_rules} 条 A2 规则, {len(scorer.a1_patterns)} 条 A1 模式, "
        f"{n_a3} 条 A3 差分模式, {n_a4} 条 A4 统计量模式")

    e3c_path = STRICT_EXPERIMENT_DIR / f"e3c_combo_construction_{lottery_type}.json"
    if e3c_path.exists():
        e3c_saved = load_json(str(e3c_path))
        best_beam_key = e3c_saved.get("best_beam_config", "bw200_rw0.2")
        beam_cfg = e3c_saved.get("beam_search", {}).get(best_beam_key, {})
        beam_width = beam_cfg.get("beam_width", 200)
        rule_weight = beam_cfg.get("rule_weight", 0.2)
    else:
        beam_width = 200
        rule_weight = 0.2

    log(f"  Beam 参数: width={beam_width}, rule_weight={rule_weight}")

    # === Layer 3: 组合排序与最终评估 ===
    log("\n  [Layer 3] 组合排序与最终评估...")

    survivals = 0
    combo_counts = []
    true_ranks = []
    top_n_hits = {10: 0, 50: 0, 100: 0, 200: 0}

    for i in range(n_samples):
        t = int(test_indices[i])
        true_vals = tuple(int(data.red_matrix[t, pos]) for pos in range(n_pos))

        # 每期预过滤 A1 模式
        if scorer is not None:
            scorer.prefilter_for_period(t)

        combos, scores = beam_search_combos(
            fused_weights[i], n_pos, red_range,
            beam_width=beam_width, rule_weight=rule_weight,
            candidates_sets=candidates_per_sample[i],
            rule_scorer=scorer, t=t)

        combo_counts.append(len(combos))

        if true_vals in set(combos):
            survivals += 1
            rank = combos.index(true_vals) + 1
            true_ranks.append(rank)
            for n_top in top_n_hits:
                if rank <= n_top:
                    top_n_hits[n_top] += 1
        else:
            true_ranks.append(len(combos) + 1)

    surv_rate = survivals / n_samples
    avg_combos = float(np.mean(combo_counts))
    reduction = 1.0 - avg_combos / baseline_combos_all
    efficiency = reduction * surv_rate

    log(f"\n  === 端到端结果 ===")
    log(f"  组合存活率: {surv_rate:.2%}")
    log(f"  平均组合数: {avg_combos:.0f}")
    log(f"  组合缩减率: {reduction:.2%}")
    log(f"  组合效率: {efficiency:.4f}")

    if survivals > 0:
        rank_arr = np.array([r for r in true_ranks if r <= max(combo_counts)])
        log(f"  命中时平均排名: {float(np.mean(rank_arr)):.1f}")

    for n_top, hits in sorted(top_n_hits.items()):
        log(f"  Top-{n_top} 命中率: {hits/n_samples:.2%}")

    # === 蓝球独立评估 ===
    log("\n  --- 蓝球独立评估 ---")
    blue_summary_path = RESULTS_DIR / "experiment_blue" / "e0_step6_blue_summary.json"
    blue_result = None
    if blue_summary_path.exists():
        blue_summary = load_json(str(blue_summary_path))
        blue_report = blue_summary.get("reports", {}).get(lottery_type, {})
        if blue_report:
            blue_result = {
                "best_method": blue_report.get("best_method"),
                "direction_accuracy": blue_report.get("direction_accuracy"),
                "avg_reduction": blue_report.get("avg_reduction"),
                "avg_survival": blue_report.get("avg_survival"),
                "efficiency": blue_report.get("efficiency"),
            }
            log(f"  蓝球最优方法: {blue_result['best_method']}")
            log(f"  蓝球方向准确率: {blue_result['direction_accuracy']:.2%}")
            log(f"  蓝球缩减率: {blue_result['avg_reduction']:.2%}")
            log(f"  蓝球存活率: {blue_result['avg_survival']:.2%}")
            log(f"  蓝球效率: {blue_result['efficiency']:.4f}")

            # 综合效率 = 红球效率 × 蓝球效率（独立事件）
            combined_efficiency = efficiency * blue_result['efficiency']
            log(f"  红球效率: {efficiency:.4f}")
            log(f"  综合效率（红×蓝）: {combined_efficiency:.4f}")
        else:
            log(f"  蓝球结果不可用（{lottery_type}）")
    else:
        log(f"  蓝球实验结果文件不存在")

    # === 与 E1/E2 基线对比 ===
    log("\n  --- 与基线对比 ---")
    e1_path = STRICT_EXPERIMENT_DIR / f"e1_combo_evaluation_{lottery_type}.json"
    if e1_path.exists():
        e1_data = load_json(str(e1_path))
        e1_surv = e1_data.get("combo_survival_rate", 0)
        e1_reduction = e1_data.get("combo_reduction_rate", 0)
        e1_efficiency = e1_surv * e1_reduction
        log(f"  E1 基线: 存活={e1_surv:.2%}, 缩减={e1_reduction:.2%}, 效率={e1_efficiency:.4f}")
        log(f"  E3 vs E1: 效率提升 {efficiency - e1_efficiency:+.4f}")
    else:
        e1_efficiency = None
        log(f"  E1 基线文件不存在，跳过对比")

    results = {
        "lottery_type": lottery_type,
        "layer0_hard_exclusion": {
            "avg_candidates_per_pos": avg_cands,
            "exclusion_error_rate": err_rate,
        },
        "layer1_signal_fusion": {
            "alphas": best_alphas,
            "optimal_ks": optimal_ks,
        },
        "layer2_beam_search": {
            "beam_width": beam_width,
            "rule_weight": rule_weight,
            "n_rules": n_rules,
            "n_a1_patterns": len(scorer.a1_patterns),
            "n_a3_patterns": n_a3,
            "n_a4_patterns": n_a4,
        },
        "layer3_final": {
            "survival_rate": surv_rate,
            "avg_combo_count": avg_combos,
            "combo_reduction": reduction,
            "efficiency": efficiency,
            "top_n_hits": {str(k): v / n_samples for k, v in top_n_hits.items()},
            "mean_rank_when_hit": float(np.mean([r for r in true_ranks if r <= max(combo_counts)])) if survivals > 0 else None,
        },
        "comparison": {
            "e1_efficiency": e1_efficiency,
            "e3_efficiency": efficiency,
            "improvement": efficiency - e1_efficiency if e1_efficiency is not None else None,
        },
        "blue_ball": blue_result,
        "combined_efficiency": efficiency * blue_result['efficiency'] if blue_result else None,
    }

    save_json(results, STRICT_EXPERIMENT_DIR / f"e3d_end_to_end_{lottery_type}.json")
    return results


# ============================================================
#  主入口
# ============================================================

def main():
    """运行 E3 完整实验"""
    setup_logging()
    log("=" * 60)
    log("  E3: 规则灵活运用框架")
    log("=" * 60)

    # 确保输出目录存在
    STRICT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    lottery_types = ["shuangseqiu", "daletou"]

    for lt in lottery_types:
        try:
            log(f"\n\n{'#'*60}")
            log(f"  彩种: {lt.upper()}")
            log(f"{'#'*60}")

            # E3a: 硬排除边界验证
            with Timer(f"E3a ({lt})"):
                e3a_results = e3a_hard_exclusion(lt)

            # E3b: 多信号融合验证
            with Timer(f"E3b ({lt})"):
                e3b_results, fused_weights, optimal_ks = e3b_signal_fusion(lt, e3a_results)

            # E3c: 组合构造验证
            with Timer(f"E3c ({lt})"):
                e3c_results = e3c_combo_construction(lt, fused_weights, optimal_ks)

            # E3d: 端到端评估
            with Timer(f"E3d ({lt})"):
                e3d_results = e3d_end_to_end(lt)

        except Exception as e:
            log(f"\n  [错误] {lt}: {e}")
            traceback.print_exc()
            continue

    log("\n" + "=" * 60)
    log("  E3 实验完成")
    log("=" * 60)


if __name__ == "__main__":
    main()
