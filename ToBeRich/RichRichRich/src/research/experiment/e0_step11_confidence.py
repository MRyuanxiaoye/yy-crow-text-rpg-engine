# -*- coding: utf-8 -*-
"""E0-Step11：置信度评分系统

综合多维度信号，为每期预测输出一个置信度评分：
  S1: 方向预测一致性（多模型投票一致度）
  S2: 幅度预测集中度（条件分布IQR / 无条件IQR）
  S3: 历史模式匹配强度（匹配的显著模式数量和质量）
  S4: 序列稳定性（近期方向预测准确率的滑动窗口）
  S5: 组合存活率（候选组合在历史上的存活比例）

最终置信度 = 加权融合 S1~S5，输出 [0, 1] 区间

用法: python3 -m src.research.experiment.e0_step11_confidence
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_json, RESULTS_DIR,
)
from research.experiment.e0_step10a_amplitude_scan import (
    encode_direction_seq, encode_amplitude_bin, encode_value_zone,
    encode_diff_sign_amp, encode_diff_change_pattern,
    compute_miss_periods, encode_miss_zone,
)

STEP10_DIR = RESULTS_DIR / "e0_step10"
STEP11_DIR = RESULTS_DIR / "e0_step11"
STEP11_DIR.mkdir(parents=True, exist_ok=True)

# 默认权重
DEFAULT_WEIGHTS = {
    'S1_direction_consistency': 0.20,
    'S2_amplitude_concentration': 0.15,
    'S3_pattern_match_strength': 0.15,
    'S4_sequence_stability': 0.15,
    'S5_combo_survival': 0.15,
    'S6_e1_rule_match': 0.20,
}


# ============================================================
#  S1: 方向预测一致性
# ============================================================

def compute_s1_direction_consistency(data, pos, t):
    """多种方法预测方向，计算投票一致度

    方法：
    1. 最近1期方向延续
    2. 最近3期多数方向
    3. 最近5期多数方向
    4. 均值回归方向（偏离均值则反向）
    5. 动量方向（最近差分的符号）

    Returns:
        score: [0, 1]，1表示完全一致
        predicted_dir: 多数投票的方向
    """
    series = data.position_series[pos]
    dir_series = data.direction_series[pos]
    diff_series = np.diff(series.astype(np.float64))

    votes = []

    # 方法1: 最近1期方向延续
    if t >= 1:
        votes.append(int(dir_series[t - 1]))

    # 方法2: 最近3期多数方向
    if t >= 3:
        recent3 = [int(dir_series[t - i]) for i in range(1, 4)]
        from collections import Counter
        mc = Counter(recent3).most_common(1)[0][0]
        votes.append(mc)

    # 方法3: 最近5期多数方向
    if t >= 5:
        recent5 = [int(dir_series[t - i]) for i in range(1, 6)]
        mc = Counter(recent5).most_common(1)[0][0]
        votes.append(mc)

    # 方法4: 均值回归
    if t >= 30:
        mean_val = np.mean(series[t-30:t])
        current = series[t]
        if current > mean_val * 1.05:
            votes.append(0)  # D: 预测下降
        elif current < mean_val * 0.95:
            votes.append(2)  # U: 预测上升
        else:
            votes.append(1)  # E: 持平

    # 方法5: 动量方向
    if t >= 2:
        d1 = diff_series[t - 2]
        d2 = diff_series[t - 1]
        momentum = d1 + d2
        if momentum > 0:
            votes.append(2)  # U
        elif momentum < 0:
            votes.append(0)  # D
        else:
            votes.append(1)  # E

    if not votes:
        return 0.5, 1  # 默认

    from collections import Counter
    counter = Counter(votes)
    predicted_dir = counter.most_common(1)[0][0]
    consistency = counter[predicted_dir] / len(votes)

    return float(consistency), int(predicted_dir)


# ============================================================
#  S2: 幅度预测集中度
# ============================================================

def compute_s2_amplitude_concentration(data, pos, t, scan_results):
    """基于匹配的显著模式，计算幅度预测的集中度

    Returns:
        score: [0, 1]，1表示非常集中（确定性高）
    """
    series = data.position_series[pos]
    diff_series = np.diff(series.astype(np.float64))
    dir_series = data.direction_series[pos]

    # 预计算边界
    train_60pct = int(data.n_draws * 0.6) - 1
    train_abs = np.abs(diff_series[:train_60pct])
    amp_bin_edges = np.percentile(train_abs, np.linspace(0, 100, 11))
    amp_5_edges = np.percentile(train_abs, np.linspace(0, 100, 6))
    zone_edges = np.linspace(1, data.red_range, 6)
    uncond_iqr = float(np.percentile(train_abs, 75) - np.percentile(train_abs, 25))

    # 查找匹配模式
    pos_key = f'P{pos}'
    patterns = []
    if pos_key in scan_results.get('positions', {}):
        for p in scan_results['positions'][pos_key].get('patterns', []):
            if p.get('validation', {}).get('status') == 'valid':
                patterns.append(p)

    matched_concentrations = []
    for pat in patterns:
        dim = pat['dimension']
        cond = pat['condition']
        label = _get_label(dim, t, diff_series, dir_series, series, data,
                           amp_bin_edges, amp_5_edges, zone_edges)
        if label is not None and str(label) == cond:
            matched_concentrations.append(pat.get('concentration', 1.0))

    if not matched_concentrations:
        return 0.5  # 无匹配，中等置信度

    # 集中度越低越好（IQR越小）
    avg_concentration = np.mean(matched_concentrations)
    # 转换为 [0, 1] 分数：concentration < 0.5 → 高分，> 1.0 → 低分
    score = max(0, min(1, 1.0 - avg_concentration))
    return float(score)


# ============================================================
#  S3: 历史模式匹配强度
# ============================================================

def compute_s3_pattern_match_strength(data, pos, t, scan_results):
    """匹配的显著模式数量和质量

    Returns:
        score: [0, 1]
    """
    series = data.position_series[pos]
    diff_series = np.diff(series.astype(np.float64))
    dir_series = data.direction_series[pos]

    train_60pct = int(data.n_draws * 0.6) - 1
    train_abs = np.abs(diff_series[:train_60pct])
    amp_bin_edges = np.percentile(train_abs, np.linspace(0, 100, 11))
    amp_5_edges = np.percentile(train_abs, np.linspace(0, 100, 6))
    zone_edges = np.linspace(1, data.red_range, 6)

    pos_key = f'P{pos}'
    patterns = []
    if pos_key in scan_results.get('positions', {}):
        for p in scan_results['positions'][pos_key].get('patterns', []):
            if p.get('validation', {}).get('status') == 'valid':
                patterns.append(p)

    n_matched = 0
    total_js = 0.0

    for pat in patterns:
        dim = pat['dimension']
        cond = pat['condition']
        label = _get_label(dim, t, diff_series, dir_series, series, data,
                           amp_bin_edges, amp_5_edges, zone_edges)
        if label is not None and str(label) == cond:
            n_matched += 1
            total_js += pat.get('js_divergence', 0)

    # 匹配数量分数：0个→0, 5个→0.5, 10+个→1.0
    count_score = min(1.0, n_matched / 10.0)

    # JS散度分数：越大越好
    js_score = min(1.0, total_js / 0.5) if n_matched > 0 else 0.0

    score = 0.6 * count_score + 0.4 * js_score
    return float(score)


# ============================================================
#  S4: 序列稳定性
# ============================================================

def compute_s4_sequence_stability(data, pos, t, window=20):
    """近期方向预测准确率的滑动窗口

    用最简单的"延续上期方向"策略，看近期准确率

    Returns:
        score: [0, 1]
    """
    dir_series = data.direction_series[pos]

    if t < window + 1:
        return 0.5

    # 最近 window 期的方向延续准确率
    correct = 0
    for i in range(t - window, t):
        if i >= 1 and dir_series[i] == dir_series[i - 1]:
            correct += 1

    acc = correct / window

    # 准确率 > 0.5 说明序列有惯性，置信度高
    # 准确率 < 0.3 说明序列反转频繁，也有规律
    # 准确率 ≈ 0.33 说明完全随机
    deviation = abs(acc - 1.0 / 3.0)
    score = min(1.0, deviation / 0.3)  # 偏离随机基线越多，分数越高

    return float(score)


# ============================================================
#  S5: 组合存活率估计
# ============================================================

def compute_s5_combo_survival(data, pos, t, scan_results):
    """估计候选集的存活率（候选号码覆盖正确答案的历史比例）

    用最近30期的回测来估计

    Returns:
        score: [0, 1]
    """
    from research.experiment.e0_step10c_combo_rerank import (
        generate_amplitude_constrained_candidates,
    )

    series = data.position_series[pos]
    lookback = min(30, t - 5)
    if lookback < 10:
        return 0.5

    hits = 0
    for bt in range(t - lookback, t):
        candidates, _ = generate_amplitude_constrained_candidates(
            data, pos, bt, scan_results, confidence_level='medium')
        true_next = int(series[bt + 1])
        if true_next in candidates:
            hits += 1

    survival_rate = hits / lookback
    return float(survival_rate)


# ============================================================
#  S6: E1 规律库匹配强度
# ============================================================

def compute_s6_e1_rule_match(data, pos, t):
    """基于 E1 规律库（D1-D12 维度）的规则匹配数量和质量

    利用 e3_rule_framework 中的辅助函数重建条件并匹配规则。

    Returns:
        score: [0, 1]，匹配规则越多且质量越高，分数越高
    """
    from research.experiment.e3_rule_framework import (
        _load_e1_rules_by_position, _rebuild_e1_condition, _precompute_e1_edges,
    )

    lt = data.lottery_type
    rules_by_pos = _load_e1_rules_by_position(lt)
    if not rules_by_pos:
        return 0.5

    pos_rules = rules_by_pos.get(pos, [])
    if not pos_rules:
        return 0.5

    # 预计算边界
    max_train_idx = int(data.n_draws * 0.6)
    edges_cache = {pos: _precompute_e1_edges(data, pos, max_train_idx)}

    # 匹配规则
    matched_qualities = []
    for rule in pos_rules:
        try:
            if _rebuild_e1_condition(rule, data, pos, t, edges_cache):
                matched_qualities.append(rule.get("quality_score", 0.5))
        except Exception:
            continue

    if not matched_qualities:
        return 0.3  # 无匹配，低置信度

    n_matched = len(matched_qualities)
    avg_quality = float(np.mean(matched_qualities))

    # 匹配数量分数：1个→0.3, 5个→0.6, 10+个→1.0
    count_score = min(1.0, 0.2 + n_matched * 0.08)

    # 质量分数：quality_score 本身就在 [0, 1]
    quality_score = avg_quality

    score = 0.5 * count_score + 0.5 * quality_score
    return float(max(0.0, min(1.0, score)))


# ============================================================
#  综合置信度
# ============================================================

def compute_confidence(data, pos, t, scan_results, weights=None):
    """计算综合置信度评分

    Returns:
        confidence: float [0, 1]
        components: dict of individual scores
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    s1, pred_dir = compute_s1_direction_consistency(data, pos, t)
    s2 = compute_s2_amplitude_concentration(data, pos, t, scan_results)
    s3 = compute_s3_pattern_match_strength(data, pos, t, scan_results)
    s4 = compute_s4_sequence_stability(data, pos, t)
    s5 = compute_s5_combo_survival(data, pos, t, scan_results)
    s6 = compute_s6_e1_rule_match(data, pos, t)

    components = {
        'S1_direction_consistency': s1,
        'S2_amplitude_concentration': s2,
        'S3_pattern_match_strength': s3,
        'S4_sequence_stability': s4,
        'S5_combo_survival': s5,
        'S6_e1_rule_match': s6,
        'predicted_direction': pred_dir,
    }

    confidence = sum(
        weights[k] * components[k]
        for k in weights.keys()
    )
    confidence = max(0.0, min(1.0, confidence))

    return float(confidence), components


# ============================================================
#  评估：置信度与实际准确率的相关性
# ============================================================

def evaluate_confidence_system(data, scan_results, test_start, test_end):
    """评估置信度系统的有效性

    核心指标：高置信度期的预测准确率 > 低置信度期
    """
    from research.experiment.e0_step10c_combo_rerank import (
        generate_amplitude_constrained_candidates,
    )

    n_pos = data.red_count
    series_all = data.position_series

    all_confidences = []
    all_hits = []

    for t in range(test_start + 5, test_end - 1):
        period_conf = []
        period_hits = []

        for pos in range(n_pos):
            conf, components = compute_confidence(data, pos, t, scan_results)
            period_conf.append(conf)

            # 检查候选集是否命中
            candidates, _ = generate_amplitude_constrained_candidates(
                data, pos, t, scan_results, confidence_level='medium')
            true_next = int(series_all[pos][t + 1])
            hit = 1 if true_next in candidates else 0
            period_hits.append(hit)

        avg_conf = float(np.mean(period_conf))
        avg_hit = float(np.mean(period_hits))
        all_confidences.append(avg_conf)
        all_hits.append(avg_hit)

    all_confidences = np.array(all_confidences)
    all_hits = np.array(all_hits)

    # 按置信度分3档
    q33 = np.percentile(all_confidences, 33)
    q66 = np.percentile(all_confidences, 66)

    low_mask = all_confidences <= q33
    mid_mask = (all_confidences > q33) & (all_confidences <= q66)
    high_mask = all_confidences > q66

    results = {
        'n_periods': len(all_confidences),
        'confidence_stats': {
            'mean': float(np.mean(all_confidences)),
            'std': float(np.std(all_confidences)),
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences)),
        },
        'hit_rate_by_confidence': {
            'low': float(np.mean(all_hits[low_mask])) if np.sum(low_mask) > 0 else 0,
            'mid': float(np.mean(all_hits[mid_mask])) if np.sum(mid_mask) > 0 else 0,
            'high': float(np.mean(all_hits[high_mask])) if np.sum(high_mask) > 0 else 0,
        },
        'n_by_confidence': {
            'low': int(np.sum(low_mask)),
            'mid': int(np.sum(mid_mask)),
            'high': int(np.sum(high_mask)),
        },
    }

    # Spearman 相关系数
    from scipy.stats import spearmanr
    corr, p_val = spearmanr(all_confidences, all_hits)
    results['spearman_correlation'] = float(corr)
    results['spearman_p_value'] = float(p_val)

    # 置信度有效性判断
    hr_low = results['hit_rate_by_confidence']['low']
    hr_high = results['hit_rate_by_confidence']['high']
    results['confidence_effective'] = hr_high > hr_low
    results['high_low_lift'] = float(hr_high / hr_low) if hr_low > 0 else float('inf')

    return results


# ============================================================
#  辅助：条件标签获取（复用）
# ============================================================

def _get_label(dim, t, diff_series, dir_series, series, data,
               amp_bin_edges, amp_5_edges, zone_edges):
    """获取条件标签"""
    if dim.startswith('D1_dir_seq_n'):
        return encode_direction_seq(dir_series, t, int(dim.split('n')[-1]))
    elif dim.startswith('D2_amp_bin_n'):
        return encode_amplitude_bin(diff_series, t, int(dim.split('n')[-1]), amp_bin_edges)
    elif dim == 'D3_value_zone':
        return encode_value_zone(series[t], zone_edges)
    elif dim == 'D4_diff_sign_amp':
        return encode_diff_sign_amp(diff_series[t-1], amp_5_edges) if t >= 1 else None
    elif dim == 'D5_diff_change':
        return encode_diff_change_pattern(diff_series, t)
    elif dim == 'D6_miss_zone':
        return encode_miss_zone(compute_miss_periods(series, t, series[t]))

    if '_x_' in dim:
        parts = dim.split('_x_')
        la = _parse_part(parts[0], t, diff_series, dir_series, series, data,
                         amp_bin_edges, amp_5_edges, zone_edges)
        lb = _parse_part(parts[1], t, diff_series, dir_series, series, data,
                         amp_bin_edges, amp_5_edges, zone_edges)
        if la is None or lb is None:
            return None
        return (la, lb)
    return None


def _parse_part(dim_part, t, diff_series, dir_series, series, data,
                amp_bin_edges, amp_5_edges, zone_edges):
    if dim_part.startswith('D1n'):
        return encode_direction_seq(dir_series, t, int(dim_part[3:]))
    elif dim_part.startswith('D2n'):
        return encode_amplitude_bin(diff_series, t, int(dim_part[3:]), amp_bin_edges)
    elif dim_part == 'D3':
        return encode_value_zone(series[t], zone_edges)
    elif dim_part == 'D4':
        return encode_diff_sign_amp(diff_series[t-1], amp_5_edges) if t >= 1 else None
    elif dim_part == 'D5':
        return encode_diff_change_pattern(diff_series, t)
    elif dim_part == 'D6':
        return encode_miss_zone(compute_miss_periods(series, t, series[t]))
    return None


# ============================================================
#  主入口
# ============================================================

def run_step11(lottery_type):
    """运行 Step11 置信度评分系统"""
    log(f"\n{'═'*60}")
    log(f"  E0-Step11: 置信度评分系统 [{lottery_type}]")
    log(f"{'═'*60}")

    data = LotteryData(lottery_type)
    n_draws = data.n_draws

    # 加载 Step10a 结果
    scan_path = STEP10_DIR / f"step10a_amplitude_scan_{lottery_type}.json"
    if not scan_path.exists():
        log(f"  Step10a 结果不存在，请先运行 Step10a")
        return None

    scan_results = load_json(scan_path)

    test_start = int(n_draws * 0.6)
    test_end = n_draws

    log(f"  总期数: {n_draws}, 测试期: [{test_start}, {test_end})")

    # 评估置信度系统
    with Timer("置信度评估"):
        eval_results = evaluate_confidence_system(data, scan_results, test_start, test_end)

    log(f"\n  置信度统计:")
    log(f"    均值: {eval_results['confidence_stats']['mean']:.3f}")
    log(f"    标准差: {eval_results['confidence_stats']['std']:.3f}")
    log(f"  命中率（按置信度分档）:")
    log(f"    低: {eval_results['hit_rate_by_confidence']['low']:.3f} "
        f"({eval_results['n_by_confidence']['low']} 期)")
    log(f"    中: {eval_results['hit_rate_by_confidence']['mid']:.3f} "
        f"({eval_results['n_by_confidence']['mid']} 期)")
    log(f"    高: {eval_results['hit_rate_by_confidence']['high']:.3f} "
        f"({eval_results['n_by_confidence']['high']} 期)")
    log(f"  Spearman 相关: {eval_results['spearman_correlation']:.4f} "
        f"(p={eval_results['spearman_p_value']:.4f})")
    log(f"  高/低提升: {eval_results['high_low_lift']:.2f}x")
    log(f"  置信度有效: {'是' if eval_results['confidence_effective'] else '否'}")

    # 输出最近5期的置信度详情
    log(f"\n  最近5期置信度详情:")
    for t in range(n_draws - 6, n_draws - 1):
        confs = []
        for pos in range(data.red_count):
            conf, comp = compute_confidence(data, pos, t, scan_results)
            confs.append(conf)
        avg = np.mean(confs)
        log(f"    期{t}: 平均置信度={avg:.3f}, 各位置={[f'{c:.2f}' for c in confs]}")

    all_results = {
        'lottery_type': lottery_type,
        'weights': DEFAULT_WEIGHTS,
        'evaluation': eval_results,
    }

    save_json(all_results, STEP11_DIR / f"step11_confidence_{lottery_type}.json")
    return all_results


def main():
    setup_logging()
    log("=" * 60)
    log("  E0-Step11: 置信度评分系统")
    log("=" * 60)

    results = {}
    for lt in ['daletou', 'shuangseqiu']:
        with Timer(f"Step11 [{lt}]"):
            results[lt] = run_step11(lt)

    # 保存汇总
    summary = {}
    for lt, r in results.items():
        if r:
            summary[lt] = {
                'confidence_effective': r['evaluation']['confidence_effective'],
                'high_low_lift': r['evaluation']['high_low_lift'],
                'spearman_corr': r['evaluation']['spearman_correlation'],
            }
    save_json(summary, STEP11_DIR / "step11_summary.json")

    log("\n  Step11 全部完成!")


if __name__ == '__main__':
    main()
