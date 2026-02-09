# -*- coding: utf-8 -*-
"""E0-Step10c：组合精排评估

利用 Step10b 的幅度预测结果，对候选组合进行精排：
  1. 加载 E1 的候选集生成逻辑
  2. 用幅度预测缩小每位置的候选号码范围
  3. 评估精排后的组合数缩减比例和命中率

评估指标：
  - 候选集缩减比例（精排后/精排前）
  - 命中率（正确号码被保留的比例）
  - Top-K 覆盖率
  - 组合数压缩比

用法: python3 -m src.research.experiment.e0_step10c_combo_rerank
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


# ============================================================
#  幅度约束候选集生成
# ============================================================

def generate_amplitude_constrained_candidates(
    data, pos, t, scan_results, confidence_level='medium'
):
    """基于幅度预测约束，生成单位置的候选号码集合

    Args:
        data: LotteryData
        pos: 位置索引
        t: 当前期索引（预测 t+1 期）
        scan_results: Step10a 扫描结果
        confidence_level: 'tight'(q25-q75), 'medium'(q10-q90), 'loose'(q05-q95)

    Returns:
        candidates: set of candidate numbers
        info: dict with debug info
    """
    series = data.position_series[pos]
    diff_series = np.diff(series.astype(np.float64))
    dir_series = data.direction_series[pos]

    current_value = int(series[t])

    # 预计算边界
    train_60pct = int(data.n_draws * 0.6) - 1
    train_abs = np.abs(diff_series[:train_60pct])
    amp_bin_edges = np.percentile(train_abs, np.linspace(0, 100, 11))
    amp_5_edges = np.percentile(train_abs, np.linspace(0, 100, 6))
    zone_edges = np.linspace(1, data.red_range, 6)

    # 无条件幅度分布
    uncond_q10 = float(np.percentile(train_abs, 10))
    uncond_q25 = float(np.percentile(train_abs, 25))
    uncond_q50 = float(np.percentile(train_abs, 50))
    uncond_q75 = float(np.percentile(train_abs, 75))
    uncond_q90 = float(np.percentile(train_abs, 90))

    # 查找匹配的显著模式
    pos_key = f'P{pos}'
    patterns = []
    if pos_key in scan_results.get('positions', {}):
        for p in scan_results['positions'][pos_key].get('patterns', []):
            if p.get('validation', {}).get('status') == 'valid':
                patterns.append(p)

    matched_patterns = []
    for pat in patterns:
        dim = pat['dimension']
        cond = pat['condition']
        label = _get_label_at_t(
            dim, t, diff_series, dir_series, series, data,
            amp_bin_edges, amp_5_edges, zone_edges)
        if label is not None and str(label) == cond:
            matched_patterns.append(pat)

    # 确定幅度范围
    if matched_patterns:
        # 用匹配模式的统计量加权
        weights = [1.0 / (pat['ks_p'] + 1e-10) for pat in matched_patterns]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # 加权分位数
        if confidence_level == 'tight':
            q_low_key, q_high_key = 'q25', 'q75'
        elif confidence_level == 'medium':
            q_low_key, q_high_key = 'q10', 'q90'
        else:
            q_low_key, q_high_key = 'q10', 'q90'

        amp_low = sum(w * pat.get(q_low_key, 0) for w, pat in zip(weights, matched_patterns))
        amp_high = sum(w * pat.get(q_high_key, 20) for w, pat in zip(weights, matched_patterns))
        amp_median = sum(w * pat['median'] for w, pat in zip(weights, matched_patterns))

        source = 'conditional'
    else:
        # 无匹配模式，使用无条件分布
        if confidence_level == 'tight':
            amp_low, amp_high = uncond_q25, uncond_q75
        elif confidence_level == 'medium':
            amp_low, amp_high = uncond_q10, uncond_q90
        else:
            amp_low, amp_high = uncond_q10, uncond_q90

        amp_median = uncond_q50
        source = 'unconditional'

    # 生成候选号码：当前值 ± 幅度范围
    amp_low_int = max(0, int(np.floor(amp_low)))
    amp_high_int = int(np.ceil(amp_high))

    candidates = set()
    for delta in range(-amp_high_int, amp_high_int + 1):
        if abs(delta) < amp_low_int:
            continue  # 排除幅度过小的（如果 tight 模式）
        new_val = current_value + delta
        if 1 <= new_val <= data.red_range:
            candidates.add(new_val)

    # 如果 tight 模式排除了太多，回退到不排除小幅度
    if len(candidates) < 3:
        candidates = set()
        for delta in range(-amp_high_int, amp_high_int + 1):
            new_val = current_value + delta
            if 1 <= new_val <= data.red_range:
                candidates.add(new_val)

    info = {
        'current_value': current_value,
        'n_matched_patterns': len(matched_patterns),
        'source': source,
        'amp_low': float(amp_low),
        'amp_high': float(amp_high),
        'amp_median': float(amp_median),
        'n_candidates': len(candidates),
        'full_range': data.red_range,
        'reduction_ratio': len(candidates) / data.red_range,
    }

    return candidates, info


def _get_label_at_t(dim, t, diff_series, dir_series, series, data,
                    amp_bin_edges, amp_5_edges, zone_edges):
    """获取时刻t的条件标签"""
    if dim.startswith('D1_dir_seq_n'):
        n = int(dim.split('n')[-1])
        return encode_direction_seq(dir_series, t, n)
    elif dim.startswith('D2_amp_bin_n'):
        n = int(dim.split('n')[-1])
        return encode_amplitude_bin(diff_series, t, n, amp_bin_edges)
    elif dim == 'D3_value_zone':
        return encode_value_zone(series[t], zone_edges)
    elif dim == 'D4_diff_sign_amp':
        return encode_diff_sign_amp(diff_series[t-1], amp_5_edges) if t >= 1 else None
    elif dim == 'D5_diff_change':
        return encode_diff_change_pattern(diff_series, t)
    elif dim == 'D6_miss_zone':
        miss = compute_miss_periods(series, t, series[t])
        return encode_miss_zone(miss)

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
    """解析交叉维度部分"""
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
#  评估：精排效果
# ============================================================

def evaluate_rerank(data, scan_results, test_start, test_end, confidence_level='medium'):
    """评估精排效果

    Returns:
        dict with evaluation metrics
    """
    series_all = data.position_series  # shape: (n_pos, n_draws)
    n_pos = data.red_count

    hit_counts = []  # 每期命中的位置数
    reduction_ratios = []  # 每期的候选集缩减比
    combo_compressions = []  # 组合数压缩比
    position_hit_rates = [[] for _ in range(n_pos)]  # 每位置的命中率

    for t in range(test_start, test_end - 1):
        # 真实下一期号码
        true_next = [int(series_all[pos][t + 1]) for pos in range(n_pos)]

        period_candidates = []
        period_reductions = []
        period_hits = 0

        for pos in range(n_pos):
            candidates, info = generate_amplitude_constrained_candidates(
                data, pos, t, scan_results, confidence_level)

            period_candidates.append(candidates)
            period_reductions.append(info['reduction_ratio'])

            hit = true_next[pos] in candidates
            if hit:
                period_hits += 1
            position_hit_rates[pos].append(1 if hit else 0)

        hit_counts.append(period_hits)
        reduction_ratios.append(np.mean(period_reductions))

        # 组合数压缩比
        combo_before = data.red_range ** n_pos  # 简化：不考虑排序约束
        combo_after = 1
        for cands in period_candidates:
            combo_after *= len(cands)
        if combo_before > 0:
            combo_compressions.append(combo_after / combo_before)

    n_periods = len(hit_counts)

    results = {
        'n_periods': n_periods,
        'confidence_level': confidence_level,
        'avg_hit_positions': float(np.mean(hit_counts)),
        'all_hit_rate': float(np.mean([h == n_pos for h in hit_counts])),
        'avg_reduction_ratio': float(np.mean(reduction_ratios)),
        'avg_combo_compression': float(np.mean(combo_compressions)) if combo_compressions else None,
        'median_combo_compression': float(np.median(combo_compressions)) if combo_compressions else None,
        'position_hit_rates': {
            f'P{pos}': float(np.mean(position_hit_rates[pos]))
            for pos in range(n_pos)
        },
    }

    # 与随机基线对比
    # 随机基线：每位置随机选同样数量的候选
    avg_cand_size = float(np.mean(reduction_ratios)) * data.red_range
    random_hit_rate = avg_cand_size / data.red_range  # 单位置随机命中率
    results['random_baseline_hit_rate'] = float(random_hit_rate)
    results['hit_rate_lift'] = float(
        np.mean([np.mean(position_hit_rates[pos]) for pos in range(n_pos)]) / random_hit_rate
    ) if random_hit_rate > 0 else 1.0

    return results


# ============================================================
#  主入口
# ============================================================

def run_step10c(lottery_type):
    """运行 Step10c 组合精排评估"""
    log(f"\n{'═'*60}")
    log(f"  E0-Step10c: 组合精排评估 [{lottery_type}]")
    log(f"{'═'*60}")

    data = LotteryData(lottery_type)
    n_draws = data.n_draws

    # 加载 Step10a 结果
    scan_path = STEP10_DIR / f"step10a_amplitude_scan_{lottery_type}.json"
    if not scan_path.exists():
        log(f"  Step10a 结果不存在，请先运行 Step10a")
        return None

    scan_results = load_json(scan_path)

    # 测试期
    test_start = int(n_draws * 0.6)
    test_end = n_draws

    log(f"  总期数: {n_draws}, 测试期: [{test_start}, {test_end})")

    all_results = {
        'lottery_type': lottery_type,
        'evaluations': {},
    }

    for level in ['tight', 'medium', 'loose']:
        with Timer(f"{level} 评估"):
            eval_result = evaluate_rerank(data, scan_results, test_start, test_end, level)

        all_results['evaluations'][level] = eval_result

        log(f"\n  [{level}] 置信度:")
        log(f"    平均命中位置: {eval_result['avg_hit_positions']:.2f}/{data.red_count}")
        log(f"    全命中率: {eval_result['all_hit_rate']:.4f}")
        log(f"    平均缩减比: {eval_result['avg_reduction_ratio']:.3f}")
        log(f"    命中率提升: {eval_result['hit_rate_lift']:.2f}x")
        if eval_result['avg_combo_compression']:
            log(f"    组合压缩比: {eval_result['avg_combo_compression']:.6f}")

        for pos_key, hr in eval_result['position_hit_rates'].items():
            log(f"    {pos_key} 命中率: {hr:.3f}")

    # 最优置信度选择
    best_level = max(
        all_results['evaluations'].keys(),
        key=lambda k: all_results['evaluations'][k]['hit_rate_lift']
    )
    all_results['best_confidence_level'] = best_level
    all_results['best_hit_rate_lift'] = all_results['evaluations'][best_level]['hit_rate_lift']

    log(f"\n  最优置信度: {best_level}, 命中率提升: {all_results['best_hit_rate_lift']:.2f}x")

    save_json(all_results, STEP10_DIR / f"step10c_combo_rerank_{lottery_type}.json")
    return all_results


def main():
    setup_logging()
    log("=" * 60)
    log("  E0-Step10c: 组合精排评估")
    log("=" * 60)

    results = {}
    for lt in ['daletou', 'shuangseqiu']:
        with Timer(f"Step10c [{lt}]"):
            results[lt] = run_step10c(lt)

    # 保存汇总
    summary = {}
    for lt, r in results.items():
        if r:
            summary[lt] = {
                'best_confidence_level': r['best_confidence_level'],
                'best_hit_rate_lift': r['best_hit_rate_lift'],
            }
    save_json(summary, STEP10_DIR / "step10c_summary.json")

    log("\n  Step10c 全部完成!")


if __name__ == '__main__':
    main()
