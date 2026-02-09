# -*- coding: utf-8 -*-
"""E0-Step10a：差分幅度条件暴力扫描

穷举所有可能的"条件→幅度分布"关系，找出哪些条件下幅度分布显著偏离无条件分布。

扫描维度：
  D1: 最近N期方向序列 (N=1~5)
  D2: 最近N期幅度区间 (N=1~3, 10分位)
  D3: 当前值在值域中的位置 (5分区)
  D4: 最近1期差分符号+幅度区间 (符号3×幅度5分位)
  D5: 最近2期差分的变化模式 (9类)
  D6: 遗漏期数区间 (5档)

交叉组合：D1×D3, D1×D4, D2×D3, D4×D6, D5×D3

筛选：KS检验 p<0.01, 支持度>=20, 幅度集中度<0.8 或均值偏移>1σ
数据切分：前60%扫描，后40%验证。

用法: python3 -m src.research.experiment.e0_step10a_amplitude_scan
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, RESULTS_DIR,
)

# === 路径 ===
STEP10_DIR = RESULTS_DIR / "e0_step10"
STEP10_DIR.mkdir(parents=True, exist_ok=True)

# === 常量 ===
MIN_SUPPORT = 20
KS_ALPHA = 0.01
CONCENTRATION_THRESHOLD = 0.8
MEAN_SHIFT_SIGMA = 1.0


# ============================================================
#  辅助函数：条件编码
# ============================================================

def encode_direction_seq(dir_series, t, n):
    """编码最近N期方向序列为元组，dir_series[t]对应第t+1期的方向"""
    if t < n:
        return None
    return tuple(int(dir_series[t - n + i]) for i in range(n))


def encode_amplitude_bin(diff_series, t, n, bin_edges):
    """编码最近N期幅度区间为元组"""
    if t < n:
        return None
    bins = []
    for i in range(n):
        idx = t - n + i
        val = abs(diff_series[idx])
        b = np.searchsorted(bin_edges, val, side='right') - 1
        b = max(0, min(b, len(bin_edges) - 2))
        bins.append(b)
    return tuple(bins)


def encode_value_zone(value, zone_edges):
    """编码当前值在值域中的位置(5分区)"""
    z = np.searchsorted(zone_edges, value, side='right') - 1
    return max(0, min(z, len(zone_edges) - 2))


def encode_diff_sign_amp(diff_val, amp_edges):
    """编码差分符号+幅度区间"""
    if diff_val > 0:
        sign = 1
    elif diff_val < 0:
        sign = -1
    else:
        sign = 0
    amp_bin = np.searchsorted(amp_edges, abs(diff_val), side='right') - 1
    amp_bin = max(0, min(amp_bin, len(amp_edges) - 2))
    return (sign, amp_bin)


def encode_diff_change_pattern(diff_series, t):
    """编码最近2期差分的变化模式(9类)
    模式: (sign_change, magnitude_change)
    sign_change: 同向/反转/含零
    magnitude_change: 加速/减速/持平
    """
    if t < 2:
        return None
    d1 = diff_series[t - 2]  # 前前期差分
    d2 = diff_series[t - 1]  # 前期差分

    # 符号变化
    if d1 * d2 > 0:
        sign_ch = 0  # 同向
    elif d1 * d2 < 0:
        sign_ch = 1  # 反转
    else:
        sign_ch = 2  # 含零

    # 幅度变化
    a1, a2 = abs(d1), abs(d2)
    if a1 == 0 and a2 == 0:
        mag_ch = 2  # 持平
    elif a2 > a1 * 1.2:
        mag_ch = 0  # 加速
    elif a2 < a1 * 0.8:
        mag_ch = 1  # 减速
    else:
        mag_ch = 2  # 持平

    return (sign_ch, mag_ch)


def compute_miss_periods(value_series, t, value):
    """计算当前值的遗漏期数（上次出现距今多少期）"""
    for lag in range(1, min(t + 1, 200)):
        if value_series[t - lag] == value:
            return lag
    return 200  # 超过200期未出现


def encode_miss_zone(miss_periods):
    """遗漏期数分5档: 1-3/4-8/9-15/16-30/31+"""
    if miss_periods <= 3:
        return 0
    elif miss_periods <= 8:
        return 1
    elif miss_periods <= 15:
        return 2
    elif miss_periods <= 30:
        return 3
    else:
        return 4


# ============================================================
#  核心：单维度扫描
# ============================================================

def scan_single_dimension(diff_abs, condition_labels, uncond_iqr, uncond_mean, uncond_std, dim_name):
    """对单个维度的条件进行扫描

    Args:
        diff_abs: 下一期差分绝对值数组
        condition_labels: 与diff_abs等长的条件标签数组（可以是元组）
        uncond_iqr: 无条件IQR
        uncond_mean: 无条件均值
        uncond_std: 无条件标准差
        dim_name: 维度名称

    Returns:
        list of significant patterns
    """
    # 按条件分组
    groups = {}
    for i, label in enumerate(condition_labels):
        if label is None:
            continue
        if label not in groups:
            groups[label] = []
        groups[label].append(diff_abs[i])

    patterns = []
    for label, values in groups.items():
        if len(values) < MIN_SUPPORT:
            continue

        values = np.array(values)
        # KS检验
        ks_stat, ks_p = sp_stats.ks_2samp(values, diff_abs)

        if ks_p >= KS_ALPHA:
            continue

        # 统计量
        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        std_val = float(np.std(values))
        q25, q75 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
        iqr = q75 - q25
        concentration = iqr / uncond_iqr if uncond_iqr > 0 else 1.0
        mean_shift = abs(mean_val - uncond_mean) / uncond_std if uncond_std > 0 else 0.0

        # JS散度
        hist_cond, edges = np.histogram(values, bins=20, density=True)
        hist_uncond, _ = np.histogram(diff_abs, bins=edges, density=True)
        # 避免零值
        eps = 1e-10
        p = hist_cond + eps
        q = hist_uncond + eps
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        js_div = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

        # 筛选：集中度<0.8 或 均值偏移>1σ
        if concentration >= CONCENTRATION_THRESHOLD and mean_shift <= MEAN_SHIFT_SIGMA:
            continue

        patterns.append({
            'dimension': dim_name,
            'condition': str(label),
            'support': len(values),
            'ks_stat': float(ks_stat),
            'ks_p': float(ks_p),
            'js_divergence': js_div,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'q10': float(np.percentile(values, 10)),
            'q25': q25,
            'q50': median_val,
            'q75': q75,
            'q90': float(np.percentile(values, 90)),
            'iqr': iqr,
            'concentration': concentration,
            'mean_shift_sigma': mean_shift,
        })

    return patterns


# ============================================================
#  核心：交叉维度扫描
# ============================================================

def scan_cross_dimension(diff_abs, labels_a, labels_b, uncond_iqr, uncond_mean, uncond_std, dim_name):
    """对两个维度的交叉条件进行扫描"""
    # 构造交叉标签
    cross_labels = []
    for i in range(len(diff_abs)):
        if labels_a[i] is None or labels_b[i] is None:
            cross_labels.append(None)
        else:
            cross_labels.append((labels_a[i], labels_b[i]))

    return scan_single_dimension(diff_abs, cross_labels, uncond_iqr, uncond_mean, uncond_std, dim_name)


# ============================================================
#  主扫描函数：单位置
# ============================================================

def scan_position(data, pos, train_end_idx, test_start_idx):
    """对单个位置进行全维度暴力扫描

    Returns:
        train_patterns: 训练期发现的显著模式
        validation: 验证期结果
    """
    series = data.position_series[pos]
    diff_series = np.diff(series.astype(np.float64))  # diff_series[t] = series[t+1] - series[t]
    dir_series = data.direction_series[pos]  # dir_series[t] 对应 diff_series[t] 的方向

    # 训练期范围：diff_series 索引 [0, train_end_idx-2]
    # 我们要预测 diff_series[t]，条件基于 t 之前的信息
    # 所以样本范围：t from some_min to train_end_idx-2
    train_diff_end = train_end_idx - 1  # diff_series 的训练期结束索引（不含）
    test_diff_start = test_start_idx - 1  # diff_series 的测试期开始索引

    # 训练期的差分绝对值（作为无条件分布）
    train_abs_diffs = np.abs(diff_series[:train_diff_end])
    uncond_mean = float(np.mean(train_abs_diffs))
    uncond_std = float(np.std(train_abs_diffs))
    q25_unc = float(np.percentile(train_abs_diffs, 25))
    q75_unc = float(np.percentile(train_abs_diffs, 75))
    uncond_iqr = q75_unc - q25_unc

    # 预计算分位数边界（基于训练期）
    amp_bin_edges = np.percentile(train_abs_diffs, np.linspace(0, 100, 11))  # 10分位
    amp_5_edges = np.percentile(train_abs_diffs, np.linspace(0, 100, 6))  # 5分位
    zone_edges = np.linspace(1, data.red_range, 6)  # 值域5分区

    log(f"    P{pos}: 训练期差分 {len(train_abs_diffs)} 个, "
        f"均值={uncond_mean:.2f}, std={uncond_std:.2f}, IQR={uncond_iqr:.2f}")

    # === 构造条件标签（训练期） ===
    # 对于每个 t（diff_series 的索引），条件基于 t 之前的信息
    # 预测目标是 diff_series[t] 的绝对值
    n_train = train_diff_end
    target_abs = np.abs(diff_series[:n_train])

    # D1: 方向序列
    d1_labels = {n: [] for n in range(1, 6)}
    for t in range(n_train):
        for n in range(1, 6):
            d1_labels[n].append(encode_direction_seq(dir_series, t, n))

    # D2: 幅度区间序列
    d2_labels = {n: [] for n in range(1, 4)}
    for t in range(n_train):
        for n in range(1, 4):
            d2_labels[n].append(encode_amplitude_bin(diff_series, t, n, amp_bin_edges))

    # D3: 值域位置
    d3_labels = []
    for t in range(n_train):
        # series[t] 是当前值（diff_series[t] = series[t+1] - series[t]）
        d3_labels.append(encode_value_zone(series[t], zone_edges))

    # D4: 上期差分符号+幅度
    d4_labels = []
    for t in range(n_train):
        if t < 1:
            d4_labels.append(None)
        else:
            d4_labels.append(encode_diff_sign_amp(diff_series[t - 1], amp_5_edges))

    # D5: 差分变化模式
    d5_labels = []
    for t in range(n_train):
        d5_labels.append(encode_diff_change_pattern(diff_series, t))

    # D6: 遗漏期数
    d6_labels = []
    for t in range(n_train):
        # 当前值 series[t] 的遗漏期数
        miss = compute_miss_periods(series, t, series[t])
        d6_labels.append(encode_miss_zone(miss))

    # === 单维度扫描 ===
    all_patterns = []

    # D1
    for n in range(1, 6):
        pats = scan_single_dimension(
            target_abs, d1_labels[n], uncond_iqr, uncond_mean, uncond_std, f'D1_dir_seq_n{n}')
        all_patterns.extend(pats)
    log(f"    P{pos} D1完成: {sum(1 for p in all_patterns if p['dimension'].startswith('D1'))} 个显著模式")

    # D2
    for n in range(1, 4):
        pats = scan_single_dimension(
            target_abs, d2_labels[n], uncond_iqr, uncond_mean, uncond_std, f'D2_amp_bin_n{n}')
        all_patterns.extend(pats)
    log(f"    P{pos} D2完成: {sum(1 for p in all_patterns if p['dimension'].startswith('D2'))} 个显著模式")

    # D3
    pats = scan_single_dimension(
        target_abs, d3_labels, uncond_iqr, uncond_mean, uncond_std, 'D3_value_zone')
    all_patterns.extend(pats)

    # D4
    pats = scan_single_dimension(
        target_abs, d4_labels, uncond_iqr, uncond_mean, uncond_std, 'D4_diff_sign_amp')
    all_patterns.extend(pats)

    # D5
    pats = scan_single_dimension(
        target_abs, d5_labels, uncond_iqr, uncond_mean, uncond_std, 'D5_diff_change')
    all_patterns.extend(pats)

    # D6
    pats = scan_single_dimension(
        target_abs, d6_labels, uncond_iqr, uncond_mean, uncond_std, 'D6_miss_zone')
    all_patterns.extend(pats)

    log(f"    P{pos} 单维度合计: {len(all_patterns)} 个显著模式")

    # === 交叉维度扫描 ===
    cross_patterns = []

    # D1×D3
    for n in range(1, 6):
        pats = scan_cross_dimension(
            target_abs, d1_labels[n], d3_labels, uncond_iqr, uncond_mean, uncond_std,
            f'D1n{n}_x_D3')
        cross_patterns.extend(pats)

    # D1×D4
    for n in range(1, 6):
        pats = scan_cross_dimension(
            target_abs, d1_labels[n], d4_labels, uncond_iqr, uncond_mean, uncond_std,
            f'D1n{n}_x_D4')
        cross_patterns.extend(pats)

    # D2×D3
    for n in range(1, 4):
        pats = scan_cross_dimension(
            target_abs, d2_labels[n], d3_labels, uncond_iqr, uncond_mean, uncond_std,
            f'D2n{n}_x_D3')
        cross_patterns.extend(pats)

    # D4×D6
    pats = scan_cross_dimension(
        target_abs, d4_labels, d6_labels, uncond_iqr, uncond_mean, uncond_std, 'D4_x_D6')
    cross_patterns.extend(pats)

    # D5×D3
    pats = scan_cross_dimension(
        target_abs, d5_labels, d3_labels, uncond_iqr, uncond_mean, uncond_std, 'D5_x_D3')
    cross_patterns.extend(pats)

    log(f"    P{pos} 交叉维度: {len(cross_patterns)} 个显著模式")

    all_patterns.extend(cross_patterns)

    # === 验证期验证 ===
    test_abs_diffs = np.abs(diff_series[test_diff_start:])
    n_test = len(test_abs_diffs)

    # 重新构造测试期条件标签
    validated = []
    n_validated = 0

    for pat in all_patterns:
        dim = pat['dimension']
        cond = pat['condition']

        # 重新构造测试期标签并筛选匹配样本
        test_values = _collect_test_values(
            dim, cond, diff_series, dir_series, series, data,
            test_diff_start, len(diff_series),
            amp_bin_edges, amp_5_edges, zone_edges)

        if len(test_values) < 10:
            pat['validation'] = {'status': 'insufficient', 'n_test': len(test_values)}
            validated.append(pat)
            continue

        # 验证期KS检验
        ks_stat, ks_p = sp_stats.ks_2samp(test_values, test_abs_diffs)
        test_mean = float(np.mean(test_values))
        test_median = float(np.median(test_values))

        # 验证通过条件：KS p < 0.05（放宽一点）
        is_valid = ks_p < 0.05

        pat['validation'] = {
            'status': 'valid' if is_valid else 'invalid',
            'n_test': len(test_values),
            'ks_stat': float(ks_stat),
            'ks_p': float(ks_p),
            'test_mean': test_mean,
            'test_median': test_median,
            'train_test_mean_diff': abs(test_mean - pat['mean']),
        }

        if is_valid:
            n_validated += 1

        validated.append(pat)

    log(f"    P{pos} 验证通过: {n_validated}/{len(all_patterns)}")

    return validated


def _collect_test_values(dim, cond_str, diff_series, dir_series, series, data,
                         test_start, test_end, amp_bin_edges, amp_5_edges, zone_edges):
    """收集测试期中满足条件的差分绝对值"""
    values = []

    for t in range(test_start, test_end):
        label = _get_label_for_dim(
            dim, t, diff_series, dir_series, series, data,
            amp_bin_edges, amp_5_edges, zone_edges)

        if label is not None and str(label) == cond_str:
            values.append(abs(diff_series[t]))

    return np.array(values) if values else np.array([])


def _get_label_for_dim(dim, t, diff_series, dir_series, series, data,
                       amp_bin_edges, amp_5_edges, zone_edges):
    """根据维度名获取时刻t的条件标签"""
    # 单维度
    if dim.startswith('D1_dir_seq_n'):
        n = int(dim.split('n')[-1])
        return encode_direction_seq(dir_series, t, n)
    elif dim.startswith('D2_amp_bin_n'):
        n = int(dim.split('n')[-1])
        return encode_amplitude_bin(diff_series, t, n, amp_bin_edges)
    elif dim == 'D3_value_zone':
        return encode_value_zone(series[t], zone_edges)
    elif dim == 'D4_diff_sign_amp':
        if t < 1:
            return None
        return encode_diff_sign_amp(diff_series[t - 1], amp_5_edges)
    elif dim == 'D5_diff_change':
        return encode_diff_change_pattern(diff_series, t)
    elif dim == 'D6_miss_zone':
        miss = compute_miss_periods(series, t, series[t])
        return encode_miss_zone(miss)

    # 交叉维度
    if '_x_' in dim:
        parts = dim.split('_x_')
        dim_a_str, dim_b_str = parts[0], parts[1]

        # 解析维度A
        label_a = _parse_cross_label(dim_a_str, t, diff_series, dir_series, series, data,
                                     amp_bin_edges, amp_5_edges, zone_edges)
        # 解析维度B
        label_b = _parse_cross_label(dim_b_str, t, diff_series, dir_series, series, data,
                                     amp_bin_edges, amp_5_edges, zone_edges)

        if label_a is None or label_b is None:
            return None
        return (label_a, label_b)

    return None


def _parse_cross_label(dim_part, t, diff_series, dir_series, series, data,
                       amp_bin_edges, amp_5_edges, zone_edges):
    """解析交叉维度中的单个维度标签"""
    if dim_part.startswith('D1n'):
        n = int(dim_part[3:])
        return encode_direction_seq(dir_series, t, n)
    elif dim_part.startswith('D2n'):
        n = int(dim_part[3:])
        return encode_amplitude_bin(diff_series, t, n, amp_bin_edges)
    elif dim_part == 'D3':
        return encode_value_zone(series[t], zone_edges)
    elif dim_part == 'D4':
        if t < 1:
            return None
        return encode_diff_sign_amp(diff_series[t - 1], amp_5_edges)
    elif dim_part == 'D5':
        return encode_diff_change_pattern(diff_series, t)
    elif dim_part == 'D6':
        miss = compute_miss_periods(series, t, series[t])
        return encode_miss_zone(miss)
    return None


# ============================================================
#  主入口
# ============================================================

def run_step10a(lottery_type):
    """运行 Step10a 幅度暴力扫描"""
    log(f"\n{'═'*60}")
    log(f"  E0-Step10a: 差分幅度条件暴力扫描 [{lottery_type}]")
    log(f"{'═'*60}")

    data = LotteryData(lottery_type)
    n_draws = data.n_draws

    # 时间切分：前60%扫描，后40%验证
    train_end_idx = int(n_draws * 0.6)
    test_start_idx = train_end_idx

    log(f"  总期数: {n_draws}, 训练期: [0, {train_end_idx}), 测试期: [{test_start_idx}, {n_draws})")
    log(f"  红球位置数: {data.red_count}, 红球范围: 1-{data.red_range}")

    all_results = {
        'lottery_type': lottery_type,
        'n_draws': n_draws,
        'train_end': train_end_idx,
        'test_start': test_start_idx,
        'positions': {},
        'summary': {},
    }

    total_patterns = 0
    total_validated = 0

    for pos in range(data.red_count):
        with Timer(f"P{pos} 扫描"):
            patterns = scan_position(data, pos, train_end_idx, test_start_idx)

        n_valid = sum(1 for p in patterns if p.get('validation', {}).get('status') == 'valid')
        total_patterns += len(patterns)
        total_validated += n_valid

        # 按JS散度排序，保留top模式
        patterns.sort(key=lambda p: p.get('js_divergence', 0), reverse=True)

        all_results['positions'][f'P{pos}'] = {
            'n_patterns': len(patterns),
            'n_validated': n_valid,
            'patterns': patterns[:200],  # 每位置保留top200
        }

        log(f"  P{pos}: {len(patterns)} 个显著模式, {n_valid} 个验证通过")

    # 汇总
    all_results['summary'] = {
        'total_patterns': total_patterns,
        'total_validated': total_validated,
        'validation_rate': total_validated / total_patterns if total_patterns > 0 else 0,
    }

    log(f"\n  汇总: {total_patterns} 个显著模式, {total_validated} 个验证通过 "
        f"({all_results['summary']['validation_rate']:.1%})")

    save_json(all_results, STEP10_DIR / f"step10a_amplitude_scan_{lottery_type}.json")
    return all_results


def main():
    setup_logging()
    log("=" * 60)
    log("  E0-Step10a: 差分幅度条件暴力扫描")
    log("=" * 60)

    results = {}
    for lt in ['daletou', 'shuangseqiu']:
        with Timer(f"Step10a [{lt}]"):
            results[lt] = run_step10a(lt)

    # 保存汇总
    summary = {
        'daletou': results['daletou']['summary'],
        'shuangseqiu': results['shuangseqiu']['summary'],
    }
    save_json(summary, STEP10_DIR / "step10a_summary.json")

    log("\n  Step10a 全部完成!")
    for lt, r in results.items():
        log(f"  {lt}: {r['summary']['total_patterns']} 模式, "
            f"{r['summary']['total_validated']} 验证通过")


if __name__ == '__main__':
    main()
