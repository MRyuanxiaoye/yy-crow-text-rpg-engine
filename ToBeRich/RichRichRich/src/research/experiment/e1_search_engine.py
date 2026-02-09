# -*- coding: utf-8 -*-
"""E1 Layer 1: 条件空间系统化搜索引擎

定义 D1-D12 条件编码函数 + 统一扫描接口。
每个原子任务 = (彩种, 位置, 维度组合)，完成后立即保存。

复用 Step10a 的 D1-D6 编码 + KS/JS 评估逻辑，新增 D7-D12。

用法: 不直接运行，由 e1_scheduler.py 调用。
"""

import numpy as np
from scipy import stats as sp_stats

from research.data_loader import LotteryData

# === 常量 ===
MIN_SUPPORT = 20          # 单/二维最小支持度
MIN_SUPPORT_3D = 30       # 三维交叉最小支持度
KS_ALPHA = 0.01           # 训练期 KS 检验阈值
KS_ALPHA_TEST = 0.05      # 测试期 KS 检验阈值（放宽）
CONCENTRATION_THRESHOLD = 0.8
MEAN_SHIFT_SIGMA = 1.0


# ============================================================
#  D1-D6 编码函数（复用 Step10a）
# ============================================================

def encode_direction_seq(dir_series, t, n):
    """D1: 最近N期方向序列"""
    if t < n:
        return None
    return tuple(int(dir_series[t - n + i]) for i in range(n))


def encode_amplitude_bin(diff_series, t, n, bin_edges):
    """D2: 最近N期幅度区间"""
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
    """D3: 当前值在值域中的位置(5分区)"""
    z = np.searchsorted(zone_edges, value, side='right') - 1
    return max(0, min(z, len(zone_edges) - 2))


def encode_diff_sign_amp(diff_val, amp_edges):
    """D4: 差分符号+幅度区间"""
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
    """D5: 最近2期差分变化模式(9类)"""
    if t < 2:
        return None
    d1 = diff_series[t - 2]
    d2 = diff_series[t - 1]
    if d1 * d2 > 0:
        sign_ch = 0  # 同向
    elif d1 * d2 < 0:
        sign_ch = 1  # 反转
    else:
        sign_ch = 2  # 含零
    a1, a2 = abs(d1), abs(d2)
    if a1 == 0 and a2 == 0:
        mag_ch = 2
    elif a2 > a1 * 1.2:
        mag_ch = 0  # 加速
    elif a2 < a1 * 0.8:
        mag_ch = 1  # 减速
    else:
        mag_ch = 2  # 持平
    return (sign_ch, mag_ch)


def compute_miss_periods(value_series, t, value):
    """计算当前值的遗漏期数"""
    for lag in range(1, min(t + 1, 200)):
        if value_series[t - lag] == value:
            return lag
    return 200


def encode_miss_zone(miss_periods):
    """D6: 遗漏期数分5档"""
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
#  D7-D12 新增编码函数
# ============================================================

def encode_cross_pos_direction(dir_series_dict, pos, t):
    """D7: 跨位置方向联合 — 当前位置与相邻位置的方向组合
    返回 (dir_prev_pos, dir_cur_pos) 或 (dir_cur_pos, dir_next_pos)
    对每个位置，取与前一个位置的联合（pos>0时）
    """
    if pos == 0:
        return None  # P0 没有前一个位置
    if t < 0 or t >= len(dir_series_dict[pos]):
        return None
    d_cur = int(dir_series_dict[pos][t])
    d_prev = int(dir_series_dict[pos - 1][t])
    return (d_prev, d_cur)


def encode_combo_stat_zone(combo_stats, stat_name, t, edges):
    """D8: 组合统计量区间 — sum/span/ac_value 的5分位"""
    if stat_name not in combo_stats:
        return None
    arr = combo_stats[stat_name]
    if t >= len(arr):
        return None
    z = np.searchsorted(edges, arr[t], side='right') - 1
    return max(0, min(z, len(edges) - 2))


def encode_odd_big_pattern(red_matrix, t, red_count, red_range):
    """D9: 号码奇偶/大小格局 — 奇数个数×大数个数的联合编码"""
    if t >= len(red_matrix):
        return None
    row = red_matrix[t]
    odd_count = int(np.sum(row % 2 == 1))
    mid = (1 + red_range) / 2
    big_count = int(np.sum(row > mid))
    return (odd_count, big_count)


def encode_long_trend(series, t, window):
    """D10: 长期趋势 — 最近 window 期的均值偏移方向
    返回: 1(上升趋势), -1(下降趋势), 0(平稳)
    """
    if t < window:
        return None
    recent = series[t - window + 1:t + 1].astype(np.float64)
    earlier = series[max(0, t - 2 * window + 1):t - window + 1].astype(np.float64)
    if len(earlier) < window // 2:
        return None
    diff = np.mean(recent) - np.mean(earlier)
    std = np.std(series[:t + 1].astype(np.float64))
    if std == 0:
        return 0
    # 偏移超过 0.3σ 视为有趋势
    if diff > 0.3 * std:
        return 1
    elif diff < -0.3 * std:
        return -1
    else:
        return 0


def encode_consec_status(red_matrix, t, pos):
    """D11: 连号状态 — 当前位置是否参与连号
    返回: 0(不连号), 1(与前一位连号), 2(与后一位连号), 3(两侧都连号)
    """
    if t >= len(red_matrix):
        return None
    row = red_matrix[t]
    val = row[pos]
    left = (pos > 0 and row[pos - 1] == val - 1)
    right = (pos < len(row) - 1 and row[pos + 1] == val + 1)
    if left and right:
        return 3
    elif left:
        return 1
    elif right:
        return 2
    else:
        return 0


def encode_hot_cold(value_series, t, value, window=10):
    """D12: 冷热度 — 最近 window 期该号码出现次数分4档
    0: 冷(0次), 1: 温(1次), 2: 热(2-3次), 3: 极热(4+次)
    """
    if t < 1:
        return 0  # 第一期默认冷
    start = max(0, t - window)
    count = int(np.sum(value_series[start:t] == value))
    if count == 0:
        return 0
    elif count == 1:
        return 1
    elif count <= 3:
        return 2
    else:
        return 3


# ============================================================
#  核心扫描函数
# ============================================================

def scan_single_dimension(diff_abs, condition_labels, uncond_iqr, uncond_mean,
                          uncond_std, dim_name, min_support=MIN_SUPPORT):
    """对单个维度的条件进行扫描，返回显著模式列表"""
    groups = {}
    for i, label in enumerate(condition_labels):
        if label is None:
            continue
        if label not in groups:
            groups[label] = []
        groups[label].append(diff_abs[i])

    patterns = []
    for label, values in groups.items():
        if len(values) < min_support:
            continue
        values = np.array(values)
        ks_stat, ks_p = sp_stats.ks_2samp(values, diff_abs)
        if ks_p >= KS_ALPHA:
            continue

        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        std_val = float(np.std(values))
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        iqr = q75 - q25
        concentration = iqr / uncond_iqr if uncond_iqr > 0 else 1.0
        mean_shift = abs(mean_val - uncond_mean) / uncond_std if uncond_std > 0 else 0.0

        # JS 散度
        hist_cond, edges = np.histogram(values, bins=20, density=True)
        hist_uncond, _ = np.histogram(diff_abs, bins=edges, density=True)
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


def scan_cross_dimension(diff_abs, labels_a, labels_b, uncond_iqr, uncond_mean,
                         uncond_std, dim_name, min_support=MIN_SUPPORT):
    """对两个维度的交叉条件进行扫描"""
    cross_labels = []
    for i in range(len(diff_abs)):
        if labels_a[i] is None or labels_b[i] is None:
            cross_labels.append(None)
        else:
            cross_labels.append((labels_a[i], labels_b[i]))
    return scan_single_dimension(diff_abs, cross_labels, uncond_iqr, uncond_mean,
                                 uncond_std, dim_name, min_support)


def scan_triple_dimension(diff_abs, labels_a, labels_b, labels_c,
                          uncond_iqr, uncond_mean, uncond_std, dim_name):
    """对三个维度的交叉条件进行扫描"""
    cross_labels = []
    for i in range(len(diff_abs)):
        if labels_a[i] is None or labels_b[i] is None or labels_c[i] is None:
            cross_labels.append(None)
        else:
            cross_labels.append((labels_a[i], labels_b[i], labels_c[i]))
    return scan_single_dimension(diff_abs, cross_labels, uncond_iqr, uncond_mean,
                                 uncond_std, dim_name, MIN_SUPPORT_3D)


# ============================================================
#  验证函数
# ============================================================

def validate_patterns(patterns, diff_abs_test, all_test_labels, dim_name_to_labels):
    """在测试期验证训练期发现的模式

    Args:
        patterns: 训练期发现的模式列表
        diff_abs_test: 测试期差分绝对值
        all_test_labels: dict, dim_name -> 测试期标签列表
        dim_name_to_labels: 同 all_test_labels（兼容）

    Returns:
        带验证结果的模式列表
    """
    validated = []
    for pat in patterns:
        dim = pat['dimension']
        cond_str = pat['condition']

        test_labels = dim_name_to_labels.get(dim, [])
        if not test_labels:
            pat['validation'] = {'status': 'no_test_labels'}
            validated.append(pat)
            continue

        # 收集匹配样本
        test_values = []
        for i, label in enumerate(test_labels):
            if label is not None and str(label) == cond_str:
                if i < len(diff_abs_test):
                    test_values.append(diff_abs_test[i])

        if len(test_values) < 10:
            pat['validation'] = {'status': 'insufficient', 'n_test': len(test_values)}
            validated.append(pat)
            continue

        test_values = np.array(test_values)
        ks_stat, ks_p = sp_stats.ks_2samp(test_values, diff_abs_test)
        test_mean = float(np.mean(test_values))
        is_valid = ks_p < KS_ALPHA_TEST

        pat['validation'] = {
            'status': 'valid' if is_valid else 'invalid',
            'n_test': len(test_values),
            'ks_stat': float(ks_stat),
            'ks_p': float(ks_p),
            'test_mean': test_mean,
            'test_median': float(np.median(test_values)),
            'train_test_mean_diff': abs(test_mean - pat['mean']),
        }
        validated.append(pat)

    return validated


# ============================================================
#  预计算边界
# ============================================================

class PrecomputedEdges:
    """预计算的分位数边界，避免重复计算"""

    def __init__(self, data, pos, train_diff_end):
        series = data.position_series[pos]
        diff_series = np.diff(series.astype(np.float64))
        train_abs = np.abs(diff_series[:train_diff_end])

        self.amp_bin_edges = np.percentile(train_abs, np.linspace(0, 100, 11))  # 10分位
        self.amp_5_edges = np.percentile(train_abs, np.linspace(0, 100, 6))     # 5分位
        self.zone_edges = np.linspace(1, data.red_range, 6)                     # 值域5分区

        self.uncond_mean = float(np.mean(train_abs))
        self.uncond_std = float(np.std(train_abs))
        q25 = float(np.percentile(train_abs, 25))
        q75 = float(np.percentile(train_abs, 75))
        self.uncond_iqr = q75 - q25

        # 组合统计量边界（基于训练期）
        combo_stats = data.get_combo_stats_series()
        self.combo_edges = {}
        for stat_name, arr in combo_stats.items():
            train_arr = arr[:train_diff_end + 1]  # +1 因为 combo_stats 对齐原始期数
            self.combo_edges[stat_name] = np.percentile(train_arr, np.linspace(0, 100, 6))


# ============================================================
#  标签构造器
# ============================================================

class LabelBuilder:
    """为指定位置构造所有维度的条件标签"""

    def __init__(self, data, pos, start_idx, end_idx, edges):
        """
        Args:
            data: LotteryData
            pos: 位置索引
            start_idx: diff_series 起始索引
            end_idx: diff_series 结束索引（不含）
            edges: PrecomputedEdges
        """
        self.data = data
        self.pos = pos
        self.edges = edges

        series = data.position_series[pos]
        diff_series = np.diff(series.astype(np.float64))
        dir_series = data.direction_series[pos]

        n_samples = end_idx - start_idx
        self.target_abs = np.abs(diff_series[start_idx:end_idx])

        # 存储所有维度的标签
        self.labels = {}

        # --- D1: 方向序列 ---
        for n in range(1, 6):
            key = f'D1_n{n}'
            labs = []
            for t in range(start_idx, end_idx):
                labs.append(encode_direction_seq(dir_series, t, n))
            self.labels[key] = labs

        # --- D2: 幅度区间 ---
        for n in range(1, 4):
            key = f'D2_n{n}'
            labs = []
            for t in range(start_idx, end_idx):
                labs.append(encode_amplitude_bin(diff_series, t, n, edges.amp_bin_edges))
            self.labels[key] = labs

        # --- D3: 值域位置 ---
        labs = []
        for t in range(start_idx, end_idx):
            labs.append(encode_value_zone(series[t], edges.zone_edges))
        self.labels['D3'] = labs

        # --- D4: 差分符号+幅度 ---
        labs = []
        for t in range(start_idx, end_idx):
            if t < 1:
                labs.append(None)
            else:
                labs.append(encode_diff_sign_amp(diff_series[t - 1], edges.amp_5_edges))
        self.labels['D4'] = labs

        # --- D5: 差分变化模式 ---
        labs = []
        for t in range(start_idx, end_idx):
            labs.append(encode_diff_change_pattern(diff_series, t))
        self.labels['D5'] = labs

        # --- D6: 遗漏期数 ---
        labs = []
        for t in range(start_idx, end_idx):
            miss = compute_miss_periods(series, t, series[t])
            labs.append(encode_miss_zone(miss))
        self.labels['D6'] = labs

        # --- D7: 跨位置方向联合 ---
        labs = []
        for t in range(start_idx, end_idx):
            labs.append(encode_cross_pos_direction(data.direction_series, pos, t))
        self.labels['D7'] = labs

        # --- D8: 组合统计量区间 ---
        combo_stats = data.get_combo_stats_series()
        for stat_name in ['sum', 'span', 'ac_value']:
            key = f'D8_{stat_name}'
            stat_edges = edges.combo_edges.get(stat_name)
            if stat_edges is None:
                self.labels[key] = [None] * n_samples
                continue
            labs = []
            for t in range(start_idx, end_idx):
                labs.append(encode_combo_stat_zone(combo_stats, stat_name, t, stat_edges))
            self.labels[key] = labs

        # --- D9: 奇偶/大小格局 ---
        labs = []
        for t in range(start_idx, end_idx):
            labs.append(encode_odd_big_pattern(data.red_matrix, t, data.red_count, data.red_range))
        self.labels['D9'] = labs

        # --- D10: 长期趋势 ---
        for window in [10, 20]:
            key = f'D10_w{window}'
            labs = []
            for t in range(start_idx, end_idx):
                labs.append(encode_long_trend(series, t, window))
            self.labels[key] = labs

        # --- D11: 连号状态 ---
        labs = []
        for t in range(start_idx, end_idx):
            labs.append(encode_consec_status(data.red_matrix, t, pos))
        self.labels['D11'] = labs

        # --- D12: 冷热度 ---
        labs = []
        for t in range(start_idx, end_idx):
            labs.append(encode_hot_cold(series, t, series[t], window=10))
        self.labels['D12'] = labs


# ============================================================
#  维度名映射（用于任务定义）
# ============================================================

# 单维度的所有子键
SINGLE_DIM_KEYS = {
    'D1': [f'D1_n{n}' for n in range(1, 6)],
    'D2': [f'D2_n{n}' for n in range(1, 4)],
    'D3': ['D3'],
    'D4': ['D4'],
    'D5': ['D5'],
    'D6': ['D6'],
    'D7': ['D7'],
    'D8': [f'D8_{s}' for s in ['sum', 'span', 'ac_value']],
    'D9': ['D9'],
    'D10': [f'D10_w{w}' for w in [10, 20]],
    'D11': ['D11'],
    'D12': ['D12'],
}


# ============================================================
#  原子任务执行函数
# ============================================================

def run_single_task(data, pos, dimensions, train_end_idx):
    """执行单个原子任务：对指定位置和维度组合进行扫描+验证

    Args:
        data: LotteryData
        pos: 位置索引
        dimensions: 维度列表，如 ['D1'] 或 ['D1', 'D3']
        train_end_idx: 训练期结束索引（原始期数）

    Returns:
        dict: 任务结果
    """
    n_draws = data.n_draws
    train_diff_end = train_end_idx - 1
    test_diff_start = train_end_idx - 1

    # 预计算边界
    edges = PrecomputedEdges(data, pos, train_diff_end)

    # 构造训练期标签
    train_builder = LabelBuilder(data, pos, 0, train_diff_end, edges)
    target_abs_train = train_builder.target_abs

    # 构造测试期标签
    diff_series = np.diff(data.position_series[pos].astype(np.float64))
    test_builder = LabelBuilder(data, pos, test_diff_start, len(diff_series), edges)
    target_abs_test = test_builder.target_abs

    # 根据维度数量选择扫描方式
    if len(dimensions) == 1:
        patterns = _scan_single_dim_task(
            target_abs_train, train_builder, dimensions[0],
            edges.uncond_iqr, edges.uncond_mean, edges.uncond_std)
    elif len(dimensions) == 2:
        patterns = _scan_cross_dim_task(
            target_abs_train, train_builder, dimensions[0], dimensions[1],
            edges.uncond_iqr, edges.uncond_mean, edges.uncond_std)
    elif len(dimensions) == 3:
        patterns = _scan_triple_dim_task(
            target_abs_train, train_builder,
            dimensions[0], dimensions[1], dimensions[2],
            edges.uncond_iqr, edges.uncond_mean, edges.uncond_std)
    else:
        patterns = []

    # 为交叉维度构造组合标签（修复：test_builder.labels 只有单维度 key）
    test_labels = dict(test_builder.labels)
    _build_cross_labels(test_labels, patterns)

    # 验证
    validated = validate_patterns(patterns, target_abs_test, None, test_labels)

    n_valid = sum(1 for p in validated if p.get('validation', {}).get('status') == 'valid')

    # 按 JS 散度排序
    validated.sort(key=lambda p: p.get('js_divergence', 0), reverse=True)

    return {
        'status': 'completed',
        'position': pos,
        'dimensions': dimensions,
        'n_patterns_found': len(patterns),
        'n_validated': n_valid,
        'n_insufficient': sum(1 for p in validated
                              if p.get('validation', {}).get('status') == 'insufficient'),
        'validation_rate': n_valid / len(patterns) if patterns else 0,
        'uncond_mean': edges.uncond_mean,
        'uncond_std': edges.uncond_std,
        'uncond_iqr': edges.uncond_iqr,
        'patterns': validated,  # 全部保留，汇总时再筛选
    }


def _build_cross_labels(labels_dict, patterns):
    """从 patterns 中提取交叉维度 key，用单维度标签组合出交叉标签并注入 labels_dict"""
    needed = set()
    for pat in patterns:
        dim = pat['dimension']
        if '_x_' in dim:
            needed.add(dim)

    for cross_key in needed:
        if cross_key in labels_dict:
            continue
        parts = cross_key.split('_x_')
        # 检查所有子维度标签是否存在
        sub_labels = []
        for p in parts:
            if p not in labels_dict:
                sub_labels = None
                break
            sub_labels.append(labels_dict[p])
        if sub_labels is None:
            continue

        n = len(sub_labels[0])
        cross = []
        for i in range(n):
            vals = [sl[i] for sl in sub_labels]
            if any(v is None for v in vals):
                cross.append(None)
            else:
                cross.append(tuple(vals))
        labels_dict[cross_key] = cross


def _scan_single_dim_task(target_abs, builder, dim, uncond_iqr, uncond_mean, uncond_std):
    """单维度扫描"""
    all_patterns = []
    sub_keys = SINGLE_DIM_KEYS.get(dim, [dim])
    for key in sub_keys:
        labels = builder.labels.get(key, [])
        if not labels:
            continue
        pats = scan_single_dimension(
            target_abs, labels, uncond_iqr, uncond_mean, uncond_std, key)
        all_patterns.extend(pats)
    return all_patterns


def _scan_cross_dim_task(target_abs, builder, dim_a, dim_b,
                         uncond_iqr, uncond_mean, uncond_std):
    """二维交叉扫描"""
    all_patterns = []
    keys_a = SINGLE_DIM_KEYS.get(dim_a, [dim_a])
    keys_b = SINGLE_DIM_KEYS.get(dim_b, [dim_b])
    for ka in keys_a:
        labels_a = builder.labels.get(ka, [])
        if not labels_a:
            continue
        for kb in keys_b:
            labels_b = builder.labels.get(kb, [])
            if not labels_b:
                continue
            dim_name = f'{ka}_x_{kb}'
            pats = scan_cross_dimension(
                target_abs, labels_a, labels_b,
                uncond_iqr, uncond_mean, uncond_std, dim_name)
            all_patterns.extend(pats)
    return all_patterns


def _scan_triple_dim_task(target_abs, builder, dim_a, dim_b, dim_c,
                          uncond_iqr, uncond_mean, uncond_std):
    """三维交叉扫描"""
    all_patterns = []
    keys_a = SINGLE_DIM_KEYS.get(dim_a, [dim_a])
    keys_b = SINGLE_DIM_KEYS.get(dim_b, [dim_b])
    keys_c = SINGLE_DIM_KEYS.get(dim_c, [dim_c])
    for ka in keys_a:
        labels_a = builder.labels.get(ka, [])
        if not labels_a:
            continue
        for kb in keys_b:
            labels_b = builder.labels.get(kb, [])
            if not labels_b:
                continue
            for kc in keys_c:
                labels_c = builder.labels.get(kc, [])
                if not labels_c:
                    continue
                dim_name = f'{ka}_x_{kb}_x_{kc}'
                pats = scan_triple_dimension(
                    target_abs, labels_a, labels_b, labels_c,
                    uncond_iqr, uncond_mean, uncond_std, dim_name)
                all_patterns.extend(pats)
    return all_patterns
