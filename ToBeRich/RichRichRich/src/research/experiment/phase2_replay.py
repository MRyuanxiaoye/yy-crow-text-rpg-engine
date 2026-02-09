"""阶段二：历史回放 + 训练数据构造

对每一期历史数据，记录规则触发状态和局面特征，生成训练/测试数据集。
"""

import sys
import numpy as np
from pathlib import Path

# 支持从项目根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.data_loader import LotteryData
from .utils import (
    log, Timer, save_json, load_json, save_npz, Normalizer,
    EXPERIMENT_DIR,
)


# === 条件匹配 ===

def check_single_condition(cond, red_matrix, direction_series, diff_series, combo_stats, t, red_range):
    """检查单个原子条件是否在第t期满足

    注意索引对齐:
      red_matrix[t] = 第t期的红球
      direction_series[pos][t] = 第t期到第t+1期的方向变化
        → 所以"第t期的方向状态"应该用 direction_series[pos][t-1]（从第t-1期到第t期的变化）
      diff_series[pos][t] = position_series[pos][t+1] - position_series[pos][t]
        → 同理，"第t期的差分"用 diff_series[pos][t-1]
      combo_stats[stat][t] = 第t期的统计量
    """
    parts = cond.split('_')

    if parts[0].startswith('P') and len(parts[0]) == 2:
        pos = int(parts[0][1])
        val = int(red_matrix[t, pos])

        if parts[1] == 'val':
            lo, hi = int(parts[2]), int(parts[3])
            return lo <= val <= hi
        elif parts[1] == 'dir':
            # 方向：从t-1期到t期的变化
            if t < 1:
                return False
            dir_map = {'U': 1, 'D': -1, 'E': 0}
            target_dir = dir_map.get(parts[2], 0)
            return int(direction_series[pos][t - 1]) == target_dir
        elif parts[1] == 'odd':
            return val % 2 == 1
        elif parts[1] == 'even':
            return val % 2 == 0
        elif parts[1] == 'big':
            return val > red_range / 2
        elif parts[1] == 'small':
            return val <= red_range / 2
        elif parts[1] == 'diff':
            if t < 1:
                return False
            lo, hi = int(parts[2]), int(parts[3])
            diff_val = int(diff_series[pos][t - 1])
            return lo <= diff_val <= hi
        return False

    elif parts[0] == 'combo':
        # combo 条件格式:
        #   combo_sum_27_83          → stat_name=sum, lo=27, hi=83
        #   combo_ac_value_5_6       → stat_name=ac_value, lo=5, hi=6
        #   combo_consec_groups_0_1  → stat_name=consec_groups, lo=0, hi=1
        #   combo_big_count_2_3      → stat_name=big_count, lo=2, hi=3
        #   combo_odd_count_3_5      → stat_name=odd_count, lo=3, hi=5
        lo, hi = int(parts[-2]), int(parts[-1])
        stat_name = '_'.join(parts[1:-2])
        # 映射到 combo_stats 的 key
        stat_key_map = {
            'sum': 'sum',
            'span': 'span',
            'odd_count': 'odd_count',
            'big_count': 'big_count',
            'ac_value': 'ac_value',
            'consec_groups': 'consec_groups',
        }
        stat_key = stat_key_map.get(stat_name)
        if stat_key is None or stat_key not in combo_stats:
            return False
        stat_val = int(combo_stats[stat_key][t])
        return lo <= stat_val <= hi

    return False


def check_rule_conditions(conditions, red_matrix, direction_series, diff_series, combo_stats, t, red_range):
    """检查A2规则的所有条件是否在第t期满足"""
    for cond in conditions:
        if not check_single_condition(cond, red_matrix, direction_series, diff_series, combo_stats, t, red_range):
            return False
    return True


def check_direction_pattern(pattern, direction_series, t):
    """检查A1方向模式是否在第t期匹配

    pattern结构:
      positions: [0,1,...]
      window: 5
      pattern_key: [242, 0, 242, 0, 242]  (每个时间步的联合编码)

    匹配逻辑: 检查最近window期的方向序列是否与pattern_key一致
    """
    positions = pattern['positions']
    window = pattern['window']
    pattern_key = pattern['pattern_key']

    # 需要 direction_series 有足够的历史
    # direction_series[pos] 长度 = n_draws - 1
    # direction_series[pos][i] = 第i期到第i+1期的方向
    # 要匹配window个时间步，需要 t-1 >= window-1，即 t >= window
    for pos in positions:
        if t - 1 < window - 1:
            return False
        if t - 1 >= len(direction_series[pos]):
            return False

    # 重建编码并比较
    for step in range(window):
        # 第 step 个时间步对应 direction_series 的索引
        dir_idx = t - window + step
        if dir_idx < 0:
            return False

        # 联合编码：多位置的方向组合
        encoded = 0
        for k, pos in enumerate(positions):
            d = int(direction_series[pos][dir_idx]) + 1  # -1→0, 0→1, 1→2
            encoded += d * (3 ** k)

        if encoded != pattern_key[step]:
            return False

    return True


# === A3/A4 规则加载与匹配 ===

# A3 差分模式的文本标签 → 数值编码
A3_LABEL_TO_CODE = {"大降": 0, "小降": 1, "平": 2, "小升": 3, "大升": 4}


def parse_a3_pattern(pattern_str):
    """将 '大升→大降→小降' 转为 (4, 0, 1)"""
    labels = pattern_str.split("→")
    return tuple(A3_LABEL_TO_CODE[lb] for lb in labels)


def load_a3_rules(rules_dir, lottery_type):
    """加载 A3 差分模式规则，返回结构化列表

    Returns:
        list of dict: [{pos, order, window, pattern_codes, chi2, prediction, prediction_confidence}, ...]
    """
    from pathlib import Path
    path = Path(rules_dir) / f"a3_diff_analysis_{lottery_type}.json"
    if not path.exists():
        log(f"  A3 规则文件不存在: {path}")
        return []

    raw = load_json(path)
    diff_patterns = raw.get("analysis", {}).get("diff_patterns", {})
    result = []
    for pos_key, orders in diff_patterns.items():
        pos = int(pos_key[1:])  # "P0" → 0
        for order_key, info in orders.items():
            order = int(order_key.split("_")[1])  # "order_1" → 1
            for pat in info.get("top_patterns", []):
                try:
                    codes = parse_a3_pattern(pat["pattern"])
                except KeyError:
                    continue
                result.append({
                    "pos": pos,
                    "order": order,
                    "window": pat["window"],
                    "pattern_codes": codes,
                    "chi2": pat["chi2"],
                    "prediction": pat["prediction"],
                    "prediction_confidence": pat["prediction_confidence"],
                })
    return result


def load_a4_rules(rules_dir, lottery_type):
    """加载 A4 统计量模式规则，返回结构化列表

    Returns:
        list of dict: [{stat_name, is_joint, window, pattern_values, chi2, prediction, prediction_confidence}, ...]
    """
    from pathlib import Path
    path = Path(rules_dir) / f"a4_sliding_stats_{lottery_type}.json"
    if not path.exists():
        log(f"  A4 规则文件不存在: {path}")
        return []

    raw = load_json(path)
    stat_patterns = raw.get("analysis", {}).get("stat_patterns", {})
    result = []
    for stat_name, info in stat_patterns.items():
        is_joint = "+" in stat_name
        for pat in info.get("top_patterns", []):
            result.append({
                "stat_name": stat_name,
                "is_joint": is_joint,
                "window": pat["window"],
                "pattern_values": tuple(pat["pattern"]),
                "chi2": pat["chi2"],
                "prediction": pat["prediction"],
                "prediction_confidence": pat["prediction_confidence"],
            })
    return result


def precompute_a3_discrete(data, split1):
    """预计算各位置各阶差分的离散化序列（用前split1期数据拟合五分位阈值）

    Returns:
        dict: {(pos, order): (discrete_array, thresholds)}
    """
    rc = data.red_count
    result = {}
    for pos in range(rc):
        for order in [1, 2]:
            diff = data.get_diff_series(pos, order)
            if len(diff) < 10:
                continue
            # 用前 split1-order 期的差分拟合阈值（差分序列比原序列短 order 期）
            fit_end = max(10, split1 - order)
            fit_data = diff[:fit_end]
            thresholds = np.percentile(fit_data, [20, 40, 60, 80])

            # 离散化完整序列
            discrete = np.zeros(len(diff), dtype=np.int8)
            discrete[diff <= thresholds[0]] = 0  # 大降
            discrete[(diff > thresholds[0]) & (diff <= thresholds[1])] = 1  # 小降
            discrete[(diff > thresholds[1]) & (diff <= thresholds[2])] = 2  # 平
            discrete[(diff > thresholds[2]) & (diff <= thresholds[3])] = 3  # 小升
            discrete[diff > thresholds[3]] = 4  # 大升

            result[(pos, order)] = (discrete, thresholds)
    return result


def precompute_a4_discrete(combo_stats, split1, n_levels=3):
    """预计算各统计量的离散化序列（用前split1期数据拟合分位阈值）

    Returns:
        dict: {stat_name: discrete_array}
              对联合统计量 "a+b"，编码为 disc_a * n_levels + disc_b
    """
    single_stats = ['sum', 'span', 'odd_count', 'big_count', 'ac_value', 'consec_groups']
    result = {}

    # 先计算单统计量的离散化和阈值
    single_discrete = {}
    single_edges = {}
    for sn in single_stats:
        if sn not in combo_stats:
            continue
        series = combo_stats[sn]
        fit_data = series[:split1].astype(float)
        percentiles = np.linspace(0, 100, n_levels + 1)
        edges = np.percentile(fit_data, percentiles)
        edges = np.unique(edges)
        actual_levels = len(edges) - 1
        if actual_levels < 1:
            disc = np.zeros(len(series), dtype=np.int8)
        else:
            disc = np.zeros(len(series), dtype=np.int8)
            for i in range(actual_levels):
                if i < actual_levels - 1:
                    mask = (series >= edges[i]) & (series < edges[i + 1])
                else:
                    mask = (series >= edges[i]) & (series <= edges[i + 1])
                disc[mask] = i
        single_discrete[sn] = disc
        single_edges[sn] = edges
        result[sn] = disc

    # 联合统计量
    from itertools import combinations as _combs
    for sn_a, sn_b in _combs(single_stats, 2):
        key = f"{sn_a}+{sn_b}"
        if sn_a in single_discrete and sn_b in single_discrete:
            joint = single_discrete[sn_a].astype(np.int16) * n_levels + single_discrete[sn_b].astype(np.int16)
            result[key] = joint.astype(np.int8)

    return result


def select_a3_top_patterns(a3_rules, rc, top_n=10):
    """按 pos×order 分组，每组取 chi2 最高的 top_n 个模式

    Returns:
        dict: {(pos, order): [pattern_list]}  每个 list 最多 top_n 个
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in a3_rules:
        if r["pos"] < rc:
            groups[(r["pos"], r["order"])].append(r)
    result = {}
    for key, pats in groups.items():
        pats.sort(key=lambda x: x["chi2"], reverse=True)
        result[key] = pats[:top_n]
    return result


def select_a4_top_patterns(a4_rules, top_n_single=10, top_n_joint=5):
    """按 stat_name 分组，单统计量取 top_n_single，联合取 top_n_joint

    Returns:
        dict: {stat_name: [pattern_list]}
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in a4_rules:
        groups[r["stat_name"]].append(r)
    result = {}
    for key, pats in groups.items():
        pats.sort(key=lambda x: x["chi2"], reverse=True)
        top_n = top_n_joint if "+" in key else top_n_single
        result[key] = pats[:top_n]
    return result


def check_a3_match(pattern_codes, window, a3_disc_series, t):
    """检查第t期是否匹配A3差分模式

    a3_disc_series: 离散化后的差分序列
    t: 期数索引（对应 red_matrix 的行号）
    差分序列索引对齐: diff[i] = val[i+1] - val[i]，所以第t期的"最近差分"是 diff[t-1]
    模式匹配: 检查 diff[t-window:t] 的离散值是否与 pattern_codes 一致
    """
    # 差分序列中，索引 t-1 对应"第t-1期到第t期的差分"
    # 要匹配 window 个差分值，需要 diff[t-window : t]
    end_idx = t  # diff[t-1] 是最新的，但 t 在差分序列中可能越界
    start_idx = t - window
    if start_idx < 0 or end_idx > len(a3_disc_series):
        return False
    segment = a3_disc_series[start_idx:end_idx]
    if len(segment) != window:
        return False
    return tuple(int(s) for s in segment) == pattern_codes


def check_a4_match(pattern_values, window, a4_disc_series, t):
    """检查第t期是否匹配A4统计量模式

    a4_disc_series: 离散化后的统计量序列
    t: 期数索引
    统计量索引对齐: combo_stats[stat][t] = 第t期的统计量
    模式匹配: 检查 series[t-window:t] 的离散值是否匹配
    """
    start_idx = t - window
    if start_idx < 0 or t > len(a4_disc_series):
        return False
    segment = a4_disc_series[start_idx:t]
    if len(segment) != window:
        return False
    return tuple(int(s) for s in segment) == pattern_values


# === 特征构造 ===

def build_features_for_period(t, data, clusters, a1_filtered, diff_series, combo_stats, normalizer,
                              a3_grouped=None, a4_grouped=None, a3_discrete=None, a4_discrete=None):
    """为第t期构造特征向量

    Args:
        t: 期数索引（red_matrix中的行号）
        data: LotteryData
        clusters: A2规则簇字典
        a1_filtered: 筛选后的A1模式列表
        diff_series: {pos: 一阶差分序列}
        combo_stats: {stat_name: 统计量序列}
        normalizer: Normalizer实例

    Returns:
        feature_dict: {feature_name: value}
    """
    features = {}
    rc = data.red_count

    # === 局面特征 ===

    # 1. 最近5期各位置的值（归一化）
    for lag in range(1, 6):
        idx = t - lag
        if idx < 0:
            for pos in range(rc):
                features[f'val_lag{lag}_P{pos}'] = 0.0
        else:
            for pos in range(rc):
                features[f'val_lag{lag}_P{pos}'] = normalizer.transform(
                    f'val_P{pos}', float(data.red_matrix[idx, pos]))

    # 2. 最近5期各位置的方向
    for lag in range(1, 6):
        dir_idx = t - lag - 1  # direction_series[pos][i] = 第i期→第i+1期
        for pos in range(rc):
            if 0 <= dir_idx < len(data.direction_series[pos]):
                features[f'dir_lag{lag}_P{pos}'] = float(data.direction_series[pos][dir_idx])
            else:
                features[f'dir_lag{lag}_P{pos}'] = 0.0

    # 3. 最近5期各位置的一阶差分（归一化）
    for lag in range(1, 6):
        diff_idx = t - lag - 1
        for pos in range(rc):
            if 0 <= diff_idx < len(diff_series[pos]):
                features[f'diff_lag{lag}_P{pos}'] = normalizer.transform(
                    f'diff_P{pos}', float(diff_series[pos][diff_idx]))
            else:
                features[f'diff_lag{lag}_P{pos}'] = 0.0

    # 4. 最近3期的组合统计量（归一化）
    stat_names = ['sum', 'span', 'odd_count', 'big_count', 'ac_value', 'consec_groups']
    for lag in range(1, 4):
        idx = t - lag
        for sn in stat_names:
            if 0 <= idx < len(combo_stats[sn]):
                features[f'stat_lag{lag}_{sn}'] = normalizer.transform(
                    f'stat_{sn}', float(combo_stats[sn][idx]))
            else:
                features[f'stat_lag{lag}_{sn}'] = 0.0

    # === A2 规则簇触发特征 ===
    n_triggered = 0
    conf_sum = 0.0
    lift_sum = 0.0

    for cid, cinfo in clusters.items():
        rule = cinfo['representative']
        triggered = check_rule_conditions(
            rule['conditions'], data.red_matrix, data.direction_series,
            diff_series, combo_stats, t, data.red_range)
        val = 1.0 if triggered else 0.0
        features[f'a2_c{cid}'] = val
        if triggered:
            n_triggered += 1
            conf_sum += cinfo['avg_confidence']
            lift_sum += cinfo['avg_lift']

    # === A1 模式匹配特征 ===
    n_a1_matched = 0
    for i, pat in enumerate(a1_filtered):
        matched = check_direction_pattern(pat, data.direction_series, t)
        features[f'a1_p{i}'] = 1.0 if matched else 0.0
        if matched:
            n_a1_matched += 1

    # === A3 差分模式匹配特征 ===
    n_a3_matched = 0
    a3_chi2_sum = 0.0
    a3_conf_sum = 0.0
    if a3_grouped and a3_discrete:
        for (pos, order), pats in a3_grouped.items():
            disc_key = (pos, order)
            if disc_key not in a3_discrete:
                for idx_p, _ in enumerate(pats):
                    features[f'a3_P{pos}_o{order}_p{idx_p}'] = 0.0
                continue
            disc_series = a3_discrete[disc_key][0]
            for idx_p, pat in enumerate(pats):
                matched = check_a3_match(
                    pat["pattern_codes"], pat["window"], disc_series, t)
                features[f'a3_P{pos}_o{order}_p{idx_p}'] = 1.0 if matched else 0.0
                if matched:
                    n_a3_matched += 1
                    a3_chi2_sum += pat["chi2"]
                    a3_conf_sum += pat["prediction_confidence"]

    # === A4 统计量模式匹配特征 ===
    n_a4_matched = 0
    a4_chi2_sum = 0.0
    a4_conf_sum = 0.0
    if a4_grouped and a4_discrete:
        for stat_name, pats in a4_grouped.items():
            if stat_name not in a4_discrete:
                for idx_p, _ in enumerate(pats):
                    features[f'a4_{stat_name}_p{idx_p}'] = 0.0
                continue
            disc_series = a4_discrete[stat_name]
            for idx_p, pat in enumerate(pats):
                matched = check_a4_match(
                    pat["pattern_values"], pat["window"], disc_series, t)
                features[f'a4_{stat_name}_p{idx_p}'] = 1.0 if matched else 0.0
                if matched:
                    n_a4_matched += 1
                    a4_chi2_sum += pat["chi2"]
                    a4_conf_sum += pat["prediction_confidence"]

    # === 触发元信息 ===
    features['n_triggered_a2'] = float(n_triggered)
    features['avg_conf_triggered'] = conf_sum / n_triggered if n_triggered > 0 else 0.0
    features['avg_lift_triggered'] = lift_sum / n_triggered if n_triggered > 0 else 0.0
    features['n_matched_a1'] = float(n_a1_matched)
    features['n_matched_a3'] = float(n_a3_matched)
    features['avg_chi2_a3'] = a3_chi2_sum / n_a3_matched if n_a3_matched > 0 else 0.0
    features['avg_conf_a3'] = a3_conf_sum / n_a3_matched if n_a3_matched > 0 else 0.0
    features['n_matched_a4'] = float(n_a4_matched)
    features['avg_chi2_a4'] = a4_chi2_sum / n_a4_matched if n_a4_matched > 0 else 0.0
    features['avg_conf_a4'] = a4_conf_sum / n_a4_matched if n_a4_matched > 0 else 0.0

    return features


def build_labels_for_period(t, data):
    """为第t期构造标签

    Returns:
        labels: {dir_P{pos}: 0/1/2, val_P{pos}: int}
    """
    labels = {}
    for pos in range(data.red_count):
        # 方向标签：第t-1期到第t期的方向
        if t - 1 >= 0 and t - 1 < len(data.direction_series[pos]):
            direction = int(data.direction_series[pos][t - 1])
            labels[f'dir_P{pos}'] = direction + 1  # -1→0, 0→1, 1→2
        else:
            labels[f'dir_P{pos}'] = 1  # 默认平

        # 值标签
        labels[f'val_P{pos}'] = int(data.red_matrix[t, pos])

    return labels


# === 主流程 ===

def run_phase2(lottery_type, clusters, a1_filtered, rules_dir=None):
    """执行阶段二：历史回放 + 数据构造

    Args:
        lottery_type: "daletou" 或 "shuangseqiu"
        clusters: 阶段一输出的A2规则簇
        a1_filtered: 阶段一输出的A1筛选模式
        rules_dir: A3/A4 规则文件所在目录（None 则不加载 A3/A4）

    Returns:
        train_X, train_Y, test_X, test_Y: numpy数组
        feature_names: 特征名列表
        train_indices, test_indices: 期数索引
    """
    log(f"\n{'─'*40}")
    log(f"阶段二：历史回放 [{lottery_type}]")
    log(f"{'─'*40}")

    # 加载数据
    with Timer("加载数据"):
        data = LotteryData(lottery_type)
        n = data.n_draws
        rc = data.red_count
        log(f"  总期数: {n}, 红球位置数: {rc}")

    # 预计算
    with Timer("预计算差分和统计量"):
        diff_series = {}
        for pos in range(rc):
            diff_series[pos] = data.get_diff_series(pos, 1)
        combo_stats = data.get_combo_stats_series()

    # 时间切分
    split1 = int(n * 0.6)
    split2 = int(n * 0.85)
    log(f"  时间切分: 规则期[0:{split1}], 训练期[{split1}:{split2}], 测试期[{split2}:{n}]")
    log(f"  训练样本: {split2 - split1} 期, 测试样本: {n - split2} 期")

    # 归一化器：用规则期数据拟合
    with Timer("拟合归一化器"):
        normalizer = Normalizer()
        for pos in range(rc):
            normalizer.fit(f'val_P{pos}', data.red_matrix[:split1, pos].astype(float))
            valid_diff = diff_series[pos][:max(0, split1 - 1)]
            if len(valid_diff) > 0:
                normalizer.fit(f'diff_P{pos}', valid_diff.astype(float))
            else:
                normalizer.fit(f'diff_P{pos}', [0.0])
        stat_names = ['sum', 'span', 'odd_count', 'big_count', 'ac_value', 'consec_groups']
        for sn in stat_names:
            normalizer.fit(f'stat_{sn}', combo_stats[sn][:split1].astype(float))

    # A3/A4 规则加载与预计算
    a3_grouped = None
    a4_grouped = None
    a3_disc = None
    a4_disc = None
    if rules_dir is not None:
        with Timer("加载 A3/A4 规则"):
            a3_rules = load_a3_rules(rules_dir, lottery_type)
            a4_rules = load_a4_rules(rules_dir, lottery_type)
            log(f"  A3 规则: {len(a3_rules)} 条, A4 规则: {len(a4_rules)} 条")

        if a3_rules:
            with Timer("A3 预计算离散化"):
                a3_grouped = select_a3_top_patterns(a3_rules, rc, top_n=10)
                a3_disc = precompute_a3_discrete(data, split1)
                n_a3_feats = sum(len(pats) for pats in a3_grouped.values())
                log(f"  A3 分组: {len(a3_grouped)} 组, 特征维度: {n_a3_feats}")

        if a4_rules:
            with Timer("A4 预计算离散化"):
                a4_grouped = select_a4_top_patterns(a4_rules, top_n_single=10, top_n_joint=5)
                a4_disc = precompute_a4_discrete(combo_stats, split1)
                n_a4_feats = sum(len(pats) for pats in a4_grouped.values())
                log(f"  A4 分组: {len(a4_grouped)} 组, 特征维度: {n_a4_feats}")

    # 构造特征和标签
    # 需要足够的历史（至少5期lag + 方向模式窗口），从 max(20, split1) 开始
    start_t = max(20, split1)

    with Timer("构造训练集特征"):
        train_features = []
        train_labels = []
        train_indices = []
        for t in range(start_t, split2):
            feat = build_features_for_period(t, data, clusters, a1_filtered,
                                             diff_series, combo_stats, normalizer,
                                             a3_grouped, a4_grouped, a3_disc, a4_disc)
            lab = build_labels_for_period(t, data)
            train_features.append(feat)
            train_labels.append(lab)
            train_indices.append(t)

            if (t - start_t + 1) % 100 == 0:
                log(f"    训练集进度: {t - start_t + 1}/{split2 - start_t}")

    with Timer("构造测试集特征"):
        test_features = []
        test_labels = []
        test_indices = []
        for t in range(split2, n):
            feat = build_features_for_period(t, data, clusters, a1_filtered,
                                             diff_series, combo_stats, normalizer,
                                             a3_grouped, a4_grouped, a3_disc, a4_disc)
            lab = build_labels_for_period(t, data)
            test_features.append(feat)
            test_labels.append(lab)
            test_indices.append(t)

            if (t - split2 + 1) % 100 == 0:
                log(f"    测试集进度: {t - split2 + 1}/{n - split2}")

    # 转换为numpy数组
    with Timer("转换为numpy数组"):
        feature_names = sorted(train_features[0].keys())
        label_names = sorted(train_labels[0].keys())

        train_X = np.array([[f[k] for k in feature_names] for f in train_features], dtype=np.float32)
        test_X = np.array([[f[k] for k in feature_names] for f in test_features], dtype=np.float32)

        # 方向标签矩阵 (n_samples, n_positions)
        dir_label_names = [f'dir_P{pos}' for pos in range(rc)]
        train_Y = np.array([[l[k] for k in dir_label_names] for l in train_labels], dtype=np.int32)
        test_Y = np.array([[l[k] for k in dir_label_names] for l in test_labels], dtype=np.int32)

        # 值标签矩阵
        val_label_names = [f'val_P{pos}' for pos in range(rc)]
        train_Y_val = np.array([[l[k] for k in val_label_names] for l in train_labels], dtype=np.int32)
        test_Y_val = np.array([[l[k] for k in val_label_names] for l in test_labels], dtype=np.int32)

        train_indices = np.array(train_indices, dtype=np.int32)
        test_indices = np.array(test_indices, dtype=np.int32)

    log(f"\n  数据集形状:")
    log(f"    train_X: {train_X.shape}, train_Y: {train_Y.shape}")
    log(f"    test_X:  {test_X.shape},  test_Y:  {test_Y.shape}")
    log(f"    特征数: {len(feature_names)}")

    # 保存
    save_npz(EXPERIMENT_DIR / f"phase2_train_{lottery_type}.npz",
             X=train_X, Y=train_Y, Y_val=train_Y_val, indices=train_indices)
    save_npz(EXPERIMENT_DIR / f"phase2_test_{lottery_type}.npz",
             X=test_X, Y=test_Y, Y_val=test_Y_val, indices=test_indices)
    save_json(feature_names, EXPERIMENT_DIR / f"phase2_feature_names_{lottery_type}.json")

    summary = {
        'lottery_type': lottery_type,
        'n_draws': n,
        'split1': split1,
        'split2': split2,
        'train_samples': int(train_X.shape[0]),
        'test_samples': int(test_X.shape[0]),
        'n_features': len(feature_names),
        'n_a2_clusters': len(clusters),
        'n_a1_patterns': len(a1_filtered),
        'n_a3_groups': len(a3_grouped) if a3_grouped else 0,
        'n_a3_features': sum(len(p) for p in a3_grouped.values()) if a3_grouped else 0,
        'n_a4_groups': len(a4_grouped) if a4_grouped else 0,
        'n_a4_features': sum(len(p) for p in a4_grouped.values()) if a4_grouped else 0,
        'train_Y_distribution': {
            f'P{pos}': {
                'D': int((train_Y[:, pos] == 0).sum()),
                'E': int((train_Y[:, pos] == 1).sum()),
                'U': int((train_Y[:, pos] == 2).sum()),
            } for pos in range(rc)
        },
    }
    save_json(summary, EXPERIMENT_DIR / f"phase2_summary_{lottery_type}.json")

    return train_X, train_Y, test_X, test_Y, train_Y_val, test_Y_val, feature_names, train_indices, test_indices, data
