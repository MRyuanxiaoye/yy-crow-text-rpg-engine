# -*- coding: utf-8 -*-
"""
阶段1：排除引擎（Exclusion Engine）

通过6个排除算法的加权投票，排除30%-50%的号码。
6个算法：
  1.1 连续重复排除器
  1.2 遗漏值异常排除器
  1.3 极端组合约束排除器
  1.4 马尔可夫转移排除器
  1.5 周期性排除器
  1.6 聚类异常排除器
"""

import math
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict

from pipeline.config import LotteryConfig, get_config


# ============================================================
# 算法1.1：连续重复排除器
# ============================================================

def algo_consecutive(
    number_features: Dict[str, Dict], config: LotteryConfig
) -> Dict[str, float]:
    """
    连续出现多期的号码，继续出现的概率指数级衰减。
    confidence = 1 - base_prob ^ consecutive_count
    """
    result = {}
    base_prob = config.red_prob

    for num_str, feat in number_features.items():
        consec = feat.get("consecutive", 0)
        if consec == 0:
            confidence = 0.0
        elif consec == 1:
            confidence = 0.1
        elif consec == 2:
            confidence = 0.4
        else:
            # 精确公式：1 - p^n
            confidence = 1.0 - base_prob ** consec
            confidence = min(confidence, 0.95)
        result[num_str] = confidence

    return result


# ============================================================
# 算法1.2：遗漏值异常排除器
# ============================================================

def algo_missing_value(
    number_features: Dict[str, Dict], config: LotteryConfig
) -> Dict[str, float]:
    """
    刚出现不久（遗漏值远低于平均间隔）的号码，短期再次出现概率较低。
    ratio = missing / avg_gap
    ratio < 0.3 时排除，confidence = 0.6 * (1 - ratio/0.3)
    """
    result = {}

    for num_str, feat in number_features.items():
        missing = feat.get("missing_value", 0)
        avg_gap = feat.get("avg_gap", 7.0)

        if avg_gap <= 0:
            avg_gap = 1.0

        ratio = missing / avg_gap

        if ratio < 0.3:
            confidence = 0.6 * (1.0 - ratio / 0.3)
        else:
            confidence = 0.0

        result[num_str] = confidence

    return result


# ============================================================
# 算法1.3：极端组合约束排除器
# ============================================================

def algo_extreme_combo(
    number_features: Dict[str, Dict],
    combo_stats: Dict[str, Any],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    从组合层面评估号码的约束违反风险。
    如果某号码的属性（奇偶、大小）在当前高权重号码中已过度集中，
    则该号码被排除的置信度升高。

    简化实现：基于号码本身属性和历史组合分布的边际概率。
    """
    result = {}
    mid = config.red_midpoint

    # 从历史分布中获取极端比例的概率
    oe_dist = combo_stats.get("odd_even_dist", {})
    bs_dist = combo_stats.get("big_small_dist", {})
    total_draws = sum(oe_dist.values()) if oe_dist else 1

    # 计算全奇/全偶的历史概率
    rc = config.red_count
    all_odd_key = f"{rc}:0"
    all_even_key = f"0:{rc}"
    extreme_oe_prob = (oe_dist.get(all_odd_key, 0) + oe_dist.get(all_even_key, 0)) / total_draws

    # 统计当前所有号码的奇偶/大小分布
    odd_nums = set()
    even_nums = set()
    big_nums = set()
    small_nums = set()

    for num_str in number_features:
        num = int(num_str)
        if num % 2 == 1:
            odd_nums.add(num_str)
        else:
            even_nums.add(num_str)
        if num > mid:
            big_nums.add(num_str)
        else:
            small_nums.add(num_str)

    # 计算每个号码的边际风险
    odd_ratio = len(odd_nums) / len(number_features) if number_features else 0.5
    even_ratio = len(even_nums) / len(number_features) if number_features else 0.5

    for num_str, feat in number_features.items():
        num = int(num_str)
        risk = 0.0

        # 如果奇数或偶数号码占比已经很高，同类号码风险升高
        is_odd = num % 2 == 1
        is_big = num > mid

        # 基于和值的风险：号码越极端（很大或很小），越容易导致和值偏离
        sum_mean = combo_stats.get("sum_mean", 90)
        sum_std = combo_stats.get("sum_std", 22)
        # 号码对和值的贡献偏离程度
        expected_per_ball = sum_mean / config.red_count
        deviation = abs(num - expected_per_ball) / (sum_std / math.sqrt(config.red_count))
        sum_risk = min(deviation * 0.15, 0.4)

        # 极端号码（1,2 或 34,35）额外风险
        if num <= 2 or num >= config.red_range - 1:
            sum_risk += 0.05

        risk = sum_risk
        # 用 sigmoid 平滑
        confidence = 1.0 / (1.0 + math.exp(-5 * (risk - 0.3)))
        result[num_str] = confidence

    return result


# ============================================================
# 算法1.4：马尔可夫转移排除器
# ============================================================

def algo_markov(
    number_features: Dict[str, Dict],
    transition_matrix: List[List[float]],
    last_red_balls: List[int],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    基于马尔可夫转移矩阵，如果上一期出现的号码转移到某号码的概率极低，
    则排除该号码。

    对每个候选号码j，计算从上一期所有号码转移到j的平均概率，
    概率极低的给予排除置信度。
    """
    result = {}
    n = config.red_range

    for num_str in number_features:
        j = int(num_str) - 1  # 0-indexed

        # 计算从上一期所有红球转移到j的平均概率
        trans_probs = []
        for prev_num in last_red_balls:
            i = prev_num - 1
            if 0 <= i < n and 0 <= j < n:
                trans_probs.append(transition_matrix[i][j])

        avg_prob = sum(trans_probs) / len(trans_probs) if trans_probs else 0.0

        # 理论均匀转移概率
        uniform_prob = config.red_count / config.red_range

        # 如果转移概率远低于均匀概率，给予排除置信度
        if uniform_prob > 0 and avg_prob < uniform_prob * 0.3:
            ratio = avg_prob / (uniform_prob * 0.3)
            confidence = 0.5 * (1.0 - ratio)
        else:
            confidence = 0.0

        result[num_str] = confidence

    return result


# ============================================================
# 算法1.5：周期性排除器
# ============================================================

def algo_periodicity(
    number_features: Dict[str, Dict],
    time_features: Dict[str, Dict],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    基于FFT分析的周期性特征，如果号码不在其主周期的出现窗口内，
    给予排除置信度。
    """
    result = {}

    for num_str, feat in number_features.items():
        tf = time_features.get(num_str, {})
        period = tf.get("fft_main_period", 0)
        amplitude = tf.get("fft_amplitude", 0)
        missing = feat.get("missing_value", 0)

        if period <= 2 or amplitude < 1.0:
            # 周期性不明显，不排除
            result[num_str] = 0.0
            continue

        # 计算当前遗漏值在周期中的位置
        phase = missing % period
        # 如果在周期的前半段（刚出现不久），排除
        # 如果在周期的后半段（接近下次出现），不排除
        half_period = period / 2

        if phase < half_period * 0.5:
            # 刚出现不久，离下次出现还远
            confidence = 0.3 * (1.0 - phase / (half_period * 0.5))
            # 振幅越大，周期性越可信
            confidence *= min(amplitude / 5.0, 1.0)
        else:
            confidence = 0.0

        result[num_str] = min(confidence, 0.6)

    return result


# ============================================================
# 算法1.6：聚类异常排除器
# ============================================================

def algo_cluster(
    number_features: Dict[str, Dict],
    time_features: Dict[str, Dict],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    基于多维特征的异常检测。
    将每个号码的特征向量与整体分布比较，偏离过大的给予排除置信度。
    使用简化的Z-score方法。
    """
    # 构建特征向量：[频率, 遗漏值/avg_gap, window_5占比, window_10占比]
    vectors = {}
    for num_str, feat in number_features.items():
        tf = time_features.get(num_str, {})
        avg_gap = feat.get("avg_gap", 7.0)
        if avg_gap <= 0:
            avg_gap = 1.0

        vec = [
            feat.get("frequency", 0),
            feat.get("missing_value", 0) / avg_gap,
            tf.get("window_5", 0) / 5.0,
            tf.get("window_10", 0) / 10.0,
        ]
        vectors[num_str] = vec

    if not vectors:
        return {k: 0.0 for k in number_features}

    # 计算均值和标准差
    all_vecs = np.array(list(vectors.values()))
    means = all_vecs.mean(axis=0)
    stds = all_vecs.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)

    result = {}
    for num_str, vec in vectors.items():
        z_scores = np.abs((np.array(vec) - means) / stds)
        max_z = float(z_scores.max())

        # Z-score > 2 开始排除
        if max_z > 2.0:
            confidence = min((max_z - 2.0) * 0.2, 0.6)
        else:
            confidence = 0.0

        result[num_str] = confidence

    return result


# ============================================================
# 算法1.7：位置模式排除器
# ============================================================

def algo_position_pattern(
    number_features: Dict[str, Dict],
    advanced_analysis: Dict[str, Any],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    基于位置预测值域排除号码。
    如果某号码不在任何位置的预测范围内，排除置信度升高。
    """
    result = {}
    pos_predictions = advanced_analysis.get("position_pattern", {}).get(
        "position_predictions", {}
    )

    # 收集所有位置预测的值域并集
    all_predicted_ranges: List[Tuple[int, int]] = []
    for pos, pred in pos_predictions.items():
        conf = pred.get("confidence", 0)
        if conf > 0.3:
            vr = pred.get("value_range", [])
            if len(vr) == 2:
                all_predicted_ranges.append((vr[0], vr[1], conf))

    for num_str in number_features:
        num = int(num_str)

        if not all_predicted_ranges:
            result[num_str] = 0.0
            continue

        # 检查号码是否落在任何位置的预测范围内
        in_range_count = 0
        max_conf = 0.0
        for low, high, conf in all_predicted_ranges:
            if low <= num <= high:
                in_range_count += 1
                max_conf = max(max_conf, conf)

        if in_range_count == 0:
            # 不在任何预测范围内，给予排除置信度
            # 置信度与预测的平均置信度成正比
            avg_conf = sum(c for _, _, c in all_predicted_ranges) / len(all_predicted_ranges)
            confidence = min(avg_conf * 0.6, 0.5)
        else:
            confidence = 0.0

        result[num_str] = confidence

    return result


# ============================================================
# 融合器：加权投票
# ============================================================

def fuse_exclusion(
    algo_results: Dict[str, Dict[str, float]],
    config: LotteryConfig,
) -> Tuple[List[int], Dict[str, Dict]]:
    """
    加权投票融合6个排除算法的结果。

    返回:
        excluded_numbers: 被排除的号码列表
        exclusion_details: 每个被排除号码的详细信息
    """
    weights = config.exclusion_weights
    threshold = config.exclusion_threshold

    # 获取所有号码
    all_nums = set()
    for algo_result in algo_results.values():
        all_nums.update(algo_result.keys())

    # 计算加权置信度
    scores = {}
    details = {}
    for num_str in sorted(all_nums, key=lambda x: int(x)):
        weighted_sum = 0.0
        reasons = {}
        for algo_name, algo_result in algo_results.items():
            conf = algo_result.get(num_str, 0.0)
            w = weights.get(algo_name, 0.0)
            weighted_sum += conf * w
            reasons[algo_name] = conf

        scores[num_str] = weighted_sum
        details[num_str] = {
            "confidence": weighted_sum,
            "reasons": reasons,
        }

    # 按阈值排除
    excluded = []
    for num_str, score in scores.items():
        if score >= threshold:
            excluded.append(int(num_str))

    # 如果排除比例不在目标范围内，动态调整
    total = len(all_nums)
    target_min = int(total * 0.25)
    target_max = int(total * 0.50)

    if len(excluded) < target_min:
        # 排除太少，降低阈值取 top N
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        excluded = [int(x[0]) for x in sorted_nums[:target_min]]
    elif len(excluded) > target_max:
        # 排除太多，只取置信度最高的
        sorted_excluded = sorted(
            [(n, scores[str(n)]) for n in excluded],
            key=lambda x: x[1], reverse=True
        )
        excluded = [x[0] for x in sorted_excluded[:target_max]]

    excluded.sort()

    # 构建排除详情（只保留被排除的）
    exclusion_details = {}
    for num in excluded:
        exclusion_details[str(num)] = details[str(num)]

    return excluded, exclusion_details


# ============================================================
# 主函数：运行阶段1
# ============================================================

def run_stage1(
    stage0_result: Dict[str, Any],
    lottery_type: str,
) -> Dict[str, Any]:
    """
    运行阶段1：排除引擎

    参数:
        stage0_result: 阶段0的输出
        lottery_type: 彩种标识

    返回:
        excluded_numbers: 被排除的号码列表
        exclusion_details: 排除详情
        remaining_numbers: 保留的号码列表
    """
    config = get_config(lottery_type)
    nf = stage0_result["number_features"]
    cs = stage0_result["combo_stats"]
    tf = stage0_result["time_features"]
    tm = stage0_result["transition_matrix"]
    meta = stage0_result["metadata"]

    # 获取最近一期的红球（用于马尔可夫）
    # 从 number_features 中找遗漏值为0的号码
    last_red = []
    for num_str, feat in nf.items():
        if feat.get("missing_value", 999) == 0:
            last_red.append(int(num_str))
    if not last_red:
        last_red = list(range(1, min(6, config.red_range + 1)))

    print(f"[阶段1] 开始排除引擎，候选号码: {config.red_range} 个")

    # 运行6个排除算法
    print("[阶段1] 算法1.1 连续重复排除器...")
    r1 = algo_consecutive(nf, config)

    print("[阶段1] 算法1.2 遗漏值异常排除器...")
    r2 = algo_missing_value(nf, config)

    print("[阶段1] 算法1.3 极端组合约束排除器...")
    r3 = algo_extreme_combo(nf, cs, config)

    print("[阶段1] 算法1.4 马尔可夫转移排除器...")
    r4 = algo_markov(nf, tm, last_red, config)

    print("[阶段1] 算法1.5 周期性排除器...")
    r5 = algo_periodicity(nf, tf, config)

    print("[阶段1] 算法1.6 聚类异常排除器...")
    r6 = algo_cluster(nf, tf, config)

    # 算法1.7：位置模式排除器
    advanced = stage0_result.get("advanced_analysis", {})
    print("[阶段1] 算法1.7 位置模式排除器...")
    r7 = algo_position_pattern(nf, advanced, config)

    # 融合
    algo_results = {
        "consecutive": r1,
        "missing_value": r2,
        "extreme_combo": r3,
        "markov": r4,
        "periodicity": r5,
        "cluster": r6,
        "position_pattern": r7,
    }

    excluded, exclusion_details = fuse_exclusion(algo_results, config)
    remaining = [n for n in config.red_numbers if n not in excluded]

    print(f"[阶段1] 完成，排除 {len(excluded)} 个，保留 {len(remaining)} 个")

    return {
        "excluded_numbers": excluded,
        "exclusion_details": exclusion_details,
        "remaining_numbers": remaining,
    }
