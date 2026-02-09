# -*- coding: utf-8 -*-
"""
阶段2：权重引擎（Weight Engine）

对未排除号码计算综合权重分数，6个权重算法归一化融合。
6个算法：
  2.1 频率回归权重
  2.2 遗漏值回归权重
  2.3 时序衰减权重
  2.4 马尔可夫转移权重
  2.5 共现关联权重
  2.6 深度学习预测权重（简化版：基于多特征的MLP打分）
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple

from pipeline.config import LotteryConfig, get_config


# ============================================================
# 工具函数
# ============================================================

def sigmoid(x: float) -> float:
    """Sigmoid 函数，防止溢出"""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def softmax(values: Dict[str, float]) -> Dict[str, float]:
    """对字典值做 softmax 归一化"""
    if not values:
        return {}
    max_v = max(values.values())
    exp_vals = {k: math.exp(v - max_v) for k, v in values.items()}
    total = sum(exp_vals.values())
    if total == 0:
        n = len(values)
        return {k: 1.0 / n for k in values}
    return {k: v / total for k, v in exp_vals.items()}


def min_max_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Min-Max 归一化到 [0, 1]"""
    if not values:
        return {}
    vals = list(values.values())
    v_min, v_max = min(vals), max(vals)
    if v_max == v_min:
        return {k: 0.5 for k in values}
    return {k: (v - v_min) / (v_max - v_min) for k, v in values.items()}


# ============================================================
# 算法2.1：频率回归权重
# ============================================================

def algo_frequency_regression(
    number_features: Dict[str, Dict],
    remaining: List[int],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    基于大数定律的频率回归。
    频率低于理论值的号码获得更高权重。
    raw_weight = sigmoid(deviation * scale_factor)
    """
    theoretical = config.red_prob
    scale = 15.0  # 灵敏度因子
    result = {}

    for num in remaining:
        feat = number_features.get(str(num), {})
        freq = feat.get("frequency", theoretical)
        deviation = theoretical - freq  # 正值=频率偏低
        result[str(num)] = sigmoid(deviation * scale)

    return min_max_normalize(result)


# ============================================================
# 算法2.2：遗漏值回归权重
# ============================================================

def algo_missing_regression(
    number_features: Dict[str, Dict],
    remaining: List[int],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    基于几何分布的累积概率。
    遗漏值越接近或超过平均间隔，出现概率越高。
    weight = 1 - (1-p)^missing  （几何分布CDF）
    """
    p = config.red_prob
    result = {}

    for num in remaining:
        feat = number_features.get(str(num), {})
        missing = feat.get("missing_value", 0)
        avg_gap = feat.get("avg_gap", 7.0)

        # 几何分布累积概率
        cdf = 1.0 - (1.0 - p) ** missing if missing > 0 else 0.0

        # 额外加成：遗漏值超过平均间隔时
        if avg_gap > 0 and missing > avg_gap:
            bonus = min((missing - avg_gap) / avg_gap * 0.2, 0.3)
            cdf = min(cdf + bonus, 1.0)

        result[str(num)] = cdf

    return min_max_normalize(result)


# ============================================================
# 算法2.3：时序衰减权重
# ============================================================

def algo_time_decay(
    number_features: Dict[str, Dict],
    time_features: Dict[str, Dict],
    remaining: List[int],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    指数衰减的近期热度。
    综合近5/10/20/50期的出现频率，近期权重更高。
    """
    # 衰减系数：近期窗口权重更高
    window_weights = {
        "window_5": 0.40,
        "window_10": 0.30,
        "window_20": 0.20,
        "window_50": 0.10,
    }
    window_sizes = {
        "window_5": 5,
        "window_10": 10,
        "window_20": 20,
        "window_50": 50,
    }

    result = {}
    for num in remaining:
        tf = time_features.get(str(num), {})
        score = 0.0
        for wkey, weight in window_weights.items():
            count = tf.get(wkey, 0)
            size = window_sizes[wkey]
            freq = count / size  # 窗口内频率
            score += freq * weight

        # 趋势加成
        trend = tf.get("trend", 0.0)
        if trend > 0:
            score *= (1.0 + min(trend * 10, 0.3))

        result[str(num)] = score

    return min_max_normalize(result)


# ============================================================
# 算法2.4：马尔可夫转移权重
# ============================================================

def algo_markov_transition(
    number_features: Dict[str, Dict],
    transition_matrix: List[List[float]],
    last_red_balls: List[int],
    remaining: List[int],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    基于转移矩阵的接力概率。
    从上一期出现的号码转移到候选号码的平均概率。
    """
    n = config.red_range
    result = {}

    for num in remaining:
        j = num - 1  # 0-indexed
        probs = []
        for prev in last_red_balls:
            i = prev - 1
            if 0 <= i < n and 0 <= j < n:
                probs.append(transition_matrix[i][j])
        avg_prob = sum(probs) / len(probs) if probs else 0.0
        result[str(num)] = avg_prob

    return min_max_normalize(result)


# ============================================================
# 算法2.5：共现关联权重
# ============================================================

def algo_co_occurrence(
    number_features: Dict[str, Dict],
    co_matrix: List[List[int]],
    last_red_balls: List[int],
    remaining: List[int],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    基于共现矩阵的协同过滤。
    与上一期号码共现次数多的候选号码获得更高权重。
    """
    result = {}

    for num in remaining:
        j = num - 1
        co_score = 0.0
        for prev in last_red_balls:
            i = prev - 1
            if 0 <= i < len(co_matrix) and 0 <= j < len(co_matrix[0]):
                co_score += co_matrix[i][j]
        result[str(num)] = co_score

    return min_max_normalize(result)


# ============================================================
# 算法2.6：深度学习预测权重（简化版MLP打分）
# ============================================================

def algo_deep_learning(
    number_features: Dict[str, Dict],
    time_features: Dict[str, Dict],
    remaining: List[int],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    简化版深度学习：基于多维特征的非线性打分。
    使用 numpy 实现一个简单的两层感知机，权重基于特征统计规律初始化。
    后续可替换为 PyTorch MPS 加速的 Transformer 模型。
    """
    # 构建特征向量
    vectors = {}
    for num in remaining:
        nf = number_features.get(str(num), {})
        tf = time_features.get(str(num), {})
        avg_gap = nf.get("avg_gap", 7.0)
        if avg_gap <= 0:
            avg_gap = 1.0

        vec = np.array([
            nf.get("frequency", 0.14),
            nf.get("missing_value", 0) / avg_gap,       # 归一化遗漏
            nf.get("consecutive", 0) / 3.0,              # 归一化连续
            tf.get("window_5", 0) / 5.0,
            tf.get("window_10", 0) / 10.0,
            tf.get("window_20", 0) / 20.0,
            tf.get("trend", 0.0) * 10,                   # 放大趋势
            tf.get("fft_amplitude", 0) / 10.0,
        ], dtype=np.float64)
        vectors[str(num)] = vec

    if not vectors:
        return {}

    # 简单两层MLP（固定权重，基于领域知识初始化）
    # 第一层：8 -> 4
    np.random.seed(42)
    W1 = np.array([
        [ 0.3, 0.5, -0.2, 0.4, 0.3, 0.2, 0.3, 0.1],   # 频率回归特征
        [ 0.1, 0.6,  0.0, 0.1, 0.1, 0.1, 0.0, 0.0],   # 遗漏值特征
        [ 0.2, 0.1,  0.0, 0.5, 0.4, 0.3, 0.4, 0.1],   # 热度特征
        [-0.1, 0.2, -0.3, 0.2, 0.2, 0.1, 0.2, 0.2],   # 周期特征
    ])
    b1 = np.array([-0.1, -0.2, -0.1, -0.1])

    # 第二层：4 -> 1
    W2 = np.array([[0.3, 0.3, 0.25, 0.15]])
    b2 = np.array([0.0])

    result = {}
    for num_str, vec in vectors.items():
        # 前向传播
        h = np.tanh(W1 @ vec + b1)
        out = float((W2 @ h + b2)[0])
        result[num_str] = out

    return min_max_normalize(result)


# ============================================================
# 算法2.7：位置模式权重
# ============================================================

def algo_position_pattern_weight(
    number_features: Dict[str, Dict],
    advanced_analysis: Dict[str, Any],
    remaining: List[int],
    config: LotteryConfig,
) -> Dict[str, float]:
    """
    基于位置预测加权。
    号码落在预测范围中心 → 权重升高（高斯衰减）。
    active_rules 支持的号码额外加分。
    """
    result = {}
    pos_predictions = advanced_analysis.get("position_pattern", {}).get(
        "position_predictions", {}
    )
    active_rules = advanced_analysis.get("sequence_mining", {}).get(
        "active_rules", []
    )

    # 从 active_rules 中提取支持的号码范围
    rule_boost_zones = []
    for rule in active_rules:
        decoded = rule.get("decoded", {})
        filt = decoded.get("filter", {})
        conf = rule.get("confidence", 0)
        if conf > 0.4:
            rule_boost_zones.append((filt, conf))

    for num in remaining:
        num_str = str(num)
        score = 0.0

        # 位置预测加分：高斯衰减
        for pos, pred in pos_predictions.items():
            conf = pred.get("confidence", 0)
            if conf <= 0.3:
                continue
            vr = pred.get("value_range", [])
            if len(vr) != 2:
                continue
            low, high = vr[0], vr[1]
            center = (low + high) / 2
            width = max((high - low) / 2, 1.0)

            if low <= num <= high:
                # 高斯衰减：越靠近中心分越高
                dist = abs(num - center) / width
                gaussian = math.exp(-0.5 * dist * dist)
                score += gaussian * conf

        # active_rules 加分
        mid = config.red_midpoint
        for filt, conf in rule_boost_zones:
            match = True
            if "size" in filt:
                if filt["size"] == "big" and num <= mid:
                    match = False
                if filt["size"] == "small" and num > mid:
                    match = False
            if "parity" in filt:
                if filt["parity"] == "odd" and num % 2 == 0:
                    match = False
                if filt["parity"] == "even" and num % 2 == 1:
                    match = False
            if "zone" in filt:
                zone_size = config.red_range / 3
                num_zone = min(int((num - 1) / zone_size), 2)
                if num_zone != filt["zone"]:
                    match = False
            if match:
                score += conf * 0.3

        result[num_str] = score

    return min_max_normalize(result)


# ============================================================
# 融合器：归一化加权求和
# ============================================================

def fuse_weights(
    algo_results: Dict[str, Dict[str, float]],
    remaining: List[int],
    config: LotteryConfig,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """
    归一化加权融合6个权重算法的结果。

    返回:
        number_weights: 每个号码的综合权重（softmax归一化）
        weight_details: 每个号码的权重分解详情
    """
    algo_weights = config.weight_algo_weights
    raw_scores = {}
    details = {}

    for num in remaining:
        num_str = str(num)
        weighted_sum = 0.0
        breakdown = {}

        for algo_name, algo_result in algo_results.items():
            score = algo_result.get(num_str, 0.0)
            w = algo_weights.get(algo_name, 0.0)
            weighted_sum += score * w
            breakdown[algo_name] = score

        raw_scores[num_str] = weighted_sum
        details[num_str] = {
            "raw_score": weighted_sum,
            "breakdown": breakdown,
        }

    # Softmax 归一化
    final_weights = softmax(raw_scores)

    # 更新 details 中的 final 权重
    for num_str in details:
        details[num_str]["final"] = final_weights.get(num_str, 0.0)

    return final_weights, details


# ============================================================
# 主函数：运行阶段2
# ============================================================

def run_stage2(
    stage0_result: Dict[str, Any],
    stage1_result: Dict[str, Any],
    lottery_type: str,
) -> Dict[str, Any]:
    """
    运行阶段2：权重引擎

    参数:
        stage0_result: 阶段0的输出
        stage1_result: 阶段1的输出
        lottery_type: 彩种标识

    返回:
        number_weights: 号码权重表（softmax归一化）
        weight_details: 权重分解详情
        top_numbers: 按权重降序排列的号码
        top_count: Top N 数量
    """
    config = get_config(lottery_type)
    nf = stage0_result["number_features"]
    tf = stage0_result["time_features"]
    tm = stage0_result["transition_matrix"]
    co = stage0_result["correlation_matrix"]
    remaining = stage1_result["remaining_numbers"]

    # 获取最近一期红球
    last_red = []
    for num_str, feat in nf.items():
        if feat.get("missing_value", 999) == 0:
            last_red.append(int(num_str))
    if not last_red:
        last_red = remaining[:config.red_count]

    print(f"[阶段2] 开始权重引擎，候选号码: {len(remaining)} 个")

    # 运行6个权重算法
    print("[阶段2] 算法2.1 频率回归权重...")
    w1 = algo_frequency_regression(nf, remaining, config)

    print("[阶段2] 算法2.2 遗漏值回归权重...")
    w2 = algo_missing_regression(nf, remaining, config)

    print("[阶段2] 算法2.3 时序衰减权重...")
    w3 = algo_time_decay(nf, tf, remaining, config)

    print("[阶段2] 算法2.4 马尔可夫转移权重...")
    w4 = algo_markov_transition(nf, tm, last_red, remaining, config)

    print("[阶段2] 算法2.5 共现关联权重...")
    w5 = algo_co_occurrence(nf, co, last_red, remaining, config)

    print("[阶段2] 算法2.6 深度学习预测权重...")
    w6 = algo_deep_learning(nf, tf, remaining, config)

    # 算法2.7：位置模式权重
    advanced = stage0_result.get("advanced_analysis", {})
    print("[阶段2] 算法2.7 位置模式权重...")
    w7 = algo_position_pattern_weight(nf, advanced, remaining, config)

    # 融合
    algo_results = {
        "frequency_regression": w1,
        "missing_regression": w2,
        "time_decay": w3,
        "markov_transition": w4,
        "co_occurrence": w5,
        "deep_learning": w6,
        "position_pattern": w7,
    }

    number_weights, weight_details = fuse_weights(algo_results, remaining, config)

    # 按权重降序排列
    sorted_nums = sorted(remaining, key=lambda n: number_weights.get(str(n), 0), reverse=True)
    top_count = min(15, len(remaining))

    print(f"[阶段2] 完成，Top5: {sorted_nums[:5]}")

    return {
        "number_weights": number_weights,
        "weight_details": weight_details,
        "top_numbers": sorted_nums,
        "top_count": top_count,
    }
