# -*- coding: utf-8 -*-
"""
阶段0：数据预处理与特征工程

将原始开奖记录转化为统一特征矩阵，供后续所有阶段共享使用。
包含4大特征提取模块：
  - 模块0.1：基础特征（频率、遗漏值、间隔期、连续出现）
  - 模块0.2：组合特征（奇偶比、大小比、和值、跨度、连号、AC值）
  - 模块0.3：时序特征（滑动窗口、趋势、周期性FFT）
  - 模块0.4：关联特征（共现矩阵、转移矩阵）
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from pipeline.config import LotteryConfig, get_config
from pipeline.advanced_analysis import run_advanced_analysis


# ============================================================
# 数据加载
# ============================================================

def load_draws(lottery_type: str) -> List[Dict]:
    """加载历史开奖数据"""
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    file_path = data_dir / f"{lottery_type}_history.json"
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("draws", [])


# ============================================================
# 模块0.1：基础特征提取
# ============================================================

def extract_basic_features(
    draws: List[Dict], config: LotteryConfig
) -> Dict[str, Dict[str, Any]]:
    """
    提取每个号码的基础特征：
    - frequency: 历史出现频率
    - missing_value: 当前遗漏值
    - avg_gap: 平均间隔期
    - max_gap: 最大间隔期
    - consecutive: 从最近一期往回连续出现的期数
    - total_count: 总出现次数
    """
    total_periods = len(draws)
    if total_periods == 0:
        return {}

    # 记录每个号码出现的期号索引列表
    appearances: Dict[int, List[int]] = defaultdict(list)
    for idx, draw in enumerate(draws):
        for num in draw.get("red_balls", []):
            appearances[num].append(idx)

    features = {}
    last_idx = total_periods - 1

    for num in config.red_numbers:
        app_list = appearances.get(num, [])
        count = len(app_list)
        freq = count / total_periods if total_periods > 0 else 0.0

        # 遗漏值：从最后一期往回，连续多少期没出现
        if app_list:
            missing = last_idx - app_list[-1]
        else:
            missing = total_periods

        # 间隔期统计
        gaps = []
        if len(app_list) >= 2:
            for i in range(1, len(app_list)):
                gaps.append(app_list[i] - app_list[i - 1])
        avg_gap = sum(gaps) / len(gaps) if gaps else float(total_periods)
        max_gap = max(gaps) if gaps else total_periods

        # 连续出现次数：从最近一期往回
        consecutive = 0
        if app_list and app_list[-1] == last_idx:
            consecutive = 1
            for i in range(len(app_list) - 2, -1, -1):
                if app_list[i] == app_list[i + 1] - 1:
                    consecutive += 1
                else:
                    break

        features[str(num)] = {
            "frequency": freq,
            "missing_value": missing,
            "avg_gap": avg_gap,
            "max_gap": max_gap,
            "consecutive": consecutive,
            "total_count": count,
        }

    return features


# ============================================================
# 模块0.2：组合特征提取
# ============================================================

def compute_ac_value(balls: List[int]) -> int:
    """计算AC值 = 两两差值绝对值的去重个数 - (选号个数 - 1)"""
    if len(balls) < 2:
        return 0
    diffs = set()
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            diffs.add(abs(balls[i] - balls[j]))
    return len(diffs) - (len(balls) - 1)


def count_consecutive_groups(balls: List[int]) -> int:
    """计算连号组数"""
    if len(balls) < 2:
        return 0
    sorted_b = sorted(balls)
    groups = 0
    in_group = False
    for i in range(1, len(sorted_b)):
        if sorted_b[i] - sorted_b[i - 1] == 1:
            if not in_group:
                groups += 1
                in_group = True
        else:
            in_group = False
    return groups


def extract_combo_features(
    draws: List[Dict], config: LotteryConfig
) -> Dict[str, Any]:
    """
    提取组合层面的统计特征：
    - odd_even_dist: 奇偶比分布
    - big_small_dist: 大小比分布
    - sum_values: 和值序列
    - span_values: 跨度序列
    - ac_values: AC值序列
    - consecutive_groups: 连号组数序列
    """
    odd_even_dist: Dict[str, int] = defaultdict(int)
    big_small_dist: Dict[str, int] = defaultdict(int)
    sum_values = []
    span_values = []
    ac_values = []
    consec_groups = []

    mid = config.red_midpoint

    for draw in draws:
        red = draw.get("red_balls", [])
        if not red:
            continue

        # 奇偶比
        odd_count = sum(1 for b in red if b % 2 == 1)
        even_count = len(red) - odd_count
        odd_even_dist[f"{odd_count}:{even_count}"] += 1

        # 大小比
        big_count = sum(1 for b in red if b > mid)
        small_count = len(red) - big_count
        big_small_dist[f"{big_count}:{small_count}"] += 1

        # 和值、跨度
        sum_values.append(sum(red))
        span_values.append(max(red) - min(red))

        # AC值
        ac_values.append(compute_ac_value(red))

        # 连号组数
        consec_groups.append(count_consecutive_groups(red))

    return {
        "odd_even_dist": dict(odd_even_dist),
        "big_small_dist": dict(big_small_dist),
        "sum_values": sum_values,
        "span_values": span_values,
        "ac_values": ac_values,
        "consecutive_groups": consec_groups,
        # 统计摘要
        "sum_mean": float(np.mean(sum_values)) if sum_values else 0,
        "sum_std": float(np.std(sum_values)) if sum_values else 0,
        "span_mean": float(np.mean(span_values)) if span_values else 0,
        "ac_mean": float(np.mean(ac_values)) if ac_values else 0,
    }


# ============================================================
# 模块0.3：时序特征提取
# ============================================================

def extract_time_features(
    draws: List[Dict], config: LotteryConfig
) -> Dict[str, Dict[str, Any]]:
    """
    提取每个号码的时序特征：
    - window_5/10/20/50: 近N期出现次数
    - trend: 趋势斜率（正=上升，负=下降）
    - trend_label: 上升/下降/平稳
    - fft_main_period: FFT主周期
    - fft_amplitude: FFT主频振幅
    """
    total = len(draws)
    if total == 0:
        return {}

    # 构建每个号码的出现序列（0/1）
    appearance_seq: Dict[int, List[int]] = {}
    for num in config.red_numbers:
        appearance_seq[num] = [0] * total

    for idx, draw in enumerate(draws):
        for num in draw.get("red_balls", []):
            if num in appearance_seq:
                appearance_seq[num][idx] = 1

    features = {}
    windows = [5, 10, 20, 50]

    for num in config.red_numbers:
        seq = appearance_seq[num]
        feat: Dict[str, Any] = {}

        # 滑动窗口统计
        for w in windows:
            if total >= w:
                feat[f"window_{w}"] = sum(seq[-w:])
            else:
                feat[f"window_{w}"] = sum(seq)

        # 趋势：对近50期做线性回归
        trend_window = min(50, total)
        recent = seq[-trend_window:]
        if trend_window >= 5:
            x = np.arange(trend_window)
            y = np.array(recent, dtype=float)
            # 最小二乘法求斜率
            x_mean = x.mean()
            y_mean = y.mean()
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            denominator = ((x - x_mean) ** 2).sum()
            slope = numerator / denominator if denominator > 0 else 0.0
            feat["trend"] = float(slope)
            if slope > 0.005:
                feat["trend_label"] = "上升"
            elif slope < -0.005:
                feat["trend_label"] = "下降"
            else:
                feat["trend_label"] = "平稳"
        else:
            feat["trend"] = 0.0
            feat["trend_label"] = "平稳"

        # FFT周期性分析
        if total >= 20:
            y = np.array(seq, dtype=float)
            y = y - y.mean()  # 去均值
            fft_vals = np.fft.rfft(y)
            magnitudes = np.abs(fft_vals)
            # 跳过直流分量（index 0）
            if len(magnitudes) > 1:
                main_idx = np.argmax(magnitudes[1:]) + 1
                feat["fft_main_period"] = float(total / main_idx) if main_idx > 0 else 0.0
                feat["fft_amplitude"] = float(magnitudes[main_idx])
            else:
                feat["fft_main_period"] = 0.0
                feat["fft_amplitude"] = 0.0
        else:
            feat["fft_main_period"] = 0.0
            feat["fft_amplitude"] = 0.0

        features[str(num)] = feat

    return features


# ============================================================
# 模块0.4：关联特征提取
# ============================================================

def extract_correlation_features(
    draws: List[Dict], config: LotteryConfig
) -> Dict[str, Any]:
    """
    提取号码间的关联特征：
    - co_occurrence_matrix: 共现矩阵 (N x N)
    - transition_matrix: 一阶马尔可夫转移矩阵 (N x N)
    """
    n = config.red_range
    total = len(draws)

    # 共现矩阵
    co_matrix = np.zeros((n, n), dtype=int)
    for draw in draws:
        red = draw.get("red_balls", [])
        for i in range(len(red)):
            for j in range(i + 1, len(red)):
                a, b = red[i] - 1, red[j] - 1  # 转为0-indexed
                co_matrix[a][b] += 1
                co_matrix[b][a] += 1

    # 转移矩阵：上一期出现号码i，下一期出现号码j的次数
    trans_count = np.zeros((n, n), dtype=int)
    for idx in range(1, total):
        prev_red = set(draws[idx - 1].get("red_balls", []))
        curr_red = set(draws[idx].get("red_balls", []))
        for i in prev_red:
            for j in curr_red:
                trans_count[i - 1][j - 1] += 1

    # 归一化为概率
    row_sums = trans_count.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # 避免除零
    trans_matrix = trans_count / row_sums

    return {
        "co_occurrence_matrix": co_matrix.tolist(),
        "transition_matrix": trans_matrix.tolist(),
    }


# ============================================================
# 出现矩阵（用于可视化）
# ============================================================

def build_appearance_matrix(
    draws: List[Dict], config: LotteryConfig, recent: int = 30
) -> List[List[int]]:
    """构建最近N期的号码出现矩阵，行=号码，列=期"""
    use_draws = draws[-recent:] if len(draws) >= recent else draws
    matrix = []
    for num in config.red_numbers:
        row = []
        for draw in use_draws:
            row.append(1 if num in draw.get("red_balls", []) else 0)
        matrix.append(row)
    return matrix


# ============================================================
# 主函数：运行阶段0
# ============================================================

def run_stage0(
    lottery_type: str,
    draws: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    运行阶段0：数据预处理与特征工程

    参数:
        lottery_type: 彩种标识 ("daletou" 或 "shuangseqiu")
        draws: 可选，直接传入开奖数据；为None时从文件加载

    返回:
        特征矩阵字典，包含:
        - number_features: 每个号码的基础特征
        - combo_stats: 组合层面统计
        - time_features: 时序特征
        - correlation_matrix: 共现矩阵
        - transition_matrix: 转移矩阵
        - appearance_matrix: 出现矩阵（可视化用）
        - metadata: 元数据
    """
    config = get_config(lottery_type)

    if draws is None:
        draws = load_draws(lottery_type)

    total = len(draws)
    print(f"[阶段0] 加载 {config.name} 数据: {total} 期")

    # 模块0.1：基础特征
    print("[阶段0] 提取基础特征...")
    number_features = extract_basic_features(draws, config)

    # 模块0.2：组合特征
    print("[阶段0] 提取组合特征...")
    combo_stats = extract_combo_features(draws, config)

    # 模块0.3：时序特征
    print("[阶段0] 提取时序特征...")
    time_features = extract_time_features(draws, config)

    # 模块0.4：关联特征
    print("[阶段0] 提取关联特征...")
    correlation = extract_correlation_features(draws, config)

    # 出现矩阵
    appearance_matrix = build_appearance_matrix(draws, config)

    # 模块0.5：高级分析（位置模式 + 序列挖掘 + 信息论）
    print("[阶段0] 运行高级分析模块...")
    advanced_analysis = run_advanced_analysis(draws, config.red_count, config.red_range)

    print(f"[阶段0] 完成，提取了 {len(number_features)} 个号码的特征")

    return {
        "number_features": number_features,
        "combo_stats": combo_stats,
        "time_features": time_features,
        "correlation_matrix": correlation["co_occurrence_matrix"],
        "transition_matrix": correlation["transition_matrix"],
        "appearance_matrix": appearance_matrix,
        "advanced_analysis": advanced_analysis,
        "metadata": {
            "lottery_type": lottery_type,
            "total_numbers": config.red_range,
            "blue_range": config.blue_range,
            "total_draws": total,
            "latest_period": draws[-1]["period"] if draws else "",
        },
    }
