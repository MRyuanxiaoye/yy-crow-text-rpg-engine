# -*- coding: utf-8 -*-
"""
A4: 滑动窗口统计量扫描

对每期的组合统计量（和值、跨度、AC值、奇偶比、大小比、连号组数）
在不同窗口大小下计算滑动统计量，寻找异常模式和周期性。

扫描内容：
  - 滑动均值/标准差的趋势
  - 统计量突变检测
  - 统计量之间的滞后相关
  - 统计量的周期性（FFT）
  - 统计量组合模式
"""

import json
import time
import numpy as np
from itertools import combinations
from collections import Counter
from typing import Dict, List, Any, Tuple
from pathlib import Path

from research.config import ResearchConfig, RULES_DIR
from research.data_loader import LotteryData


# ============================================================
# 滑动窗口计算
# ============================================================

def sliding_mean_std(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """计算滑动均值和标准差"""
    n = len(series)
    if n < window:
        return np.array([]), np.array([])

    means = np.zeros(n - window + 1)
    stds = np.zeros(n - window + 1)
    for i in range(n - window + 1):
        w = series[i:i + window].astype(np.float64)
        means[i] = np.mean(w)
        stds[i] = np.std(w)
    return means, stds


def detect_changepoints(series: np.ndarray, window: int = 20, threshold: float = 2.0) -> List[Dict]:
    """
    突变点检测：比较前后窗口的均值差异。
    当差异超过 threshold 倍标准差时标记为突变。
    """
    n = len(series)
    if n < 2 * window:
        return []

    changes = []
    global_std = np.std(series)
    if global_std == 0:
        return []

    for t in range(window, n - window):
        before = series[t - window:t].astype(np.float64)
        after = series[t:t + window].astype(np.float64)
        diff = abs(np.mean(after) - np.mean(before))
        if diff > threshold * global_std:
            changes.append({
                "position": int(t),
                "before_mean": round(float(np.mean(before)), 2),
                "after_mean": round(float(np.mean(after)), 2),
                "diff_in_std": round(float(diff / global_std), 2),
            })

    return changes


def compute_fft_periods(series: np.ndarray, top_k: int = 5) -> List[Dict]:
    """
    用 FFT 检测周期性。
    返回最强的 top_k 个周期。
    """
    n = len(series)
    if n < 10:
        return []

    # 去均值
    centered = series.astype(np.float64) - np.mean(series)
    fft_vals = np.fft.rfft(centered)
    magnitudes = np.abs(fft_vals)

    # 跳过直流分量（index 0）
    if len(magnitudes) < 2:
        return []

    magnitudes[0] = 0

    # 找最大的 top_k 个频率
    top_indices = np.argsort(magnitudes)[::-1][:top_k]

    results = []
    total_power = np.sum(magnitudes ** 2)
    for idx in top_indices:
        if idx == 0:
            continue
        period = n / idx
        power = float(magnitudes[idx] ** 2)
        power_ratio = power / total_power if total_power > 0 else 0
        if period < 2 or power_ratio < 0.01:
            continue
        results.append({
            "period": round(period, 2),
            "frequency_index": int(idx),
            "magnitude": round(float(magnitudes[idx]), 2),
            "power_ratio": round(power_ratio, 4),
        })

    return results


def lagged_correlation(series_a: np.ndarray, series_b: np.ndarray, max_lag: int = 10) -> List[Dict]:
    """计算两个序列之间的滞后相关"""
    n = min(len(series_a), len(series_b))
    if n < max_lag + 10:
        return []

    a = series_a[:n].astype(np.float64)
    b = series_b[:n].astype(np.float64)

    std_a = np.std(a)
    std_b = np.std(b)
    if std_a == 0 or std_b == 0:
        return []

    results = []
    threshold = 2 / (n ** 0.5)

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = np.corrcoef(a[:n - lag], b[lag:n])[0, 1] if n - lag > 5 else 0
        else:
            corr = np.corrcoef(a[-lag:n], b[:n + lag])[0, 1] if n + lag > 5 else 0

        if abs(corr) > threshold:
            results.append({
                "lag": lag,
                "correlation": round(float(corr), 4),
            })

    return results


# ============================================================
# 统计量离散化 + 模式扫描
# ============================================================

def discretize_to_levels(series: np.ndarray, n_levels: int = 3) -> np.ndarray:
    """将连续序列离散化为 n_levels 个等级"""
    percentiles = np.linspace(0, 100, n_levels + 1)
    edges = np.percentile(series, percentiles)
    edges = np.unique(edges)
    actual_levels = len(edges) - 1
    if actual_levels < 1:
        return np.zeros(len(series), dtype=np.int8)

    result = np.zeros(len(series), dtype=np.int8)
    for i in range(actual_levels):
        if i < actual_levels - 1:
            mask = (series >= edges[i]) & (series < edges[i + 1])
        else:
            mask = (series >= edges[i]) & (series <= edges[i + 1])
        result[mask] = i
    return result


def scan_stat_patterns(
    discrete_series: np.ndarray,
    n_levels: int,
    window_range: Tuple[int, int],
    min_support: int = 5,
) -> List[Dict]:
    """扫描离散化统计量序列的模式"""
    n = len(discrete_series)
    results = []

    for window in range(window_range[0], min(window_range[1] + 1, n)):
        pattern_counts = Counter()
        pattern_next = {}

        for t in range(n - window):
            key = tuple(int(discrete_series[t + k]) for k in range(window))
            pattern_counts[key] += 1
            if t + window < n:
                nxt = int(discrete_series[t + window])
                if key not in pattern_next:
                    pattern_next[key] = Counter()
                pattern_next[key][nxt] += 1

        total = n - window
        n_possible = n_levels ** window
        expected = total / n_possible if n_possible > 0 else 1

        for pattern_key, observed in pattern_counts.items():
            if observed < min_support:
                continue
            chi2 = (observed - expected) ** 2 / expected if expected > 0 else 0
            if chi2 < 6.63:
                continue

            next_counter = pattern_next.get(pattern_key, Counter())
            next_total = sum(next_counter.values())
            best = next_counter.most_common(1)
            pred_conf = best[0][1] / next_total if best and next_total > 0 else 0

            results.append({
                "window": window,
                "pattern": list(pattern_key),
                "observed": observed,
                "expected": round(expected, 2),
                "chi2": round(chi2, 2),
                "prediction": int(best[0][0]) if best else -1,
                "prediction_confidence": round(pred_conf, 4),
            })

    results.sort(key=lambda x: x["chi2"], reverse=True)
    return results


# ============================================================
# 主分析器
# ============================================================

class SlidingStatsScanner:
    """滑动窗口统计量扫描器"""

    def __init__(self, data: LotteryData, config: ResearchConfig):
        self.data = data
        self.config = config
        self.results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """执行全部滑动窗口分析"""
        start = time.time()

        combo_stats = self.data.get_combo_stats_series()
        stat_names = list(combo_stats.keys())
        print(f"[A4] 滑动窗口分析: {len(stat_names)} 个统计量, "
              f"窗口 {self.config.sliding_window_range}")

        results = {
            "sliding_stats": {},
            "changepoints": {},
            "periodicity": {},
            "cross_stat_correlation": {},
            "stat_patterns": {},
        }

        # 1. 每个统计量的滑动分析
        for stat_name in stat_names:
            series = combo_stats[stat_name]
            print(f"[A4] 分析统计量: {stat_name} (长度 {len(series)})")

            # 滑动均值/标准差（选几个代表性窗口）
            windows_to_check = [5, 10, 20, 30, 50]
            sliding_info = {}
            for w in windows_to_check:
                if w > len(series):
                    continue
                means, stds = sliding_mean_std(series, w)
                if len(means) == 0:
                    continue
                sliding_info[f"window_{w}"] = {
                    "mean_of_means": round(float(np.mean(means)), 4),
                    "std_of_means": round(float(np.std(means)), 4),
                    "mean_of_stds": round(float(np.mean(stds)), 4),
                    "trend": round(float(np.corrcoef(
                        np.arange(len(means)), means
                    )[0, 1]) if len(means) > 2 else 0, 4),
                }
            results["sliding_stats"][stat_name] = sliding_info

            # 突变检测
            changes = detect_changepoints(series, window=20, threshold=2.0)
            results["changepoints"][stat_name] = {
                "n_changepoints": len(changes),
                "changepoints": changes[:20],
            }

            # 周期性
            periods = compute_fft_periods(series, top_k=5)
            results["periodicity"][stat_name] = {
                "dominant_periods": periods,
            }

        elapsed = time.time() - start
        print(f"[A4] 基础分析完成, 耗时 {elapsed:.1f}s")

        # 2. 统计量之间的滞后相关
        print(f"[A4] 计算统计量间滞后相关...")
        for name_a, name_b in combinations(stat_names, 2):
            key = f"{name_a}_vs_{name_b}"
            lag_corrs = lagged_correlation(
                combo_stats[name_a], combo_stats[name_b], max_lag=10
            )
            results["cross_stat_correlation"][key] = {
                "significant_lags": lag_corrs,
                "n_significant": len(lag_corrs),
            }

        elapsed = time.time() - start
        print(f"[A4] 滞后相关完成, 耗时 {elapsed:.1f}s")

        # 3. 统计量模式扫描
        print(f"[A4] 统计量模式扫描...")
        total_patterns = 0
        # 限制窗口范围避免组合爆炸
        pattern_window = (3, min(15, self.config.sliding_window_range[1]))
        n_levels = getattr(self.config, 'sliding_n_levels', 3)

        for stat_name in stat_names:
            series = combo_stats[stat_name]
            discrete = discretize_to_levels(series, n_levels=n_levels)
            patterns = scan_stat_patterns(
                discrete, n_levels, pattern_window,
                self.config.direction_min_support,
            )
            results["stat_patterns"][stat_name] = {
                "n_patterns": len(patterns),
                "top_patterns": patterns[:100],
            }
            total_patterns += len(patterns)

        # 4. 联合统计量模式（两两组合）
        print(f"[A4] 联合统计量模式扫描...")
        joint_pattern_window = (3, min(10, self.config.sliding_window_range[1]))
        for name_a, name_b in combinations(stat_names, 2):
            key = f"{name_a}+{name_b}"
            disc_a = discretize_to_levels(combo_stats[name_a], n_levels=n_levels)
            disc_b = discretize_to_levels(combo_stats[name_b], n_levels=n_levels)
            # 联合编码
            joint = disc_a * n_levels + disc_b
            patterns = scan_stat_patterns(
                joint, n_levels * n_levels, joint_pattern_window,
                self.config.direction_min_support,
            )
            results["stat_patterns"][key] = {
                "n_patterns": len(patterns),
                "top_patterns": patterns[:50],
            }
            total_patterns += len(patterns)

        elapsed = time.time() - start
        print(f"[A4] 完成: 共发现 {total_patterns} 个统计量模式, 耗时 {elapsed:.1f}s")

        self.results = results
        return results

    def save(self, filename: str = "a4_sliding_stats.json"):
        """保存结果"""
        RULES_DIR.mkdir(parents=True, exist_ok=True)
        path = RULES_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "lottery_type": self.data.lottery_type,
                "analysis": self.results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[A4] 结果已保存: {path}")

    def summary(self) -> Dict[str, Any]:
        """生成摘要"""
        if not self.results:
            return {"status": "未运行"}

        # 统计突变点
        total_changes = sum(
            info.get("n_changepoints", 0)
            for info in self.results.get("changepoints", {}).values()
        )

        # 统计周期
        all_periods = []
        for info in self.results.get("periodicity", {}).values():
            for p in info.get("dominant_periods", []):
                if p.get("power_ratio", 0) > 0.05:
                    all_periods.append(p)

        # 统计模式
        total_patterns = sum(
            info.get("n_patterns", 0)
            for info in self.results.get("stat_patterns", {}).values()
        )

        return {
            "total_changepoints": total_changes,
            "strong_periods": all_periods[:10],
            "total_stat_patterns": total_patterns,
        }
