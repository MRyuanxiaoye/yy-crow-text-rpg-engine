# -*- coding: utf-8 -*-
"""
A3: 差分空间全分析

对每个位置的时间序列进行多阶差分分析：
  - 1-4阶差分的分布统计
  - 差分序列的自相关函数（ACF）
  - 差分序列的游程检验（随机性检验）
  - 跨位置差分的相关性
  - 差分模式扫描（类似A1但在差分空间）
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
# 自相关函数
# ============================================================

def compute_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """计算自相关函数"""
    n = len(series)
    if n < max_lag + 1:
        max_lag = n - 1
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    centered = series - mean
    for lag in range(1, max_lag + 1):
        acf[lag] = np.sum(centered[:n - lag] * centered[lag:]) / (n * var)
    return acf


# ============================================================
# 游程检验
# ============================================================

def runs_test(series: np.ndarray) -> Dict[str, Any]:
    """
    游程检验：检验序列是否随机。
    将序列按中位数二值化，统计游程数。
    """
    median = np.median(series)
    binary = (series > median).astype(int)
    n = len(binary)
    if n < 10:
        return {"z_stat": 0, "p_approx": 1.0, "n_runs": 0}

    # 计算游程数
    runs = 1
    for i in range(1, n):
        if binary[i] != binary[i - 1]:
            runs += 1

    n1 = int(np.sum(binary))
    n0 = n - n1

    if n0 == 0 or n1 == 0:
        return {"z_stat": 0, "p_approx": 1.0, "n_runs": runs}

    # 期望游程数和标准差
    expected = 1 + 2 * n0 * n1 / n
    var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
    if var <= 0:
        return {"z_stat": 0, "p_approx": 1.0, "n_runs": runs}

    std = var ** 0.5
    z = (runs - expected) / std

    # 近似 p 值（双侧）
    # |z| > 1.96 → p < 0.05, |z| > 2.58 → p < 0.01
    p_approx = 2 * (1 - _normal_cdf(abs(z)))

    return {
        "n_runs": runs,
        "expected_runs": round(expected, 2),
        "z_stat": round(z, 4),
        "p_approx": round(p_approx, 6),
        "is_random": abs(z) < 1.96,
    }


def _normal_cdf(x: float) -> float:
    """标准正态分布 CDF 近似"""
    # Abramowitz and Stegun 近似
    if x < 0:
        return 1 - _normal_cdf(-x)
    t = 1 / (1 + 0.2316419 * x)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
           t * (-1.821255978 + t * 1.330274429))))
    return 1 - d * np.exp(-x * x / 2) * poly


# ============================================================
# 差分模式扫描
# ============================================================

def scan_diff_patterns(
    diff_series: np.ndarray,
    window_range: Tuple[int, int],
    min_support: int = 5,
) -> List[Dict]:
    """
    在差分序列上扫描模式。
    将差分值离散化为：大降/小降/平/小升/大升 五类。
    """
    n = len(diff_series)
    if n < window_range[0] + 1:
        return []

    # 离散化：按五分位
    percentiles = [20, 40, 60, 80]
    thresholds = np.percentile(diff_series, percentiles)

    def discretize(val):
        if val <= thresholds[0]:
            return 0  # 大降
        elif val <= thresholds[1]:
            return 1  # 小降
        elif val <= thresholds[2]:
            return 2  # 平
        elif val <= thresholds[3]:
            return 3  # 小升
        else:
            return 4  # 大升

    discrete = np.array([discretize(v) for v in diff_series], dtype=np.int8)
    label_map = {0: "大降", 1: "小降", 2: "平", 3: "小升", 4: "大升"}

    results = []
    for window in range(window_range[0], min(window_range[1] + 1, n)):
        pattern_counts = Counter()
        pattern_next = {}

        for t in range(n - window):
            key = tuple(int(discrete[t + k]) for k in range(window))
            pattern_counts[key] += 1

            if t + window < n:
                next_val = int(discrete[t + window])
                if key not in pattern_next:
                    pattern_next[key] = Counter()
                pattern_next[key][next_val] += 1

        total = n - window
        n_possible = 5 ** window
        expected = total / n_possible if n_possible > 0 else 1

        for pattern_key, observed in pattern_counts.items():
            if observed < min_support:
                continue

            chi2 = (observed - expected) ** 2 / expected if expected > 0 else 0
            if chi2 < 6.63:  # p < 0.01
                continue

            next_counter = pattern_next.get(pattern_key, Counter())
            next_total = sum(next_counter.values())
            best_next = next_counter.most_common(1)
            pred_conf = best_next[0][1] / next_total if best_next and next_total > 0 else 0

            pattern_str = "→".join(label_map[d] for d in pattern_key)
            pred_label = label_map.get(best_next[0][0], "?") if best_next else "?"

            results.append({
                "window": window,
                "pattern": pattern_str,
                "observed": observed,
                "expected": round(expected, 2),
                "chi2": round(chi2, 2),
                "prediction": pred_label,
                "prediction_confidence": round(pred_conf, 4),
            })

    results.sort(key=lambda x: x["chi2"], reverse=True)
    return results


# ============================================================
# 主分析器
# ============================================================

class DiffAnalyzer:
    """差分空间全分析器"""

    def __init__(self, data: LotteryData, config: ResearchConfig):
        self.data = data
        self.config = config
        self.results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """执行全部差分分析"""
        start = time.time()
        rc = self.data.red_count

        results = {
            "diff_distributions": {},
            "acf_analysis": {},
            "runs_tests": {},
            "cross_position_correlation": {},
            "diff_patterns": {},
        }

        # 1. 各位置各阶差分的分布统计 + ACF + 游程检验
        print(f"[A3] 开始差分分析: {rc} 个位置, 最大 {self.config.diff_max_order} 阶")

        for pos in range(rc):
            pos_key = f"P{pos}"
            results["diff_distributions"][pos_key] = {}
            results["acf_analysis"][pos_key] = {}
            results["runs_tests"][pos_key] = {}

            for order in range(1, self.config.diff_max_order + 1):
                diff = self.data.get_diff_series(pos, order)
                if len(diff) < 10:
                    continue

                order_key = f"order_{order}"

                # 分布统计
                results["diff_distributions"][pos_key][order_key] = {
                    "mean": round(float(np.mean(diff)), 4),
                    "std": round(float(np.std(diff)), 4),
                    "median": round(float(np.median(diff)), 4),
                    "min": round(float(np.min(diff)), 4),
                    "max": round(float(np.max(diff)), 4),
                    "skewness": round(float(_skewness(diff)), 4),
                    "kurtosis": round(float(_kurtosis(diff)), 4),
                    "n": len(diff),
                    "positive_ratio": round(float(np.mean(diff > 0)), 4),
                    "negative_ratio": round(float(np.mean(diff < 0)), 4),
                    "zero_ratio": round(float(np.mean(diff == 0)), 4),
                }

                # ACF
                acf = compute_acf(diff, self.config.diff_acf_max_lag)
                # 找显著滞后（超过 2/sqrt(n) 的置信带）
                threshold = 2 / (len(diff) ** 0.5)
                significant_lags = [
                    {"lag": int(lag), "acf": round(float(acf[lag]), 4)}
                    for lag in range(1, len(acf))
                    if abs(acf[lag]) > threshold
                ]
                results["acf_analysis"][pos_key][order_key] = {
                    "significant_lags": significant_lags[:20],
                    "n_significant": len(significant_lags),
                    "threshold": round(threshold, 4),
                }

                # 游程检验
                results["runs_tests"][pos_key][order_key] = runs_test(diff)

        elapsed = time.time() - start
        print(f"[A3] 差分分布+ACF+游程检验完成, 耗时 {elapsed:.1f}s")

        # 2. 跨位置差分相关性
        print(f"[A3] 计算跨位置差分相关性...")
        for pos_i, pos_j in combinations(range(rc), 2):
            cross_diff = self.data.get_cross_diff_series(pos_i, pos_j)
            key = f"P{pos_i}-P{pos_j}"

            # 一阶差分的相关性
            diff_i = self.data.get_diff_series(pos_i, 1)
            diff_j = self.data.get_diff_series(pos_j, 1)
            min_len = min(len(diff_i), len(diff_j))
            if min_len > 10:
                corr = float(np.corrcoef(diff_i[:min_len], diff_j[:min_len])[0, 1])
            else:
                corr = 0

            results["cross_position_correlation"][key] = {
                "diff_correlation": round(corr, 4),
                "cross_diff_mean": round(float(np.mean(cross_diff)), 4),
                "cross_diff_std": round(float(np.std(cross_diff)), 4),
                "runs_test": runs_test(cross_diff),
            }

        elapsed = time.time() - start
        print(f"[A3] 跨位置分析完成, 耗时 {elapsed:.1f}s")

        # 3. 差分模式扫描
        print(f"[A3] 差分模式扫描...")
        total_patterns = 0
        for pos in range(rc):
            pos_key = f"P{pos}"
            results["diff_patterns"][pos_key] = {}

            for order in range(1, min(3, self.config.diff_max_order + 1)):
                diff = self.data.get_diff_series(pos, order)
                if len(diff) < self.config.diff_window_range[0] + 1:
                    continue

                patterns = scan_diff_patterns(
                    diff,
                    self.config.diff_window_range,
                    self.config.direction_min_support,
                )
                results["diff_patterns"][pos_key][f"order_{order}"] = {
                    "n_patterns": len(patterns),
                    "top_patterns": patterns[:50],
                }
                total_patterns += len(patterns)

        elapsed = time.time() - start
        print(f"[A3] 完成: 共发现 {total_patterns} 个差分模式, 耗时 {elapsed:.1f}s")

        self.results = results
        return results

    def save(self, filename: str = "a3_diff_analysis.json"):
        """保存结果"""
        RULES_DIR.mkdir(parents=True, exist_ok=True)
        path = RULES_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "lottery_type": self.data.lottery_type,
                "analysis": self.results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[A3] 结果已保存: {path}")

    def summary(self) -> Dict[str, Any]:
        """生成摘要"""
        if not self.results:
            return {"status": "未运行"}

        # 统计非随机的序列
        non_random = []
        for pos_key, orders in self.results.get("runs_tests", {}).items():
            for order_key, rt in orders.items():
                if isinstance(rt, dict) and not rt.get("is_random", True):
                    non_random.append(f"{pos_key}_{order_key}")

        # 统计显著ACF
        sig_acf_count = 0
        for pos_key, orders in self.results.get("acf_analysis", {}).items():
            for order_key, acf_info in orders.items():
                sig_acf_count += acf_info.get("n_significant", 0)

        # 差分模式总数
        total_patterns = 0
        for pos_key, orders in self.results.get("diff_patterns", {}).items():
            for order_key, info in orders.items():
                total_patterns += info.get("n_patterns", 0)

        return {
            "non_random_series": non_random,
            "n_non_random": len(non_random),
            "total_significant_acf_lags": sig_acf_count,
            "total_diff_patterns": total_patterns,
        }


# ============================================================
# 辅助统计函数
# ============================================================

def _skewness(x: np.ndarray) -> float:
    """偏度"""
    n = len(x)
    if n < 3:
        return 0
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return float(np.mean(((x - mean) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """峰度（超额）"""
    n = len(x)
    if n < 4:
        return 0
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return float(np.mean(((x - mean) / std) ** 4) - 3)
