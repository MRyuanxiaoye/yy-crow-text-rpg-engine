# -*- coding: utf-8 -*-
"""
方向1：位置序列模式挖掘 + 差分分析

核心思路：
  大乐透前区红球 a<b<c<d<e 严格升序，每个位置有隐含范围。
  相邻期同位置号码的变化方向（升/降/平）构成方向序列，
  通过滑动窗口扫描 + 卡方检验发现显著偏离随机的模式，
  再利用当前前缀匹配历史模式进行方向预测。

输出：
  - position_predictions: 每个位置的预测值域 + 方向 + 置信度
  - direction_patterns: 每个位置的显著模式列表
  - diff_analysis: 差分统计（一阶/二阶）
  - optimal_windows: 每个位置的最优窗口大小
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict


class PositionPatternAnalyzer:
    """位置序列模式分析器"""

    def __init__(self, draws: List[Dict], red_count: int, red_range: int):
        """
        参数:
            draws: 历史开奖数据列表，每条含 "red_balls" 字段（已升序）
            red_count: 每期红球个数（大乐透=5，双色球=6）
            red_range: 红球号码范围上限（大乐透=35，双色球=33）
        """
        self.draws = draws
        self.red_count = red_count
        self.red_range = red_range

        # 按位置拆分为时间序列：position_series[pos] = [v1, v2, ...]
        self.position_series: Dict[int, List[int]] = {}
        self._extract_position_series()

    # ============================================================
    # 1. 提取位置序列
    # ============================================================

    def _extract_position_series(self):
        """从 draws 中按位置拆分为独立时间序列"""
        for pos in range(self.red_count):
            self.position_series[pos] = []

        for draw in self.draws:
            red = draw.get("red_balls", [])
            if len(red) < self.red_count:
                continue
            for pos in range(self.red_count):
                self.position_series[pos].append(red[pos])

    # ============================================================
    # 2. 方向编码
    # ============================================================

    @staticmethod
    def encode_directions(series: List[int]) -> List[str]:
        """
        相邻期变化编码为 U(升)/D(降)/E(平)
        返回长度 = len(series) - 1
        """
        directions = []
        for i in range(1, len(series)):
            diff = series[i] - series[i - 1]
            if diff > 0:
                directions.append("U")
            elif diff < 0:
                directions.append("D")
            else:
                directions.append("E")
        return directions

    # ============================================================
    # 3. 模式扫描 + 频率统计
    # ============================================================

    @staticmethod
    def scan_patterns(
        directions: List[str], window_size: int
    ) -> Counter:
        """
        滑动窗口扫描方向序列，统计每种模式的出现频率。
        返回 Counter: {pattern_tuple: count}
        """
        counts = Counter()
        for i in range(len(directions) - window_size + 1):
            pattern = tuple(directions[i:i + window_size])
            counts[pattern] += 1
        return counts

    # ============================================================
    # 4. 卡方检验：找出显著偏离随机的模式
    # ============================================================

    @staticmethod
    def chi_square_test(
        observed_counts: Counter, total_windows: int, n_categories: int
    ) -> Dict[tuple, Dict[str, float]]:
        """
        对每种模式做卡方检验，与均匀分布期望对比。

        参数:
            observed_counts: 模式频率计数
            total_windows: 总窗口数
            n_categories: 该窗口大小下的理论模式总数（3^window_size）

        返回:
            {pattern: {"observed": int, "expected": float,
                       "chi2": float, "significant": bool}}
        """
        expected = total_windows / n_categories if n_categories > 0 else 1.0
        results = {}

        for pattern, obs in observed_counts.items():
            chi2 = (obs - expected) ** 2 / expected if expected > 0 else 0.0
            # 自由度=1 的卡方分布，p<0.05 对应 chi2>3.84，p<0.01 对应 chi2>6.63
            significant = chi2 > 3.84
            results[pattern] = {
                "observed": obs,
                "expected": round(expected, 2),
                "chi2": round(chi2, 2),
                "significant": significant,
                "ratio": round(obs / expected, 2) if expected > 0 else 0,
            }

        return results

    # ============================================================
    # 5. 最优窗口搜索
    # ============================================================

    def find_optimal_window(
        self, pos: int, window_range: Tuple[int, int] = (3, 8)
    ) -> Dict[str, Any]:
        """
        对指定位置回测不同窗口大小，选预测准确率最高的。

        方法：用历史前缀匹配预测下一步方向，统计命中率。
        """
        series = self.position_series.get(pos, [])
        directions = self.encode_directions(series)

        if len(directions) < window_range[1] + 10:
            return {"best_window": 3, "accuracy": 0.0, "details": {}}

        best_window = window_range[0]
        best_accuracy = 0.0
        details = {}

        for w in range(window_range[0], window_range[1] + 1):
            # 回测：从第 w 个方向开始，用前 w 个方向预测第 w+1 个
            correct = 0
            total = 0

            # 构建历史模式库（只用到当前位置之前的数据）
            for t in range(w, len(directions)):
                prefix = tuple(directions[t - w:t])
                actual = directions[t]

                # 在 t 之前的历史中查找相同前缀
                next_dirs = []
                for h in range(w, t):
                    if tuple(directions[h - w:h]) == prefix:
                        next_dirs.append(directions[h])

                if not next_dirs:
                    continue

                # 多数投票预测
                vote = Counter(next_dirs)
                predicted = vote.most_common(1)[0][0]
                total += 1
                if predicted == actual:
                    correct += 1

            accuracy = correct / total if total > 0 else 0.0
            details[w] = {
                "accuracy": round(accuracy, 4),
                "test_count": total,
            }

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_window = w

        return {
            "best_window": best_window,
            "accuracy": round(best_accuracy, 4),
            "details": details,
        }

    # ============================================================
    # 6. 方向预测：当前前缀匹配历史
    # ============================================================

    def predict_direction(
        self, pos: int, window_size: int
    ) -> Dict[str, Any]:
        """
        用当前最近 window_size 个方向作为前缀，
        匹配历史中相同前缀后的下一步方向分布，预测下一期方向和值域。
        """
        series = self.position_series.get(pos, [])
        directions = self.encode_directions(series)

        if len(directions) < window_size:
            return {
                "direction": "unknown",
                "confidence": 0.0,
                "value_range": [1, self.red_range],
                "distribution": {},
            }

        # 当前前缀
        current_prefix = tuple(directions[-window_size:])

        # 在历史中查找匹配
        next_dirs = []
        next_values = []
        for t in range(window_size, len(directions)):
            if tuple(directions[t - window_size:t]) == current_prefix:
                next_dirs.append(directions[t])
                # 对应的实际值（t+1 因为 directions 比 series 少1）
                if t + 1 < len(series):
                    next_values.append(series[t + 1])

        if not next_dirs:
            return {
                "direction": "unknown",
                "confidence": 0.0,
                "value_range": [1, self.red_range],
                "distribution": {},
                "match_count": 0,
            }

        # 方向分布
        dir_counter = Counter(next_dirs)
        total_matches = len(next_dirs)
        distribution = {
            d: round(c / total_matches, 3)
            for d, c in dir_counter.most_common()
        }

        # 预测方向
        predicted_dir = dir_counter.most_common(1)[0][0]
        confidence = dir_counter[predicted_dir] / total_matches

        # 预测值域
        current_value = series[-1]
        if next_values:
            val_mean = np.mean(next_values)
            val_std = max(np.std(next_values), 1.0)
            val_low = max(1, int(val_mean - 1.5 * val_std))
            val_high = min(self.red_range, int(val_mean + 1.5 * val_std))
        else:
            # 根据方向估算
            if predicted_dir == "U":
                val_low = current_value + 1
                val_high = min(current_value + 8, self.red_range)
            elif predicted_dir == "D":
                val_low = max(1, current_value - 8)
                val_high = current_value - 1
            else:
                val_low = max(1, current_value - 2)
                val_high = min(current_value + 2, self.red_range)

        # 确保值域合法
        val_low = max(1, val_low)
        val_high = min(self.red_range, val_high)
        if val_low > val_high:
            val_low, val_high = val_high, val_low

        return {
            "direction": predicted_dir,
            "confidence": round(confidence, 3),
            "value_range": [val_low, val_high],
            "current_value": current_value,
            "distribution": distribution,
            "match_count": total_matches,
        }

    # ============================================================
    # 7. 差分分析
    # ============================================================

    def analyze_diff(self, pos: int) -> Dict[str, Any]:
        """
        对指定位置进行差分分析：
        - 一阶差分分布（均值、标准差、正负比）
        - 正负交替比（方向变化频率）
        - 二阶差分趋势（加速/减速）
        """
        series = self.position_series.get(pos, [])
        if len(series) < 3:
            return {}

        arr = np.array(series, dtype=float)

        # 一阶差分
        diff1 = np.diff(arr)
        pos_count = int(np.sum(diff1 > 0))
        neg_count = int(np.sum(diff1 < 0))
        zero_count = int(np.sum(diff1 == 0))
        total_d1 = len(diff1)

        # 正负交替比：相邻两个差分符号不同的比例
        alternation = 0
        for i in range(1, len(diff1)):
            if (diff1[i] > 0 and diff1[i - 1] < 0) or \
               (diff1[i] < 0 and diff1[i - 1] > 0):
                alternation += 1
        alt_ratio = alternation / (len(diff1) - 1) if len(diff1) > 1 else 0

        # 二阶差分
        diff2 = np.diff(diff1)
        diff2_mean = float(np.mean(diff2)) if len(diff2) > 0 else 0.0
        diff2_std = float(np.std(diff2)) if len(diff2) > 0 else 0.0

        # 近期趋势（最近10期的一阶差分）
        recent_diff1 = diff1[-10:] if len(diff1) >= 10 else diff1
        recent_mean = float(np.mean(recent_diff1))
        recent_std = float(np.std(recent_diff1))

        return {
            "diff1_mean": round(float(np.mean(diff1)), 3),
            "diff1_std": round(float(np.std(diff1)), 3),
            "diff1_median": round(float(np.median(diff1)), 3),
            "positive_ratio": round(pos_count / total_d1, 3) if total_d1 > 0 else 0,
            "negative_ratio": round(neg_count / total_d1, 3) if total_d1 > 0 else 0,
            "zero_ratio": round(zero_count / total_d1, 3) if total_d1 > 0 else 0,
            "alternation_ratio": round(alt_ratio, 3),
            "diff2_mean": round(diff2_mean, 3),
            "diff2_std": round(diff2_std, 3),
            "recent_diff1_mean": round(recent_mean, 3),
            "recent_diff1_std": round(recent_std, 3),
            # 趋势判断
            "trend": "加速上升" if recent_mean > 0.5 and diff2_mean > 0
                     else "减速上升" if recent_mean > 0.5 and diff2_mean <= 0
                     else "加速下降" if recent_mean < -0.5 and diff2_mean < 0
                     else "减速下降" if recent_mean < -0.5 and diff2_mean >= 0
                     else "震荡",
        }

    # ============================================================
    # 8. 主入口：运行完整分析
    # ============================================================

    def analyze(self) -> Dict[str, Any]:
        """运行完整的位置模式分析，返回所有结果"""
        if not self.position_series or not self.position_series.get(0):
            return {
                "position_predictions": {},
                "direction_patterns": {},
                "diff_analysis": {},
                "optimal_windows": {},
            }

        position_predictions = {}
        direction_patterns = {}
        diff_analysis = {}
        optimal_windows = {}

        for pos in range(self.red_count):
            series = self.position_series[pos]
            directions = self.encode_directions(series)

            if len(directions) < 10:
                continue

            # 最优窗口搜索
            opt = self.find_optimal_window(pos)
            optimal_windows[pos] = opt
            best_w = opt["best_window"]

            # 方向预测
            pred = self.predict_direction(pos, best_w)
            position_predictions[pos] = pred

            # 显著模式扫描（对多个窗口大小）
            pos_patterns = []
            for w in range(3, min(7, len(directions))):
                counts = self.scan_patterns(directions, w)
                total_windows = len(directions) - w + 1
                n_categories = 3 ** w  # U/D/E 三种方向

                chi_results = self.chi_square_test(counts, total_windows, n_categories)

                # 只保留显著的模式
                for pattern, result in chi_results.items():
                    if result["significant"] and result["ratio"] > 1.5:
                        pos_patterns.append({
                            "pattern": "".join(pattern),
                            "window": w,
                            "observed": result["observed"],
                            "expected": result["expected"],
                            "chi2": result["chi2"],
                            "ratio": result["ratio"],
                        })

            # 按 chi2 降序排列，取 top 20
            pos_patterns.sort(key=lambda x: x["chi2"], reverse=True)
            direction_patterns[pos] = pos_patterns[:20]

            # 差分分析
            diff_analysis[pos] = self.analyze_diff(pos)

        return {
            "position_predictions": position_predictions,
            "direction_patterns": direction_patterns,
            "diff_analysis": diff_analysis,
            "optimal_windows": optimal_windows,
        }
