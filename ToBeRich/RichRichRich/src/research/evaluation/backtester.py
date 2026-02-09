# -*- coding: utf-8 -*-
"""
C1: 回测验证器

对发现的规则进行滚动窗口回测：
  - 用前 70% 数据发现规则
  - 在后 30% 数据上逐期验证
  - 统计命中率、覆盖率、排除效率
"""

import json
import time
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

from research.config import ResearchConfig, RULES_DIR, REPORTS_DIR
from research.data_loader import LotteryData


class Backtester:
    """回测验证器"""

    def __init__(self, data: LotteryData, config: ResearchConfig):
        self.data = data
        self.config = config
        self.results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """对 A1 方向模式规则进行回测"""
        start = time.time()
        n = self.data.n_draws
        split = int(n * self.config.backtest_train_ratio)
        split = max(split, self.config.backtest_min_train)
        test_start = split
        test_end = n

        print(f"[C1] 回测: 训练期 0-{split}, 测试期 {test_start}-{test_end}")

        # 加载 A1 规则
        a1_path = RULES_DIR / f"a1_direction_patterns_{self.data.lottery_type}.json"
        if not a1_path.exists():
            print("[C1] A1 规则文件不存在，跳过")
            return {}

        with open(a1_path, "r", encoding="utf-8") as f:
            a1_data = json.load(f)

        patterns = a1_data.get("patterns", [])
        if not patterns:
            print("[C1] 无规则可回测")
            return {}

        rc = self.data.red_count

        # 按位置筛选单位置规则（取每个位置 top 100）
        pos_patterns = {pos: [] for pos in range(rc)}
        for pat in patterns:
            positions = pat.get("positions", [])
            if len(positions) == 1:
                pos = positions[0]
                if pos < rc and len(pos_patterns[pos]) < 100:
                    pos_patterns[pos].append(pat)

        total_rules = sum(len(v) for v in pos_patterns.values())
        print(f"[C1] 回测 {total_rules} 条单位置规则")

        # 为每个位置建立模式查找表：pattern_key_tuple -> pattern
        pos_lookup = {}
        for pos in range(rc):
            lookup = {}
            for pat in pos_patterns[pos]:
                key = tuple(pat.get("pattern_key", []))
                if key not in lookup:
                    lookup[key] = pat
            pos_lookup[pos] = lookup

        # 逐期回测
        hits = 0
        misses = 0
        no_match = 0
        per_pos_hits = {pos: 0 for pos in range(rc)}
        per_pos_total = {pos: 0 for pos in range(rc)}

        for t in range(test_start, test_end - 1):
            for pos in range(rc):
                dir_seq = self.data.direction_series[pos]
                if t >= len(dir_seq):
                    continue
                actual_dir = int(dir_seq[t])

                # 尝试不同窗口大小匹配
                matched = False
                lookup = pos_lookup[pos]
                for pat_key, pat in lookup.items():
                    window = len(pat_key)
                    if t < window:
                        continue

                    # 构建当前窗口编码: dir+1 映射 -1→0, 0→1, 1→2
                    current = tuple(
                        int(dir_seq[t - window + k]) + 1
                        for k in range(window)
                        if t - window + k >= 0 and t - window + k < len(dir_seq)
                    )
                    if len(current) != window:
                        continue

                    if current == pat_key:
                        matched = True
                        next_dist = pat.get("next_distribution", {})
                        if next_dist:
                            best_next = max(next_dist.items(),
                                            key=lambda x: x[1].get("prob", 0))
                            # next key 格式: "(-1,)" 或 "(0,)" 或 "(1,)"
                            try:
                                pred_tuple = eval(best_next[0])
                                if isinstance(pred_tuple, tuple):
                                    pred_dir = pred_tuple[0]
                                else:
                                    pred_dir = pred_tuple
                            except:
                                pred_dir = None

                            if pred_dir is not None:
                                per_pos_total[pos] += 1
                                if pred_dir == actual_dir:
                                    hits += 1
                                    per_pos_hits[pos] += 1
                                else:
                                    misses += 1
                        break

                if not matched:
                    no_match += 1

        total_predictions = hits + misses
        accuracy = hits / total_predictions if total_predictions > 0 else 0
        coverage = total_predictions / ((test_end - test_start - 1) * rc) if (test_end - test_start - 1) > 0 else 0

        per_pos_accuracy = {}
        for pos in range(rc):
            t = per_pos_total[pos]
            h = per_pos_hits[pos]
            per_pos_accuracy[f"P{pos}"] = {
                "predictions": t,
                "hits": h,
                "accuracy": round(h / t, 4) if t > 0 else 0,
            }

        elapsed = time.time() - start
        self.results = {
            "test_range": [test_start, test_end],
            "n_rules_tested": total_rules,
            "total_predictions": total_predictions,
            "hits": hits,
            "misses": misses,
            "no_match": no_match,
            "accuracy": round(accuracy, 4),
            "coverage": round(coverage, 4),
            "per_position": per_pos_accuracy,
            "random_baseline": round(1.0 / 3, 4),
            "time": round(elapsed, 1),
        }

        print(f"[C1] 回测完成: 准确率 {accuracy:.4f} (随机基线 0.3333), "
              f"覆盖率 {coverage:.4f}, 预测 {total_predictions} 次, "
              f"耗时 {elapsed:.1f}s")

        return self.results

    def save(self, filename: str = "c1_backtest_results.json"):
        """保存结果"""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "lottery_type": self.data.lottery_type,
                "backtest": self.results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[C1] 结果已保存: {path}")

    def summary(self) -> Dict[str, Any]:
        return self.results
