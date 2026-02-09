# -*- coding: utf-8 -*-
"""
A1: 全维度方向模式扫描

穷举所有位置组合 × 所有窗口大小 × 所有方向模式，
统计每种模式的出现频率、后续方向分布、卡方检验显著性。

扫描范围：
  - 单位置：5个位置 × 窗口3-15
  - 双位置联合：C(5,2)=10 × 窗口3-10
  - 三位置联合：C(5,3)=10 × 窗口3-8
  - 四位置联合：C(5,4)=5 × 窗口3-6
  - 五位置联合：窗口3-5
"""

import json
import time
import numpy as np
from itertools import combinations, product
from collections import Counter
from typing import Dict, List, Any, Tuple
from pathlib import Path

from research.config import ResearchConfig, RULES_DIR
from research.data_loader import LotteryData


# ============================================================
# 核心扫描函数（可被并行调用）
# ============================================================

def scan_single_task(args: tuple) -> List[Dict]:
    """
    扫描单个任务：指定位置组合 + 窗口大小。
    返回显著模式列表。

    args = (direction_arrays, pos_combo, window, min_support, p_threshold)
    direction_arrays: dict {pos: np.array}  方向序列
    pos_combo: tuple  位置组合，如 (0,) 或 (0, 2)
    window: int  窗口大小
    min_support: int  最小支持度
    p_threshold: float  p值阈值
    """
    direction_arrays, pos_combo, window, min_support, p_threshold = args

    n_positions = len(pos_combo)

    # 获取各位置的方向序列
    dir_seqs = [direction_arrays[pos] for pos in pos_combo]

    # 找到最短序列长度
    min_len = min(len(s) for s in dir_seqs)
    if min_len < window + 1:
        return []

    # 截断到相同长度
    dir_seqs = [s[:min_len] for s in dir_seqs]

    # 构建联合方向序列：每个时间步是一个元组 (d0, d1, ...)
    # 编码为整数以加速：每个方向 -1/0/1 映射到 0/1/2，然后用3进制编码
    encoded = np.zeros(min_len, dtype=np.int64)
    for i, seq in enumerate(dir_seqs):
        encoded += (seq + 1).astype(np.int64) * (3 ** i)

    # 滑动窗口扫描
    # 模式编码：窗口内每个时间步的联合编码，用更高位的3进制
    base = 3 ** n_positions  # 每个时间步的编码基数
    pattern_counts = Counter()
    # 模式后续方向统计：pattern -> Counter of next_direction_tuple
    pattern_next = {}

    for t in range(min_len - window):
        # 编码当前窗口的模式
        pattern_key = tuple(int(encoded[t + k]) for k in range(window))
        pattern_counts[pattern_key] += 1

        # 记录下一步的方向
        if t + window < min_len:
            next_dir = tuple(int(dir_seqs[i][t + window]) for i in range(n_positions))
            if pattern_key not in pattern_next:
                pattern_next[pattern_key] = Counter()
            pattern_next[pattern_key][next_dir] += 1

    # 统计检验
    total_windows = min_len - window
    n_possible_patterns = base ** window  # 理论模式总数
    expected = total_windows / n_possible_patterns if n_possible_patterns > 0 else 1.0

    results = []
    for pattern_key, observed in pattern_counts.items():
        if observed < min_support:
            continue

        # 卡方检验
        chi2 = (observed - expected) ** 2 / expected if expected > 0 else 0.0

        # p值近似（自由度=1的卡方分布）
        # chi2 > 6.63 对应 p < 0.01, chi2 > 10.83 对应 p < 0.001
        if chi2 < 3.84:  # p > 0.05，不显著
            continue

        # 效应量 Cramér's V（简化版）
        cramers_v = (chi2 / total_windows) ** 0.5 if total_windows > 0 else 0

        # 后续方向分布
        next_counter = pattern_next.get(pattern_key, Counter())
        next_total = sum(next_counter.values())
        next_dist = {}
        if next_total > 0:
            for next_dir, cnt in next_counter.most_common():
                next_dist[str(next_dir)] = {
                    "count": cnt,
                    "prob": round(cnt / next_total, 4),
                }

        # 解码模式为可读字符串
        dir_map = {0: "D", 1: "E", 2: "U"}
        pattern_str = ""
        for step_code in pattern_key:
            step_dirs = []
            code = step_code
            for _ in range(n_positions):
                step_dirs.append(dir_map[code % 3])
                code //= 3
            if n_positions == 1:
                pattern_str += step_dirs[0]
            else:
                pattern_str += "(" + "".join(step_dirs) + ")"

        # 预测方向（多数投票）
        if next_counter:
            best_next = next_counter.most_common(1)[0]
            pred_dirs = []
            code = best_next[0] if isinstance(best_next[0], tuple) else (best_next[0],)
            for d in code:
                pred_dirs.append(dir_map.get(d + 1, "?") if isinstance(d, int) and d in (-1, 0, 1) else "?")
            pred_confidence = best_next[1] / next_total if next_total > 0 else 0
        else:
            pred_dirs = []
            pred_confidence = 0

        results.append({
            "positions": list(pos_combo),
            "window": window,
            "pattern": pattern_str,
            "pattern_key": list(pattern_key),
            "observed": observed,
            "expected": round(expected, 2),
            "chi2": round(chi2, 2),
            "cramers_v": round(cramers_v, 4),
            "ratio": round(observed / expected, 2) if expected > 0 else 0,
            "next_distribution": next_dist,
            "prediction_confidence": round(pred_confidence, 4),
            "total_windows": total_windows,
        })

    return results


# ============================================================
# 主扫描器
# ============================================================

class DirectionScanner:
    """全维度方向模式扫描器"""

    def __init__(self, data: LotteryData, config: ResearchConfig):
        self.data = data
        self.config = config
        self.results: List[Dict] = []

    def build_tasks(self) -> List[tuple]:
        """构建所有扫描任务"""
        tasks = []
        rc = self.data.red_count
        cfg = self.config

        # 将方向序列转为普通 dict（multiprocessing 需要可序列化）
        dir_arrays = {
            pos: self.data.direction_series[pos]
            for pos in range(rc)
        }

        # 单位置
        for pos in range(rc):
            for w in range(cfg.direction_single_windows[0],
                           cfg.direction_single_windows[1] + 1):
                tasks.append((
                    dir_arrays, (pos,), w,
                    cfg.direction_min_support, cfg.direction_p_threshold
                ))

        # 双位置
        for combo in combinations(range(rc), 2):
            for w in range(cfg.direction_dual_windows[0],
                           cfg.direction_dual_windows[1] + 1):
                tasks.append((
                    dir_arrays, combo, w,
                    cfg.direction_min_support, cfg.direction_p_threshold
                ))

        # 三位置
        for combo in combinations(range(rc), 3):
            for w in range(cfg.direction_triple_windows[0],
                           cfg.direction_triple_windows[1] + 1):
                tasks.append((
                    dir_arrays, combo, w,
                    cfg.direction_min_support, cfg.direction_p_threshold
                ))

        # 四位置
        for combo in combinations(range(rc), 4):
            for w in range(cfg.direction_quad_windows[0],
                           cfg.direction_quad_windows[1] + 1):
                tasks.append((
                    dir_arrays, combo, w,
                    cfg.direction_min_support, cfg.direction_p_threshold
                ))

        # 五位置（大乐透5个位置，双色球6个位置）
        if rc <= 6:
            for combo in combinations(range(rc), min(5, rc)):
                for w in range(cfg.direction_quint_windows[0],
                               cfg.direction_quint_windows[1] + 1):
                    tasks.append((
                        dir_arrays, combo, w,
                        cfg.direction_min_support, cfg.direction_p_threshold
                    ))

        return tasks

    def run(self) -> List[Dict]:
        """执行扫描（并行版）"""
        tasks = self.build_tasks()
        total = len(tasks)
        print(f"[A1] 方向模式扫描: 共 {total} 个扫描任务")

        start = time.time()

        from research.brute_force.parallel_engine import parallel_map
        batch_results = parallel_map(
            scan_single_task, tasks,
            n_workers=self.config.n_workers,
            chunk_size=max(1, total // (self.config.n_workers * 4)),
            desc="A1 方向模式扫描",
        )

        all_results = []
        for batch in batch_results:
            all_results.extend(batch)

        # 按 chi2 降序排列
        all_results.sort(key=lambda x: x["chi2"], reverse=True)
        self.results = all_results

        elapsed = time.time() - start
        print(f"[A1] 完成: {len(all_results)} 个显著模式, 耗时 {elapsed:.1f}s")

        return all_results

    def save(self, filename: str = "a1_direction_patterns.json"):
        """保存结果"""
        RULES_DIR.mkdir(parents=True, exist_ok=True)
        path = RULES_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "lottery_type": self.data.lottery_type,
                "total_patterns": len(self.results),
                "patterns": self.results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[A1] 结果已保存: {path}")

    def summary(self) -> Dict[str, Any]:
        """生成摘要"""
        if not self.results:
            return {"total": 0}

        # 按位置组合数分组统计
        by_n_pos = {}
        for r in self.results:
            n = len(r["positions"])
            by_n_pos.setdefault(n, []).append(r)

        summary = {
            "total": len(self.results),
            "by_position_count": {
                n: {
                    "count": len(patterns),
                    "top_chi2": patterns[0]["chi2"] if patterns else 0,
                    "avg_confidence": round(
                        np.mean([p["prediction_confidence"] for p in patterns]), 4
                    ),
                }
                for n, patterns in sorted(by_n_pos.items())
            },
            "top10": self.results[:10],
        }
        return summary
