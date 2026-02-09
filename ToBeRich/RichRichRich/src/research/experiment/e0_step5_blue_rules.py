# -*- coding: utf-8 -*-
"""E0-Step5: 蓝球规则挖掘

对蓝球区域运行 A1 方向模式扫描 + A2 条件排除规则挖掘。
使用适配器模式，将蓝球数据映射到红球接口，复用现有 DirectionScanner/RuleMiner。

用法: python3 -m src.research.experiment.e0_step5_blue_rules
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.config import ResearchConfig, DATA_DIR
from research.data_loader import LotteryData
from research.brute_force.direction_scanner import DirectionScanner
from research.brute_force.rule_miner import RuleMiner

# === 路径常量 ===
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_RULES_DIR = RESULTS_DIR / "rules_strict"


class BlueLotteryDataAdapter:
    """将蓝球数据伪装成红球格式，复用 DirectionScanner/RuleMiner。

    核心映射：
      red_count     ← blue_count (1 或 2)
      red_range     ← blue_range (16 或 12)
      position_series ← blue_position_series
      direction_series ← blue_direction_series
      red_matrix    ← blue_matrix
    """

    def __init__(self, data: LotteryData, n_draws_limit: int = None):
        """
        Args:
            data: 原始 LotteryData 对象
            n_draws_limit: 截断期数（严格模式用前60%数据）
        """
        self._data = data
        self.lottery_type = data.lottery_type
        self.red_count = data.blue_count
        self.red_range = data.blue_range

        # 截断数据
        if n_draws_limit and n_draws_limit < data.n_draws:
            self.n_draws = n_draws_limit
            self.position_series = {
                pos: arr[:n_draws_limit]
                for pos, arr in data.blue_position_series.items()
            }
            self.red_matrix = data.blue_matrix[:n_draws_limit]
        else:
            self.n_draws = data.n_draws
            self.position_series = dict(data.blue_position_series)
            self.red_matrix = data.blue_matrix.copy()

        # 重建方向序列（基于截断后的 position_series）
        self.direction_series = {}
        for pos, series in self.position_series.items():
            diff = np.diff(series)
            d = np.zeros(len(diff), dtype=np.int8)
            d[diff > 0] = 1
            d[diff < 0] = -1
            self.direction_series[pos] = d

    def get_diff_series(self, pos: int, order: int = 1) -> np.ndarray:
        """获取蓝球指定位置的N阶差分序列"""
        series = self.position_series[pos].astype(np.float64)
        for _ in range(order):
            series = np.diff(series)
        return series

    def get_cross_diff_series(self, pos_i: int, pos_j: int) -> np.ndarray:
        """获取跨位置差分序列（仅大乐透后区有意义）"""
        return (self.position_series[pos_i].astype(np.float64) -
                self.position_series[pos_j].astype(np.float64))

    def get_combo_stats_series(self) -> Dict[str, np.ndarray]:
        """获取蓝球组合统计量（双色球只有1个蓝球，返回空dict）"""
        if self.red_count < 2:
            return {}

        sums = []
        spans = []
        odd_counts = []
        for row in self.red_matrix:
            sums.append(int(np.sum(row)))
            spans.append(int(np.max(row) - np.min(row)))
            odd_counts.append(int(np.sum(row % 2 == 1)))

        return {
            "blue_sum": np.array(sums, dtype=np.int32),
            "blue_span": np.array(spans, dtype=np.int32),
            "blue_odd_count": np.array(odd_counts, dtype=np.int32),
        }

    def direction_to_str(self, d: int) -> str:
        if d == 1:
            return "U"
        elif d == -1:
            return "D"
        return "E"


def build_blue_config(blue_count: int) -> ResearchConfig:
    """根据蓝球数量构建适配的扫描配置。

    蓝球数量少（1-2个），位置组合空间小，适当调整窗口和条件数。
    """
    if blue_count == 1:
        # 双色球：只有1个蓝球，无多位置联合扫描
        return ResearchConfig(
            # A1: 单位置窗口 3-30，无多位置
            direction_single_windows=(3, 30),
            direction_dual_windows=(3, 3),      # 不会被使用（只有1个位置）
            direction_triple_windows=(3, 3),
            direction_quad_windows=(3, 3),
            direction_quint_windows=(3, 3),
            direction_min_support=3,
            direction_p_threshold=0.01,
            # A2: 最大3条件组合（原子条件少）
            rule_max_conditions=3,
            rule_min_support=10,
            rule_min_lift=1.2,
            rule_value_bins=10,
            rule_max_rules=0,
            # 并行
            n_workers=7,
        )
    else:
        # 大乐透后区：2个蓝球，有双位置联合扫描
        return ResearchConfig(
            # A1: 单位置 3-30，双位置 3-20
            direction_single_windows=(3, 30),
            direction_dual_windows=(3, 20),
            direction_triple_windows=(3, 3),    # 不会被使用（只有2个位置）
            direction_quad_windows=(3, 3),
            direction_quint_windows=(3, 3),
            direction_min_support=3,
            direction_p_threshold=0.01,
            # A2: 最大4条件组合
            rule_max_conditions=4,
            rule_min_support=10,
            rule_min_lift=1.2,
            rule_value_bins=10,
            rule_max_rules=0,
            # 并行
            n_workers=7,
        )


def run_blue_rules(lottery_type: str) -> Dict[str, Any]:
    """对单个彩种运行蓝球规则挖掘"""
    print(f"\n{'═' * 50}")
    print(f"  蓝球规则挖掘: {lottery_type}")
    print(f"{'═' * 50}")

    # 1. 加载数据
    data = LotteryData(lottery_type)
    print(f"蓝球: {data.blue_count} 个, 范围 1-{data.blue_range}")

    # 2. 严格模式：截断到前60%
    n_sliced = int(data.n_draws * 0.6)
    print(f"严格模式: 使用前 {n_sliced}/{data.n_draws} 期 (60%)")

    # 3. 创建适配器
    adapter = BlueLotteryDataAdapter(data, n_draws_limit=n_sliced)
    print(f"适配器: red_count={adapter.red_count}, red_range={adapter.red_range}, "
          f"n_draws={adapter.n_draws}")

    # 4. 构建蓝球专用配置
    config = build_blue_config(data.blue_count)

    # 5. A1 方向模式扫描
    print(f"\n--- A1 蓝球方向模式扫描 ---")
    t0 = time.time()
    scanner = DirectionScanner(adapter, config)
    a1_results = scanner.run()
    a1_time = time.time() - t0

    # 保存 A1
    STRICT_RULES_DIR.mkdir(parents=True, exist_ok=True)
    a1_output = {
        "lottery_type": lottery_type,
        "ball_type": "blue",
        "blue_count": data.blue_count,
        "blue_range": data.blue_range,
        "n_draws_used": n_sliced,
        "total_patterns": len(a1_results),
        "patterns": a1_results,
    }
    a1_path = STRICT_RULES_DIR / f"a1_blue_direction_patterns_{lottery_type}.json"
    with open(a1_path, "w", encoding="utf-8") as f:
        json.dump(a1_output, f, ensure_ascii=False, indent=2)
    print(f"A1 保存: {len(a1_results)} 个模式 → {a1_path.name} ({a1_time:.1f}s)")

    # 6. A2 条件排除规则挖掘
    print(f"\n--- A2 蓝球条件排除规则挖掘 ---")
    t0 = time.time()
    miner = RuleMiner(adapter, config)
    a2_results = miner.run()
    a2_time = time.time() - t0

    # 按 lift 降序排序
    a2_results.sort(key=lambda x: x["lift"], reverse=True)

    # 保存 A2
    a2_output = {
        "lottery_type": lottery_type,
        "ball_type": "blue",
        "blue_count": data.blue_count,
        "blue_range": data.blue_range,
        "n_draws_used": n_sliced,
        "total_rules": len(a2_results),
        "rules": a2_results[:10000],
    }
    a2_path = STRICT_RULES_DIR / f"a2_blue_exclusion_rules_{lottery_type}.json"
    with open(a2_path, "w", encoding="utf-8") as f:
        json.dump(a2_output, f, ensure_ascii=False, indent=2)
    print(f"A2 保存: {len(a2_results)} 条规则 → {a2_path.name} ({a2_time:.1f}s)")

    return {
        "lottery_type": lottery_type,
        "blue_count": data.blue_count,
        "blue_range": data.blue_range,
        "n_draws_total": data.n_draws,
        "n_draws_used": n_sliced,
        "a1_patterns": len(a1_results),
        "a1_time": round(a1_time, 1),
        "a2_rules": len(a2_results),
        "a2_time": round(a2_time, 1),
    }


def main():
    print("=" * 60)
    print("  E0-Step5: 蓝球规则挖掘（严格版）")
    print("=" * 60)

    total_start = time.time()
    summary = {}

    for lottery_type in ["daletou", "shuangseqiu"]:
        result = run_blue_rules(lottery_type)
        summary[lottery_type] = result

    # 保存汇总
    summary_path = STRICT_RULES_DIR / "e0_step5_blue_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    total_time = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"  E0-Step5 完成 (总耗时 {total_time:.1f}s)")
    print(f"{'=' * 60}")
    for lt, r in summary.items():
        print(f"  {lt}: A1={r['a1_patterns']} 模式, A2={r['a2_rules']} 规则")
    print(f"  汇总: {summary_path.name}")


if __name__ == "__main__":
    main()
