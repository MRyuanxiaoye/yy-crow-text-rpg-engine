# -*- coding: utf-8 -*-
"""
研究系统主运行器

支持：
  - 选择运行模块（A/B/C）
  - 后台运行（nohup / caffeinate）
  - 进度日志输出到文件
  - 结果自动保存
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

# 将 src 加入路径
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research.config import get_research_config, RESULTS_DIR, RULES_DIR, REPORTS_DIR
from research.data_loader import LotteryData


def run_module_a(data: LotteryData, config, modules: str = "all"):
    """运行模块A：穷举搜索"""
    results = {}
    lt = data.lottery_type

    if modules in ("all", "a1"):
        print("\n" + "=" * 60)
        print("  A1: 方向模式扫描")
        print("=" * 60)
        from research.brute_force.direction_scanner import DirectionScanner
        scanner = DirectionScanner(data, config)
        scanner.run()
        scanner.save(f"a1_direction_patterns_{lt}.json")
        results["a1"] = scanner.summary()
        print(f"[A1] 摘要: {json.dumps(results['a1'], ensure_ascii=False, indent=2)[:500]}")

    if modules in ("all", "a2"):
        print("\n" + "=" * 60)
        print("  A2: 条件排除规则挖掘")
        print("=" * 60)
        from research.brute_force.rule_miner import RuleMiner
        miner = RuleMiner(data, config)
        miner.run()
        miner.save(f"a2_exclusion_rules_{lt}.json")
        results["a2"] = miner.summary()
        print(f"[A2] 摘要: {json.dumps(results['a2'], ensure_ascii=False, indent=2)[:500]}")

    if modules in ("all", "a3"):
        print("\n" + "=" * 60)
        print("  A3: 差分空间分析")
        print("=" * 60)
        from research.brute_force.diff_analyzer import DiffAnalyzer
        analyzer = DiffAnalyzer(data, config)
        analyzer.run()
        analyzer.save(f"a3_diff_analysis_{lt}.json")
        results["a3"] = analyzer.summary()
        print(f"[A3] 摘要: {json.dumps(results['a3'], ensure_ascii=False, indent=2)[:500]}")

    if modules in ("all", "a4"):
        print("\n" + "=" * 60)
        print("  A4: 滑动窗口统计量扫描")
        print("=" * 60)
        from research.brute_force.sliding_stats import SlidingStatsScanner
        scanner = SlidingStatsScanner(data, config)
        scanner.run()
        scanner.save(f"a4_sliding_stats_{lt}.json")
        results["a4"] = scanner.summary()
        print(f"[A4] 摘要: {json.dumps(results['a4'], ensure_ascii=False, indent=2)[:500]}")

    return results


def run_module_b(data: LotteryData, config, modules: str = "all"):
    """运行模块B：深度学习"""
    results = {}
    lt = data.lottery_type

    if modules in ("all", "b1"):
        print("\n" + "=" * 60)
        print("  B1: Transformer 序列预测")
        print("=" * 60)
        try:
            import torch
            from research.deep_learning.trainer import TransformerTrainer
            trainer = TransformerTrainer(data, config)
            trainer.run()
            trainer.save(f"b1_transformer_results_{lt}.json")
            results["b1"] = trainer.summary()
            print(f"[B1] 摘要: {json.dumps(results['b1'], ensure_ascii=False, indent=2)[:500]}")
        except ImportError as e:
            print(f"[B1] PyTorch 未安装，跳过: {e}")
        except Exception as e:
            print(f"[B1] 运行出错: {e}")

    return results


def run_module_c(data: LotteryData, config, results_a: dict, results_b: dict):
    """运行模块C：评估验证"""
    results = {}
    lt = data.lottery_type

    print("\n" + "=" * 60)
    print("  C1: 回测验证")
    print("=" * 60)
    try:
        from research.evaluation.backtester import Backtester
        bt = Backtester(data, config)
        bt.run()
        bt.save(f"c1_backtest_results_{lt}.json")
        results["c1"] = bt.summary()
        print(f"[C1] 摘要: {json.dumps(results['c1'], ensure_ascii=False, indent=2)[:500]}")
    except Exception as e:
        print(f"[C1] 运行出错: {e}")

    print("\n" + "=" * 60)
    print("  C2: 稳定性检验 + 规则排名")
    print("=" * 60)
    try:
        from research.evaluation.stability import StabilityChecker
        sc = StabilityChecker(data, config)
        sc.run()
        sc.save(f"c2_stability_results_{lt}.json")
        results["c2"] = sc.summary()
        print(f"[C2] 摘要: {json.dumps(results['c2'], ensure_ascii=False, indent=2)[:500]}")
    except Exception as e:
        print(f"[C2] 运行出错: {e}")

    return results


def generate_report(lottery_type: str, results: dict):
    """生成研究报告"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"research_report_{lottery_type}_{timestamp}.json"

    report = {
        "lottery_type": lottery_type,
        "timestamp": timestamp,
        "modules": results,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[报告] 研究报告已保存: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="彩票研究系统")
    parser.add_argument("--type", default="daletou",
                        choices=["daletou", "shuangseqiu"],
                        help="彩种类型")
    parser.add_argument("--module", default="a",
                        choices=["a", "b", "c", "all",
                                 "a1", "a2", "a3", "a4", "b1"],
                        help="运行模块: a=穷举, b=深度学习, c=评估, all=全部, a1-a4/b1=单个子模块")
    parser.add_argument("--report", action="store_true",
                        help="生成研究报告")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  彩票研究系统")
    print(f"  彩种: {args.type}")
    print(f"  模块: {args.module}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start = time.time()

    # 加载配置和数据
    config = get_research_config(args.type)
    data = LotteryData(args.type)

    all_results = {}

    # 运行模块
    if args.module in ("a", "all"):
        all_results["module_a"] = run_module_a(data, config, "all")
    elif args.module.startswith("a"):
        all_results["module_a"] = run_module_a(data, config, args.module)

    if args.module in ("b", "all"):
        all_results["module_b"] = run_module_b(data, config, "all")
    elif args.module.startswith("b"):
        all_results["module_b"] = run_module_b(data, config, args.module)

    if args.module in ("c", "all"):
        all_results["module_c"] = run_module_c(
            data, config,
            all_results.get("module_a", {}),
            all_results.get("module_b", {}),
        )

    # 生成报告
    if args.report or args.module == "all":
        generate_report(args.type, all_results)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  全部完成! 总耗时 {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
