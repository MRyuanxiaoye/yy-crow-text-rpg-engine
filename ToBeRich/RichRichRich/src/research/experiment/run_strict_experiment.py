"""严格时间切分验证实验

用前60%数据重新挖掘规则，在完全未见过的测试集上评估，
与原始结果对比，判断数据泄露的影响程度。

用法: python3 -m src.research.experiment.run_strict_experiment
"""

import json
import sys
import time
import traceback
import numpy as np
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.config import ResearchConfig, DATA_DIR, get_research_config
from research.data_loader import LotteryData
from research.brute_force.direction_scanner import DirectionScanner
from research.brute_force.rule_miner import RuleMiner
from research.brute_force.diff_analyzer import DiffAnalyzer
from research.brute_force.sliding_stats import SlidingStatsScanner

import research.experiment.utils as exp_utils
from research.experiment.utils import setup_logging, log, Timer, save_json, load_json
from research.experiment.phase1_compress import run_phase1
from research.experiment.phase2_replay import run_phase2
from research.experiment.phase3_train import run_phase3
from research.experiment.phase4_analyze import run_phase4

# === 路径常量 ===
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_RULES_DIR = RESULTS_DIR / "rules_strict"
STRICT_EXPERIMENT_DIR = RESULTS_DIR / "experiment_strict"
ORIGINAL_EXPERIMENT_DIR = RESULTS_DIR / "experiment"


def generate_sliced_data(lottery_type, ratio=0.6):
    """生成切片数据JSON，只保留前ratio比例的draws

    Returns:
        sliced_path: 切片数据文件路径
        n_total: 原始总期数
        n_sliced: 切片后期数
    """
    src_path = DATA_DIR / f"{lottery_type}_history.json"
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    draws = data.get("draws", [])
    n_total = len(draws)
    n_sliced = int(n_total * ratio)

    sliced_data = dict(data)
    sliced_data["draws"] = draws[:n_sliced]

    sliced_path = DATA_DIR / f"{lottery_type}_history_60pct.json"
    with open(sliced_path, "w", encoding="utf-8") as f:
        json.dump(sliced_data, f, ensure_ascii=False, indent=2)

    log(f"  切片数据: {n_total} → {n_sliced} 期 ({ratio:.0%})")
    log(f"  保存到: {sliced_path}")
    return sliced_path, n_total, n_sliced


def mine_rules_on_sliced_data(lottery_type, sliced_path):
    """用切片数据重新挖掘A1/A2规则

    Returns:
        a1_count: A1模式数量
        a2_count: A2规则数量
    """
    STRICT_RULES_DIR.mkdir(parents=True, exist_ok=True)

    # 加载切片数据
    sliced_data = LotteryData(lottery_type, data_path=str(sliced_path))
    config = get_research_config(lottery_type)

    # A1 方向模式扫描
    log(f"  开始A1方向模式扫描...")
    scanner = DirectionScanner(sliced_data, config)
    a1_results = scanner.run()

    a1_output = {
        "lottery_type": lottery_type,
        "total_patterns": len(a1_results),
        "patterns": a1_results,
    }
    a1_path = STRICT_RULES_DIR / f"a1_direction_patterns_{lottery_type}.json"
    with open(a1_path, "w", encoding="utf-8") as f:
        json.dump(a1_output, f, ensure_ascii=False, indent=2)
    log(f"  A1保存: {len(a1_results)} 个模式 → {a1_path.name}")

    # A2 条件排除规则挖掘
    log(f"  开始A2条件排除规则挖掘...")
    miner = RuleMiner(sliced_data, config)
    a2_results = miner.run()

    # 按lift降序，保存top10000
    a2_results.sort(key=lambda x: x["lift"], reverse=True)
    a2_output = {
        "lottery_type": lottery_type,
        "total_rules": len(a2_results),
        "rules": a2_results[:10000],
    }
    a2_path = STRICT_RULES_DIR / f"a2_exclusion_rules_{lottery_type}.json"
    with open(a2_path, "w", encoding="utf-8") as f:
        json.dump(a2_output, f, ensure_ascii=False, indent=2)
    log(f"  A2保存: {len(a2_results)} 条规则 → {a2_path.name}")

    return len(a1_results), len(a2_results)


def mine_a3a4_on_sliced_data(lottery_type, sliced_path):
    """用切片数据挖掘A3差分模式和A4统计量模式"""
    STRICT_RULES_DIR.mkdir(parents=True, exist_ok=True)
    sliced_data = LotteryData(lottery_type, data_path=str(sliced_path))
    config = get_research_config(lottery_type)

    # A3 差分分析
    log(f"  开始A3差分空间分析...")
    analyzer = DiffAnalyzer(sliced_data, config)
    a3_results = analyzer.run()
    a3_summary = analyzer.summary()

    a3_path = STRICT_RULES_DIR / f"a3_diff_analysis_{lottery_type}.json"
    with open(a3_path, "w", encoding="utf-8") as f:
        json.dump({"lottery_type": lottery_type, "analysis": a3_results},
                  f, ensure_ascii=False, indent=2)
    a3_count = a3_summary.get("total_diff_patterns", 0)
    log(f"  A3保存: {a3_count} 个差分模式 → {a3_path.name}")

    # A4 滑动窗口统计量
    log(f"  开始A4滑动窗口统计量扫描...")
    scanner = SlidingStatsScanner(sliced_data, config)
    a4_results = scanner.run()
    a4_summary = scanner.summary()

    a4_path = STRICT_RULES_DIR / f"a4_sliding_stats_{lottery_type}.json"
    with open(a4_path, "w", encoding="utf-8") as f:
        json.dump({"lottery_type": lottery_type, "analysis": a4_results},
                  f, ensure_ascii=False, indent=2)
    a4_count = a4_summary.get("total_stat_patterns", 0)
    log(f"  A4保存: {a4_count} 个统计量模式 → {a4_path.name}")

    return a3_count, a4_count


def run_strict_pipeline(lottery_type):
    """对单个彩种运行严格验证管线"""
    log(f"\n{'═'*50}")
    log(f"  严格验证: {lottery_type}")
    log(f"{'═'*50}")

    # 3.1 生成切片数据
    with Timer("生成切片数据"):
        sliced_path, n_total, n_sliced = generate_sliced_data(lottery_type)

    # 3.2 用切片数据挖掘规则
    with Timer(f"规则挖掘 ({lottery_type})"):
        a1_count, a2_count = mine_rules_on_sliced_data(lottery_type, sliced_path)

    # 3.3 运行实验管线 phases 1-4
    # 临时替换 EXPERIMENT_DIR
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STRICT_EXPERIMENT_DIR
    STRICT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with Timer(f"阶段一 [{lottery_type}]"):
            clusters, a1_filtered = run_phase1(lottery_type, rules_dir=STRICT_RULES_DIR)

        with Timer(f"阶段二 [{lottery_type}]"):
            (train_X, train_Y, test_X, test_Y,
             train_Y_val, test_Y_val,
             feature_names, train_indices, test_indices, data) = run_phase2(
                lottery_type, clusters, a1_filtered, rules_dir=STRICT_RULES_DIR)

        with Timer(f"阶段三 [{lottery_type}]"):
            results_3a, results_3b, results_3c = run_phase3(
                train_X, train_Y, test_X, test_Y,
                feature_names, data, test_indices)

        with Timer(f"阶段四 [{lottery_type}]"):
            final_report = run_phase4(lottery_type, results_3a, results_3b, results_3c)
    finally:
        # 恢复原始路径
        exp_utils.EXPERIMENT_DIR = original_exp_dir

    return final_report, a1_count, a2_count, n_total, n_sliced


def generate_comparison_report(lottery_type, strict_report, a1_count, a2_count,
                                n_total, n_sliced):
    """生成严格验证 vs 原始实验的对比报告"""
    comparison = {
        "lottery_type": lottery_type,
        "data_split": {
            "total_draws": n_total,
            "sliced_draws": n_sliced,
            "ratio": round(n_sliced / n_total, 2),
        },
        "rule_counts": {
            "strict_a1": a1_count,
            "strict_a2": a2_count,
        },
        "strict_results": None,
        "original_results": None,
        "delta": None,
    }

    # 提取严格验证结果
    if strict_report and strict_report.get("conclusion"):
        c = strict_report["conclusion"]
        comparison["strict_results"] = {
            "best_method": c.get("best_method"),
            "efficiency": c.get("best_efficiency", 0),
            "direction_accuracy": c.get("direction_accuracy", 0),
            "avg_reduction": c.get("avg_reduction", 0),
            "avg_survival": c.get("avg_survival", 0),
        }

    # 加载原始实验结果
    orig_path = ORIGINAL_EXPERIMENT_DIR / f"phase4_final_report_{lottery_type}.json"
    if orig_path.exists():
        orig_report = load_json(orig_path)
        if orig_report.get("conclusion"):
            c = orig_report["conclusion"]
            comparison["original_results"] = {
                "best_method": c.get("best_method"),
                "efficiency": c.get("best_efficiency", 0),
                "direction_accuracy": c.get("direction_accuracy", 0),
                "avg_reduction": c.get("avg_reduction", 0),
                "avg_survival": c.get("avg_survival", 0),
            }

        # 加载原始规则数量
        for rule_type in ["a1_direction_patterns", "a2_exclusion_rules"]:
            rp = RESULTS_DIR / "rules" / f"{rule_type}_{lottery_type}.json"
            if rp.exists():
                rd = load_json(rp)
                key = "total_patterns" if "a1" in rule_type else "total_rules"
                tag = "original_a1" if "a1" in rule_type else "original_a2"
                comparison["rule_counts"][tag] = rd.get(key, 0)

    # 计算差值
    s = comparison["strict_results"]
    o = comparison["original_results"]
    if s and o:
        comparison["delta"] = {
            "efficiency_drop": round(o["efficiency"] - s["efficiency"], 4),
            "accuracy_drop": round(o["direction_accuracy"] - s["direction_accuracy"], 4),
            "reduction_drop": round(o["avg_reduction"] - s["avg_reduction"], 4),
            "survival_drop": round(o["avg_survival"] - s["avg_survival"], 4),
        }

        # 判断数据泄露影响
        eff_drop = comparison["delta"]["efficiency_drop"]
        if eff_drop < 0.05:
            comparison["verdict"] = "数据泄露影响较小（efficiency下降<0.05），原始结果基本可信"
        elif eff_drop < 0.10:
            comparison["verdict"] = "数据泄露有一定影响（efficiency下降0.05-0.10），需谨慎解读原始结果"
        else:
            comparison["verdict"] = "数据泄露影响显著（efficiency下降>0.10），原始结果存在过拟合"
    else:
        comparison["verdict"] = "缺少原始实验结果，无法对比"

    return comparison


def run_a3a4_only():
    """独立运行A3/A4严格挖掘（不重跑A1/A2和Phase1-4）"""
    STRICT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STRICT_EXPERIMENT_DIR
    logger = setup_logging()
    exp_utils.EXPERIMENT_DIR = original_exp_dir

    log("=" * 60)
    log("  E0-Step1: 补挖严格版 A3/A4 规则")
    log("=" * 60)

    for lottery_type in ["daletou", "shuangseqiu"]:
        log(f"\n{'═'*50}")
        log(f"  {lottery_type}: A3/A4 严格挖掘")
        log(f"{'═'*50}")

        sliced_path = DATA_DIR / f"{lottery_type}_history_60pct.json"
        if not sliced_path.exists():
            with Timer("生成切片数据"):
                sliced_path, _, _ = generate_sliced_data(lottery_type)
        else:
            log(f"  复用已有切片数据: {sliced_path}")

        with Timer(f"A3/A4 规则挖掘 ({lottery_type})"):
            a3_count, a4_count = mine_a3a4_on_sliced_data(lottery_type, sliced_path)

        log(f"  结果: A3={a3_count} 个差分模式, A4={a4_count} 个统计量模式")

    log(f"\n{'═'*60}")
    log(f"  E0-Step1 完成！输出目录: {STRICT_RULES_DIR}")
    log(f"{'═'*60}")


def run_a3a4_phase2_only():
    """独立运行带 A3/A4 特征的 Phase2（验证用）"""
    STRICT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STRICT_EXPERIMENT_DIR
    logger = setup_logging()

    log("=" * 60)
    log("  E0-Step2 验证: 带 A3/A4 特征的 Phase2")
    log("=" * 60)

    try:
        for lottery_type in ["daletou", "shuangseqiu"]:
            log(f"\n{'═'*50}")
            log(f"  {lottery_type}: Phase1 + Phase2 (含 A3/A4)")
            log(f"{'═'*50}")

            with Timer(f"阶段一 [{lottery_type}]"):
                clusters, a1_filtered = run_phase1(lottery_type, rules_dir=STRICT_RULES_DIR)

            with Timer(f"阶段二 [{lottery_type}]"):
                (train_X, train_Y, test_X, test_Y,
                 train_Y_val, test_Y_val,
                 feature_names, train_indices, test_indices, data) = run_phase2(
                    lottery_type, clusters, a1_filtered, rules_dir=STRICT_RULES_DIR)

            # 验证输出
            n_a3 = sum(1 for f in feature_names if f.startswith('a3_'))
            n_a4 = sum(1 for f in feature_names if f.startswith('a4_'))
            log(f"\n  验证结果:")
            log(f"    总特征数: {len(feature_names)}")
            log(f"    A3 特征数: {n_a3}")
            log(f"    A4 特征数: {n_a4}")
            log(f"    train_X 形状: {train_X.shape}")
            log(f"    test_X 形状: {test_X.shape}")
            log(f"    NaN 检查: train={np.isnan(train_X).sum()}, test={np.isnan(test_X).sum()}")

            # 检查 A3/A4 特征不全为 0
            a3_cols = [i for i, f in enumerate(feature_names) if f.startswith('a3_') and '_p' in f]
            a4_cols = [i for i, f in enumerate(feature_names) if f.startswith('a4_') and '_p' in f]
            if a3_cols:
                a3_nonzero = (train_X[:, a3_cols] != 0).sum()
                log(f"    A3 匹配特征非零数: {a3_nonzero} / {train_X.shape[0] * len(a3_cols)}")
            if a4_cols:
                a4_nonzero = (train_X[:, a4_cols] != 0).sum()
                log(f"    A4 匹配特征非零数: {a4_nonzero} / {train_X.shape[0] * len(a4_cols)}")

    finally:
        exp_utils.EXPERIMENT_DIR = original_exp_dir

    log(f"\n{'═'*60}")
    log(f"  E0-Step2 验证完成！")
    log(f"{'═'*60}")


    # 配置日志到严格实验目录
    STRICT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # 临时设置日志目录
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STRICT_EXPERIMENT_DIR
    logger = setup_logging()
    exp_utils.EXPERIMENT_DIR = original_exp_dir

    start_time = time.time()

    log("=" * 60)
    log("  严格时间切分验证实验")
    log("  用前60%数据重新挖掘规则，消除数据泄露")
    log("=" * 60)

    all_comparisons = {}

    for lottery_type in ["daletou", "shuangseqiu"]:
        try:
            report, a1_count, a2_count, n_total, n_sliced = run_strict_pipeline(lottery_type)
            comparison = generate_comparison_report(
                lottery_type, report, a1_count, a2_count, n_total, n_sliced)
            all_comparisons[lottery_type] = comparison

            save_json(comparison,
                      STRICT_EXPERIMENT_DIR / f"strict_vs_original_comparison_{lottery_type}.json")

            log(f"\n  ═══ {lottery_type} 对比结果 ═══")
            if comparison.get("delta"):
                d = comparison["delta"]
                log(f"  efficiency 下降: {d['efficiency_drop']:.4f}")
                log(f"  accuracy 下降:   {d['accuracy_drop']:.4f}")
                log(f"  reduction 下降:  {d['reduction_drop']:.4f}")
                log(f"  survival 下降:   {d['survival_drop']:.4f}")
            log(f"  结论: {comparison.get('verdict', 'N/A')}")

        except Exception as e:
            log(f"\n[错误] {lottery_type} 严格验证失败: {e}")
            log(traceback.format_exc())
            all_comparisons[lottery_type] = {"error": str(e)}

    # 汇总
    total_time = time.time() - start_time
    summary = {
        "total_time_seconds": round(total_time, 1),
        "total_time_minutes": round(total_time / 60, 1),
        "comparisons": all_comparisons,
    }

    # 临时切换目录保存
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STRICT_EXPERIMENT_DIR
    save_json(summary, STRICT_EXPERIMENT_DIR / "experiment_summary.json")
    exp_utils.EXPERIMENT_DIR = original_exp_dir

    log(f"\n{'═'*60}")
    log(f"  严格验证实验完成！总耗时: {summary['total_time_minutes']:.1f} 分钟")
    log(f"  结果目录: {STRICT_EXPERIMENT_DIR}")
    log(f"{'═'*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="严格时间切分验证实验")
    parser.add_argument("--mode", default="full",
                        choices=["full", "a3a4", "a3a4_phase2"],
                        help="运行模式: full=完整管线, a3a4=仅补挖A3/A4, a3a4_phase2=验证A3/A4特征构造")
    args = parser.parse_args()

    if args.mode == "a3a4":
        run_a3a4_only()
    elif args.mode == "a3a4_phase2":
        run_a3a4_phase2_only()
    else:
        main()
