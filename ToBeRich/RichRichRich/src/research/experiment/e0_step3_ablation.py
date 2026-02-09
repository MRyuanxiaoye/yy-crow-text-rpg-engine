"""E0-Step3: A3/A4 增量贡献消融实验

4组对比实验：
  G0: A1 + A2 + 局面特征（基线，无A3/A4）
  G1: G0 + A3 差分模式
  G2: G0 + A4 统计量模式
  G3: G0 + A3 + A4（全量）

每组跑 XGBoost + MLP，两个彩种各跑4组 = 16个模型。
输出：增量贡献报告 + McNemar 显著性检验。

用法: python3 -m src.research.experiment.e0_step3_ablation
"""

import sys
import numpy as np
from pathlib import Path
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import research.experiment.utils as exp_utils
from research.experiment.utils import setup_logging, log, Timer, save_json, load_json
from research.experiment.phase1_compress import run_phase1
from research.experiment.phase2_replay import run_phase2
from research.experiment.phase3_train import (
    experiment_3b, experiment_3c, evaluate_direction_predictions,
)
from research.data_loader import LotteryData

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_RULES_DIR = RESULTS_DIR / "rules_strict"
STEP3_DIR = RESULTS_DIR / "e0_step3_ablation"


def filter_features_by_group(feature_names, train_X, test_X, group):
    """按组别筛选特征列

    Args:
        group: "G0" | "G1" | "G2" | "G3"

    Returns:
        filtered_names, filtered_train_X, filtered_test_X
    """
    keep = []
    for i, name in enumerate(feature_names):
        is_a3 = name.startswith('a3_')
        is_a4 = name.startswith('a4_')
        # A3/A4 元信息特征也要对应过滤
        is_a3_meta = name in ('n_matched_a3', 'avg_chi2_a3', 'avg_conf_a3')
        is_a4_meta = name in ('n_matched_a4', 'avg_chi2_a4', 'avg_conf_a4')

        if group == "G0":
            if not is_a3 and not is_a4 and not is_a3_meta and not is_a4_meta:
                keep.append(i)
        elif group == "G1":
            if not is_a4 and not is_a4_meta:
                keep.append(i)
        elif group == "G2":
            if not is_a3 and not is_a3_meta:
                keep.append(i)
        elif group == "G3":
            keep.append(i)

    keep = np.array(keep)
    filtered_names = [feature_names[i] for i in keep]
    filtered_train = train_X[:, keep]
    filtered_test = test_X[:, keep]
    return filtered_names, filtered_train, filtered_test


def mcnemar_test(pred_a, pred_b, true_y):
    """McNemar 检验：比较两个模型的预测差异是否显著

    Args:
        pred_a, pred_b: (n_samples, n_pos) 预测方向
        true_y: (n_samples, n_pos) 真实方向

    Returns:
        chi2, p_value
    """
    from scipy.stats import chi2 as chi2_dist

    correct_a = (pred_a == true_y).all(axis=1)  # 所有位置都对
    correct_b = (pred_b == true_y).all(axis=1)

    # 2x2 列联表
    n01 = int((correct_a & ~correct_b).sum())  # A对B错
    n10 = int((~correct_a & correct_b).sum())  # A错B对

    if n01 + n10 == 0:
        return 0.0, 1.0

    chi2_val = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_val = 1.0 - chi2_dist.cdf(chi2_val, df=1)
    return float(chi2_val), float(p_val)


def run_ablation_for_lottery(lottery_type):
    """对单个彩种运行4组消融实验"""
    log(f"\n{'═'*60}")
    log(f"  E0-Step3 消融实验: {lottery_type}")
    log(f"{'═'*60}")

    # Phase1: 规则压缩
    with Timer(f"Phase1 [{lottery_type}]"):
        clusters, a1_filtered = run_phase1(lottery_type, rules_dir=STRICT_RULES_DIR)

    # Phase2: 构造全量特征（含A3/A4）
    with Timer(f"Phase2 [{lottery_type}]"):
        (train_X, train_Y, test_X, test_Y,
         train_Y_val, test_Y_val,
         feature_names, train_indices, test_indices, data) = run_phase2(
            lottery_type, clusters, a1_filtered, rules_dir=STRICT_RULES_DIR)

    log(f"\n  全量特征: {len(feature_names)} 维")
    n_a3 = sum(1 for f in feature_names if f.startswith('a3_'))
    n_a4 = sum(1 for f in feature_names if f.startswith('a4_'))
    log(f"  A3 特征: {n_a3} 维, A4 特征: {n_a4} 维")

    # 4组消融实验
    groups = OrderedDict([
        ("G0", "基线 (A1+A2+局面)"),
        ("G1", "G0 + A3 差分模式"),
        ("G2", "G0 + A4 统计量模式"),
        ("G3", "全量 (A1+A2+A3+A4)"),
    ])

    all_results = {}
    all_preds = {}  # 保存预测结果用于McNemar检验

    for group_id, group_desc in groups.items():
        log(f"\n{'─'*50}")
        log(f"  {group_id}: {group_desc}")
        log(f"{'─'*50}")

        g_names, g_train, g_test = filter_features_by_group(
            feature_names, train_X, test_X, group_id)
        log(f"  特征维度: {len(g_names)}")

        group_results = {"group": group_id, "description": group_desc,
                         "n_features": len(g_names)}

        # XGBoost
        log(f"\n  --- XGBoost ---")
        res_xgb, probs_xgb = experiment_3b(
            g_train, train_Y, g_test, test_Y, g_names, data, test_indices)
        group_results["xgboost"] = {
            "direction_accuracy": res_xgb["overall"]["direction_accuracy"],
            "avg_reduction": res_xgb["overall"]["avg_reduction"],
            "avg_survival": res_xgb["overall"]["avg_survival"],
            "efficiency": res_xgb["overall"]["efficiency"],
            "per_position": res_xgb["per_position"],
        }
        pred_xgb = np.argmax(probs_xgb, axis=2)
        all_preds[f"{group_id}_xgb"] = pred_xgb

        # MLP
        log(f"\n  --- MLP ---")
        res_mlp, probs_mlp = experiment_3c(
            g_train, train_Y, g_test, test_Y, g_names, data, test_indices)
        if res_mlp is not None:
            group_results["mlp"] = {
                "direction_accuracy": res_mlp["overall"]["direction_accuracy"],
                "avg_reduction": res_mlp["overall"]["avg_reduction"],
                "avg_survival": res_mlp["overall"]["avg_survival"],
                "efficiency": res_mlp["overall"]["efficiency"],
                "per_position": res_mlp["per_position"],
            }
            pred_mlp = np.argmax(probs_mlp, axis=2)
            all_preds[f"{group_id}_mlp"] = pred_mlp
        else:
            group_results["mlp"] = None

        all_results[group_id] = group_results

    # McNemar 检验：G0 vs G1, G0 vs G2, G0 vs G3
    log(f"\n{'─'*50}")
    log(f"  McNemar 显著性检验")
    log(f"{'─'*50}")

    mcnemar_results = {}
    for model_type in ["xgb", "mlp"]:
        baseline_key = f"G0_{model_type}"
        if baseline_key not in all_preds:
            continue
        for compare_group in ["G1", "G2", "G3"]:
            compare_key = f"{compare_group}_{model_type}"
            if compare_key not in all_preds:
                continue
            chi2_val, p_val = mcnemar_test(
                all_preds[baseline_key], all_preds[compare_key], test_Y)
            key = f"G0_vs_{compare_group}_{model_type}"
            mcnemar_results[key] = {"chi2": chi2_val, "p_value": p_val,
                                     "significant": p_val < 0.05}
            sig = "显著" if p_val < 0.05 else "不显著"
            log(f"  {key}: chi2={chi2_val:.4f}, p={p_val:.4f} ({sig})")

    # 汇总报告
    log(f"\n{'═'*60}")
    log(f"  消融实验汇总: {lottery_type}")
    log(f"{'═'*60}")

    summary_table = []
    for group_id in groups:
        r = all_results[group_id]
        row = {"group": group_id, "description": groups[group_id],
               "n_features": r["n_features"]}
        for model_type in ["xgboost", "mlp"]:
            if r.get(model_type):
                row[f"{model_type}_accuracy"] = r[model_type]["direction_accuracy"]
                row[f"{model_type}_efficiency"] = r[model_type]["efficiency"]
            else:
                row[f"{model_type}_accuracy"] = None
                row[f"{model_type}_efficiency"] = None
        summary_table.append(row)

    # 打印汇总表
    log(f"\n  {'组别':<6} {'特征数':<8} {'XGB准确率':<12} {'XGB效率':<10} {'MLP准确率':<12} {'MLP效率':<10}")
    log(f"  {'─'*58}")
    for row in summary_table:
        xgb_acc = f"{row['xgboost_accuracy']:.4f}" if row['xgboost_accuracy'] else "N/A"
        xgb_eff = f"{row['xgboost_efficiency']:.4f}" if row['xgboost_efficiency'] else "N/A"
        mlp_acc = f"{row['mlp_accuracy']:.4f}" if row['mlp_accuracy'] else "N/A"
        mlp_eff = f"{row['mlp_efficiency']:.4f}" if row['mlp_efficiency'] else "N/A"
        log(f"  {row['group']:<6} {row['n_features']:<8} {xgb_acc:<12} {xgb_eff:<10} {mlp_acc:<12} {mlp_eff:<10}")

    # 计算增量
    log(f"\n  增量分析（相对G0基线）:")
    g0_xgb_acc = all_results["G0"]["xgboost"]["direction_accuracy"]
    g0_mlp_acc = all_results["G0"]["mlp"]["direction_accuracy"] if all_results["G0"]["mlp"] else None
    g0_xgb_eff = all_results["G0"]["xgboost"]["efficiency"]
    g0_mlp_eff = all_results["G0"]["mlp"]["efficiency"] if all_results["G0"]["mlp"] else None

    increments = {}
    for group_id in ["G1", "G2", "G3"]:
        r = all_results[group_id]
        inc = {"group": group_id}
        inc["xgb_acc_delta"] = r["xgboost"]["direction_accuracy"] - g0_xgb_acc
        inc["xgb_eff_delta"] = r["xgboost"]["efficiency"] - g0_xgb_eff
        if r["mlp"] and g0_mlp_acc is not None:
            inc["mlp_acc_delta"] = r["mlp"]["direction_accuracy"] - g0_mlp_acc
            inc["mlp_eff_delta"] = r["mlp"]["efficiency"] - g0_mlp_eff
        else:
            inc["mlp_acc_delta"] = None
            inc["mlp_eff_delta"] = None
        increments[group_id] = inc

        xgb_delta = f"{inc['xgb_acc_delta']:+.4f}"
        mlp_delta = f"{inc['mlp_acc_delta']:+.4f}" if inc['mlp_acc_delta'] is not None else "N/A"
        log(f"  {group_id}: XGB准确率 {xgb_delta}, MLP准确率 {mlp_delta}")

    # 结论
    log(f"\n  结论:")
    # 判断A3是否有用
    a3_useful_xgb = increments["G1"]["xgb_acc_delta"] > 0.005
    a3_useful_mlp = (increments["G1"]["mlp_acc_delta"] or 0) > 0.005
    a4_useful_xgb = increments["G2"]["xgb_acc_delta"] > 0.005
    a4_useful_mlp = (increments["G2"]["mlp_acc_delta"] or 0) > 0.005

    a3_verdict = "有用" if (a3_useful_xgb or a3_useful_mlp) else "无用"
    a4_verdict = "有用" if (a4_useful_xgb or a4_useful_mlp) else "无用"
    log(f"    A3 差分模式: {a3_verdict} (XGB: {increments['G1']['xgb_acc_delta']:+.4f})")
    log(f"    A4 统计量模式: {a4_verdict} (XGB: {increments['G2']['xgb_acc_delta']:+.4f})")

    # 保存完整结果
    report = {
        "lottery_type": lottery_type,
        "groups": all_results,
        "mcnemar_tests": mcnemar_results,
        "summary_table": summary_table,
        "increments": increments,
        "conclusion": {
            "a3_useful": a3_verdict,
            "a4_useful": a4_verdict,
            "a3_xgb_delta": increments["G1"]["xgb_acc_delta"],
            "a3_mlp_delta": increments["G1"]["mlp_acc_delta"],
            "a4_xgb_delta": increments["G2"]["xgb_acc_delta"],
            "a4_mlp_delta": increments["G2"]["mlp_acc_delta"],
        },
    }
    save_json(report, STEP3_DIR / f"e0_step3_ablation_{lottery_type}.json")
    return report


def main():
    STEP3_DIR.mkdir(parents=True, exist_ok=True)
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STEP3_DIR
    logger = setup_logging()

    log("=" * 60)
    log("  E0-Step3: A3/A4 增量贡献消融实验")
    log("=" * 60)

    all_reports = {}
    try:
        for lottery_type in ["daletou", "shuangseqiu"]:
            report = run_ablation_for_lottery(lottery_type)
            all_reports[lottery_type] = report

        # 跨彩种汇总
        log(f"\n{'═'*60}")
        log(f"  跨彩种汇总结论")
        log(f"{'═'*60}")

        for lt, rpt in all_reports.items():
            c = rpt["conclusion"]
            log(f"\n  {lt}:")
            log(f"    A3: {c['a3_useful']} (XGB {c['a3_xgb_delta']:+.4f}, MLP {c['a3_mlp_delta']:+.4f})" if c['a3_mlp_delta'] is not None else f"    A3: {c['a3_useful']} (XGB {c['a3_xgb_delta']:+.4f})")
            log(f"    A4: {c['a4_useful']} (XGB {c['a4_xgb_delta']:+.4f}, MLP {c['a4_mlp_delta']:+.4f})" if c['a4_mlp_delta'] is not None else f"    A4: {c['a4_useful']} (XGB {c['a4_xgb_delta']:+.4f})")

        # 最终决策
        log(f"\n  最终决策:")
        for lt, rpt in all_reports.items():
            c = rpt["conclusion"]
            if c["a3_useful"] == "有用":
                log(f"    {lt}: 保留 A3 特征")
            else:
                log(f"    {lt}: 移除 A3 特征（无增量贡献）")
            if c["a4_useful"] == "有用":
                log(f"    {lt}: 保留 A4 特征")
            else:
                log(f"    {lt}: 移除 A4 特征（无增量贡献）")

        save_json(all_reports, STEP3_DIR / "e0_step3_summary.json")

    finally:
        exp_utils.EXPERIMENT_DIR = original_exp_dir

    log(f"\n{'═'*60}")
    log(f"  E0-Step3 消融实验完成！")
    log(f"  结果目录: {STEP3_DIR}")
    log(f"{'═'*60}")


if __name__ == "__main__":
    main()
