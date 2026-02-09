# -*- coding: utf-8 -*-
"""E0-Step6: 蓝球规则适配实验

对蓝球运行完整的 Phase1-4 规则适配实验管线：
  Phase1: 蓝球 A2 规则聚类压缩 + A1 模式筛选
  Phase2: 蓝球历史回放 + 特征构造
  Phase3: 模型训练（3A/3B/3C）
  Phase4: 结果分析

复用 BlueLotteryDataAdapter 将蓝球数据映射到红球接口。

用法: python3 -m src.research.experiment.e0_step6_blue_experiment
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData
from research.experiment.e0_step5_blue_rules import BlueLotteryDataAdapter
from research.experiment.phase1_compress import compress_a2_rules, filter_a1_patterns
from research.experiment.phase2_replay import (
    check_rule_conditions, check_direction_pattern,
)
from research.experiment.phase3_train import run_phase3
from research.experiment.phase4_analyze import run_phase4
import research.experiment.utils as exp_utils
from research.experiment.utils import (
    log, Timer, save_json, load_json, save_npz, Normalizer,
    setup_logging,
)

# === 路径常量 ===
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_RULES_DIR = RESULTS_DIR / "rules_strict"
BLUE_EXPERIMENT_DIR = RESULTS_DIR / "experiment_blue"


# ============================================================
#  Phase1: 蓝球规则压缩
# ============================================================

def run_blue_phase1(lottery_type):
    """蓝球版 Phase1：加载蓝球 A1/A2 规则，聚类压缩 + 筛选

    Returns:
        clusters: A2 规则簇
        a1_filtered: 筛选后的 A1 模式
    """
    log(f"\n{'─'*40}")
    log(f"蓝球阶段一：规则压缩 [{lottery_type}]")
    log(f"{'─'*40}")

    # 加载蓝球 A2 规则
    with Timer("加载蓝球 A2 规则"):
        a2_path = STRICT_RULES_DIR / f"a2_blue_exclusion_rules_{lottery_type}.json"
        a2_data = load_json(a2_path)
        rules = a2_data['rules']
        log(f"  A2 规则数: {len(rules)} (保存的 top 规则)")
        log(f"  A2 总规则数: {a2_data['total_rules']}")

    # 聚类压缩
    # 蓝球规则数量可能较少（双色球仅 129 条），调整候选簇数
    with Timer("A2 规则聚类"):
        n_rules = len(rules)
        if n_rules < 100:
            candidate_ks = tuple(k for k in (10, 20, 30, 50) if k < n_rules)
        elif n_rules < 500:
            candidate_ks = tuple(k for k in (20, 50, 100, 200) if k < n_rules)
        else:
            candidate_ks = (100, 200, 500, 800, 1000)
        log(f"  候选簇数: {candidate_ks}")
        clusters, best_k, sil_score = compress_a2_rules(rules, candidate_ks=candidate_ks)

    # 加载并筛选蓝球 A1 模式
    with Timer("蓝球 A1 模式筛选"):
        a1_path = STRICT_RULES_DIR / f"a1_blue_direction_patterns_{lottery_type}.json"
        a1_data = load_json(a1_path)
        a1_patterns = a1_data['patterns']
        log(f"  A1 总模式数: {len(a1_patterns)}")
        a1_filtered = filter_a1_patterns(a1_patterns)

    # 保存
    output = {
        'lottery_type': lottery_type,
        'ball_type': 'blue',
        'n_clusters': best_k,
        'silhouette_score': sil_score,
        'n_a2_rules_input': len(rules),
        'a1_filtered_count': len(a1_filtered),
    }
    save_json(output, BLUE_EXPERIMENT_DIR / f"blue_phase1_summary_{lottery_type}.json")

    log(f"\n  蓝球阶段一汇总:")
    log(f"    A2: {len(rules)} 条 → {best_k} 个簇 (silhouette={sil_score:.4f})")
    log(f"    A1: {len(a1_patterns)} 条 → {len(a1_filtered)} 条")

    return clusters, a1_filtered


# ============================================================
#  Phase2: 蓝球历史回放 + 特征构造
# ============================================================

def build_blue_features_for_period(t, adapter, clusters, a1_filtered, diff_series, normalizer):
    """为第 t 期构造蓝球特征向量

    与红球版的区别：
    - 使用 adapter（BlueLotteryDataAdapter）的蓝球数据
    - 无 A3/A4 特征（蓝球未挖掘 A3/A4）
    - 无 combo_stats 特征（双色球 1 球无组合统计量，大乐透 2 球统计量有限）
    """
    features = {}
    rc = adapter.red_count  # 蓝球位置数（1 或 2）

    # === 局面特征 ===

    # 1. 最近 5 期各位置的值（归一化）
    for lag in range(1, 6):
        idx = t - lag
        if idx < 0:
            for pos in range(rc):
                features[f'val_lag{lag}_P{pos}'] = 0.0
        else:
            for pos in range(rc):
                features[f'val_lag{lag}_P{pos}'] = normalizer.transform(
                    f'val_P{pos}', float(adapter.red_matrix[idx, pos]))

    # 2. 最近 5 期各位置的方向
    for lag in range(1, 6):
        dir_idx = t - lag - 1
        for pos in range(rc):
            if 0 <= dir_idx < len(adapter.direction_series[pos]):
                features[f'dir_lag{lag}_P{pos}'] = float(adapter.direction_series[pos][dir_idx])
            else:
                features[f'dir_lag{lag}_P{pos}'] = 0.0

    # 3. 最近 5 期各位置的一阶差分（归一化）
    for lag in range(1, 6):
        diff_idx = t - lag - 1
        for pos in range(rc):
            if 0 <= diff_idx < len(diff_series[pos]):
                features[f'diff_lag{lag}_P{pos}'] = normalizer.transform(
                    f'diff_P{pos}', float(diff_series[pos][diff_idx]))
            else:
                features[f'diff_lag{lag}_P{pos}'] = 0.0

    # === A2 规则簇触发特征 ===
    n_triggered = 0
    conf_sum = 0.0
    lift_sum = 0.0

    for cid, cinfo in clusters.items():
        rule = cinfo['representative']
        triggered = check_rule_conditions(
            rule['conditions'], adapter.red_matrix, adapter.direction_series,
            diff_series, {}, t, adapter.red_range)
        val = 1.0 if triggered else 0.0
        features[f'a2_c{cid}'] = val
        if triggered:
            n_triggered += 1
            conf_sum += cinfo['avg_confidence']
            lift_sum += cinfo['avg_lift']

    # === A1 模式匹配特征 ===
    n_a1_matched = 0
    for i, pat in enumerate(a1_filtered):
        matched = check_direction_pattern(pat, adapter.direction_series, t)
        features[f'a1_p{i}'] = 1.0 if matched else 0.0
        if matched:
            n_a1_matched += 1

    # === 触发元信息 ===
    features['n_triggered_a2'] = float(n_triggered)
    features['avg_conf_triggered'] = conf_sum / n_triggered if n_triggered > 0 else 0.0
    features['avg_lift_triggered'] = lift_sum / n_triggered if n_triggered > 0 else 0.0
    features['n_matched_a1'] = float(n_a1_matched)

    return features


def build_blue_labels_for_period(t, adapter):
    """为第 t 期构造蓝球标签"""
    labels = {}
    for pos in range(adapter.red_count):
        if t - 1 >= 0 and t - 1 < len(adapter.direction_series[pos]):
            direction = int(adapter.direction_series[pos][t - 1])
            labels[f'dir_P{pos}'] = direction + 1  # -1→0, 0→1, 1→2
        else:
            labels[f'dir_P{pos}'] = 1  # 默认平

        labels[f'val_P{pos}'] = int(adapter.red_matrix[t, pos])

    return labels


def run_blue_phase2(lottery_type, clusters, a1_filtered):
    """蓝球版 Phase2：历史回放 + 数据构造

    Returns:
        train_X, train_Y, test_X, test_Y, feature_names, train_indices, test_indices, adapter
    """
    log(f"\n{'─'*40}")
    log(f"蓝球阶段二：历史回放 [{lottery_type}]")
    log(f"{'─'*40}")

    # 加载数据并创建适配器（使用全量数据，时间切分在内部完成）
    with Timer("加载数据"):
        data = LotteryData(lottery_type)
        adapter = BlueLotteryDataAdapter(data)  # 全量数据
        n = adapter.n_draws
        rc = adapter.red_count
        log(f"  总期数: {n}, 蓝球位置数: {rc}, 范围: 1-{adapter.red_range}")

    # 预计算差分
    with Timer("预计算差分"):
        diff_series = {}
        for pos in range(rc):
            diff_series[pos] = adapter.get_diff_series(pos, 1)

    # 时间切分
    split1 = int(n * 0.6)
    split2 = int(n * 0.85)
    log(f"  时间切分: 规则期[0:{split1}], 训练期[{split1}:{split2}], 测试期[{split2}:{n}]")
    log(f"  训练样本: {split2 - split1} 期, 测试样本: {n - split2} 期")

    # 归一化器：用规则期数据拟合
    with Timer("拟合归一化器"):
        normalizer = Normalizer()
        for pos in range(rc):
            normalizer.fit(f'val_P{pos}', adapter.red_matrix[:split1, pos].astype(float))
            valid_diff = diff_series[pos][:max(0, split1 - 1)]
            if len(valid_diff) > 0:
                normalizer.fit(f'diff_P{pos}', valid_diff.astype(float))
            else:
                normalizer.fit(f'diff_P{pos}', [0.0])

    # 构造特征和标签
    start_t = max(20, split1)

    with Timer("构造训练集特征"):
        train_features = []
        train_labels = []
        train_indices = []
        for t in range(start_t, split2):
            feat = build_blue_features_for_period(
                t, adapter, clusters, a1_filtered, diff_series, normalizer)
            lab = build_blue_labels_for_period(t, adapter)
            train_features.append(feat)
            train_labels.append(lab)
            train_indices.append(t)

            if (t - start_t + 1) % 200 == 0:
                log(f"    训练集进度: {t - start_t + 1}/{split2 - start_t}")

    with Timer("构造测试集特征"):
        test_features = []
        test_labels = []
        test_indices = []
        for t in range(split2, n):
            feat = build_blue_features_for_period(
                t, adapter, clusters, a1_filtered, diff_series, normalizer)
            lab = build_blue_labels_for_period(t, adapter)
            test_features.append(feat)
            test_labels.append(lab)
            test_indices.append(t)

            if (t - split2 + 1) % 200 == 0:
                log(f"    测试集进度: {t - split2 + 1}/{n - split2}")

    # 转换为 numpy 数组
    with Timer("转换为 numpy 数组"):
        feature_names = sorted(train_features[0].keys())
        train_X = np.array([[f[k] for k in feature_names] for f in train_features], dtype=np.float32)
        test_X = np.array([[f[k] for k in feature_names] for f in test_features], dtype=np.float32)

        dir_label_names = [f'dir_P{pos}' for pos in range(rc)]
        train_Y = np.array([[l[k] for k in dir_label_names] for l in train_labels], dtype=np.int32)
        test_Y = np.array([[l[k] for k in dir_label_names] for l in test_labels], dtype=np.int32)

        val_label_names = [f'val_P{pos}' for pos in range(rc)]
        train_Y_val = np.array([[l[k] for k in val_label_names] for l in train_labels], dtype=np.int32)
        test_Y_val = np.array([[l[k] for k in val_label_names] for l in test_labels], dtype=np.int32)

        train_indices = np.array(train_indices, dtype=np.int32)
        test_indices = np.array(test_indices, dtype=np.int32)

    log(f"\n  数据集形状:")
    log(f"    train_X: {train_X.shape}, train_Y: {train_Y.shape}")
    log(f"    test_X:  {test_X.shape},  test_Y:  {test_Y.shape}")
    log(f"    特征数: {len(feature_names)}")

    # 保存
    save_npz(BLUE_EXPERIMENT_DIR / f"blue_phase2_train_{lottery_type}.npz",
             X=train_X, Y=train_Y, Y_val=train_Y_val, indices=train_indices)
    save_npz(BLUE_EXPERIMENT_DIR / f"blue_phase2_test_{lottery_type}.npz",
             X=test_X, Y=test_Y, Y_val=test_Y_val, indices=test_indices)
    save_json(feature_names, BLUE_EXPERIMENT_DIR / f"blue_phase2_feature_names_{lottery_type}.json")

    summary = {
        'lottery_type': lottery_type,
        'ball_type': 'blue',
        'n_draws': n,
        'blue_count': rc,
        'blue_range': adapter.red_range,
        'split1': split1,
        'split2': split2,
        'train_samples': int(train_X.shape[0]),
        'test_samples': int(test_X.shape[0]),
        'n_features': len(feature_names),
        'n_a2_clusters': len(clusters),
        'n_a1_patterns': len(a1_filtered),
        'train_Y_distribution': {
            f'P{pos}': {
                'D': int((train_Y[:, pos] == 0).sum()),
                'E': int((train_Y[:, pos] == 1).sum()),
                'U': int((train_Y[:, pos] == 2).sum()),
            } for pos in range(rc)
        },
    }
    save_json(summary, BLUE_EXPERIMENT_DIR / f"blue_phase2_summary_{lottery_type}.json")

    return train_X, train_Y, test_X, test_Y, train_Y_val, test_Y_val, feature_names, train_indices, test_indices, adapter


# ============================================================
#  主流程
# ============================================================

def run_blue_pipeline(lottery_type):
    """对单个彩种运行蓝球适配实验管线"""
    log(f"\n{'═'*50}")
    log(f"  蓝球适配实验: {lottery_type}")
    log(f"{'═'*50}")

    with Timer(f"蓝球阶段一 [{lottery_type}]"):
        clusters, a1_filtered = run_blue_phase1(lottery_type)

    with Timer(f"蓝球阶段二 [{lottery_type}]"):
        (train_X, train_Y, test_X, test_Y,
         train_Y_val, test_Y_val,
         feature_names, train_indices, test_indices, adapter) = run_blue_phase2(
            lottery_type, clusters, a1_filtered)

    with Timer(f"蓝球阶段三 [{lottery_type}]"):
        results_3a, results_3b, results_3c = run_phase3(
            train_X, train_Y, test_X, test_Y,
            feature_names, adapter, test_indices)

    with Timer(f"蓝球阶段四 [{lottery_type}]"):
        final_report = run_phase4(lottery_type, results_3a, results_3b, results_3c)

    return final_report


def main():
    BLUE_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # 临时替换 EXPERIMENT_DIR 让 Phase3/Phase4 的输出写到蓝球目录
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = BLUE_EXPERIMENT_DIR
    setup_logging()

    start_time = time.time()

    log("=" * 60)
    log("  E0-Step6: 蓝球规则适配实验")
    log("=" * 60)

    all_reports = {}

    try:
        for lottery_type in ["daletou", "shuangseqiu"]:
            try:
                report = run_blue_pipeline(lottery_type)
                all_reports[lottery_type] = report
            except Exception as e:
                import traceback
                log(f"\n[错误] {lottery_type} 蓝球实验失败: {e}")
                log(traceback.format_exc())
                all_reports[lottery_type] = {"error": str(e)}
    finally:
        exp_utils.EXPERIMENT_DIR = original_exp_dir

    # 汇总
    total_time = time.time() - start_time
    summary = {
        "total_time_seconds": round(total_time, 1),
        "reports": {},
    }

    for lt, report in all_reports.items():
        if report and report.get("conclusion"):
            c = report["conclusion"]
            summary["reports"][lt] = {
                "best_method": c.get("best_method"),
                "efficiency": c.get("best_efficiency", 0),
                "direction_accuracy": c.get("direction_accuracy", 0),
                "avg_reduction": c.get("avg_reduction", 0),
                "avg_survival": c.get("avg_survival", 0),
                "verdict": c.get("verdict"),
            }
        else:
            summary["reports"][lt] = report

    # 临时切换目录保存汇总
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = BLUE_EXPERIMENT_DIR
    save_json(summary, BLUE_EXPERIMENT_DIR / "e0_step6_blue_summary.json")
    exp_utils.EXPERIMENT_DIR = original_exp_dir

    log(f"\n{'═'*60}")
    log(f"  E0-Step6 完成！总耗时: {total_time:.1f}s")
    log(f"{'═'*60}")
    for lt, r in summary["reports"].items():
        if isinstance(r, dict) and "efficiency" in r:
            log(f"  {lt}: 方向准确率={r['direction_accuracy']:.4f}, "
                f"缩减率={r['avg_reduction']:.4f}, "
                f"存活率={r['avg_survival']:.4f}, "
                f"效率={r['efficiency']:.4f} [{r['verdict']}]")
        else:
            log(f"  {lt}: {r}")
    log(f"  结果目录: {BLUE_EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
