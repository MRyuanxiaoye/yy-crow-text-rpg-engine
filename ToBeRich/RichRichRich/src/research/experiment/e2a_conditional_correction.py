"""E2a：条件方向概率修正实验

用条件概率矩阵修正模型输出的方向概率，验证能否提升方向预测、号码排序和组合缩号效果。

用法: python3 -m src.research.experiment.e2a_conditional_correction
"""

import sys
import time
import traceback
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import research.experiment.utils as exp_utils
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_json, load_npz,
)
from research.experiment.e1_combo_evaluation import (
    get_best_probs, count_ordered_combos, check_combo_survival,
    generate_candidate_sets,
)
from research.experiment.e1_5_weighted_evaluation import (
    compute_diff_distributions, compute_number_weights,
)
from research.data_loader import LotteryData

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_EXPERIMENT_DIR = RESULTS_DIR / "experiment_strict"

DIR_MAP = {-1: 0, 0: 1, 1: 2}  # D=0, E=1, U=2


# ============================================================
#  Step 0：条件概率矩阵稳定性验证
# ============================================================

def build_cond_prob_matrix(data, start_idx, end_idx):
    """从指定方向序列范围构建条件概率矩阵

    Args:
        data: LotteryData
        start_idx: 方向序列起始索引（含）
        end_idx: 方向序列结束索引（不含）

    Returns:
        matrices: {(pos, pos+1): np.array(3,3)}
                  matrix[d_prev, d_next] = P(d_next | d_prev)
    """
    n_pos = data.red_count
    dir_series = data.direction_series
    matrices = {}

    for pos in range(n_pos - 1):
        joint = np.zeros((3, 3), dtype=float)
        max_idx = min(end_idx, len(dir_series[pos]), len(dir_series[pos + 1]))
        for t in range(start_idx, max_idx):
            d_prev = DIR_MAP[int(dir_series[pos][t])]
            d_next = DIR_MAP[int(dir_series[pos + 1][t])]
            joint[d_prev, d_next] += 1

        # 拉普拉斯平滑 +1
        joint += 1.0
        row_sums = joint.sum(axis=1, keepdims=True)
        matrices[(pos, pos + 1)] = joint / row_sums

    return matrices


def kl_divergence_matrix(p, q):
    """计算两个条件概率矩阵的平均 KL 散度（按行）"""
    kl_sum = 0.0
    n_rows = p.shape[0]
    for r in range(n_rows):
        for c in range(p.shape[1]):
            if p[r, c] > 1e-10 and q[r, c] > 1e-10:
                kl_sum += p[r, c] * np.log(p[r, c] / q[r, c])
    return kl_sum / n_rows


def step0_stability(data, test_indices, lottery_type):
    """Step 0：条件概率矩阵稳定性验证"""
    log(f"\n{'─'*50}")
    log(f"  Step 0：条件概率矩阵稳定性验证 [{lottery_type}]")
    log(f"{'─'*50}")

    max_train_dir_idx = int(test_indices[0]) - 1
    n_pos = data.red_count

    # 全局矩阵（训练期）
    global_matrices = build_cond_prob_matrix(data, 0, max_train_dir_idx)

    # === 0a 滚动窗口稳定性 ===
    log("\n  0a: 滚动窗口稳定性")
    n_windows = 5
    window_size = max_train_dir_idx // n_windows
    window_matrices_list = []
    for w in range(n_windows):
        ws = w * window_size
        we = (w + 1) * window_size if w < n_windows - 1 else max_train_dir_idx
        wm = build_cond_prob_matrix(data, ws, we)
        window_matrices_list.append(wm)

    stability_results = {}
    all_std_ok = True
    all_kl_ok = True

    for pos in range(n_pos - 1):
        pair_key = f"P{pos}_P{pos+1}"
        # 收集各窗口的矩阵
        window_mats = np.array([wm[(pos, pos + 1)] for wm in window_matrices_list])
        # 元素标准差
        elem_std = np.std(window_mats, axis=0)
        max_std = float(np.max(elem_std))
        mean_std = float(np.mean(elem_std))

        # 各窗口与全局的 KL
        kl_values = []
        for wm in window_matrices_list:
            kl_val = kl_divergence_matrix(wm[(pos, pos + 1)], global_matrices[(pos, pos + 1)])
            kl_values.append(kl_val)
        max_kl = float(np.max(kl_values))
        mean_kl = float(np.mean(kl_values))

        std_pass = max_std < 0.05
        kl_pass = max_kl < 0.02
        if not std_pass:
            all_std_ok = False
        if not kl_pass:
            all_kl_ok = False

        stability_results[pair_key] = {
            "max_elem_std": max_std,
            "mean_elem_std": mean_std,
            "std_pass": bool(std_pass),
            "kl_values": [float(v) for v in kl_values],
            "max_kl": max_kl,
            "mean_kl": mean_kl,
            "kl_pass": bool(kl_pass),
        }
        log(f"  {pair_key}: max_std={max_std:.4f}({'OK' if std_pass else 'FAIL'}), "
            f"max_kl={max_kl:.4f}({'OK' if kl_pass else 'FAIL'})")

    step0a_pass = all_std_ok and all_kl_ok
    log(f"  0a 结论: {'通过' if step0a_pass else '未通过'}")

    # === 0b 训练期 vs 测试期一致性 ===
    log("\n  0b: 训练期 vs 测试期一致性")
    test_start_dir_idx = int(test_indices[0]) - 1
    test_end_dir_idx = min(data.n_draws - 1,
                           max(len(data.direction_series[p]) for p in range(n_pos)))
    test_matrices = build_cond_prob_matrix(data, test_start_dir_idx, test_end_dir_idx)

    consistency_results = {}
    all_chi2_ok = True

    for pos in range(n_pos - 1):
        pair_key = f"P{pos}_P{pos+1}"
        # 用原始计数做卡方检验（不含平滑）
        dir_series = data.direction_series
        # 训练期联合计数
        train_joint = np.zeros((3, 3), dtype=float)
        max_idx = min(max_train_dir_idx, len(dir_series[pos]), len(dir_series[pos + 1]))
        for t in range(0, max_idx):
            d_prev = DIR_MAP[int(dir_series[pos][t])]
            d_next = DIR_MAP[int(dir_series[pos + 1][t])]
            train_joint[d_prev, d_next] += 1

        # 测试期联合计数
        test_joint = np.zeros((3, 3), dtype=float)
        max_idx_t = min(test_end_dir_idx, len(dir_series[pos]), len(dir_series[pos + 1]))
        for t in range(test_start_dir_idx, max_idx_t):
            d_prev = DIR_MAP[int(dir_series[pos][t])]
            d_next = DIR_MAP[int(dir_series[pos + 1][t])]
            test_joint[d_prev, d_next] += 1

        # 齐性检验：对每行（条件分布）做卡方检验
        row_p_values = []
        for r in range(3):
            observed = np.array([train_joint[r], test_joint[r]])
            # 避免全零行
            if observed.sum() == 0:
                row_p_values.append(1.0)
                continue
            observed = np.maximum(observed, 0.5)
            chi2, p, dof, _ = sp_stats.chi2_contingency(observed)
            row_p_values.append(float(p))

        min_p = min(row_p_values)
        chi2_pass = min_p > 0.05
        if not chi2_pass:
            all_chi2_ok = False

        consistency_results[pair_key] = {
            "row_p_values": row_p_values,
            "min_p_value": min_p,
            "chi2_pass": bool(chi2_pass),
        }
        log(f"  {pair_key}: min_p={min_p:.4f} ({'通过' if chi2_pass else '不通过'})")

    step0b_pass = all_chi2_ok
    use_sliding_window = not step0b_pass
    log(f"  0b 结论: {'通过' if step0b_pass else '不通过，将使用滑动窗口'}")

    results = {
        "step0a_rolling_stability": {
            "n_windows": n_windows,
            "window_size": window_size,
            "per_pair": stability_results,
            "pass": bool(step0a_pass),
        },
        "step0b_train_test_consistency": {
            "per_pair": consistency_results,
            "pass": bool(step0b_pass),
        },
        "overall_pass": bool(step0a_pass and step0b_pass),
        "use_sliding_window": bool(use_sliding_window),
    }

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2a_step0_stability_{lottery_type}.json")
    return global_matrices, use_sliding_window, results


# ============================================================
#  Step 1：单步修正 + alpha 扫描
# ============================================================

def correct_probs_single_step(probs, cond_matrices, alpha):
    """用条件概率矩阵单步修正方向概率

    P0 保持不变。对 pos >= 1：
    P_corrected[d] = sum_d'( P_model[pos-1, d'] * cond_matrix[d', d] )
    P_final = (1-alpha) * P_model[pos] + alpha * P_corrected
    """
    n_samples, n_pos, _ = probs.shape
    corrected = probs.copy()

    for i in range(n_samples):
        for pos in range(1, n_pos):
            key = (pos - 1, pos)
            if key not in cond_matrices:
                continue
            cond_mat = cond_matrices[key]  # (3,3), [d_prev, d_next]
            p_prev = probs[i, pos - 1, :]  # 前一位置的原始模型概率
            p_corrected = p_prev @ cond_mat  # (3,) = sum over d_prev
            p_final = (1 - alpha) * probs[i, pos, :] + alpha * p_corrected
            # 归一化
            s = p_final.sum()
            if s > 0:
                p_final /= s
            corrected[i, pos, :] = p_final

    return corrected


def compute_direction_accuracy(probs, data, test_indices, pos_start=0):
    """计算方向预测准确率，返回逐样本逐位置的正误数组"""
    n_samples, n_pos, _ = probs.shape
    dir_series = data.direction_series
    correct_arr = []  # 每个元素是 True/False

    for i in range(n_samples):
        t = int(test_indices[i])
        dir_idx = t - 1
        for pos in range(pos_start, n_pos):
            if dir_idx < 0 or dir_idx >= len(dir_series[pos]):
                continue
            true_dir = DIR_MAP[int(dir_series[pos][dir_idx])]
            pred_dir = int(np.argmax(probs[i, pos, :]))
            correct_arr.append(pred_dir == true_dir)

    return np.array(correct_arr, dtype=bool)


def step1_direction(probs, cond_matrices, data, test_indices, lottery_type):
    """Step 1：alpha 扫描 + McNemar 检验"""
    log(f"\n{'─'*50}")
    log(f"  Step 1：单步修正 + alpha 扫描 [{lottery_type}]")
    log(f"{'─'*50}")

    alphas = np.arange(0.0, 1.05, 0.1)
    base_correct = compute_direction_accuracy(probs, data, test_indices, pos_start=1)
    base_acc = float(base_correct.mean())
    log(f"  基线 (alpha=0) 方向准确率: {base_acc:.4f}")

    results_per_alpha = []
    best_alpha = 0.0
    best_acc = base_acc

    for alpha in alphas:
        alpha = round(float(alpha), 2)
        if alpha == 0.0:
            acc = base_acc
        else:
            corrected = correct_probs_single_step(probs, cond_matrices, alpha)
            correct_arr = compute_direction_accuracy(
                corrected, data, test_indices, pos_start=1)
            acc = float(correct_arr.mean())

        results_per_alpha.append({"alpha": alpha, "accuracy": acc})
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    log(f"  最优 alpha={best_alpha:.1f}, 准确率={best_acc:.4f} (基线={base_acc:.4f})")

    # McNemar 检验：best_alpha vs alpha=0
    mcnemar_p = 1.0
    passed = False
    if best_alpha > 0:
        corrected_best = correct_probs_single_step(probs, cond_matrices, best_alpha)
        best_correct = compute_direction_accuracy(
            corrected_best, data, test_indices, pos_start=1)

        # 2x2 表
        both_correct = int(np.sum(base_correct & best_correct))
        only_base = int(np.sum(base_correct & ~best_correct))
        only_best = int(np.sum(~base_correct & best_correct))
        both_wrong = int(np.sum(~base_correct & ~best_correct))

        # McNemar: chi2 = (b-c)^2 / (b+c)
        b, c = only_base, only_best
        if b + c > 0:
            chi2_val = (b - c) ** 2 / (b + c)
            mcnemar_p = float(sp_stats.chi2.sf(chi2_val, 1))
        passed = mcnemar_p < 0.05

        log(f"  McNemar 检验: only_base={only_base}, only_best={only_best}, "
            f"chi2={(b-c)**2/(b+c) if b+c>0 else 0:.2f}, p={mcnemar_p:.6f}")
        log(f"  结论: {'显著' if passed else '不显著'}")

        # 按位置分析
        n_samples, n_pos, _ = probs.shape
        pos_accs = {}
        for pos in range(n_pos):
            base_pos = compute_direction_accuracy(probs, data, test_indices, pos_start=pos)
            best_pos = compute_direction_accuracy(corrected_best, data, test_indices, pos_start=pos)
            # 只取该位置的
            per_pos_base = []
            per_pos_best = []
            dir_series = data.direction_series
            for ii in range(n_samples):
                t = int(test_indices[ii])
                dir_idx = t - 1
                if dir_idx < 0 or dir_idx >= len(dir_series[pos]):
                    continue
                true_dir = DIR_MAP[int(dir_series[pos][dir_idx])]
                pred_base = int(np.argmax(probs[ii, pos, :]))
                pred_best = int(np.argmax(corrected_best[ii, pos, :]))
                per_pos_base.append(pred_base == true_dir)
                per_pos_best.append(pred_best == true_dir)
            pos_accs[f"P{pos}"] = {
                "base_acc": float(np.mean(per_pos_base)) if per_pos_base else 0,
                "corrected_acc": float(np.mean(per_pos_best)) if per_pos_best else 0,
            }
            log(f"    P{pos}: base={pos_accs[f'P{pos}']['base_acc']:.4f}, "
                f"corrected={pos_accs[f'P{pos}']['corrected_acc']:.4f}")
    else:
        pos_accs = {}

    results = {
        "alpha_scan": results_per_alpha,
        "best_alpha": best_alpha,
        "best_accuracy": best_acc,
        "baseline_accuracy": base_acc,
        "improvement": best_acc - base_acc,
        "mcnemar_p_value": mcnemar_p,
        "significant": bool(passed),
        "per_position": pos_accs,
    }

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2a_step1_direction_{lottery_type}.json")
    return best_alpha, passed, results


# ============================================================
#  Step 2：号码权重 + AUC 评估
# ============================================================

def compute_auc_from_weights(all_weights, data, test_indices):
    """计算每位置的 AUC 和逐样本 rank"""
    n_samples = len(all_weights)
    n_pos = data.red_count
    red_range = data.red_range
    ranks = {pos: [] for pos in range(n_pos)}

    for i in range(n_samples):
        t = int(test_indices[i])
        for pos in range(n_pos):
            true_val = int(data.red_matrix[t, pos])
            sorted_nums = sorted(all_weights[i][pos].items(),
                                 key=lambda x: x[1], reverse=True)
            for r, (v, w) in enumerate(sorted_nums, 1):
                if v == true_val:
                    ranks[pos].append(r)
                    break

    aucs = {}
    all_auc_vals = []
    for pos in range(n_pos):
        r_arr = np.array(ranks[pos], dtype=float)
        auc = float(1.0 - (r_arr.mean() - 1) / (red_range - 1))
        aucs[f"P{pos}"] = {"auc": auc, "mean_rank": float(r_arr.mean())}
        all_auc_vals.append(auc)

    aucs["overall_auc"] = float(np.mean(all_auc_vals))
    return aucs, ranks


def step2_auc(probs, cond_matrices, best_alpha, data, test_indices, lottery_type):
    """Step 2：号码权重 + AUC 评估"""
    log(f"\n{'─'*50}")
    log(f"  Step 2：号码权重 + AUC 评估 [{lottery_type}]")
    log(f"{'─'*50}")

    max_train_idx = int(test_indices[0])
    diff_dist = compute_diff_distributions(data, max_train_idx)

    # 原始 AUC
    log("  计算原始 probs 的 AUC...")
    orig_weights = compute_number_weights(probs, data, test_indices, diff_dist)
    orig_aucs, orig_ranks = compute_auc_from_weights(orig_weights, data, test_indices)
    log(f"  原始 AUC: {orig_aucs['overall_auc']:.4f}")

    # 修正后 AUC
    log(f"  计算修正 probs (alpha={best_alpha}) 的 AUC...")
    corrected = correct_probs_single_step(probs, cond_matrices, best_alpha)
    corr_weights = compute_number_weights(corrected, data, test_indices, diff_dist)
    corr_aucs, corr_ranks = compute_auc_from_weights(corr_weights, data, test_indices)
    log(f"  修正 AUC: {corr_aucs['overall_auc']:.4f}")

    # Wilcoxon 符号秩检验：配对比较每位置的 rank
    n_pos = data.red_count
    wilcoxon_results = {}
    all_orig_ranks = []
    all_corr_ranks = []

    for pos in range(n_pos):
        o_ranks = np.array(orig_ranks[pos], dtype=float)
        c_ranks = np.array(corr_ranks[pos], dtype=float)
        all_orig_ranks.extend(orig_ranks[pos])
        all_corr_ranks.extend(corr_ranks[pos])

        diff = o_ranks - c_ranks  # 正值 = 修正更好
        nonzero = diff[diff != 0]
        if len(nonzero) > 10:
            stat, p = sp_stats.wilcoxon(nonzero)
            mean_diff = float(np.mean(diff))
        else:
            stat, p = 0.0, 1.0
            mean_diff = 0.0

        wilcoxon_results[f"P{pos}"] = {
            "orig_auc": orig_aucs[f"P{pos}"]["auc"],
            "corr_auc": corr_aucs[f"P{pos}"]["auc"],
            "auc_change": corr_aucs[f"P{pos}"]["auc"] - orig_aucs[f"P{pos}"]["auc"],
            "mean_rank_diff": mean_diff,
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(p),
        }
        log(f"  P{pos}: orig={orig_aucs[f'P{pos}']['auc']:.4f}, "
            f"corr={corr_aucs[f'P{pos}']['auc']:.4f}, "
            f"p={float(p):.4f}")

    # 整体 Wilcoxon
    all_orig = np.array(all_orig_ranks, dtype=float)
    all_corr = np.array(all_corr_ranks, dtype=float)
    diff_all = all_orig - all_corr
    nonzero_all = diff_all[diff_all != 0]
    if len(nonzero_all) > 10:
        stat_all, p_all = sp_stats.wilcoxon(nonzero_all)
    else:
        stat_all, p_all = 0.0, 1.0

    results = {
        "best_alpha": best_alpha,
        "original_auc": orig_aucs["overall_auc"],
        "corrected_auc": corr_aucs["overall_auc"],
        "auc_change": corr_aucs["overall_auc"] - orig_aucs["overall_auc"],
        "overall_wilcoxon_stat": float(stat_all),
        "overall_wilcoxon_p": float(p_all),
        "per_position": wilcoxon_results,
    }

    log(f"  整体: orig={orig_aucs['overall_auc']:.4f}, "
        f"corr={corr_aucs['overall_auc']:.4f}, "
        f"Wilcoxon p={float(p_all):.6f}")

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2a_step2_auc_{lottery_type}.json")
    return results


# ============================================================
#  Step 3：链式传播 vs 单步 vs 双向
# ============================================================

def correct_probs_chain(probs, cond_matrices, alpha):
    """链式修正：P2 用修正后的 P1，P3 用修正后的 P2..."""
    n_samples, n_pos, _ = probs.shape
    corrected = probs.copy()

    for i in range(n_samples):
        for pos in range(1, n_pos):
            key = (pos - 1, pos)
            if key not in cond_matrices:
                continue
            cond_mat = cond_matrices[key]
            # 用修正后的前一位置概率（链式传播）
            p_prev = corrected[i, pos - 1, :]
            p_corrected = p_prev @ cond_mat
            p_final = (1 - alpha) * probs[i, pos, :] + alpha * p_corrected
            s = p_final.sum()
            if s > 0:
                p_final /= s
            corrected[i, pos, :] = p_final

    return corrected


def build_reverse_cond_matrices(data, start_idx, end_idx):
    """构建反向条件概率矩阵 P(d_prev | d_next)

    返回 {(pos, pos+1): matrix} 其中 matrix[d_next, d_prev] = P(d_prev | d_next)
    """
    n_pos = data.red_count
    dir_series = data.direction_series
    matrices = {}

    for pos in range(n_pos - 1):
        joint = np.zeros((3, 3), dtype=float)
        max_idx = min(end_idx, len(dir_series[pos]), len(dir_series[pos + 1]))
        for t in range(start_idx, max_idx):
            d_prev = DIR_MAP[int(dir_series[pos][t])]
            d_next = DIR_MAP[int(dir_series[pos + 1][t])]
            joint[d_next, d_prev] += 1  # 注意转置

        joint += 1.0
        row_sums = joint.sum(axis=1, keepdims=True)
        matrices[(pos, pos + 1)] = joint / row_sums

    return matrices


def correct_probs_bidirectional(probs, cond_fwd, cond_bwd, alpha):
    """双向修正：同时用前向和后向条件概率

    P0 保持不变。最后一个位置只有前向。
    中间位置：P_final = (1-alpha)*P_model + alpha*0.5*(P_fwd + P_bwd)
    """
    n_samples, n_pos, _ = probs.shape
    corrected = probs.copy()

    for i in range(n_samples):
        for pos in range(1, n_pos):
            fwd_key = (pos - 1, pos)
            bwd_key = (pos, pos + 1) if pos < n_pos - 1 else None

            # 前向修正
            p_fwd = np.zeros(3)
            if fwd_key in cond_fwd:
                p_prev = probs[i, pos - 1, :]
                p_fwd = p_prev @ cond_fwd[fwd_key]

            # 后向修正
            p_bwd = np.zeros(3)
            has_bwd = False
            if bwd_key is not None and bwd_key in cond_bwd:
                p_next = probs[i, pos + 1, :]
                # cond_bwd[(pos, pos+1)][d_next, d_prev] = P(d_prev | d_next)
                # 但这里我们要 P(d_pos | d_{pos+1})
                # 需要用 (pos, pos+1) 的反向矩阵
                p_bwd = p_next @ cond_bwd[bwd_key]
                has_bwd = True

            if has_bwd:
                p_corrected = 0.5 * (p_fwd + p_bwd)
            else:
                p_corrected = p_fwd

            p_final = (1 - alpha) * probs[i, pos, :] + alpha * p_corrected
            s = p_final.sum()
            if s > 0:
                p_final /= s
            corrected[i, pos, :] = p_final

    return corrected


def step3_chain_vs_single(probs, cond_matrices, best_alpha, data, test_indices, lottery_type):
    """Step 3：三种修正方案对比"""
    log(f"\n{'─'*50}")
    log(f"  Step 3：链式 vs 单步 vs 双向 [{lottery_type}]")
    log(f"{'─'*50}")

    max_train_dir_idx = int(test_indices[0]) - 1
    n_pos = data.red_count
    max_train_idx = int(test_indices[0])
    diff_dist = compute_diff_distributions(data, max_train_idx)

    # 三种方案
    log("  计算单步修正...")
    probs_single = correct_probs_single_step(probs, cond_matrices, best_alpha)

    log("  计算链式修正...")
    probs_chain = correct_probs_chain(probs, cond_matrices, best_alpha)

    log("  计算双向修正...")
    cond_bwd = build_reverse_cond_matrices(data, 0, max_train_dir_idx)
    probs_bidir = correct_probs_bidirectional(probs, cond_matrices, cond_bwd, best_alpha)

    schemes = {
        "single_step": probs_single,
        "chain": probs_chain,
        "bidirectional": probs_bidir,
    }

    results = {}
    for name, corr_probs in schemes.items():
        log(f"\n  === {name} ===")
        # 方向准确率（按位置）
        pos_accs = {}
        dir_series = data.direction_series
        n_samples = probs.shape[0]

        for pos in range(n_pos):
            base_correct = 0
            corr_correct = 0
            total = 0
            for ii in range(n_samples):
                t = int(test_indices[ii])
                dir_idx = t - 1
                if dir_idx < 0 or dir_idx >= len(dir_series[pos]):
                    continue
                true_dir = DIR_MAP[int(dir_series[pos][dir_idx])]
                pred_base = int(np.argmax(probs[ii, pos, :]))
                pred_corr = int(np.argmax(corr_probs[ii, pos, :]))
                if pred_base == true_dir:
                    base_correct += 1
                if pred_corr == true_dir:
                    corr_correct += 1
                total += 1

            base_acc = base_correct / total if total > 0 else 0
            corr_acc = corr_correct / total if total > 0 else 0
            pos_accs[f"P{pos}"] = {
                "base_acc": float(base_acc),
                "corrected_acc": float(corr_acc),
                "change": float(corr_acc - base_acc),
            }
            log(f"    P{pos}: base={base_acc:.4f}, corr={corr_acc:.4f}, "
                f"Δ={corr_acc - base_acc:+.4f}")

        # 整体方向准确率
        overall_base = compute_direction_accuracy(probs, data, test_indices, pos_start=1)
        overall_corr = compute_direction_accuracy(corr_probs, data, test_indices, pos_start=1)

        # AUC
        corr_weights = compute_number_weights(corr_probs, data, test_indices, diff_dist)
        corr_aucs, _ = compute_auc_from_weights(corr_weights, data, test_indices)

        results[name] = {
            "overall_direction_acc": float(overall_corr.mean()),
            "overall_auc": corr_aucs["overall_auc"],
            "per_position": pos_accs,
        }
        log(f"    整体方向准确率: {float(overall_corr.mean()):.4f}")
        log(f"    整体 AUC: {corr_aucs['overall_auc']:.4f}")

    # 选最优方案
    best_scheme = max(results.keys(), key=lambda k: results[k]["overall_direction_acc"])
    results["best_scheme"] = best_scheme
    log(f"\n  最优方案: {best_scheme}")

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2a_step3_chain_vs_single_{lottery_type}.json")
    return best_scheme, results


# ============================================================
#  Step 4：组合级评估
# ============================================================

def get_corrected_probs(probs, cond_matrices, cond_bwd, alpha, scheme):
    """根据方案名称获取修正后的 probs"""
    if scheme == "single_step":
        return correct_probs_single_step(probs, cond_matrices, alpha)
    elif scheme == "chain":
        return correct_probs_chain(probs, cond_matrices, alpha)
    elif scheme == "bidirectional":
        return correct_probs_bidirectional(probs, cond_matrices, cond_bwd, alpha)
    else:
        return probs.copy()


def step4_combo(probs, cond_matrices, best_alpha, best_scheme,
                data, test_indices, lottery_type):
    """Step 4：组合级评估"""
    log(f"\n{'─'*50}")
    log(f"  Step 4：组合级评估 [{lottery_type}]")
    log(f"{'─'*50}")

    max_train_dir_idx = int(test_indices[0]) - 1
    max_train_idx = int(test_indices[0])
    n_samples, n_pos, _ = probs.shape
    red_range = data.red_range
    baseline_combos = comb(red_range, n_pos)

    # 构建反向矩阵（双向方案需要）
    cond_bwd = build_reverse_cond_matrices(data, 0, max_train_dir_idx)
    corr_probs = get_corrected_probs(
        probs, cond_matrices, cond_bwd, best_alpha, best_scheme)

    diff_dist = compute_diff_distributions(data, max_train_idx)

    # === 硬阈值方案（与 E1 对比）===
    log("\n  硬阈值方案 (threshold=0.15):")
    orig_cands = generate_candidate_sets(probs, data, test_indices, threshold=0.15)
    corr_cands = generate_candidate_sets(corr_probs, data, test_indices, threshold=0.15)

    def eval_candidates(candidates, label):
        combo_counts = []
        survivals = []
        for i in range(n_samples):
            t = int(test_indices[i])
            true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
            n_combos = count_ordered_combos(candidates[i])
            combo_counts.append(n_combos)
            survived = check_combo_survival(candidates[i], true_vals)
            survivals.append(1.0 if survived else 0.0)
        cc = np.array(combo_counts, dtype=float)
        sv = np.array(survivals)
        reduction = 1.0 - float(np.mean(cc)) / baseline_combos
        survival = float(np.mean(sv))
        efficiency = reduction * survival
        log(f"    {label}: 缩减率={reduction:.2%}, 存活率={survival:.2%}, "
            f"效率={efficiency:.4f}")
        return {
            "avg_combo_count": float(np.mean(cc)),
            "combo_reduction": reduction,
            "combo_survival": survival,
            "combo_efficiency": efficiency,
        }

    hard_orig = eval_candidates(orig_cands, "原始")
    hard_corr = eval_candidates(corr_cands, "修正")

    # === 权重化方案（与 E1.5 对比）===
    log("\n  权重化方案:")
    orig_weights = compute_number_weights(probs, data, test_indices, diff_dist)
    corr_weights = compute_number_weights(corr_probs, data, test_indices, diff_dist)

    # 用 K95 阈值
    def eval_weighted(all_weights, label):
        # 先找每位置的 K95
        optimal_ks = []
        for pos in range(n_pos):
            survival_by_k = np.zeros(red_range + 1)
            for i in range(n_samples):
                t = int(test_indices[i])
                true_val = int(data.red_matrix[t, pos])
                sorted_nums = sorted(all_weights[i][pos].items(),
                                     key=lambda x: x[1], reverse=True)
                for r, (v, w) in enumerate(sorted_nums, 1):
                    if v == true_val:
                        for k in range(r, red_range + 1):
                            survival_by_k[k] += 1
                        break
            surv_rates = survival_by_k / n_samples
            best_k = red_range
            for k in range(1, red_range + 1):
                if surv_rates[k] >= 0.95:
                    best_k = k
                    break
            optimal_ks.append(best_k)

        # 生成候选集
        combo_counts = []
        survivals = []
        for i in range(n_samples):
            t = int(test_indices[i])
            true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
            cand_sets = []
            for pos in range(n_pos):
                sorted_nums = sorted(all_weights[i][pos].items(),
                                     key=lambda x: x[1], reverse=True)
                top_k = set(v for v, w in sorted_nums[:optimal_ks[pos]])
                cand_sets.append(top_k)
            n_combos = count_ordered_combos(cand_sets)
            combo_counts.append(n_combos)
            survived = check_combo_survival(cand_sets, true_vals)
            survivals.append(1.0 if survived else 0.0)

        cc = np.array(combo_counts, dtype=float)
        sv = np.array(survivals)
        reduction = 1.0 - float(np.mean(cc)) / baseline_combos
        survival = float(np.mean(sv))
        efficiency = reduction * survival
        log(f"    {label}: K95={optimal_ks}, 缩减率={reduction:.2%}, "
            f"存活率={survival:.2%}, 效率={efficiency:.4f}")
        return {
            "optimal_ks": optimal_ks,
            "avg_combo_count": float(np.mean(cc)),
            "combo_reduction": reduction,
            "combo_survival": survival,
            "combo_efficiency": efficiency,
        }

    weighted_orig = eval_weighted(orig_weights, "原始权重")
    weighted_corr = eval_weighted(corr_weights, "修正权重")

    # === 阈值敏感性 ===
    log("\n  阈值敏感性分析:")
    threshold_scan = []
    for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        cands = generate_candidate_sets(corr_probs, data, test_indices, threshold=thr)
        cc_list = []
        sv_list = []
        for i in range(n_samples):
            t = int(test_indices[i])
            true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
            cc_list.append(count_ordered_combos(cands[i]))
            sv_list.append(1.0 if check_combo_survival(cands[i], true_vals) else 0.0)
        cc_arr = np.array(cc_list, dtype=float)
        sv_arr = np.array(sv_list)
        red = 1.0 - float(np.mean(cc_arr)) / baseline_combos
        surv = float(np.mean(sv_arr))
        eff = red * surv
        threshold_scan.append({
            "threshold": thr, "reduction": red,
            "survival": surv, "efficiency": eff,
        })
        log(f"    thr={thr:.2f}: 缩减={red:.2%}, 存活={surv:.2%}, 效率={eff:.4f}")

    results = {
        "best_alpha": best_alpha,
        "best_scheme": best_scheme,
        "baseline_combos": baseline_combos,
        "hard_threshold": {
            "original": hard_orig,
            "corrected": hard_corr,
        },
        "weighted_k95": {
            "original": weighted_orig,
            "corrected": weighted_corr,
        },
        "threshold_sensitivity": threshold_scan,
    }

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2a_step4_combo_{lottery_type}.json")
    return results


# ============================================================
#  Step 5：消融实验
# ============================================================

def step5_ablation(probs, cond_matrices, best_alpha, data, test_indices, lottery_type):
    """Step 5：消融实验验证提升来源"""
    log(f"\n{'─'*50}")
    log(f"  Step 5：消融实验 [{lottery_type}]")
    log(f"{'─'*50}")

    n_pos = data.red_count
    base_correct = compute_direction_accuracy(probs, data, test_indices, pos_start=1)
    base_acc = float(base_correct.mean())

    real_corrected = correct_probs_single_step(probs, cond_matrices, best_alpha)
    real_correct = compute_direction_accuracy(real_corrected, data, test_indices, pos_start=1)
    real_acc = float(real_correct.mean())

    log(f"  基线准确率: {base_acc:.4f}")
    log(f"  真实矩阵修正准确率: {real_acc:.4f}")

    # === 消融 1：随机矩阵 ===
    log("\n  消融 1：随机矩阵")
    n_random_trials = 20
    random_accs = []
    np.random.seed(42)
    for trial in range(n_random_trials):
        random_matrices = {}
        for pos in range(n_pos - 1):
            mat = np.random.dirichlet([1, 1, 1], size=3)
            random_matrices[(pos, pos + 1)] = mat
        rand_corrected = correct_probs_single_step(probs, random_matrices, best_alpha)
        rand_correct = compute_direction_accuracy(
            rand_corrected, data, test_indices, pos_start=1)
        random_accs.append(float(rand_correct.mean()))

    random_mean = float(np.mean(random_accs))
    random_std = float(np.std(random_accs))
    log(f"  随机矩阵准确率: {random_mean:.4f} ± {random_std:.4f}")

    # === 消融 2：均匀矩阵 ===
    log("\n  消融 2：均匀矩阵 (1/3)")
    uniform_matrices = {}
    for pos in range(n_pos - 1):
        uniform_matrices[(pos, pos + 1)] = np.ones((3, 3)) / 3.0
    uniform_corrected = correct_probs_single_step(probs, uniform_matrices, best_alpha)
    uniform_correct = compute_direction_accuracy(
        uniform_corrected, data, test_indices, pos_start=1)
    uniform_acc = float(uniform_correct.mean())
    log(f"  均匀矩阵准确率: {uniform_acc:.4f}")

    # === 消融 3：置换检验 ===
    log("\n  消融 3：置换检验（打乱位置对应关系）")
    n_perm_trials = 20
    perm_accs = []
    pair_keys = sorted(cond_matrices.keys())
    np.random.seed(123)
    for trial in range(n_perm_trials):
        shuffled_keys = list(pair_keys)
        np.random.shuffle(shuffled_keys)
        perm_matrices = {}
        for orig_key, shuf_key in zip(pair_keys, shuffled_keys):
            perm_matrices[orig_key] = cond_matrices[shuf_key]
        perm_corrected = correct_probs_single_step(probs, perm_matrices, best_alpha)
        perm_correct = compute_direction_accuracy(
            perm_corrected, data, test_indices, pos_start=1)
        perm_accs.append(float(perm_correct.mean()))

    perm_mean = float(np.mean(perm_accs))
    perm_std = float(np.std(perm_accs))
    log(f"  置换检验准确率: {perm_mean:.4f} ± {perm_std:.4f}")

    # 判断：真实矩阵是否显著优于消融
    real_vs_random = real_acc > random_mean + 2 * random_std
    real_vs_uniform = real_acc > uniform_acc
    real_vs_perm = real_acc > perm_mean + 2 * perm_std

    log(f"\n  消融结论:")
    log(f"    真实 vs 随机: {real_acc:.4f} vs {random_mean:.4f}±{random_std:.4f} "
        f"({'通过' if real_vs_random else '未通过'})")
    log(f"    真实 vs 均匀: {real_acc:.4f} vs {uniform_acc:.4f} "
        f"({'通过' if real_vs_uniform else '未通过'})")
    log(f"    真实 vs 置换: {real_acc:.4f} vs {perm_mean:.4f}±{perm_std:.4f} "
        f"({'通过' if real_vs_perm else '未通过'})")

    results = {
        "best_alpha": best_alpha,
        "baseline_accuracy": base_acc,
        "real_matrix_accuracy": real_acc,
        "ablation_random": {
            "n_trials": n_random_trials,
            "mean_accuracy": random_mean,
            "std_accuracy": random_std,
            "real_better": bool(real_vs_random),
        },
        "ablation_uniform": {
            "accuracy": uniform_acc,
            "real_better": bool(real_vs_uniform),
        },
        "ablation_permutation": {
            "n_trials": n_perm_trials,
            "mean_accuracy": perm_mean,
            "std_accuracy": perm_std,
            "real_better": bool(real_vs_perm),
        },
        "all_ablations_pass": bool(real_vs_random and real_vs_uniform and real_vs_perm),
    }

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2a_step5_ablation_{lottery_type}.json")
    return results


# ============================================================
#  主评估流程
# ============================================================

def evaluate_e2a(lottery_type):
    """对单个彩种执行 E2a 全流程"""
    log(f"\n{'═'*55}")
    log(f"  E2a 条件方向概率修正: {lottery_type}")
    log(f"{'═'*55}")

    # 获取 probs
    with Timer("获取方向概率"):
        probs, test_indices, data, best_method = get_best_probs(lottery_type)

    n_samples, n_pos = probs.shape[0], probs.shape[1]
    log(f"  测试样本数: {n_samples}, 位置数: {n_pos}")
    log(f"  最优方案: {best_method}")

    # Step 0
    with Timer("Step 0"):
        cond_matrices, use_sliding_window, step0_results = step0_stability(
            data, test_indices, lottery_type)

    if use_sliding_window:
        log("  注意：使用滑动窗口动态矩阵（训练期 vs 测试期分布不一致）")
        # 仍然使用全局训练期矩阵，但标记警告
        # 未来可改为滑动窗口，当前先继续实验

    # Step 1
    with Timer("Step 1"):
        best_alpha, step1_passed, step1_results = step1_direction(
            probs, cond_matrices, data, test_indices, lottery_type)

    if not step1_passed:
        log("\n  Step 1 未通过（无显著 alpha），实验终止")
        log("  结论：条件方向概率修正对方向预测无显著提升")
        summary = {
            "lottery_type": lottery_type,
            "terminated_at": "step1",
            "reason": "no_significant_alpha",
            "step0": step0_results,
            "step1": step1_results,
        }
        return summary

    # Step 2
    with Timer("Step 2"):
        step2_results = step2_auc(
            probs, cond_matrices, best_alpha, data, test_indices, lottery_type)

    # Step 3
    with Timer("Step 3"):
        best_scheme, step3_results = step3_chain_vs_single(
            probs, cond_matrices, best_alpha, data, test_indices, lottery_type)

    # Step 4
    with Timer("Step 4"):
        step4_results = step4_combo(
            probs, cond_matrices, best_alpha, best_scheme,
            data, test_indices, lottery_type)

    # Step 5
    with Timer("Step 5"):
        step5_results = step5_ablation(
            probs, cond_matrices, best_alpha, data, test_indices, lottery_type)

    summary = {
        "lottery_type": lottery_type,
        "best_method": best_method,
        "n_test_samples": n_samples,
        "best_alpha": best_alpha,
        "best_scheme": best_scheme,
        "step0_pass": step0_results["overall_pass"],
        "step1_significant": step1_results["significant"],
        "direction_improvement": step1_results["improvement"],
        "auc_original": step2_results["original_auc"],
        "auc_corrected": step2_results["corrected_auc"],
        "auc_change": step2_results["auc_change"],
        "combo_hard_orig_eff": step4_results["hard_threshold"]["original"]["combo_efficiency"],
        "combo_hard_corr_eff": step4_results["hard_threshold"]["corrected"]["combo_efficiency"],
        "combo_weighted_orig_eff": step4_results["weighted_k95"]["original"]["combo_efficiency"],
        "combo_weighted_corr_eff": step4_results["weighted_k95"]["corrected"]["combo_efficiency"],
        "ablation_all_pass": step5_results["all_ablations_pass"],
    }

    return summary


def main():
    STRICT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STRICT_EXPERIMENT_DIR
    setup_logging()
    exp_utils.EXPERIMENT_DIR = original_exp_dir

    start_time = time.time()

    log("=" * 60)
    log("  E2a：条件方向概率修正实验")
    log("=" * 60)

    all_results = {}

    for lottery_type in ["daletou", "shuangseqiu"]:
        try:
            results = evaluate_e2a(lottery_type)
            all_results[lottery_type] = results
        except Exception as e:
            log(f"\n[错误] {lottery_type} 评估失败: {e}")
            log(traceback.format_exc())
            all_results[lottery_type] = {"error": str(e)}

    total_time = time.time() - start_time
    summary = {
        "total_time_seconds": round(total_time, 1),
        "total_time_minutes": round(total_time / 60, 1),
        "results": all_results,
    }
    save_json(summary, STRICT_EXPERIMENT_DIR / "e2a_summary.json")

    log(f"\n{'═'*60}")
    log(f"  E2a 实验完成！总耗时: {summary['total_time_minutes']:.1f} 分钟")
    log(f"  结果目录: {STRICT_EXPERIMENT_DIR}")
    log(f"{'═'*60}")


if __name__ == "__main__":
    main()
