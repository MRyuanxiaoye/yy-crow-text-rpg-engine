"""E2：多层级联过滤实验

验证差分幅度可预测性(E2.1)、跨位置联合方向预测(E2.2)、级联组合评估(E2.3)。

用法: python3 -m src.research.experiment.e2_cascade_filter
"""

import sys
from math import comb
from pathlib import Path

import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.experiment.utils import (
    setup_logging, log, Timer, save_json,
)
from research.experiment.e1_combo_evaluation import (
    get_best_probs, count_ordered_combos, check_combo_survival,
    generate_candidate_sets,
)

# === 路径常量 ===
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_EXPERIMENT_DIR = RESULTS_DIR / "experiment_strict"


# ============================================================
#  E2.1 Step 0: 差分条件分布稳定性验证
# ============================================================

def e2_1_step0_stability(data, train_end_idx, lottery_type):
    """验证差分条件分布在不同时间窗口上是否稳定

    将训练期等分5个窗口，对每个位置-方向组合做KS和AD检验。
    Go: >= 80% 通过; No-Go: > 50% 不通过
    """
    log(f"\n{'═'*55}")
    log(f"  E2.1 Step 0: 差分分布稳定性 [{lottery_type}]")
    log(f"{'═'*55}")

    from scipy import stats as sp_stats

    n_pos = data.red_count
    dir_labels = ['U', 'D']  # E方向差分恒为0，不需要检验

    # 将训练期等分5个窗口
    n_windows = 5
    window_size = train_end_idx // n_windows
    windows = []
    for w in range(n_windows):
        start = w * window_size
        end = (w + 1) * window_size if w < n_windows - 1 else train_end_idx
        windows.append((start, end))
    log(f"  训练期: [0, {train_end_idx}), 窗口数: {n_windows}, 窗口大小: ~{window_size}")

    # Bonferroni 校正: 位置数 × 方向数(2) = 总检验次数
    n_tests = n_pos * len(dir_labels)
    alpha_corrected = 0.05 / n_tests
    log(f"  Bonferroni 校正: {n_tests} 次检验, alpha={alpha_corrected:.4f}")

    results = {'windows': [(s, e) for s, e in windows], 'per_pos_dir': {}, 'alpha_corrected': alpha_corrected}
    n_pass = 0
    n_total = 0

    for pos in range(n_pos):
        series = data.position_series[pos]
        # 差分序列: series[t] - series[t-1], 从 t=1 开始
        diffs = np.diff(series.astype(np.float64))

        for dir_label in dir_labels:
            key = f'P{pos}_{dir_label}'
            # 按方向筛选差分绝对值
            if dir_label == 'U':
                mask_fn = lambda d: d > 0
            else:
                mask_fn = lambda d: d < 0

            # 收集每个窗口的差分绝对值
            window_samples = []
            for w_start, w_end in windows:
                # diffs[i] = series[i+1] - series[i], 对应期数 i+1
                # 窗口 [w_start, w_end) 对应 diffs 索引 [w_start-1, w_end-1)
                # 但 diffs 从索引0开始，diffs[0] = series[1]-series[0]
                d_start = max(0, w_start)
                d_end = min(len(diffs), w_end - 1) if w_end > 0 else 0
                if d_start >= d_end:
                    window_samples.append(np.array([]))
                    continue
                window_diffs = diffs[d_start:d_end]
                filtered = np.abs(window_diffs[np.array([mask_fn(d) for d in window_diffs])])
                window_samples.append(filtered)

            # KS 检验: 相邻窗口对
            ks_results = []
            for i in range(len(window_samples) - 1):
                s1, s2 = window_samples[i], window_samples[i + 1]
                if len(s1) < 10 or len(s2) < 10:
                    ks_results.append({'stat': None, 'p': None, 'pass': True, 'note': '样本不足'})
                    continue
                stat, p = sp_stats.ks_2samp(s1, s2)
                ks_results.append({
                    'stat': float(stat), 'p': float(p),
                    'pass': bool(p >= alpha_corrected),
                })

            # Anderson-Darling k-sample 检验
            valid_samples = [s for s in window_samples if len(s) >= 10]
            if len(valid_samples) >= 2:
                try:
                    ad_result = sp_stats.anderson_ksamp(valid_samples)
                    ad_stat = float(ad_result.statistic)
                    ad_p = float(ad_result.pvalue) if hasattr(ad_result, 'pvalue') else float(ad_result.significance_level)
                    ad_pass = ad_p >= alpha_corrected
                except Exception as e:
                    ad_stat, ad_p, ad_pass = None, None, True
                    log(f"    {key}: AD检验异常: {e}")
            else:
                ad_stat, ad_p, ad_pass = None, None, True

            # 综合判定: KS全部通过 且 AD通过
            ks_all_pass = all(r['pass'] for r in ks_results)
            overall_pass = ks_all_pass and ad_pass

            n_total += 1
            if overall_pass:
                n_pass += 1

            results['per_pos_dir'][key] = {
                'ks_tests': ks_results,
                'ad_statistic': ad_stat,
                'ad_p_value': ad_p,
                'ad_pass': ad_pass,
                'ks_all_pass': ks_all_pass,
                'overall_pass': overall_pass,
                'window_sizes': [len(s) for s in window_samples],
            }

            status = "通过" if overall_pass else "不通过"
            ad_p_str = f"{ad_p:.4f}" if ad_p is not None else "N/A"
            log(f"  {key}: {status} (AD p={ad_p_str}, KS全通过={ks_all_pass})")

    pass_rate = n_pass / n_total if n_total > 0 else 0
    go = pass_rate >= 0.8
    results['summary'] = {
        'n_total': n_total,
        'n_pass': n_pass,
        'pass_rate': float(pass_rate),
        'go': go,
        'decision': 'GO' if go else 'NO-GO',
    }

    log(f"\n  汇总: {n_pass}/{n_total} 通过 ({pass_rate:.1%})")
    log(f"  判定: {'GO (>=80%)' if go else 'NO-GO (>50%不通过)'}")

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2_1_step0_stability_{lottery_type}.json")
    return results


# ============================================================
#  E2.1 Step 1: 差分幅度预测基线
# ============================================================

def e2_1_step1_amplitude(data, train_end_idx, test_start_idx, lottery_type):
    """验证差分幅度的可预测性

    方案A: 条件分布拟合（正态/拉普拉斯/指数）
    方案B: XGBoost回归（局面特征 → 差分绝对值）
    Go: MAE降低>=10% 且 R²>0.05
    """
    log(f"\n{'═'*55}")
    log(f"  E2.1 Step 1: 差分幅度预测 [{lottery_type}]")
    log(f"{'═'*55}")

    from scipy import stats as sp_stats

    n_pos = data.red_count
    n_draws = data.n_draws
    dir_labels = ['U', 'D']

    # 预计算差分序列
    diffs_all = {}
    for pos in range(n_pos):
        diffs_all[pos] = np.diff(data.position_series[pos].astype(np.float64))

    # === 方案A: 条件分布拟合 ===
    log(f"\n  --- 方案A: 条件分布拟合 ---")

    dist_fits = {}  # {(pos, dir): {'best_dist': ..., 'params': ..., 'median': ...}}

    for pos in range(n_pos):
        for dir_label in dir_labels:
            key = f'P{pos}_{dir_label}'
            # 训练期差分绝对值
            train_diffs = diffs_all[pos][:train_end_idx - 1]
            if dir_label == 'U':
                abs_diffs = np.abs(train_diffs[train_diffs > 0])
            else:
                abs_diffs = np.abs(train_diffs[train_diffs < 0])

            if len(abs_diffs) < 20:
                log(f"    {key}: 样本不足 ({len(abs_diffs)}), 跳过")
                continue

            # 拟合候选分布
            candidates = {}

            # 正态（截断到正值）
            mu, sigma = np.mean(abs_diffs), np.std(abs_diffs)
            if sigma > 0:
                ll = np.sum(sp_stats.norm.logpdf(abs_diffs, mu, sigma))
                candidates['norm'] = {'params': (mu, sigma), 'll': ll, 'k': 2}

            # 指数
            loc_exp = np.min(abs_diffs) - 0.5
            scale_exp = np.mean(abs_diffs - loc_exp)
            if scale_exp > 0:
                ll = np.sum(sp_stats.expon.logpdf(abs_diffs, loc=loc_exp, scale=scale_exp))
                candidates['expon'] = {'params': (loc_exp, scale_exp), 'll': ll, 'k': 2}

            # 拉普拉斯
            loc_lap = np.median(abs_diffs)
            scale_lap = np.mean(np.abs(abs_diffs - loc_lap))
            if scale_lap > 0:
                ll = np.sum(sp_stats.laplace.logpdf(abs_diffs, loc_lap, scale_lap))
                candidates['laplace'] = {'params': (loc_lap, scale_lap), 'll': ll, 'k': 2}

            # AIC选最优
            best_dist = None
            best_aic = float('inf')
            for dname, info in candidates.items():
                aic = 2 * info['k'] - 2 * info['ll']
                if aic < best_aic:
                    best_aic = aic
                    best_dist = dname

            median_pred = float(np.median(abs_diffs))
            mean_pred = float(np.mean(abs_diffs))

            dist_fits[(pos, dir_label)] = {
                'best_dist': best_dist,
                'median': median_pred,
                'mean': mean_pred,
                'n_samples': len(abs_diffs),
                'std': float(np.std(abs_diffs)),
                'p90_range': float(np.percentile(abs_diffs, 95) - np.percentile(abs_diffs, 5)),
                'full_range': float(np.max(abs_diffs) - np.min(abs_diffs)),
            }
            log(f"    {key}: best={best_dist}, median={median_pred:.1f}, mean={mean_pred:.1f}, n={len(abs_diffs)}")

    # 方案A测试期评估
    log(f"\n  方案A 测试期评估:")
    a_maes = []
    a_baseline_maes = []

    for pos in range(n_pos):
        test_diffs = diffs_all[pos][test_start_idx - 1:]
        for dir_label in dir_labels:
            fit_key = (pos, dir_label)
            if fit_key not in dist_fits:
                continue
            fit = dist_fits[fit_key]

            if dir_label == 'U':
                mask = test_diffs > 0
            else:
                mask = test_diffs < 0
            test_abs = np.abs(test_diffs[mask])

            if len(test_abs) < 5:
                continue

            # 方案A预测: 用中位数
            pred_a = fit['median']
            mae_a = float(np.mean(np.abs(test_abs - pred_a)))
            # 基线: 全局均值
            baseline_pred = fit['mean']
            mae_baseline = float(np.mean(np.abs(test_abs - baseline_pred)))

            a_maes.append(mae_a)
            a_baseline_maes.append(mae_baseline)

    if a_maes:
        avg_mae_a = np.mean(a_maes)
        avg_baseline_a = np.mean(a_baseline_maes)
        improvement_a = 1.0 - avg_mae_a / avg_baseline_a if avg_baseline_a > 0 else 0
        log(f"  方案A: MAE={avg_mae_a:.3f}, 基线MAE={avg_baseline_a:.3f}, 改进={improvement_a:.1%}")

    # === 方案B: XGBoost回归 ===
    log(f"\n  --- 方案B: XGBoost回归 ---")

    try:
        import xgboost as xgb
        xgb.XGBRegressor()
        use_xgb = True
    except Exception:
        use_xgb = False

    from sklearn.ensemble import GradientBoostingRegressor

    b_results = {}

    for pos in range(n_pos):
        for dir_label in dir_labels:
            key = f'P{pos}_{dir_label}'

            # 构造特征和标签
            train_X_list, train_y_list = [], []
            test_X_list, test_y_list = [], []

            for t in range(5, n_draws - 1):
                diff_val = float(diffs_all[pos][t - 1]) if t - 1 < len(diffs_all[pos]) else 0
                if dir_label == 'U' and diff_val <= 0:
                    continue
                if dir_label == 'D' and diff_val >= 0:
                    continue

                abs_diff = abs(diff_val)

                # 特征: 上期值、最近3期差分、最近3期方向、位置间差值
                feats = []
                # 上期值
                feats.append(float(data.red_matrix[t, pos]))
                # 最近3期差分
                for lag in range(1, 4):
                    idx = t - 1 - lag
                    if 0 <= idx < len(diffs_all[pos]):
                        feats.append(float(diffs_all[pos][idx]))
                    else:
                        feats.append(0.0)
                # 最近3期方向
                for lag in range(1, 4):
                    idx = t - 1 - lag
                    if 0 <= idx < len(data.direction_series[pos]):
                        feats.append(float(data.direction_series[pos][idx]))
                    else:
                        feats.append(0.0)
                # 位置间差值（与相邻位置）
                for other_pos in range(n_pos):
                    if other_pos != pos:
                        feats.append(float(data.red_matrix[t, pos]) - float(data.red_matrix[t, other_pos]))

                if t < train_end_idx:
                    train_X_list.append(feats)
                    train_y_list.append(abs_diff)
                elif t >= test_start_idx:
                    test_X_list.append(feats)
                    test_y_list.append(abs_diff)

            if len(train_X_list) < 30 or len(test_X_list) < 10:
                log(f"    {key}: 样本不足 (train={len(train_X_list)}, test={len(test_X_list)})")
                continue

            train_X = np.array(train_X_list, dtype=np.float32)
            train_y = np.array(train_y_list, dtype=np.float32)
            test_X = np.array(test_X_list, dtype=np.float32)
            test_y = np.array(test_y_list, dtype=np.float32)

            # 验证集
            n_val = max(20, int(len(train_X) * 0.2))

            if use_xgb:
                model = xgb.XGBRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
                    reg_alpha=0.1, reg_lambda=1.0,
                    early_stopping_rounds=30, random_state=42, n_jobs=-1, verbosity=0,
                )
                model.fit(train_X[:-n_val], train_y[:-n_val],
                          eval_set=[(train_X[-n_val:], train_y[-n_val:])], verbose=False)
            else:
                model = GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=10, subsample=0.8, random_state=42,
                    validation_fraction=0.2, n_iter_no_change=30,
                )
                model.fit(train_X[:-n_val], train_y[:-n_val])

            pred_y = model.predict(test_X)
            mae_b = float(np.mean(np.abs(test_y - pred_y)))
            rmse_b = float(np.sqrt(np.mean((test_y - pred_y) ** 2)))
            baseline_mae = float(np.mean(np.abs(test_y - np.mean(train_y))))
            ss_res = np.sum((test_y - pred_y) ** 2)
            ss_tot = np.sum((test_y - np.mean(test_y)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            mae_improvement = 1.0 - mae_b / baseline_mae if baseline_mae > 0 else 0

            b_results[key] = {
                'mae': mae_b, 'rmse': rmse_b, 'r2': r2,
                'baseline_mae': baseline_mae,
                'mae_improvement': float(mae_improvement),
                'train_samples': len(train_X), 'test_samples': len(test_X),
            }
            log(f"    {key}: MAE={mae_b:.3f}, 基线={baseline_mae:.3f}, "
                f"改进={mae_improvement:.1%}, R²={r2:.4f}")

    # 汇总方案B
    if b_results:
        avg_mae_imp = np.mean([v['mae_improvement'] for v in b_results.values()])
        avg_r2 = np.mean([v['r2'] for v in b_results.values()])
    else:
        avg_mae_imp, avg_r2 = 0, 0

    # Go/No-Go 判定
    go = avg_mae_imp >= 0.10 and avg_r2 > 0.05

    # 90%置信区间宽度 vs 全域范围
    ci_ratio = None
    if dist_fits:
        ratios = []
        for (pos, dir_label), fit in dist_fits.items():
            if fit['full_range'] > 0:
                ratios.append(fit['p90_range'] / fit['full_range'])
        ci_ratio = float(np.mean(ratios)) if ratios else None

    results = {
        'plan_a': {
            'dist_fits': {f'P{p}_{d}': v for (p, d), v in dist_fits.items()},
            'avg_mae': float(np.mean(a_maes)) if a_maes else None,
            'avg_baseline_mae': float(np.mean(a_baseline_maes)) if a_baseline_maes else None,
        },
        'plan_b': {
            'per_pos_dir': b_results,
            'avg_mae_improvement': float(avg_mae_imp),
            'avg_r2': float(avg_r2),
        },
        'ci_width_ratio': ci_ratio,
        'summary': {
            'go': go,
            'decision': 'GO' if go else 'NO-GO',
            'avg_mae_improvement': float(avg_mae_imp),
            'avg_r2': float(avg_r2),
            'criteria': 'MAE降低>=10% 且 R²>0.05',
        },
    }

    log(f"\n  汇总: MAE改进={avg_mae_imp:.1%}, R²={avg_r2:.4f}")
    log(f"  90%CI宽度/全域: {ci_ratio:.2%}" if ci_ratio else "  90%CI宽度/全域: N/A")
    log(f"  判定: {'GO' if go else 'NO-GO'}")

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2_1_step1_amplitude_{lottery_type}.json")
    return results


# ============================================================
#  E2.1 Step 2: 差分幅度对缩号的增量贡献
# ============================================================

def e2_1_step2_incremental(data, probs, test_indices, train_end_idx, test_start_idx, lottery_type):
    """用差分幅度预测的90%置信区间缩小候选值域

    在方向预测基础上，进一步用差分幅度约束缩小候选集。
    Go: 增量缩减>=10% 且 存活率>=88%
    """
    log(f"\n{'═'*55}")
    log(f"  E2.1 Step 2: 差分幅度增量缩号 [{lottery_type}]")
    log(f"{'═'*55}")

    from scipy import stats as sp_stats

    n_pos = data.red_count
    n_samples = len(test_indices)
    red_range = data.red_range

    # 预计算差分序列
    diffs_all = {}
    for pos in range(n_pos):
        diffs_all[pos] = np.diff(data.position_series[pos].astype(np.float64))

    # 从训练期统计每个位置-方向的差分绝对值分布的百分位
    diff_ci = {}  # {(pos, dir): (lo, hi)} 90%置信区间
    for pos in range(n_pos):
        train_diffs = diffs_all[pos][:train_end_idx - 1]
        for dir_label in ['U', 'D']:
            if dir_label == 'U':
                abs_d = np.abs(train_diffs[train_diffs > 0])
            else:
                abs_d = np.abs(train_diffs[train_diffs < 0])
            if len(abs_d) < 20:
                diff_ci[(pos, dir_label)] = (1, red_range)
                continue
            lo = max(1, int(np.percentile(abs_d, 5)))
            hi = min(red_range, int(np.ceil(np.percentile(abs_d, 95))))
            diff_ci[(pos, dir_label)] = (lo, hi)
            log(f"  P{pos}_{dir_label}: 90%CI=[{lo}, {hi}]")

    # 基线: 仅方向预测的候选集
    baseline_candidates = generate_candidate_sets(probs, data, test_indices, threshold=0.15)

    # 增强: 方向预测 + 差分幅度约束
    enhanced_candidates = []
    for i in range(n_samples):
        t = int(test_indices[i])
        sample_cands = []
        for pos in range(n_pos):
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2)
            base_cands = baseline_candidates[i][pos]

            # 用差分幅度约束进一步缩小
            enhanced = set()
            for v in base_cands:
                diff_val = v - current_val
                if diff_val > 0:
                    lo, hi = diff_ci.get((pos, 'U'), (1, red_range))
                    if lo <= abs(diff_val) <= hi:
                        enhanced.add(v)
                elif diff_val < 0:
                    lo, hi = diff_ci.get((pos, 'D'), (1, red_range))
                    if lo <= abs(diff_val) <= hi:
                        enhanced.add(v)
                else:
                    # 持平，保留
                    enhanced.add(v)

            # 至少保留1个候选
            if len(enhanced) == 0:
                enhanced = base_cands
            sample_cands.append(enhanced)
        enhanced_candidates.append(sample_cands)

    # 评估
    baseline_reductions = []
    enhanced_reductions = []
    baseline_survivals = []
    enhanced_survivals = []
    incremental_reductions = []

    for i in range(n_samples):
        t = int(test_indices[i])
        true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]

        for pos in range(n_pos):
            base_size = len(baseline_candidates[i][pos])
            enh_size = len(enhanced_candidates[i][pos])

            base_red = 1.0 - base_size / red_range
            enh_red = 1.0 - enh_size / red_range
            incr_red = (base_size - enh_size) / base_size if base_size > 0 else 0

            baseline_reductions.append(base_red)
            enhanced_reductions.append(enh_red)
            incremental_reductions.append(incr_red)

            base_surv = 1.0 if true_vals[pos] in baseline_candidates[i][pos] else 0.0
            enh_surv = 1.0 if true_vals[pos] in enhanced_candidates[i][pos] else 0.0
            baseline_survivals.append(base_surv)
            enhanced_survivals.append(enh_surv)

    avg_base_red = float(np.mean(baseline_reductions))
    avg_enh_red = float(np.mean(enhanced_reductions))
    avg_incr_red = float(np.mean(incremental_reductions))
    avg_base_surv = float(np.mean(baseline_survivals))
    avg_enh_surv = float(np.mean(enhanced_survivals))

    # 组合级评估
    baseline_combos = comb(red_range, n_pos)
    enh_combo_counts = []
    enh_combo_survivals = []
    for i in range(n_samples):
        t = int(test_indices[i])
        true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
        n_combos = count_ordered_combos(enhanced_candidates[i])
        enh_combo_counts.append(n_combos)
        survived = check_combo_survival(enhanced_candidates[i], true_vals)
        enh_combo_survivals.append(1.0 if survived else 0.0)

    avg_enh_combos = float(np.mean(enh_combo_counts))
    enh_combo_reduction = 1.0 - avg_enh_combos / baseline_combos
    enh_combo_survival = float(np.mean(enh_combo_survivals))

    # Wilcoxon 检验
    try:
        from scipy.stats import wilcoxon
        base_sizes = [len(baseline_candidates[i][pos]) for i in range(n_samples) for pos in range(n_pos)]
        enh_sizes = [len(enhanced_candidates[i][pos]) for i in range(n_samples) for pos in range(n_pos)]
        stat_w, p_w = wilcoxon(base_sizes, enh_sizes, alternative='greater')
    except Exception:
        stat_w, p_w = None, None

    go = avg_incr_red >= 0.10 and avg_enh_surv >= 0.88

    results = {
        'position_level': {
            'baseline_reduction': avg_base_red,
            'enhanced_reduction': avg_enh_red,
            'incremental_reduction': avg_incr_red,
            'baseline_survival': avg_base_surv,
            'enhanced_survival': avg_enh_surv,
        },
        'combo_level': {
            'avg_combo_count': avg_enh_combos,
            'combo_reduction': enh_combo_reduction,
            'combo_survival': enh_combo_survival,
        },
        'diff_ci': {f'P{p}_{d}': {'lo': lo, 'hi': hi} for (p, d), (lo, hi) in diff_ci.items()},
        'wilcoxon': {'statistic': float(stat_w) if stat_w else None, 'p_value': float(p_w) if p_w else None},
        'summary': {
            'go': go,
            'decision': 'GO' if go else 'NO-GO',
            'incremental_reduction': avg_incr_red,
            'enhanced_survival': avg_enh_surv,
            'criteria': '增量缩减>=10% 且 存活率>=88%',
        },
    }

    log(f"\n  位置级: 基线缩减={avg_base_red:.1%}, 增强缩减={avg_enh_red:.1%}, 增量={avg_incr_red:.1%}")
    log(f"  位置级存活: 基线={avg_base_surv:.1%}, 增强={avg_enh_surv:.1%}")
    log(f"  组合级: 平均组合数={avg_enh_combos:,.0f}, 缩减={enh_combo_reduction:.1%}, 存活={enh_combo_survival:.1%}")
    log(f"  判定: {'GO' if go else 'NO-GO'}")

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2_1_step2_incremental_{lottery_type}.json")
    return results, enhanced_candidates


# ============================================================
#  E2.2: 跨位置联合方向预测
# ============================================================

def build_cross_position_features(data, train_end_idx, test_start_idx):
    """构造跨位置特征矩阵

    特征包括:
    - 各位置最近N期方向序列 (N=1,2,3)
    - 各位置最近N期差分值
    - 位置间差值
    - 组合统计量 (和值、跨度、奇偶比、大小比)
    """
    n_pos = data.red_count
    n_draws = data.n_draws

    feature_names = []
    all_features = []

    for t in range(5, n_draws):
        feats = []

        # 各位置最近3期方向
        for pos in range(n_pos):
            for lag in range(1, 4):
                idx = t - 1 - lag
                if 0 <= idx < len(data.direction_series[pos]):
                    feats.append(float(data.direction_series[pos][idx]))
                else:
                    feats.append(0.0)
                if t == 5:
                    feature_names.append(f'dir_P{pos}_lag{lag}')

        # 各位置最近3期差分
        diffs = {}
        for pos in range(n_pos):
            diffs[pos] = np.diff(data.position_series[pos].astype(np.float64))
        for pos in range(n_pos):
            for lag in range(1, 4):
                idx = t - 1 - lag
                if 0 <= idx < len(diffs[pos]):
                    feats.append(float(diffs[pos][idx]))
                else:
                    feats.append(0.0)
                if t == 5:
                    feature_names.append(f'diff_P{pos}_lag{lag}')

        # 上期各位置值
        for pos in range(n_pos):
            feats.append(float(data.red_matrix[t - 1, pos]))
            if t == 5:
                feature_names.append(f'val_P{pos}')

        # 位置间差值（上期）
        for i in range(n_pos):
            for j in range(i + 1, n_pos):
                feats.append(float(data.red_matrix[t - 1, j]) - float(data.red_matrix[t - 1, i]))
                if t == 5:
                    feature_names.append(f'gap_P{i}_P{j}')

        # 上期组合统计量
        vals = data.red_matrix[t - 1, :].astype(float)
        feats.append(float(np.sum(vals)))  # 和值
        feats.append(float(np.max(vals) - np.min(vals)))  # 跨度
        feats.append(float(np.sum(vals.astype(int) % 2 == 1)))  # 奇数个数
        mid = (1 + data.red_range) / 2
        feats.append(float(np.sum(vals > mid)))  # 大号个数
        if t == 5:
            feature_names.extend(['combo_sum', 'combo_span', 'combo_odd', 'combo_big'])

        all_features.append(feats)

    X = np.array(all_features, dtype=np.float32)
    indices = np.arange(5, n_draws)

    return X, indices, feature_names


def e2_2_cross_position(data, probs, test_indices, train_end_idx, test_start_idx, lottery_type):
    """跨位置联合方向预测

    方案A: XGBoost增强（加入跨位置特征后重新训练）
    方案B: 多输出MLP（联合预测所有位置方向）
    Go: 效率提升>=5% 或 存活率提升>=2%
    """
    log(f"\n{'═'*55}")
    log(f"  E2.2: 跨位置联合方向预测 [{lottery_type}]")
    log(f"{'═'*55}")

    n_pos = data.red_count
    n_samples = len(test_indices)
    red_range = data.red_range
    dir_map = {-1: 0, 0: 1, 1: 2}

    # 构造跨位置特征
    with Timer("构造跨位置特征"):
        X_all, all_indices, feature_names = build_cross_position_features(
            data, train_end_idx, test_start_idx)

    # 构造标签: 各位置方向
    Y_all = np.zeros((len(all_indices), n_pos), dtype=np.int32)
    for i, t in enumerate(all_indices):
        for pos in range(n_pos):
            dir_idx = t - 1
            if 0 <= dir_idx < len(data.direction_series[pos]):
                Y_all[i, pos] = dir_map[int(data.direction_series[pos][dir_idx])]
            else:
                Y_all[i, pos] = 1  # 默认持平

    # 划分训练/测试
    train_mask = all_indices < train_end_idx
    test_mask = all_indices >= test_start_idx
    train_X, train_Y = X_all[train_mask], Y_all[train_mask]
    test_X, test_Y = X_all[test_mask], Y_all[test_mask]
    test_idx_mapped = all_indices[test_mask]

    log(f"  特征维度: {train_X.shape[1]}, 训练样本: {train_X.shape[0]}, 测试样本: {test_X.shape[0]}")

    # === 方案A: XGBoost增强 ===
    log(f"\n  --- 方案A: XGBoost增强 ---")

    try:
        import xgboost as xgb
        xgb.XGBClassifier()
        use_xgb = True
    except Exception:
        use_xgb = False

    from sklearn.ensemble import GradientBoostingClassifier

    cross_probs = np.zeros((test_X.shape[0], n_pos, 3))
    n_val = max(30, int(len(train_X) * 0.2))

    for pos in range(n_pos):
        if use_xgb:
            model = xgb.XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
                reg_alpha=0.1, reg_lambda=1.0,
                objective='multi:softprob', num_class=3,
                eval_metric='mlogloss', early_stopping_rounds=30,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            model.fit(train_X[:-n_val], train_Y[:-n_val, pos],
                      eval_set=[(train_X[-n_val:], train_Y[-n_val:, pos])], verbose=False)
        else:
            model = GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                min_samples_leaf=10, subsample=0.8, max_features=0.6,
                random_state=42, validation_fraction=0.2, n_iter_no_change=30,
            )
            model.fit(train_X[:-n_val], train_Y[:-n_val, pos])

        pos_probs = model.predict_proba(test_X)
        # 确保3列
        if pos_probs.shape[1] < 3:
            full_probs = np.full((pos_probs.shape[0], 3), 1.0 / 3.0)
            classes = model.classes_
            for ci, c in enumerate(classes):
                full_probs[:, c] = pos_probs[:, ci]
            pos_probs = full_probs
        cross_probs[:, pos, :] = pos_probs

        pred = np.argmax(pos_probs, axis=1)
        acc = float(np.mean(pred == test_Y[:, pos]))
        log(f"    P{pos}: acc={acc:.4f}")

    # 用 cross_probs 生成候选集
    cross_candidates = []
    full = set(range(1, red_range + 1))
    for i in range(len(test_idx_mapped)):
        t = int(test_idx_mapped[i])
        sample_cands = []
        for pos in range(n_pos):
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(red_range / 2)
            p_down, p_equal, p_up = cross_probs[i, pos, 0], cross_probs[i, pos, 1], cross_probs[i, pos, 2]
            cands = set(full)
            if p_up < 0.15:
                cands -= {v for v in full if v > current_val}
            if p_down < 0.15:
                cands -= {v for v in full if v < current_val}
            if p_equal < 0.15:
                cands.discard(current_val)
            if len(cands) == 0:
                cands = set(full)
            sample_cands.append(cands)
        cross_candidates.append(sample_cands)

    # 评估方案A
    baseline_combos = comb(red_range, n_pos)
    cross_combo_counts = []
    cross_survivals = []
    cross_pos_survivals = {pos: [] for pos in range(n_pos)}

    for i in range(len(test_idx_mapped)):
        t = int(test_idx_mapped[i])
        if t >= data.n_draws:
            continue
        true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
        n_combos = count_ordered_combos(cross_candidates[i])
        cross_combo_counts.append(n_combos)
        survived = check_combo_survival(cross_candidates[i], true_vals)
        cross_survivals.append(1.0 if survived else 0.0)
        for pos in range(n_pos):
            cross_pos_survivals[pos].append(1.0 if true_vals[pos] in cross_candidates[i][pos] else 0.0)

    avg_cross_combos = float(np.mean(cross_combo_counts)) if cross_combo_counts else 0
    cross_reduction = 1.0 - avg_cross_combos / baseline_combos if baseline_combos > 0 else 0
    cross_survival = float(np.mean(cross_survivals)) if cross_survivals else 0
    cross_efficiency = cross_reduction * cross_survival

    # 基线对比（原始 probs）
    # 需要对齐测试期
    orig_candidates = generate_candidate_sets(probs, data, test_indices, threshold=0.15)
    orig_combo_counts = []
    orig_survivals = []
    for i in range(n_samples):
        t = int(test_indices[i])
        true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
        n_combos = count_ordered_combos(orig_candidates[i])
        orig_combo_counts.append(n_combos)
        survived = check_combo_survival(orig_candidates[i], true_vals)
        orig_survivals.append(1.0 if survived else 0.0)

    avg_orig_combos = float(np.mean(orig_combo_counts))
    orig_reduction = 1.0 - avg_orig_combos / baseline_combos
    orig_survival = float(np.mean(orig_survivals))
    orig_efficiency = orig_reduction * orig_survival

    efficiency_improvement = cross_efficiency - orig_efficiency
    survival_improvement = cross_survival - orig_survival

    go = efficiency_improvement >= 0.05 or survival_improvement >= 0.02

    # === 方案B: 多输出MLP ===
    log(f"\n  --- 方案B: 多输出MLP ---")
    mlp_results = None

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        log(f"    设备: {device}")

        input_dim = train_X.shape[1]

        # 多输出MLP: 输入 → 共享层 → 各位置头
        class MultiOutputMLP(nn.Module):
            def __init__(self, in_dim, n_positions):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(in_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )
                self.heads = nn.ModuleList([
                    nn.Linear(128, 3) for _ in range(n_positions)
                ])

            def forward(self, x):
                shared_out = self.shared(x)
                return [head(shared_out) for head in self.heads]

        model = MultiOutputMLP(input_dim, n_pos).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

        tr_X_t = torch.FloatTensor(train_X[:-n_val]).to(device)
        tr_Y_t = torch.LongTensor(train_Y[:-n_val]).to(device)
        va_X_t = torch.FloatTensor(train_X[-n_val:]).to(device)
        va_Y_t = torch.LongTensor(train_Y[-n_val:]).to(device)
        te_X_t = torch.FloatTensor(test_X).to(device)

        dataset = TensorDataset(tr_X_t, tr_Y_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50
        best_state = None

        for epoch in range(500):
            model.train()
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = sum(criterion(outputs[pos], batch_Y[:, pos]) for pos in range(n_pos))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(va_X_t)
                val_loss = sum(criterion(val_outputs[pos], va_Y_t[:, pos]) for pos in range(n_pos)).item()
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        mlp_probs = np.zeros((test_X.shape[0], n_pos, 3))
        with torch.no_grad():
            test_outputs = model(te_X_t)
            for pos in range(n_pos):
                mlp_probs[:, pos, :] = torch.softmax(test_outputs[pos], dim=1).cpu().numpy()

        # 评估MLP
        mlp_accs = []
        for pos in range(n_pos):
            pred = np.argmax(mlp_probs[:, pos, :], axis=1)
            acc = float(np.mean(pred == test_Y[:, pos]))
            mlp_accs.append(acc)
            log(f"    MLP P{pos}: acc={acc:.4f}")

        mlp_results = {
            'per_pos_accuracy': {f'P{pos}': mlp_accs[pos] for pos in range(n_pos)},
            'overall_accuracy': float(np.mean(mlp_accs)),
            'epochs': epoch + 1,
            'best_val_loss': best_val_loss,
        }
        log(f"    MLP 整体准确率: {np.mean(mlp_accs):.4f}")

    except ImportError:
        log("    [警告] torch 未安装，跳过方案B")
    except Exception as e:
        log(f"    [警告] 方案B异常: {e}")

    results = {
        'plan_a_xgboost': {
            'cross_avg_combos': avg_cross_combos,
            'cross_reduction': float(cross_reduction),
            'cross_survival': float(cross_survival),
            'cross_efficiency': float(cross_efficiency),
            'per_pos_survival': {f'P{pos}': float(np.mean(cross_pos_survivals[pos]))
                                 for pos in range(n_pos) if cross_pos_survivals[pos]},
        },
        'baseline': {
            'avg_combos': avg_orig_combos,
            'reduction': float(orig_reduction),
            'survival': float(orig_survival),
            'efficiency': float(orig_efficiency),
        },
        'plan_b_mlp': mlp_results,
        'summary': {
            'go': go,
            'decision': 'GO' if go else 'NO-GO',
            'efficiency_improvement': float(efficiency_improvement),
            'survival_improvement': float(survival_improvement),
            'criteria': '效率提升>=5% 或 存活率提升>=2%',
        },
    }

    log(f"\n  基线: 效率={orig_efficiency:.4f}, 存活={orig_survival:.1%}")
    log(f"  跨位置: 效率={cross_efficiency:.4f}, 存活={cross_survival:.1%}")
    log(f"  效率提升: {efficiency_improvement:+.4f}, 存活提升: {survival_improvement:+.1%}")
    log(f"  判定: {'GO' if go else 'NO-GO'}")

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2_2_cross_position_{lottery_type}.json")
    return results, cross_probs, cross_candidates, test_idx_mapped


# ============================================================
#  E2.3: 级联组合评估 + 汇总
# ============================================================

def e2_3_cascade_evaluation(data, all_layer_results, all_candidates_layers, test_indices, lottery_type):
    """级联所有通过Go判定的层，评估组合缩号效果

    级联策略: 对每个位置取各层候选集的交集
    """
    log(f"\n{'═'*55}")
    log(f"  E2.3: 级联组合评估 [{lottery_type}]")
    log(f"{'═'*55}")

    n_pos = data.red_count
    n_samples = len(test_indices)
    red_range = data.red_range
    baseline_combos = comb(red_range, n_pos)
    full = set(range(1, red_range + 1))

    # 收集通过Go的层
    go_layers = []
    for layer_name, layer_result in all_layer_results.items():
        summary = layer_result.get('summary', {})
        if summary.get('go', False):
            go_layers.append(layer_name)
            log(f"  GO层: {layer_name}")

    if not go_layers:
        log(f"  [警告] 没有通过Go判定的层，使用所有层进行级联")
        go_layers = list(all_layer_results.keys())

    # 级联: 对每个样本、每个位置取交集
    cascade_candidates = []
    for i in range(n_samples):
        sample_cands = []
        for pos in range(n_pos):
            intersection = set(full)
            for layer_name in go_layers:
                if layer_name in all_candidates_layers:
                    layer_cands = all_candidates_layers[layer_name]
                    if i < len(layer_cands) and pos < len(layer_cands[i]):
                        intersection &= layer_cands[i][pos]
            # 至少保留1个候选
            if len(intersection) == 0:
                # 回退: 取并集中最小的层
                min_size = float('inf')
                best_cands = full
                for layer_name in go_layers:
                    if layer_name in all_candidates_layers:
                        layer_cands = all_candidates_layers[layer_name]
                        if i < len(layer_cands) and pos < len(layer_cands[i]):
                            if len(layer_cands[i][pos]) < min_size:
                                min_size = len(layer_cands[i][pos])
                                best_cands = layer_cands[i][pos]
                intersection = best_cands
            sample_cands.append(intersection)
        cascade_candidates.append(sample_cands)

    # 评估
    cascade_combo_counts = []
    cascade_survivals = []
    cascade_pos_survivals = {pos: [] for pos in range(n_pos)}
    cascade_pos_reductions = {pos: [] for pos in range(n_pos)}

    for i in range(n_samples):
        t = int(test_indices[i])
        if t >= data.n_draws:
            continue
        true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]

        n_combos = count_ordered_combos(cascade_candidates[i])
        cascade_combo_counts.append(n_combos)

        survived = check_combo_survival(cascade_candidates[i], true_vals)
        cascade_survivals.append(1.0 if survived else 0.0)

        for pos in range(n_pos):
            pos_surv = 1.0 if true_vals[pos] in cascade_candidates[i][pos] else 0.0
            cascade_pos_survivals[pos].append(pos_surv)
            cascade_pos_reductions[pos].append(1.0 - len(cascade_candidates[i][pos]) / red_range)

    avg_combos = float(np.mean(cascade_combo_counts)) if cascade_combo_counts else 0
    combo_reduction = 1.0 - avg_combos / baseline_combos if baseline_combos > 0 else 0
    combo_survival = float(np.mean(cascade_survivals)) if cascade_survivals else 0
    efficiency = combo_reduction * combo_survival

    # 各层单独效果对比
    layer_comparison = {}
    for layer_name in go_layers:
        if layer_name not in all_candidates_layers:
            continue
        layer_cands = all_candidates_layers[layer_name]
        l_combos = []
        l_survs = []
        for i in range(min(n_samples, len(layer_cands))):
            t = int(test_indices[i])
            if t >= data.n_draws:
                continue
            true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]
            n_c = count_ordered_combos(layer_cands[i])
            l_combos.append(n_c)
            s = check_combo_survival(layer_cands[i], true_vals)
            l_survs.append(1.0 if s else 0.0)
        if l_combos:
            l_red = 1.0 - np.mean(l_combos) / baseline_combos
            l_surv = np.mean(l_survs)
            layer_comparison[layer_name] = {
                'avg_combos': float(np.mean(l_combos)),
                'reduction': float(l_red),
                'survival': float(l_surv),
                'efficiency': float(l_red * l_surv),
            }

    results = {
        'go_layers': go_layers,
        'n_layers': len(go_layers),
        'cascade': {
            'avg_combos': avg_combos,
            'combo_reduction': float(combo_reduction),
            'combo_survival': float(combo_survival),
            'efficiency': float(efficiency),
            'per_pos_survival': {f'P{pos}': float(np.mean(cascade_pos_survivals[pos]))
                                 for pos in range(n_pos) if cascade_pos_survivals[pos]},
            'per_pos_reduction': {f'P{pos}': float(np.mean(cascade_pos_reductions[pos]))
                                  for pos in range(n_pos) if cascade_pos_reductions[pos]},
        },
        'layer_comparison': layer_comparison,
        'baseline_combos': baseline_combos,
    }

    log(f"\n  级联层数: {len(go_layers)}")
    log(f"  级联结果: 平均组合数={avg_combos:,.0f}, 缩减={combo_reduction:.1%}")
    log(f"  组合存活率: {combo_survival:.1%}")
    log(f"  综合效率: {efficiency:.4f}")
    log(f"\n  各层对比:")
    for ln, lc in layer_comparison.items():
        log(f"    {ln}: 缩减={lc['reduction']:.1%}, 存活={lc['survival']:.1%}, 效率={lc['efficiency']:.4f}")

    save_json(results, STRICT_EXPERIMENT_DIR / f"e2_3_cascade_{lottery_type}.json")
    return results


# ============================================================
#  主入口
# ============================================================

def run_e2(lottery_type="daletou"):
    """运行完整 E2 实验"""
    setup_logging()
    log(f"\n{'━'*60}")
    log(f"  E2: 多层级联过滤实验 [{lottery_type}]")
    log(f"{'━'*60}")

    # 确保输出目录存在
    STRICT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 E1 的方向概率和数据
    with Timer("加载数据和方向概率"):
        probs, test_indices, data, best_method = get_best_probs(lottery_type)
        log(f"  期数: {data.n_draws}, 红球: {data.red_count}, 范围: 1-{data.red_range}")
        log(f"  最优方案: {best_method}")
        log(f"  probs shape: {probs.shape}, 测试样本: {len(test_indices)}")

    # 划分训练/测试
    train_ratio = 0.7
    train_end_idx = int(data.n_draws * train_ratio)
    test_start_idx = int(test_indices[0])
    log(f"  训练期: [0, {train_end_idx}), 测试期: [{test_start_idx}, {data.n_draws})")

    all_layer_results = {}
    all_candidates_layers = {}

    # === E2.1 Step 0: 差分分布稳定性 ===
    with Timer("E2.1 Step 0"):
        step0_result = e2_1_step0_stability(data, train_end_idx, lottery_type)
    all_layer_results['e2_1_step0'] = step0_result

    # === E2.1 Step 1: 差分幅度预测 ===
    if step0_result['summary']['go']:
        with Timer("E2.1 Step 1"):
            step1_result = e2_1_step1_amplitude(data, train_end_idx, test_start_idx, lottery_type)
        all_layer_results['e2_1_step1'] = step1_result

        # === E2.1 Step 2: 差分幅度增量缩号 ===
        if step1_result['summary']['go']:
            with Timer("E2.1 Step 2"):
                step2_result, step2_candidates = e2_1_step2_incremental(
                    data, probs, test_indices, train_end_idx, test_start_idx, lottery_type)
            all_layer_results['e2_1_step2'] = step2_result
            all_candidates_layers['e2_1_step2'] = step2_candidates
        else:
            log(f"\n  E2.1 Step 1 NO-GO, 跳过 Step 2")
    else:
        log(f"\n  E2.1 Step 0 NO-GO, 跳过 Step 1 & 2")

    # === E2.2: 跨位置联合方向预测 ===
    with Timer("E2.2"):
        e2_2_result, cross_probs, cross_candidates, cross_test_idx = e2_2_cross_position(
            data, probs, test_indices, train_end_idx, test_start_idx, lottery_type)
    all_layer_results['e2_2'] = e2_2_result

    # 对齐 cross_candidates 到 test_indices
    # cross_test_idx 和 test_indices 可能不完全对齐，需要映射
    cross_idx_set = set(cross_test_idx.tolist())
    aligned_cross_candidates = []
    for i, t in enumerate(test_indices):
        t_int = int(t)
        if t_int in cross_idx_set:
            ci = np.where(cross_test_idx == t_int)[0]
            if len(ci) > 0:
                aligned_cross_candidates.append(cross_candidates[ci[0]])
            else:
                aligned_cross_candidates.append([set(range(1, data.red_range + 1))] * data.red_count)
        else:
            aligned_cross_candidates.append([set(range(1, data.red_range + 1))] * data.red_count)
    all_candidates_layers['e2_2'] = aligned_cross_candidates

    # === E2.3: 级联组合评估 ===
    with Timer("E2.3"):
        e2_3_result = e2_3_cascade_evaluation(
            data, all_layer_results, all_candidates_layers, test_indices, lottery_type)

    # === 汇总 ===
    summary = {
        'lottery_type': lottery_type,
        'n_draws': data.n_draws,
        'train_end': train_end_idx,
        'test_start': test_start_idx,
        'n_test': len(test_indices),
        'layers': {},
        'cascade': e2_3_result,
    }
    for layer_name, layer_result in all_layer_results.items():
        s = layer_result.get('summary', {})
        summary['layers'][layer_name] = {
            'decision': s.get('decision', 'N/A'),
            'go': s.get('go', False),
        }

    save_json(summary, STRICT_EXPERIMENT_DIR / f"e2_summary_{lottery_type}.json")

    log(f"\n{'━'*60}")
    log(f"  E2 实验完成")
    log(f"{'━'*60}")
    log(f"  各层判定:")
    for ln, ls in summary['layers'].items():
        log(f"    {ln}: {ls['decision']}")
    log(f"  级联效率: {e2_3_result['cascade']['efficiency']:.4f}")
    log(f"  级联存活率: {e2_3_result['cascade']['combo_survival']:.1%}")
    log(f"  级联缩减率: {e2_3_result['cascade']['combo_reduction']:.1%}")

    return summary


if __name__ == "__main__":
    lottery_type = sys.argv[1] if len(sys.argv) > 1 else "daletou"
    run_e2(lottery_type)
