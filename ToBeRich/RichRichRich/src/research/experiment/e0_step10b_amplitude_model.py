# -*- coding: utf-8 -*-
"""E0-Step10b：幅度预测模型训练

基于 Step10a 筛选出的显著模式，训练幅度预测器：
  方案A: XGBoost/MLP 回归 → diff 数值 (MSE/MAE)
  方案B: XGBoost/MLP 分类 → 7区间 (CrossEntropy)
  方案C: 分位数回归 → 幅度的概率分布 (Pinball Loss)

评估指标：MAE, RMSE, R², Top-2准确率, Top-100命中率提升

用法: python3 -m src.research.experiment.e0_step10b_amplitude_model
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_json, RESULTS_DIR,
)
from research.experiment.e0_step10a_amplitude_scan import (
    encode_direction_seq, encode_amplitude_bin, encode_value_zone,
    encode_diff_sign_amp, encode_diff_change_pattern,
    compute_miss_periods, encode_miss_zone,
)

STEP10_DIR = RESULTS_DIR / "e0_step10"

# 7区间分类标签
AMP_CLASSES = ['大降', '中降', '小降', '平', '小升', '中升', '大升']


# ============================================================
#  特征构造
# ============================================================

def build_amplitude_features(data, pos, start_idx, end_idx, scan_results):
    """为单个位置构造幅度预测特征矩阵

    特征包括：
    1. 基线局面特征（上期值、差分、方向等）
    2. Step10a 显著模式的匹配状态 + 条件下的幅度统计量

    Args:
        data: LotteryData
        pos: 位置索引
        start_idx: 起始期数索引（含）
        end_idx: 结束期数索引（不含）
        scan_results: Step10a 的扫描结果

    Returns:
        X: 特征矩阵 (n_samples, n_features)
        y_reg: 回归标签（差分值）
        y_cls: 分类标签（7区间索引）
        feature_names: 特征名列表
    """
    series = data.position_series[pos]
    diff_series = np.diff(series.astype(np.float64))
    dir_series = data.direction_series[pos]

    # 预计算分位数边界（基于训练期前60%）
    train_60pct = int(data.n_draws * 0.6) - 1
    train_abs = np.abs(diff_series[:train_60pct])
    amp_bin_edges = np.percentile(train_abs, np.linspace(0, 100, 11))
    amp_5_edges = np.percentile(train_abs, np.linspace(0, 100, 6))
    zone_edges = np.linspace(1, data.red_range, 6)

    # 7区间边界（基于训练期）
    cls_edges = np.percentile(diff_series[:train_60pct],
                              [100/7*i for i in range(1, 7)])

    # 获取验证通过的显著模式
    pos_key = f'P{pos}'
    patterns = []
    if pos_key in scan_results.get('positions', {}):
        for p in scan_results['positions'][pos_key].get('patterns', []):
            if p.get('validation', {}).get('status') == 'valid':
                patterns.append(p)

    features_list = []
    y_reg_list = []
    y_cls_list = []
    feature_names = None

    for t in range(max(start_idx - 1, 5), min(end_idx - 1, len(diff_series))):
        # diff_series[t] = series[t+1] - series[t]，条件基于 t 及之前
        feat = {}

        # === 基线局面特征 ===
        # 当前值
        feat['current_value'] = float(series[t])
        feat['value_zone'] = float(encode_value_zone(series[t], zone_edges))

        # 最近1-3期差分
        for lag in range(1, 4):
            if t >= lag:
                feat[f'diff_lag{lag}'] = float(diff_series[t - lag])
                feat[f'abs_diff_lag{lag}'] = float(abs(diff_series[t - lag]))
            else:
                feat[f'diff_lag{lag}'] = 0.0
                feat[f'abs_diff_lag{lag}'] = 0.0

        # 最近1-3期方向
        for lag in range(1, 4):
            if t >= lag:
                feat[f'dir_lag{lag}'] = float(dir_series[t - lag])
            else:
                feat[f'dir_lag{lag}'] = 0.0

        # 最近5期差分均值/标准差
        if t >= 5:
            recent = diff_series[t-5:t]
            feat['recent5_diff_mean'] = float(np.mean(recent))
            feat['recent5_diff_std'] = float(np.std(recent))
            feat['recent5_abs_mean'] = float(np.mean(np.abs(recent)))
        else:
            feat['recent5_diff_mean'] = 0.0
            feat['recent5_diff_std'] = 0.0
            feat['recent5_abs_mean'] = 0.0

        # 最近10期差分均值/标准差
        if t >= 10:
            recent10 = diff_series[t-10:t]
            feat['recent10_diff_mean'] = float(np.mean(recent10))
            feat['recent10_abs_mean'] = float(np.mean(np.abs(recent10)))
        else:
            feat['recent10_diff_mean'] = 0.0
            feat['recent10_abs_mean'] = 0.0

        # 遗漏期数
        miss = compute_miss_periods(series, t, series[t])
        feat['miss_periods'] = float(miss)
        feat['miss_zone'] = float(encode_miss_zone(miss))

        # 差分变化模式
        if t >= 2:
            d1, d2 = diff_series[t-2], diff_series[t-1]
            feat['diff_accel'] = float(d2 - d1)  # 差分加速度
            feat['diff_sign_change'] = float(1 if d1 * d2 < 0 else 0)
        else:
            feat['diff_accel'] = 0.0
            feat['diff_sign_change'] = 0.0

        # 值域位置归一化
        feat['value_normalized'] = float((series[t] - 1) / (data.red_range - 1))

        # 距值域边界的距离
        feat['dist_to_min'] = float(series[t] - 1)
        feat['dist_to_max'] = float(data.red_range - series[t])

        # === Step10a 显著模式匹配特征 ===
        n_matched = 0
        matched_means = []
        matched_medians = []
        matched_concentrations = []

        for pi, pat in enumerate(patterns[:50]):  # 最多50个模式
            dim = pat['dimension']
            cond = pat['condition']

            # 获取当前时刻的条件标签
            label = _get_label(dim, t, diff_series, dir_series, series, data,
                               amp_bin_edges, amp_5_edges, zone_edges)

            matched = 1 if (label is not None and str(label) == cond) else 0
            feat[f'pat_{pi}_match'] = float(matched)

            if matched:
                n_matched += 1
                matched_means.append(pat['mean'])
                matched_medians.append(pat['median'])
                matched_concentrations.append(pat['concentration'])

        feat['n_patterns_matched'] = float(n_matched)
        feat['matched_mean_avg'] = float(np.mean(matched_means)) if matched_means else 0.0
        feat['matched_median_avg'] = float(np.mean(matched_medians)) if matched_medians else 0.0
        feat['matched_concentration_avg'] = float(np.mean(matched_concentrations)) if matched_concentrations else 1.0

        # 构造特征向量
        if feature_names is None:
            feature_names = sorted(feat.keys())

        features_list.append([feat[k] for k in feature_names])

        # 标签
        diff_val = diff_series[t]
        y_reg_list.append(float(diff_val))

        # 7区间分类
        cls = np.searchsorted(cls_edges, diff_val)
        y_cls_list.append(int(cls))

    X = np.array(features_list, dtype=np.float32)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_cls = np.array(y_cls_list, dtype=np.int32)

    return X, y_reg, y_cls, feature_names


def _get_label(dim, t, diff_series, dir_series, series, data,
               amp_bin_edges, amp_5_edges, zone_edges):
    """获取条件标签（复用 Step10a 的逻辑）"""
    if dim.startswith('D1_dir_seq_n'):
        n = int(dim.split('n')[-1])
        return encode_direction_seq(dir_series, t, n)
    elif dim.startswith('D2_amp_bin_n'):
        n = int(dim.split('n')[-1])
        return encode_amplitude_bin(diff_series, t, n, amp_bin_edges)
    elif dim == 'D3_value_zone':
        return encode_value_zone(series[t], zone_edges)
    elif dim == 'D4_diff_sign_amp':
        return encode_diff_sign_amp(diff_series[t-1], amp_5_edges) if t >= 1 else None
    elif dim == 'D5_diff_change':
        return encode_diff_change_pattern(diff_series, t)
    elif dim == 'D6_miss_zone':
        miss = compute_miss_periods(series, t, series[t])
        return encode_miss_zone(miss)

    # 交叉维度
    if '_x_' in dim:
        parts = dim.split('_x_')
        la = _get_cross_part(parts[0], t, diff_series, dir_series, series, data,
                             amp_bin_edges, amp_5_edges, zone_edges)
        lb = _get_cross_part(parts[1], t, diff_series, dir_series, series, data,
                             amp_bin_edges, amp_5_edges, zone_edges)
        if la is None or lb is None:
            return None
        return (la, lb)
    return None


def _get_cross_part(dim_part, t, diff_series, dir_series, series, data,
                    amp_bin_edges, amp_5_edges, zone_edges):
    """解析交叉维度的单个部分"""
    if dim_part.startswith('D1n'):
        return encode_direction_seq(dir_series, t, int(dim_part[3:]))
    elif dim_part.startswith('D2n'):
        return encode_amplitude_bin(diff_series, t, int(dim_part[3:]), amp_bin_edges)
    elif dim_part == 'D3':
        return encode_value_zone(series[t], zone_edges)
    elif dim_part == 'D4':
        return encode_diff_sign_amp(diff_series[t-1], amp_5_edges) if t >= 1 else None
    elif dim_part == 'D5':
        return encode_diff_change_pattern(diff_series, t)
    elif dim_part == 'D6':
        return encode_miss_zone(compute_miss_periods(series, t, series[t]))
    return None


# ============================================================
#  方案A：回归模型
# ============================================================

def train_regression(X_train, y_train, X_test, y_test):
    """训练回归模型（XGBoost + MLP）"""
    results = {}

    # 基线：预测均值
    baseline_pred = np.full_like(y_test, np.mean(y_train))
    baseline_mae = float(np.mean(np.abs(y_test - baseline_pred)))
    baseline_rmse = float(np.sqrt(np.mean((y_test - baseline_pred) ** 2)))
    results['baseline'] = {'mae': baseline_mae, 'rmse': baseline_rmse}

    # XGBoost 回归
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)

        mae_xgb = float(np.mean(np.abs(y_test - pred_xgb)))
        rmse_xgb = float(np.sqrt(np.mean((y_test - pred_xgb) ** 2)))
        ss_res = np.sum((y_test - pred_xgb) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_xgb = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        results['xgboost'] = {
            'mae': mae_xgb, 'rmse': rmse_xgb, 'r2': r2_xgb,
            'mae_improvement': float(1 - mae_xgb / baseline_mae) if baseline_mae > 0 else 0,
        }
    except Exception as e:
        log(f"    XGBoost 回归不可用: {e}")
        # 降级到 GradientBoostingRegressor
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            gbr = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                min_samples_leaf=10, subsample=0.8, random_state=42,
            )
            gbr.fit(X_train, y_train)
            pred_gbr = gbr.predict(X_test)

            mae_gbr = float(np.mean(np.abs(y_test - pred_gbr)))
            rmse_gbr = float(np.sqrt(np.mean((y_test - pred_gbr) ** 2)))
            ss_res = np.sum((y_test - pred_gbr) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_gbr = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            results['xgboost'] = {
                'mae': mae_gbr, 'rmse': rmse_gbr, 'r2': r2_gbr,
                'mae_improvement': float(1 - mae_gbr / baseline_mae) if baseline_mae > 0 else 0,
                'note': 'sklearn GBR fallback',
            }
        except Exception as e2:
            log(f"    GBR 降级也失败: {e2}")
            results['xgboost'] = None

    # MLP 回归
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=42, learning_rate_init=0.001,
        )
        mlp.fit(X_tr_s, y_train)
        pred_mlp = mlp.predict(X_te_s)

        mae_mlp = float(np.mean(np.abs(y_test - pred_mlp)))
        rmse_mlp = float(np.sqrt(np.mean((y_test - pred_mlp) ** 2)))
        ss_res = np.sum((y_test - pred_mlp) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_mlp = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        results['mlp'] = {
            'mae': mae_mlp, 'rmse': rmse_mlp, 'r2': r2_mlp,
            'mae_improvement': float(1 - mae_mlp / baseline_mae) if baseline_mae > 0 else 0,
        }
    except Exception as e:
        log(f"    MLP 回归异常: {e}")
        results['mlp'] = None

    return results


# ============================================================
#  方案B：细粒度分类
# ============================================================

def train_classification(X_train, y_cls_train, X_test, y_cls_test):
    """训练7区间分类模型"""
    results = {}

    # 基线：预测最频繁类
    from collections import Counter
    most_common = Counter(y_cls_train.tolist()).most_common(1)[0][0]
    baseline_acc = float(np.mean(y_cls_test == most_common))
    results['baseline'] = {'accuracy': baseline_acc}

    # XGBoost 分类
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='mlogloss',
        )
        xgb.fit(X_train, y_cls_train)
        pred = xgb.predict(X_test)
        proba = xgb.predict_proba(X_test)

        acc = float(np.mean(pred == y_cls_test))
        # Top-2 准确率
        top2_acc = 0
        for i in range(len(y_cls_test)):
            top2 = np.argsort(proba[i])[-2:]
            if y_cls_test[i] in top2:
                top2_acc += 1
        top2_acc = float(top2_acc / len(y_cls_test))

        results['xgboost'] = {'accuracy': acc, 'top2_accuracy': top2_acc}
    except Exception as e:
        log(f"    XGBoost 分类不可用: {e}")
        # 降级到 GradientBoostingClassifier
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            gbc = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                min_samples_leaf=10, subsample=0.8, random_state=42,
            )
            gbc.fit(X_train, y_cls_train)
            pred = gbc.predict(X_test)
            proba = gbc.predict_proba(X_test)

            acc = float(np.mean(pred == y_cls_test))
            top2_acc = 0
            for i in range(len(y_cls_test)):
                top2 = np.argsort(proba[i])[-2:]
                if y_cls_test[i] in top2:
                    top2_acc += 1
            top2_acc = float(top2_acc / len(y_cls_test))

            results['xgboost'] = {
                'accuracy': acc, 'top2_accuracy': top2_acc,
                'note': 'sklearn GBC fallback',
            }
        except Exception as e2:
            log(f"    GBC 降级也失败: {e2}")
            results['xgboost'] = None

    # MLP 分类
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=42, learning_rate_init=0.001,
        )
        mlp.fit(X_tr_s, y_cls_train)
        pred = mlp.predict(X_te_s)
        proba = mlp.predict_proba(X_te_s)

        acc = float(np.mean(pred == y_cls_test))
        top2_acc = 0
        for i in range(len(y_cls_test)):
            top2 = np.argsort(proba[i])[-2:]
            if y_cls_test[i] in top2:
                top2_acc += 1
        top2_acc = float(top2_acc / len(y_cls_test))

        results['mlp'] = {'accuracy': acc, 'top2_accuracy': top2_acc}
    except Exception as e:
        log(f"    MLP 分类异常: {e}")
        results['mlp'] = None

    return results


# ============================================================
#  方案C：分位数回归
# ============================================================

def train_quantile_regression(X_train, y_train, X_test, y_test):
    """分位数回归，输出幅度的概率分布"""
    results = {}
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    try:
        from sklearn.ensemble import GradientBoostingRegressor

        preds = {}
        for q in quantiles:
            gbr = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                loss='quantile', alpha=q, random_state=42,
            )
            gbr.fit(X_train, y_train)
            preds[q] = gbr.predict(X_test)

        # 评估：校准度（实际落在预测区间内的比例）
        # 80% 区间: [q10, q90]
        in_80 = np.mean((y_test >= preds[0.1]) & (y_test <= preds[0.9]))
        # 50% 区间: [q25, q75]
        in_50 = np.mean((y_test >= preds[0.25]) & (y_test <= preds[0.75]))

        # Pinball loss
        pinball_losses = {}
        for q in quantiles:
            errors = y_test - preds[q]
            loss = np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))
            pinball_losses[str(q)] = float(loss)

        # CRPS 近似（用分位数近似）
        # CRPS ≈ 2 * mean(pinball_loss) 对所有分位数
        crps_approx = float(2 * np.mean(list(pinball_losses.values())))

        # 区间宽度（越窄越好）
        interval_80_width = float(np.mean(preds[0.9] - preds[0.1]))
        interval_50_width = float(np.mean(preds[0.75] - preds[0.25]))

        results['quantile_gbr'] = {
            'calibration_80': float(in_80),
            'calibration_50': float(in_50),
            'pinball_losses': pinball_losses,
            'crps_approx': crps_approx,
            'interval_80_width': interval_80_width,
            'interval_50_width': interval_50_width,
        }
    except Exception as e:
        log(f"    分位数回归异常: {e}")
        results['quantile_gbr'] = None

    return results


# ============================================================
#  主入口
# ============================================================

def run_step10b(lottery_type):
    """运行 Step10b 幅度预测模型"""
    log(f"\n{'═'*60}")
    log(f"  E0-Step10b: 幅度预测模型 [{lottery_type}]")
    log(f"{'═'*60}")

    data = LotteryData(lottery_type)
    n_draws = data.n_draws

    # 加载 Step10a 结果
    scan_path = STEP10_DIR / f"step10a_amplitude_scan_{lottery_type}.json"
    if not scan_path.exists():
        log(f"  Step10a 结果不存在: {scan_path.name}，请先运行 Step10a")
        return None

    scan_results = load_json(scan_path)

    # 时间切分：前60%训练，后40%测试
    train_end = int(n_draws * 0.6)
    test_start = train_end

    log(f"  总期数: {n_draws}, 训练: [0, {train_end}), 测试: [{test_start}, {n_draws})")

    all_results = {
        'lottery_type': lottery_type,
        'positions': {},
        'summary': {},
    }

    reg_maes = []
    cls_accs = []

    for pos in range(data.red_count):
        log(f"\n  --- P{pos} ---")

        with Timer(f"P{pos} 特征构造"):
            X_train, y_reg_train, y_cls_train, feat_names = build_amplitude_features(
                data, pos, 0, train_end, scan_results)
            X_test, y_reg_test, y_cls_test, _ = build_amplitude_features(
                data, pos, test_start, n_draws, scan_results)

        log(f"  P{pos}: 训练 {X_train.shape}, 测试 {X_test.shape}, 特征 {len(feat_names)}")

        # 方案A：回归
        with Timer(f"P{pos} 回归"):
            reg_results = train_regression(X_train, y_reg_train, X_test, y_reg_test)

        best_reg = None
        for method in ['xgboost', 'mlp']:
            r = reg_results.get(method)
            if r and (best_reg is None or r['mae'] < best_reg['mae']):
                best_reg = r
                best_reg['method'] = method

        if best_reg:
            reg_maes.append(best_reg['mae'])
            log(f"  P{pos} 回归最优: {best_reg['method']}, MAE={best_reg['mae']:.3f}, "
                f"R²={best_reg['r2']:.4f}, 改进={best_reg['mae_improvement']:.1%}")

        # 方案B：分类
        with Timer(f"P{pos} 分类"):
            cls_results = train_classification(X_train, y_cls_train, X_test, y_cls_test)

        best_cls = None
        for method in ['xgboost', 'mlp']:
            r = cls_results.get(method)
            if r and (best_cls is None or r['accuracy'] > best_cls['accuracy']):
                best_cls = r
                best_cls['method'] = method

        if best_cls:
            cls_accs.append(best_cls['accuracy'])
            log(f"  P{pos} 分类最优: {best_cls['method']}, Acc={best_cls['accuracy']:.3f}, "
                f"Top2={best_cls['top2_accuracy']:.3f}")

        # 方案C：分位数回归
        with Timer(f"P{pos} 分位数回归"):
            qr_results = train_quantile_regression(X_train, y_reg_train, X_test, y_reg_test)

        if qr_results.get('quantile_gbr'):
            qr = qr_results['quantile_gbr']
            log(f"  P{pos} 分位数: 80%校准={qr['calibration_80']:.3f}, "
                f"50%校准={qr['calibration_50']:.3f}, CRPS={qr['crps_approx']:.3f}")

        all_results['positions'][f'P{pos}'] = {
            'n_train': int(X_train.shape[0]),
            'n_test': int(X_test.shape[0]),
            'n_features': int(X_train.shape[1]),
            'regression': reg_results,
            'classification': cls_results,
            'quantile': qr_results,
        }

    # 汇总
    all_results['summary'] = {
        'avg_regression_mae': float(np.mean(reg_maes)) if reg_maes else None,
        'avg_classification_acc': float(np.mean(cls_accs)) if cls_accs else None,
        'n_positions': data.red_count,
    }

    log(f"\n  汇总: 平均回归MAE={all_results['summary']['avg_regression_mae']:.3f}, "
        f"平均分类Acc={all_results['summary']['avg_classification_acc']:.3f}")

    save_json(all_results, STEP10_DIR / f"step10b_amplitude_model_{lottery_type}.json")
    return all_results


def main():
    setup_logging()
    log("=" * 60)
    log("  E0-Step10b: 幅度预测模型训练")
    log("=" * 60)

    results = {}
    for lt in ['daletou', 'shuangseqiu']:
        with Timer(f"Step10b [{lt}]"):
            results[lt] = run_step10b(lt)

    # 保存汇总
    summary = {}
    for lt, r in results.items():
        if r:
            summary[lt] = r['summary']
    save_json(summary, STEP10_DIR / "step10b_summary.json")

    log("\n  Step10b 全部完成!")


if __name__ == '__main__':
    main()
