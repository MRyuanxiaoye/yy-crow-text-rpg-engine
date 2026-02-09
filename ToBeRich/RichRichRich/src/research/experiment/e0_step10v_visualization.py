# -*- coding: utf-8 -*-
"""E0-Step10v：可视化辅助分析

V1: t-SNE/UMAP 降维 + HDBSCAN 聚类（检测局面类型）
V2: 递归图 + RQA 量化分析（检测差分序列确定性成分）
V3: FFT 频谱分析（检测隐藏周期性）

用法: python3 -m src.research.experiment.e0_step10v_visualization
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_npz, RESULTS_DIR,
)

STEP10_DIR = RESULTS_DIR / "e0_step10"
STEP10_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  V1: UMAP 降维 + HDBSCAN 聚类
# ============================================================

def v1_umap_clustering(lottery_type):
    """用 UMAP 对 Phase2 特征矩阵降维，HDBSCAN 检测簇结构"""
    log(f"\n{'═'*55}")
    log(f"  V1: UMAP 降维 + 聚类 [{lottery_type}]")
    log(f"{'═'*55}")

    # 加载 Phase2 特征矩阵
    strict_dir = RESULTS_DIR / "experiment_strict"
    train_path = strict_dir / f"phase2_train_{lottery_type}.npz"
    test_path = strict_dir / f"phase2_test_{lottery_type}.npz"

    if not train_path.exists():
        log(f"  Phase2 特征文件不存在: {train_path.name}，跳过 V1")
        return None

    train_data = load_npz(train_path)
    test_data = load_npz(test_path)
    X_train = train_data['X']
    Y_train = train_data['Y']
    X_test = test_data['X']
    Y_test = test_data['Y']

    X_all = np.vstack([X_train, X_test])
    Y_all = np.concatenate([Y_train, Y_test])

    log(f"  特征矩阵: {X_all.shape}, 标签: {Y_all.shape}")

    # 处理 NaN/Inf
    X_clean = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        from umap import UMAP
    except ImportError:
        log("  umap-learn 未安装，尝试用 t-SNE 替代")
        return _v1_tsne_fallback(X_clean, Y_all, lottery_type)

    with Timer("UMAP 降维"):
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X_clean)

    log(f"  UMAP 完成: {embedding.shape}")

    # HDBSCAN 聚类
    try:
        from hdbscan import HDBSCAN
        with Timer("HDBSCAN 聚类"):
            clusterer = HDBSCAN(min_cluster_size=30, min_samples=10)
            cluster_labels = clusterer.fit_predict(embedding)
    except ImportError:
        log("  hdbscan 未安装，用 KMeans 替代")
        from sklearn.cluster import KMeans
        with Timer("KMeans 聚类"):
            km = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(embedding)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int(np.sum(cluster_labels == -1))
    log(f"  簇数: {n_clusters}, 噪声点: {n_noise}")

    # 分析每个簇的幅度分布差异
    data = LotteryData(lottery_type)
    n_pos = data.red_count

    # Y_all 的方向标签，需要对应到差分幅度
    # Y_all shape: (n_samples, n_pos) 或 (n_samples * n_pos,)
    # 这里分析簇内的方向准确率分布
    cluster_analysis = {}
    for c in sorted(set(cluster_labels)):
        if c == -1:
            continue
        mask = cluster_labels == c
        n_in_cluster = int(np.sum(mask))

        # 分析该簇内的方向分布
        y_cluster = Y_all[mask]
        if y_cluster.ndim == 1:
            # 单维标签
            dir_dist = {
                'n_samples': n_in_cluster,
                'label_mean': float(np.mean(y_cluster)),
                'label_std': float(np.std(y_cluster)),
            }
        else:
            # 多维标签（每位置一个方向）
            dir_dist = {
                'n_samples': n_in_cluster,
            }
            for p in range(min(n_pos, y_cluster.shape[1])):
                vals = y_cluster[:, p]
                dir_dist[f'P{p}_mean'] = float(np.mean(vals))
                dir_dist[f'P{p}_std'] = float(np.std(vals))

        cluster_analysis[f'cluster_{c}'] = dir_dist

    # 轮廓系数
    try:
        from sklearn.metrics import silhouette_score
        valid_mask = cluster_labels >= 0
        if np.sum(valid_mask) > 100 and n_clusters >= 2:
            sil = float(silhouette_score(embedding[valid_mask], cluster_labels[valid_mask],
                                         sample_size=min(5000, np.sum(valid_mask))))
        else:
            sil = -1.0
    except Exception:
        sil = -1.0

    log(f"  轮廓系数: {sil:.4f}")

    results = {
        'lottery_type': lottery_type,
        'method': 'UMAP+HDBSCAN',
        'n_samples': int(X_all.shape[0]),
        'n_features': int(X_all.shape[1]),
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': sil,
        'cluster_analysis': cluster_analysis,
    }

    save_json(results, STEP10_DIR / f"step10v_v1_umap_{lottery_type}.json")
    return results


def _v1_tsne_fallback(X, Y, lottery_type):
    """t-SNE 降维替代方案"""
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    # t-SNE 对大数据集慢，采样
    n_max = 3000
    if X.shape[0] > n_max:
        idx = np.random.RandomState(42).choice(X.shape[0], n_max, replace=False)
        X_sub = X[idx]
        Y_sub = Y[idx]
    else:
        X_sub = X
        Y_sub = Y

    with Timer("t-SNE 降维"):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        embedding = tsne.fit_transform(X_sub)

    with Timer("KMeans 聚类"):
        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(embedding)

    from sklearn.metrics import silhouette_score
    sil = float(silhouette_score(embedding, cluster_labels,
                                 sample_size=min(2000, len(embedding))))

    log(f"  t-SNE + KMeans: 5 簇, 轮廓系数={sil:.4f}")

    results = {
        'lottery_type': lottery_type,
        'method': 'tSNE+KMeans',
        'n_samples': int(X_sub.shape[0]),
        'n_features': int(X.shape[1]),
        'n_clusters': 5,
        'silhouette_score': sil,
    }

    save_json(results, STEP10_DIR / f"step10v_v1_tsne_{lottery_type}.json")
    return results


# ============================================================
#  V2: 递归图 + RQA 量化分析
# ============================================================

def v2_recurrence_analysis(lottery_type):
    """递归图 + RQA 指标，检测差分序列的确定性成分"""
    log(f"\n{'═'*55}")
    log(f"  V2: 递归图 + RQA 分析 [{lottery_type}]")
    log(f"{'═'*55}")

    data = LotteryData(lottery_type)
    results = {'lottery_type': lottery_type, 'positions': {}}

    for pos in range(data.red_count):
        diff_series = data.get_diff_series(pos, order=1)
        abs_diff = np.abs(diff_series)

        with Timer(f"P{pos} RQA"):
            rqa = _compute_rqa(abs_diff)

        # Monte Carlo 显著性检验：打乱1000次
        n_surrogates = 1000
        surrogate_dets = []
        rng = np.random.RandomState(42 + pos)

        with Timer(f"P{pos} Monte Carlo ({n_surrogates}次)"):
            for _ in range(n_surrogates):
                shuffled = rng.permutation(abs_diff)
                s_rqa = _compute_rqa(shuffled)
                surrogate_dets.append(s_rqa['DET'])

        surrogate_dets = np.array(surrogate_dets)
        p_value = float(np.mean(surrogate_dets >= rqa['DET']))

        results['positions'][f'P{pos}'] = {
            'n_points': len(abs_diff),
            'RR': rqa['RR'],
            'DET': rqa['DET'],
            'LAM': rqa['LAM'],
            'L_max': rqa['L_max'],
            'L_mean': rqa['L_mean'],
            'ENTR': rqa['ENTR'],
            'surrogate_DET_mean': float(np.mean(surrogate_dets)),
            'surrogate_DET_std': float(np.std(surrogate_dets)),
            'p_value': p_value,
            'significant': p_value < 0.05,
        }

        sig = "显著" if p_value < 0.05 else "不显著"
        log(f"  P{pos}: DET={rqa['DET']:.4f}, 基线={np.mean(surrogate_dets):.4f}, "
            f"p={p_value:.4f} ({sig})")

    # 汇总
    n_sig = sum(1 for v in results['positions'].values() if v['significant'])
    results['summary'] = {
        'n_positions': data.red_count,
        'n_significant': n_sig,
        'has_deterministic_component': n_sig > 0,
    }

    log(f"\n  汇总: {n_sig}/{data.red_count} 个位置 DET 显著高于随机基线")

    save_json(results, STEP10_DIR / f"step10v_v2_rqa_{lottery_type}.json")
    return results


def _compute_rqa(series, threshold_pct=10, l_min=2):
    """计算 RQA 指标

    Args:
        series: 1D 时间序列
        threshold_pct: 递归阈值（距离矩阵的百分位数）
        l_min: 最小对角线长度

    Returns:
        dict with RR, DET, LAM, L_max, L_mean, ENTR
    """
    n = len(series)
    # 限制大小避免内存爆炸
    max_n = 1500
    if n > max_n:
        series = series[-max_n:]
        n = max_n

    # 距离矩阵
    s = series.reshape(-1, 1)
    dist = np.abs(s - s.T)

    # 递归阈值
    threshold = np.percentile(dist[np.triu_indices(n, k=1)], threshold_pct)
    if threshold == 0:
        threshold = 0.5
    recurrence = (dist <= threshold).astype(np.int8)

    # RR: 递归率
    rr = float(np.sum(recurrence) - n) / (n * (n - 1)) if n > 1 else 0

    # 对角线长度分布
    diag_lengths = []
    for k in range(1, n):
        diag = np.diag(recurrence, k)
        length = 0
        for val in diag:
            if val:
                length += 1
            else:
                if length >= l_min:
                    diag_lengths.append(length)
                length = 0
        if length >= l_min:
            diag_lengths.append(length)

    # DET: 确定性
    total_recurrence = np.sum(recurrence) - n
    if total_recurrence > 0 and diag_lengths:
        det = float(sum(diag_lengths)) / (total_recurrence / 2)
        det = min(det, 1.0)
    else:
        det = 0.0

    # L_max, L_mean
    l_max = max(diag_lengths) if diag_lengths else 0
    l_mean = float(np.mean(diag_lengths)) if diag_lengths else 0.0

    # ENTR: 对角线长度分布的香农熵
    if diag_lengths:
        from collections import Counter
        counts = Counter(diag_lengths)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        entr = float(-np.sum(probs * np.log(probs + 1e-10)))
    else:
        entr = 0.0

    # LAM: 层流性（垂直线）
    vert_lengths = []
    for col in range(n):
        length = 0
        for row in range(n):
            if recurrence[row, col]:
                length += 1
            else:
                if length >= l_min:
                    vert_lengths.append(length)
                length = 0
        if length >= l_min:
            vert_lengths.append(length)

    if total_recurrence > 0 and vert_lengths:
        lam = float(sum(vert_lengths)) / (total_recurrence / 2)
        lam = min(lam, 1.0)
    else:
        lam = 0.0

    return {
        'RR': rr, 'DET': det, 'LAM': lam,
        'L_max': l_max, 'L_mean': l_mean, 'ENTR': entr,
    }


# ============================================================
#  V3: FFT 频谱分析
# ============================================================

def v3_fft_analysis(lottery_type):
    """FFT 频谱分析，检测差分序列的周期性成分"""
    log(f"\n{'═'*55}")
    log(f"  V3: FFT 频谱分析 [{lottery_type}]")
    log(f"{'═'*55}")

    data = LotteryData(lottery_type)
    results = {'lottery_type': lottery_type, 'positions': {}}

    for pos in range(data.red_count):
        diff_series = data.get_diff_series(pos, order=1)

        # 去均值
        centered = diff_series - np.mean(diff_series)
        n = len(centered)

        # FFT
        fft_vals = np.fft.rfft(centered)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n)

        # 排除直流分量
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]

        # Fisher's g 检验：最大周期图值 / 总和
        g_stat = float(np.max(power_no_dc) / np.sum(power_no_dc))
        # Fisher's g 的近似 p 值
        m = len(power_no_dc)
        # Bonferroni 近似: P(max > g*sum) ≈ m * (1-g)^(m-1)
        fisher_p = min(1.0, m * (1 - g_stat) ** (m - 1))

        # Top-5 频率
        top_idx = np.argsort(power_no_dc)[-5:][::-1]
        top_freqs = []
        for idx in top_idx:
            freq = freqs_no_dc[idx]
            period = 1.0 / freq if freq > 0 else float('inf')
            top_freqs.append({
                'frequency': float(freq),
                'period': float(period),
                'power': float(power_no_dc[idx]),
                'power_ratio': float(power_no_dc[idx] / np.sum(power_no_dc)),
            })

        # 号码出现/不出现的二值序列 FFT（遗漏周期检测）
        # 对每个号码值做一次
        value_series = data.position_series[pos]
        significant_periods = []

        # 只检测出现频率最高的 top-5 号码
        from collections import Counter
        value_counts = Counter(value_series.tolist())
        top_values = [v for v, _ in value_counts.most_common(5)]

        for val in top_values:
            binary = (value_series == val).astype(np.float64)
            binary_centered = binary - np.mean(binary)
            if np.std(binary_centered) < 1e-10:
                continue

            b_fft = np.fft.rfft(binary_centered)
            b_power = np.abs(b_fft) ** 2
            b_power_no_dc = b_power[1:]

            b_g = float(np.max(b_power_no_dc) / np.sum(b_power_no_dc))
            b_m = len(b_power_no_dc)
            b_p = min(1.0, b_m * (1 - b_g) ** (b_m - 1))

            if b_p < 0.05:
                peak_idx = np.argmax(b_power_no_dc)
                peak_freq = freqs_no_dc[peak_idx] if peak_idx < len(freqs_no_dc) else 0
                significant_periods.append({
                    'value': int(val),
                    'fisher_g': b_g,
                    'fisher_p': b_p,
                    'peak_period': float(1.0 / peak_freq) if peak_freq > 0 else float('inf'),
                })

        results['positions'][f'P{pos}'] = {
            'n_points': n,
            'fisher_g': g_stat,
            'fisher_p': fisher_p,
            'significant_diff_periodicity': fisher_p < 0.05,
            'top_frequencies': top_freqs,
            'significant_value_periods': significant_periods,
        }

        sig = "显著" if fisher_p < 0.05 else "不显著"
        log(f"  P{pos}: Fisher's g={g_stat:.4f}, p={fisher_p:.4f} ({sig}), "
            f"值周期: {len(significant_periods)} 个显著")

    # 汇总
    n_sig_diff = sum(1 for v in results['positions'].values() if v['significant_diff_periodicity'])
    n_sig_val = sum(len(v['significant_value_periods']) for v in results['positions'].values())

    results['summary'] = {
        'n_positions': data.red_count,
        'n_significant_diff_periodicity': n_sig_diff,
        'n_significant_value_periods': n_sig_val,
        'has_periodic_component': n_sig_diff > 0 or n_sig_val > 0,
    }

    log(f"\n  汇总: 差分周期性 {n_sig_diff}/{data.red_count} 显著, "
        f"值周期 {n_sig_val} 个显著")

    save_json(results, STEP10_DIR / f"step10v_v3_fft_{lottery_type}.json")
    return results


# ============================================================
#  主入口
# ============================================================

def run_step10v(lottery_type):
    """运行全部可视化分析"""
    log(f"\n{'═'*60}")
    log(f"  E0-Step10v: 可视化辅助分析 [{lottery_type}]")
    log(f"{'═'*60}")

    r1 = v1_umap_clustering(lottery_type)
    r2 = v2_recurrence_analysis(lottery_type)
    r3 = v3_fft_analysis(lottery_type)

    summary = {
        'lottery_type': lottery_type,
        'v1_clusters': r1['n_clusters'] if r1 else None,
        'v1_silhouette': r1['silhouette_score'] if r1 else None,
        'v2_n_significant': r2['summary']['n_significant'] if r2 else None,
        'v2_has_deterministic': r2['summary']['has_deterministic_component'] if r2 else None,
        'v3_n_periodic_diff': r3['summary']['n_significant_diff_periodicity'] if r3 else None,
        'v3_n_periodic_val': r3['summary']['n_significant_value_periods'] if r3 else None,
    }

    save_json(summary, STEP10_DIR / f"step10v_summary_{lottery_type}.json")
    return summary


def main():
    setup_logging()
    log("=" * 60)
    log("  E0-Step10v: 可视化辅助分析")
    log("=" * 60)

    for lt in ['daletou', 'shuangseqiu']:
        with Timer(f"Step10v [{lt}]"):
            run_step10v(lt)

    log("\n  Step10v 全部完成!")


if __name__ == '__main__':
    main()
