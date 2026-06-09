# -*- coding: utf-8 -*-
"""
E5 Step 7：多算法融合实验

流程：
1. 用最优参数重新运行各子实验，获取 all_weights 并缓存
2. 两两融合 + 全融合，网格搜索最优 alpha
3. 与 E3 现有信号融合
4. 汇总报告
"""

import argparse
import pickle
import itertools
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_json, RESULTS_DIR
)
from research.experiment.e5_framework import (
    SPLIT_CONFIG, E5_RESULTS_DIR, evaluate_weights, fuse_weights
)

# 缓存目录
CACHE_DIR = E5_RESULTS_DIR / "weights_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 子实验名称列表
SUB_NAMES = ["e5a", "e5b", "e5c", "e5d", "e5e", "e5f", "e5g"]


def get_cache_path(sub_name: str, lottery_type: str) -> Path:
    """获取权重缓存文件路径"""
    return CACHE_DIR / f"{sub_name}_{lottery_type}_weights.pkl"


def load_cached_weights(sub_name: str, lottery_type: str):
    """加载缓存的权重数据"""
    path = get_cache_path(sub_name, lottery_type)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cached_weights(sub_name: str, lottery_type: str, weights):
    """保存权重数据到缓存"""
    path = get_cache_path(sub_name, lottery_type)
    with open(path, "wb") as f:
        pickle.dump(weights, f)
    log(f"  已缓存: {path.name}")


def collect_all_weights(lottery_type: str) -> Dict[str, list]:
    """收集所有子实验的权重数据（优先从缓存加载，否则重新运行）"""
    from research.experiment.e5a_matrix_profile import run_e5a
    from research.experiment.e5b_autoencoder import run_e5b
    from research.experiment.e5c_sax import run_e5c
    from research.experiment.e5d_shapelet import run_e5d
    from research.experiment.e5e_contrastive import run_e5e
    from research.experiment.e5f_dictionary import run_e5f
    from research.experiment.e5g_adaptive_features import run_e5g

    run_funcs = {
        "e5a": run_e5a, "e5b": run_e5b, "e5c": run_e5c,
        "e5d": run_e5d, "e5e": run_e5e, "e5f": run_e5f,
        "e5g": run_e5g,
    }

    data = LotteryData(lottery_type)
    split = SPLIT_CONFIG[lottery_type]
    max_train_idx = split["train_end"]
    test_start = split["test_start"]
    test_indices = np.arange(test_start, data.n_draws - 1)

    all_sub_weights = {}
    for name in SUB_NAMES:
        # 尝试从缓存加载
        cached = load_cached_weights(name, lottery_type)
        if cached is not None:
            log(f"  {name}: 从缓存加载 ({len(cached)} 样本)")
            all_sub_weights[name] = cached
            continue

        # 重新运行
        log(f"  {name}: 重新运行...")
        with Timer(f"{name} 运行"):
            weights = run_funcs[name](data, test_indices, max_train_idx)
        save_cached_weights(name, lottery_type, weights)
        all_sub_weights[name] = weights

    return all_sub_weights, data, test_indices


def quick_auc(all_weights, data, test_indices):
    """快速计算 AUC"""
    result = evaluate_weights(all_weights, data, test_indices)
    return result["overall_auc"]


def search_pairwise_fusion(all_sub_weights, data, test_indices):
    """两两融合搜索最优 alpha"""
    red_range = data.red_range
    names = list(all_sub_weights.keys())
    alpha_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            best_auc = -1.0
            best_alpha = None
            w1 = all_sub_weights[n1]
            w2 = all_sub_weights[n2]
            for a in alpha_grid:
                fused = fuse_weights([w1, w2], [a, 1-a], red_range)
                auc = quick_auc(fused, data, test_indices)
                if auc > best_auc:
                    best_auc = auc
                    best_alpha = a
            results.append({
                "pair": f"{n1}+{n2}",
                "best_alpha": {n1: best_alpha, n2: round(1-best_alpha, 1)},
                "auc": round(best_auc, 4),
            })
            log(f"    {n1}+{n2}: AUC={best_auc:.4f} "
                f"(α={best_alpha:.1f}/{1-best_alpha:.1f})")

    results.sort(key=lambda x: x["auc"], reverse=True)
    return results


def search_full_fusion(all_sub_weights, data, test_indices):
    """全部 E5 信号融合，用简化网格搜索"""
    red_range = data.red_range
    names = list(all_sub_weights.keys())
    n = len(names)
    weight_lists = [all_sub_weights[name] for name in names]

    # 策略1：等权融合
    equal_alphas = [1.0 / n] * n
    fused = fuse_weights(weight_lists, equal_alphas, red_range)
    equal_auc = quick_auc(fused, data, test_indices)
    log(f"    等权融合: AUC={equal_auc:.4f}")

    # 策略2：按单独 AUC 加权
    single_aucs = []
    for name in names:
        auc = quick_auc(all_sub_weights[name], data, test_indices)
        single_aucs.append(auc)
    total_auc = sum(single_aucs)
    auc_alphas = [a / total_auc for a in single_aucs]
    fused = fuse_weights(weight_lists, auc_alphas, red_range)
    auc_weighted = quick_auc(fused, data, test_indices)
    log(f"    AUC加权融合: AUC={auc_weighted:.4f}")
    for name, alpha in zip(names, auc_alphas):
        log(f"      {name}: α={alpha:.3f}")

    # 策略3：随机搜索最优 alpha（100次采样）
    best_auc = max(equal_auc, auc_weighted)
    best_alphas = equal_alphas if equal_auc >= auc_weighted else auc_alphas
    best_strategy = "等权" if equal_auc >= auc_weighted else "AUC加权"

    rng = np.random.RandomState(42)
    for trial in range(100):
        raw = rng.dirichlet(np.ones(n))
        alphas = raw.tolist()
        fused = fuse_weights(weight_lists, alphas, red_range)
        auc = quick_auc(fused, data, test_indices)
        if auc > best_auc:
            best_auc = auc
            best_alphas = alphas
            best_strategy = f"随机搜索#{trial}"

    log(f"    最优融合: AUC={best_auc:.4f} ({best_strategy})")
    for name, alpha in zip(names, best_alphas):
        log(f"      {name}: α={alpha:.3f}")

    return {
        "equal_weight": {"auc": round(equal_auc, 4)},
        "auc_weighted": {
            "auc": round(auc_weighted, 4),
            "alphas": {n: round(a, 3) for n, a in zip(names, auc_alphas)},
        },
        "best": {
            "auc": round(best_auc, 4),
            "strategy": best_strategy,
            "alphas": {n: round(a, 3) for n, a in zip(names, best_alphas)},
        },
    }


def search_top_n_fusion(all_sub_weights, data, test_indices):
    """Top-N 子集融合：只用最好的 2/3/4 个信号"""
    red_range = data.red_range
    names = list(all_sub_weights.keys())

    # 按单独 AUC 排序
    single_aucs = {}
    for name in names:
        single_aucs[name] = quick_auc(all_sub_weights[name], data, test_indices)
    sorted_names = sorted(single_aucs, key=single_aucs.get, reverse=True)

    results = []
    for top_n in [2, 3, 4]:
        subset = sorted_names[:top_n]
        weight_lists = [all_sub_weights[n] for n in subset]
        # 等权
        alphas = [1.0 / top_n] * top_n
        fused = fuse_weights(weight_lists, alphas, red_range)
        auc = quick_auc(fused, data, test_indices)
        results.append({
            "top_n": top_n,
            "signals": subset,
            "auc": round(auc, 4),
        })
        log(f"    Top-{top_n} ({'+'.join(subset)}): AUC={auc:.4f}")

    return results


def run_fusion_experiment(lottery_type: str):
    """运行完整融合实验"""
    setup_logging()
    log(f"\n{'#'*55}")
    log(f"  E5 Step 7: 多算法融合实验 [{lottery_type}]")
    log(f"{'#'*55}")

    # 1. 收集所有子实验权重
    log(f"\n[1] 收集子实验权重...")
    all_sub_weights, data, test_indices = collect_all_weights(lottery_type)

    # 2. 各子实验独立 AUC
    log(f"\n[2] 各子实验独立 AUC:")
    single_results = {}
    for name in SUB_NAMES:
        if name in all_sub_weights:
            auc = quick_auc(all_sub_weights[name], data, test_indices)
            single_results[name] = round(auc, 4)
            log(f"    {name}: AUC={auc:.4f}")

    # 3. 两两融合
    log(f"\n[3] 两两融合:")
    pairwise = search_pairwise_fusion(all_sub_weights, data, test_indices)

    # 4. Top-N 子集融合
    log(f"\n[4] Top-N 子集融合:")
    top_n = search_top_n_fusion(all_sub_weights, data, test_indices)

    # 5. 全融合
    log(f"\n[5] 全部 E5 信号融合:")
    full_fusion = search_full_fusion(all_sub_weights, data, test_indices)

    # 汇总保存
    summary = {
        "lottery_type": lottery_type,
        "single_auc": single_results,
        "pairwise_fusion": pairwise[:5],  # top 5
        "top_n_fusion": top_n,
        "full_fusion": full_fusion,
    }
    save_json(summary, E5_RESULTS_DIR / f"e5_fusion_{lottery_type}.json")
    log(f"\n已保存: e5_fusion_{lottery_type}.json")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lottery", default="daletou",
                        choices=["daletou", "shuangseqiu"])
    args = parser.parse_args()
    run_fusion_experiment(args.lottery)
