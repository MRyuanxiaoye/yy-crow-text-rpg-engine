"""E1：端到端组合缩号评估

评估组合层面的实际缩号效果：
- 每期每位置生成候选集（复用 phase3 方向概率）
- 用 DP 计算满足排序约束的合法组合数
- 计算组合缩减率、存活率、概率提升倍数

用法: python3 -m src.research.experiment.e1_combo_evaluation
"""

import sys
import time
import traceback
from bisect import bisect_left
from math import comb
from pathlib import Path

import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import research.experiment.utils as exp_utils
from research.experiment.utils import setup_logging, log, Timer, save_json, load_json, save_npz, load_npz
from research.experiment.phase1_compress import run_phase1
from research.experiment.phase2_replay import run_phase2
from research.experiment.phase3_train import experiment_3a, experiment_3b, experiment_3c
from research.data_loader import LotteryData

# === 路径常量 ===
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_RULES_DIR = RESULTS_DIR / "rules_strict"
STRICT_EXPERIMENT_DIR = RESULTS_DIR / "experiment_strict"


# === 核心：获取最优 probs ===

def get_best_probs(lottery_type):
    """获取最优方案的方向概率数组

    优先从缓存加载，否则运行 phase1-3 管线。

    Returns:
        probs: (n_test, n_pos, 3) 方向概率
        test_indices: (n_test,) 期数索引
        data: LotteryData 实例
        best_method: 最优方案名称
    """
    cache_path = STRICT_EXPERIMENT_DIR / f"phase3_best_probs_{lottery_type}.npz"

    if cache_path.exists():
        log(f"  从缓存加载 probs: {cache_path.name}")
        loaded = load_npz(cache_path)
        probs = loaded['probs']
        test_indices = loaded['test_indices']
        best_method = str(loaded['best_method'])
        data = LotteryData(lottery_type)
        return probs, test_indices, data, best_method

    # 运行管线
    log(f"  缓存不存在，运行 phase1-3 管线...")

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
                lottery_type, clusters, a1_filtered)

        # 运行三个实验，收集 probs
        with Timer("实验 3A"):
            results_3a, probs_3a = experiment_3a(
                train_X, train_Y, test_X, test_Y, feature_names, data, test_indices)

        with Timer("实验 3B"):
            results_3b, probs_3b = experiment_3b(
                train_X, train_Y, test_X, test_Y, feature_names, data, test_indices)

        probs_3c = None
        results_3c = None
        if results_3b is None or results_3b['overall']['efficiency'] < 0.25:
            with Timer("实验 3C"):
                results_3c, probs_3c = experiment_3c(
                    train_X, train_Y, test_X, test_Y, feature_names, data, test_indices)

        # 选最优
        candidates = []
        if results_3a is not None:
            candidates.append(('3A_statistical', results_3a['overall']['efficiency'], probs_3a))
        if results_3b is not None:
            candidates.append(('3B_xgboost', results_3b['overall']['efficiency'], probs_3b))
        if results_3c is not None:
            candidates.append(('3C_mlp', results_3c['overall']['efficiency'], probs_3c))

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_method, best_eff, best_probs = candidates[0]
        log(f"  最优方案: {best_method}, 效率={best_eff:.4f}")

        # 缓存
        save_npz(cache_path, probs=best_probs, test_indices=test_indices,
                 best_method=np.array(best_method))

        return best_probs, test_indices, data, best_method

    finally:
        exp_utils.EXPERIMENT_DIR = original_exp_dir


# === 候选集生成 ===

def generate_candidate_sets(probs, data, test_indices, threshold=0.15):
    """根据方向概率生成每期每位置的候选号码集合

    Args:
        probs: (n_samples, n_pos, 3) 方向概率 [P(D), P(E), P(U)]
        data: LotteryData
        test_indices: 期数索引
        threshold: 概率排除阈值，低于此值的方向对应号码被排除

    Returns:
        candidates: list[list[set]]  # [sample][pos] = set of candidate numbers
    """
    n_samples, n_pos, _ = probs.shape
    full = set(range(1, data.red_range + 1))
    candidates = []

    for i in range(n_samples):
        t = int(test_indices[i])
        sample_cands = []

        for pos in range(n_pos):
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(data.red_range / 2)
            p_down, p_equal, p_up = probs[i, pos, 0], probs[i, pos, 1], probs[i, pos, 2]

            cands = set(full)
            if p_up < threshold:
                cands -= {v for v in full if v > current_val}
            if p_down < threshold:
                cands -= {v for v in full if v < current_val}
            if p_equal < threshold:
                cands.discard(current_val)

            # 至少保留1个候选
            if len(cands) == 0:
                cands = set(full)

            sample_cands.append(cands)
        candidates.append(sample_cands)

    return candidates


# === DP 计算合法组合数 ===

def count_ordered_combos(candidate_sets):
    """用 DP 计算满足严格递增约束的组合数

    candidate_sets: [set, set, ...] 每个位置的候选集
    返回满足 v0 < v1 < ... < v_{n-1} 的组合数

    dp[pos] = {val: count}
    """
    n_pos = len(candidate_sets)

    # P0：每个候选值各有 1 种方案
    dp = {v: 1 for v in sorted(candidate_sets[0])}

    for pos in range(1, n_pos):
        new_dp = {}
        # 构建前缀和：按 key 排序
        sorted_keys = sorted(dp.keys())
        if not sorted_keys:
            return 0

        # prefix_vals[i] = sorted_keys[i]
        # prefix_sums[i] = sum(dp[sorted_keys[0..i]])
        prefix_sums = []
        cumsum = 0
        for k in sorted_keys:
            cumsum += dp[k]
            prefix_sums.append(cumsum)

        for v in sorted(candidate_sets[pos]):
            # 需要前一个位置的值严格 < v
            # 二分查找 sorted_keys 中 < v 的最大索引
            idx = bisect_left(sorted_keys, v) - 1
            if idx >= 0:
                new_dp[v] = prefix_sums[idx]

        dp = new_dp
        if not dp:
            return 0

    return sum(dp.values())


def check_combo_survival(candidate_sets, true_values):
    """检查正确号码是否全部在对应位置的候选集中"""
    for pos, val in enumerate(true_values):
        if val not in candidate_sets[pos]:
            return False
    return True


# === 基线验证 ===

def verify_baseline(red_range, red_count):
    """验证 DP 在无排除时返回 C(red_range, red_count)"""
    full_sets = [set(range(1, red_range + 1)) for _ in range(red_count)]
    dp_count = count_ordered_combos(full_sets)
    expected = comb(red_range, red_count)
    assert dp_count == expected, f"基线验证失败: DP={dp_count}, C({red_range},{red_count})={expected}"
    log(f"  基线验证通过: DP={dp_count} == C({red_range},{red_count})={expected}")


# === 主评估 ===

def evaluate_combo(lottery_type):
    """对单个彩种执行组合级评估"""
    log(f"\n{'═'*50}")
    log(f"  E1 组合评估: {lottery_type}")
    log(f"{'═'*50}")

    # 获取 probs
    with Timer("获取方向概率"):
        probs, test_indices, data, best_method = get_best_probs(lottery_type)

    n_samples, n_pos = probs.shape[0], probs.shape[1]
    log(f"  测试样本数: {n_samples}, 位置数: {n_pos}")
    log(f"  最优方案: {best_method}")
    log(f"  红球范围: 1-{data.red_range}, 红球个数: {data.red_count}")

    # 基线验证
    with Timer("基线验证"):
        verify_baseline(data.red_range, data.red_count)

    baseline_combos = comb(data.red_range, data.red_count)
    log(f"  全组合基线: C({data.red_range},{data.red_count}) = {baseline_combos:,}")

    # 生成候选集
    with Timer("生成候选集"):
        candidates = generate_candidate_sets(probs, data, test_indices)

    # 逐期计算
    log(f"\n  逐期计算组合数...")
    combo_counts = []
    survivals = []
    per_pos_sizes = {pos: [] for pos in range(n_pos)}

    for i in range(n_samples):
        t = int(test_indices[i])
        true_vals = [int(data.red_matrix[t, pos]) for pos in range(n_pos)]

        # 记录每位置候选集大小
        for pos in range(n_pos):
            per_pos_sizes[pos].append(len(candidates[i][pos]))

        # DP 计算组合数
        n_combos = count_ordered_combos(candidates[i])
        combo_counts.append(n_combos)

        # 存活检查
        survived = check_combo_survival(candidates[i], true_vals)
        survivals.append(1.0 if survived else 0.0)

        if (i + 1) % 50 == 0:
            log(f"    进度: {i+1}/{n_samples}, 当前平均组合数: {np.mean(combo_counts):,.0f}")

    combo_counts = np.array(combo_counts, dtype=np.float64)
    survivals = np.array(survivals)

    # 汇总统计
    avg_combos = float(np.mean(combo_counts))
    median_combos = float(np.median(combo_counts))
    combo_reduction = 1.0 - avg_combos / baseline_combos
    combo_survival = float(np.mean(survivals))
    combo_efficiency = combo_reduction * combo_survival
    probability_uplift = baseline_combos / avg_combos if avg_combos > 0 else 0

    # 每位置统计
    per_pos_stats = {}
    for pos in range(n_pos):
        sizes = np.array(per_pos_sizes[pos])
        per_pos_stats[f'P{pos}'] = {
            'avg_size': float(np.mean(sizes)),
            'avg_reduction': float(1.0 - np.mean(sizes) / data.red_range),
        }

    # 实用性评估（仅红球部分）
    avg_tickets = avg_combos
    cost_per_draw = avg_tickets * 2  # 每注2元

    # 大乐透一等奖约1000万，双色球约500万
    jackpot = 10_000_000 if lottery_type == 'daletou' else 5_000_000
    # 蓝球全组合数
    blue_combos = comb(data.blue_range, data.blue_count)
    # 完整中奖概率 = combo_survival / (avg_combos * blue_combos)
    # 基线中奖概率 = 1 / (baseline_combos * blue_combos)
    total_baseline = baseline_combos * blue_combos

    results = {
        'lottery_type': lottery_type,
        'best_method': best_method,
        'n_test_samples': n_samples,
        'baseline': {
            'red_combos': baseline_combos,
            'blue_combos': blue_combos,
            'total_combos': total_baseline,
        },
        'combo_stats': {
            'avg_combo_count': avg_combos,
            'median_combo_count': median_combos,
            'min_combo_count': float(np.min(combo_counts)),
            'max_combo_count': float(np.max(combo_counts)),
            'std_combo_count': float(np.std(combo_counts)),
            'avg_combo_reduction': combo_reduction,
            'combo_survival_rate': combo_survival,
            'combo_efficiency': combo_efficiency,
        },
        'practical_value': {
            'avg_tickets_needed': avg_tickets,
            'cost_per_draw_yuan': cost_per_draw,
            'probability_uplift_red': probability_uplift,
            'jackpot_yuan': jackpot,
            'note': '仅评估红球部分，蓝球未纳入缩号',
        },
        'per_position': per_pos_stats,
        'distribution': {
            'combo_count_percentiles': {
                'p10': float(np.percentile(combo_counts, 10)),
                'p25': float(np.percentile(combo_counts, 25)),
                'p50': float(np.percentile(combo_counts, 50)),
                'p75': float(np.percentile(combo_counts, 75)),
                'p90': float(np.percentile(combo_counts, 90)),
            },
        },
        'consistency_check': {
            'position_survival_product': None,
            'combo_survival': combo_survival,
            'note': '组合存活率应 >= 各位置存活率乘积（独立下界）',
        },
    }

    # 一致性检查：计算各位置独立存活率的乘积
    pos_survivals = []
    for i in range(n_samples):
        t = int(test_indices[i])
        for pos in range(n_pos):
            true_val = int(data.red_matrix[t, pos])
            # 这里不逐位置统计了，直接用 phase3 的逻辑
    # 简化：从 candidates 直接算各位置存活率
    pos_surv_rates = []
    for pos in range(n_pos):
        surv = 0
        for i in range(n_samples):
            t = int(test_indices[i])
            true_val = int(data.red_matrix[t, pos])
            if true_val in candidates[i][pos]:
                surv += 1
        pos_surv_rates.append(surv / n_samples)

    pos_surv_product = 1.0
    for r in pos_surv_rates:
        pos_surv_product *= r

    results['consistency_check']['position_survival_rates'] = {
        f'P{pos}': pos_surv_rates[pos] for pos in range(n_pos)
    }
    results['consistency_check']['position_survival_product'] = pos_surv_product

    return results


def print_report(results):
    """打印评估报告"""
    lt = results['lottery_type']
    log(f"\n{'═'*55}")
    log(f"  E1 组合缩号评估报告: {lt}")
    log(f"{'═'*55}")

    bs = results['baseline']
    log(f"\n  基线:")
    log(f"    红球组合: {bs['red_combos']:,}")
    log(f"    蓝球组合: {bs['blue_combos']:,}")
    log(f"    总组合:   {bs['total_combos']:,}")

    cs = results['combo_stats']
    log(f"\n  组合级指标:")
    log(f"    平均组合数:   {cs['avg_combo_count']:,.0f}")
    log(f"    中位数组合数: {cs['median_combo_count']:,.0f}")
    log(f"    最小组合数:   {cs['min_combo_count']:,.0f}")
    log(f"    最大组合数:   {cs['max_combo_count']:,.0f}")
    log(f"    组合缩减率:   {cs['avg_combo_reduction']:.2%}")
    log(f"    组合存活率:   {cs['combo_survival_rate']:.2%}")
    log(f"    组合效率:     {cs['combo_efficiency']:.4f}")

    pv = results['practical_value']
    log(f"\n  实用性评估（仅红球）:")
    log(f"    平均需购注数: {pv['avg_tickets_needed']:,.0f}")
    log(f"    每期成本:     {pv['cost_per_draw_yuan']:,.0f} 元")
    log(f"    概率提升倍数: {pv['probability_uplift_red']:.2f}x")

    log(f"\n  每位置候选集:")
    for pos_key, ps in results['per_position'].items():
        log(f"    {pos_key}: 平均大小={ps['avg_size']:.1f}, 缩减率={ps['avg_reduction']:.2%}")

    dist = results['distribution']['combo_count_percentiles']
    log(f"\n  组合数分布:")
    log(f"    P10={dist['p10']:,.0f}  P25={dist['p25']:,.0f}  P50={dist['p50']:,.0f}  P75={dist['p75']:,.0f}  P90={dist['p90']:,.0f}")

    cc = results['consistency_check']
    log(f"\n  一致性检查:")
    log(f"    组合存活率:         {cc['combo_survival']:.4f}")
    log(f"    位置存活率乘积:     {cc['position_survival_product']:.4f}")
    if cc['position_survival_rates']:
        rates = [f"{k}={v:.4f}" for k, v in cc['position_survival_rates'].items()]
        log(f"    各位置存活率:       {', '.join(rates)}")

    # 实用性判断
    avg_tickets = pv['avg_tickets_needed']
    log(f"\n  实用性判断:")
    if avg_tickets > 100_000:
        log(f"    不实用（平均 {avg_tickets:,.0f} 注，每期成本 {avg_tickets*2:,.0f} 元）")
    elif avg_tickets > 10_000:
        log(f"    理论有效但成本高（平均 {avg_tickets:,.0f} 注，每期成本 {avg_tickets*2:,.0f} 元）")
    else:
        log(f"    有一定实用价值（平均 {avg_tickets:,.0f} 注，每期成本 {avg_tickets*2:,.0f} 元）")


# === 主入口 ===

def main():
    STRICT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # 配置日志
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STRICT_EXPERIMENT_DIR
    setup_logging()
    exp_utils.EXPERIMENT_DIR = original_exp_dir

    start_time = time.time()

    log("=" * 60)
    log("  E1：端到端组合缩号评估")
    log("=" * 60)

    all_results = {}

    for lottery_type in ["daletou", "shuangseqiu"]:
        try:
            results = evaluate_combo(lottery_type)
            all_results[lottery_type] = results

            # 保存单彩种结果
            save_json(results,
                      STRICT_EXPERIMENT_DIR / f"e1_combo_evaluation_{lottery_type}.json")

            # 打印报告
            print_report(results)

        except Exception as e:
            log(f"\n[错误] {lottery_type} 评估失败: {e}")
            log(traceback.format_exc())
            all_results[lottery_type] = {"error": str(e)}

    # 汇总
    total_time = time.time() - start_time
    summary = {
        'total_time_seconds': round(total_time, 1),
        'total_time_minutes': round(total_time / 60, 1),
        'results': all_results,
    }
    save_json(summary, STRICT_EXPERIMENT_DIR / "e1_summary.json")

    log(f"\n{'═'*60}")
    log(f"  E1 评估完成！总耗时: {summary['total_time_minutes']:.1f} 分钟")
    log(f"  结果目录: {STRICT_EXPERIMENT_DIR}")
    log(f"{'═'*60}")


if __name__ == "__main__":
    main()
