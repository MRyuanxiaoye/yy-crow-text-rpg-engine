"""
E6: 约束传播缩号实验
核心思路：不预测具体号码，而是用多重约束条件缩小候选空间
"""
import sys, json, time
import numpy as np
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.data_loader import LotteryData

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "e6_constraint"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 时间切分
SPLIT = {
    "daletou":      {"train_end": 2400, "total": 2833},
    "shuangseqiu":  {"train_end": 2900, "total": 3413},
}


def compute_constraints(data: LotteryData, train_end: int, percentiles=(5, 95)):
    """从训练期数据计算所有约束的区间范围"""
    R = data.red_matrix[:train_end]   # (train_end, n_pos)
    n_pos = data.red_count
    lo, hi = percentiles

    constraints = {}

    # --- 约束1: 组合跨度 (max - min) ---
    spans = R[:, -1] - R[:, 0]
    constraints["span"] = (float(np.percentile(spans, lo)),
                           float(np.percentile(spans, hi)))

    # --- 约束2: 同位置差分 (本期 - 上期) ---
    diffs = R[1:] - R[:-1]  # (train_end-1, n_pos)
    constraints["pos_diff"] = {}
    for p in range(n_pos):
        d = diffs[:, p]
        constraints["pos_diff"][p] = (float(np.percentile(d, lo)),
                                      float(np.percentile(d, hi)))

    # --- 约束3: 相邻位置间距 (P_{i+1} - P_i) ---
    gaps = np.diff(R, axis=1)  # (train_end, n_pos-1)
    constraints["gap"] = {}
    for g in range(n_pos - 1):
        col = gaps[:, g]
        constraints["gap"][g] = (int(np.percentile(col, lo)),
                                 int(np.percentile(col, hi)))

    # --- 约束4: 和值 ---
    sums = R.sum(axis=1)
    constraints["sum"] = (float(np.percentile(sums, lo)),
                          float(np.percentile(sums, hi)))

    # --- 约束5: 两两位置差值 (Pi - Pj, j<i) ---
    constraints["pair_diff"] = {}
    for i in range(n_pos):
        for j in range(i + 1, n_pos):
            d = R[:, j] - R[:, i]
            constraints["pair_diff"][(i, j)] = (int(np.percentile(d, lo)),
                                                 int(np.percentile(d, hi)))

    return constraints


def candidate_by_pos_diff(prev_vals, constraints, red_range):
    """约束2: 根据同位置差分范围，算出每个位置的候选号码"""
    n_pos = len(prev_vals)
    candidates = []
    for p in range(n_pos):
        lo, hi = constraints["pos_diff"][p]
        v_min = max(1, int(prev_vals[p] + lo))
        v_max = min(red_range, int(prev_vals[p] + hi))
        candidates.append(set(range(v_min, v_max + 1)))
    return candidates


def count_valid_combos_dp(candidates, constraints, red_range):
    """
    用 DP 精确计数满足所有约束的合法组合数。
    约束: 排序(P0<P1<...), 跨度, 间距, 和值, 两两差值
    """
    n_pos = len(candidates)
    span_lo, span_hi = constraints["span"]
    sum_lo, sum_hi = constraints["sum"]

    # 转为排序列表
    cands = [sorted(c) for c in candidates]

    # 暴力枚举（对于缩减后的候选集，通常可行）
    count = 0
    total_checked = 0

    # 用递归+剪枝
    def backtrack(pos, chosen, current_sum):
        nonlocal count, total_checked
        if pos == n_pos:
            # 检查跨度约束
            span = chosen[-1] - chosen[0]
            if span_lo <= span <= span_hi:
                # 检查和值约束
                if sum_lo <= current_sum <= sum_hi:
                    count += 1
            total_checked += 1
            return

        min_val = chosen[-1] + 1 if chosen else 1
        for v in cands[pos]:
            if v < min_val:
                continue

            # 间距剪枝
            if pos > 0:
                gap = v - chosen[-1]
                g_lo, g_hi = constraints["gap"][pos - 1]
                if gap < g_lo or gap > g_hi:
                    continue

            # 两两差值剪枝
            prune = False
            for prev_p in range(len(chosen)):
                diff = v - chosen[prev_p]
                key = (prev_p, pos)
                if key in constraints["pair_diff"]:
                    d_lo, d_hi = constraints["pair_diff"][key]
                    if diff < d_lo or diff > d_hi:
                        prune = True
                        break
            if prune:
                continue

            # 和值上界剪枝: 剩余位置至少各加 v+1, v+2, ...
            remaining = n_pos - pos - 1
            min_remaining_sum = sum(v + k for k in range(1, remaining + 1))
            if current_sum + v + min_remaining_sum > sum_hi:
                continue

            # 和值下界剪枝
            max_remaining_sum = sum(red_range - k for k in range(remaining))
            if current_sum + v + max_remaining_sum < sum_lo:
                continue

            backtrack(pos + 1, chosen + [v], current_sum + v)

    backtrack(0, [], 0)
    return count, total_checked


def run_experiment(lottery_type="daletou", n_test_samples=50):
    """运行约束传播实验"""
    print(f"\n{'='*60}")
    print(f"  E6 约束传播实验: {lottery_type}")
    print(f"{'='*60}\n")

    data = LotteryData(lottery_type)
    cfg = SPLIT[lottery_type]
    train_end = cfg["train_end"]
    red_range = data.red_range

    # 计算全组合数 C(red_range, n_pos)
    from math import comb
    total_combos = comb(red_range, data.red_count)
    print(f"全组合空间: C({red_range},{data.red_count}) = {total_combos:,}")

    # 从训练期计算约束
    print(f"\n[1] 计算约束区间 (训练期 0-{train_end})...")
    cons = compute_constraints(data, train_end, percentiles=(5, 95))

    print(f"  跨度约束: [{cons['span'][0]:.0f}, {cons['span'][1]:.0f}]")
    print(f"  和值约束: [{cons['sum'][0]:.0f}, {cons['sum'][1]:.0f}]")
    for p in range(data.red_count):
        lo, hi = cons["pos_diff"][p]
        print(f"  P{p} 差分约束: [{lo:.0f}, {hi:.0f}]")
    for g in range(data.red_count - 1):
        lo, hi = cons["gap"][g]
        print(f"  P{g}-P{g+1} 间距约束: [{lo}, {hi}]")

    # 测试期评估
    test_start = train_end + 1
    test_end = min(test_start + n_test_samples, cfg["total"])
    test_indices = list(range(test_start, test_end))

    print(f"\n[2] 测试期评估 ({len(test_indices)} 期)...")

    results = []
    for idx in test_indices:
        prev = data.red_matrix[idx - 1]  # 上一期号码
        true = data.red_matrix[idx]       # 本期真实号码

        # 每位置候选号码（仅用差分约束）
        cands = candidate_by_pos_diff(prev, cons, red_range)
        cand_sizes = [len(c) for c in cands]

        # 精确计数满足所有约束的组合
        t0 = time.time()
        valid_count, checked = count_valid_combos_dp(cands, cons, red_range)
        elapsed = time.time() - t0

        # 检查真实号码是否存活
        survived = True
        for p in range(data.red_count):
            if true[p] not in cands[p]:
                survived = False
                break

        # 如果位置级存活，还要检查组合级约束
        combo_survived = False
        if survived:
            span = true[-1] - true[0]
            s = sum(true)
            combo_survived = (cons["span"][0] <= span <= cons["span"][1] and
                              cons["sum"][0] <= s <= cons["sum"][1])
            if combo_survived:
                for g in range(data.red_count - 1):
                    gap = true[g + 1] - true[g]
                    g_lo, g_hi = cons["gap"][g]
                    if gap < g_lo or gap > g_hi:
                        combo_survived = False
                        break

        reduction = 1.0 - valid_count / total_combos if total_combos > 0 else 0

        results.append({
            "idx": idx,
            "prev": prev.tolist(),
            "true": true.tolist(),
            "cand_sizes": cand_sizes,
            "valid_combos": valid_count,
            "reduction_rate": reduction,
            "pos_survived": survived,
            "combo_survived": combo_survived,
            "time_s": round(elapsed, 2),
        })

        if (len(results) % 10 == 0) or len(results) <= 3:
            print(f"  期{idx}: 候选{cand_sizes} → {valid_count:,}组合 "
                  f"(缩减{reduction*100:.2f}%) "
                  f"存活={'✓' if combo_survived else '✗'} "
                  f"({elapsed:.1f}s)")

    # 汇总
    valid_counts = [r["valid_combos"] for r in results]
    reductions = [r["reduction_rate"] for r in results]
    pos_survivals = [r["pos_survived"] for r in results]
    combo_survivals = [r["combo_survived"] for r in results]

    summary = {
        "lottery_type": lottery_type,
        "total_combos": total_combos,
        "constraints": {
            "span": cons["span"],
            "sum": cons["sum"],
            "pos_diff": {str(k): v for k, v in cons["pos_diff"].items()},
            "gap": {str(k): v for k, v in cons["gap"].items()},
        },
        "test_samples": len(results),
        "avg_valid_combos": int(np.mean(valid_counts)),
        "median_valid_combos": int(np.median(valid_counts)),
        "min_valid_combos": int(np.min(valid_counts)),
        "max_valid_combos": int(np.max(valid_counts)),
        "avg_reduction_rate": round(float(np.mean(reductions)), 4),
        "pos_survival_rate": round(sum(pos_survivals) / len(pos_survivals), 4),
        "combo_survival_rate": round(sum(combo_survivals) / len(combo_survivals), 4),
        "efficiency": round(float(np.mean(reductions)) *
                           (sum(combo_survivals) / len(combo_survivals)), 4),
    }

    print(f"\n{'='*60}")
    print(f"  汇总结果")
    print(f"{'='*60}")
    print(f"  全组合空间:     {total_combos:>12,}")
    print(f"  平均候选组合:   {summary['avg_valid_combos']:>12,}")
    print(f"  中位候选组合:   {summary['median_valid_combos']:>12,}")
    print(f"  平均缩减率:     {summary['avg_reduction_rate']*100:>11.2f}%")
    print(f"  位置存活率:     {summary['pos_survival_rate']*100:>11.2f}%")
    print(f"  组合存活率:     {summary['combo_survival_rate']*100:>11.2f}%")
    print(f"  综合效率:       {summary['efficiency']:>11.4f}")

    # 保存
    out_path = RESULTS_DIR / f"e6_results_{lottery_type}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path.name}")

    return summary


if __name__ == "__main__":
    lt = sys.argv[1] if len(sys.argv) > 1 else "daletou"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    run_experiment(lt, n)
