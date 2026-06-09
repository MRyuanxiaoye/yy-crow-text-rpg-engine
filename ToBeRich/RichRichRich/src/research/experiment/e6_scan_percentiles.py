"""
E6 Step2: 约束宽度扫描 — 找最优分位数
"""
import sys, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.data_loader import LotteryData
from research.experiment.e6_constraint_propagation import (
    compute_constraints, candidate_by_pos_diff,
    count_valid_combos_dp, SPLIT, RESULTS_DIR
)
from math import comb


def scan_percentiles(lottery_type="daletou"):
    """扫描不同分位数宽度，找最优平衡点"""
    print(f"\n{'='*60}")
    print(f"  E6 分位数扫描: {lottery_type}")
    print(f"{'='*60}\n")

    data = LotteryData(lottery_type)
    cfg = SPLIT[lottery_type]
    train_end = cfg["train_end"]
    red_range = data.red_range
    n_pos = data.red_count
    total_combos = comb(red_range, n_pos)

    # 测试期
    test_start = train_end + 1
    test_end = min(test_start + 100, cfg["total"])
    test_indices = list(range(test_start, test_end))

    # 扫描不同分位数
    percentile_pairs = [
        (1, 99), (2, 98), (3, 97), (5, 95), (7, 93), (10, 90)
    ]

    all_results = []

    for lo_p, hi_p in percentile_pairs:
        cons = compute_constraints(data, train_end, percentiles=(lo_p, hi_p))

        valid_counts = []
        combo_survivals = []

        for idx in test_indices:
            prev = data.red_matrix[idx - 1]
            true = data.red_matrix[idx]

            cands = candidate_by_pos_diff(prev, cons, red_range)
            vc, _ = count_valid_combos_dp(cands, cons, red_range)
            valid_counts.append(vc)

            # 检查组合存活
            survived = all(true[p] in cands[p] for p in range(n_pos))
            if survived:
                span = true[-1] - true[0]
                s = int(sum(true))
                survived = (cons["span"][0] <= span <= cons["span"][1] and
                           cons["sum"][0] <= s <= cons["sum"][1])
                if survived:
                    for g in range(n_pos - 1):
                        gap = true[g+1] - true[g]
                        if not (cons["gap"][g][0] <= gap <= cons["gap"][g][1]):
                            survived = False
                            break
            combo_survivals.append(survived)

        avg_vc = int(np.mean(valid_counts))
        reduction = 1.0 - avg_vc / total_combos
        survival = sum(combo_survivals) / len(combo_survivals)
        efficiency = reduction * survival

        row = {
            "percentiles": f"{lo_p}-{hi_p}",
            "avg_combos": avg_vc,
            "reduction": round(reduction, 4),
            "survival": round(survival, 4),
            "efficiency": round(efficiency, 4),
        }
        all_results.append(row)

        print(f"  [{lo_p:2d}%-{hi_p:2d}%] "
              f"候选={avg_vc:>9,}  "
              f"缩减={reduction*100:5.1f}%  "
              f"存活={survival*100:5.1f}%  "
              f"效率={efficiency:.4f}")

    # 保存
    out = {"lottery_type": lottery_type, "total_combos": total_combos,
           "scan_results": all_results}
    path = RESULTS_DIR / f"e6_scan_{lottery_type}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {path.name}")
    return out


if __name__ == "__main__":
    lt = sys.argv[1] if len(sys.argv) > 1 else "daletou"
    scan_percentiles(lt)
