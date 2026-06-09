# -*- coding: utf-8 -*-
"""E9b: 直接走势匹配实验

思路：要预测第t期，拿t-1,t-2,...,t-w期的走势（方向序列），
在历史中找到每个位置走势都完全一样的片段z,z-1,...,z-w，
然后看z+1的走势，作为预测。

回测窗口从3期到7期，看哪个窗口最合适。

用法: python3 -m src.research.experiment.e9b_trend_match
"""

import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "e9_cross_position"
BACKTEST_HOLD = 200


def build_diff_matrix(data: LotteryData) -> np.ndarray:
    n_pos = data.red_count
    n = data.n_draws - 1
    diff_mat = np.zeros((n, n_pos), dtype=np.int32)
    for pos in range(n_pos):
        diff_mat[:, pos] = np.diff(data.position_series[pos])
    return diff_mat


def encode_direction(diff_val: int) -> int:
    """升=1, 平=0, 降=-1"""
    if diff_val > 0: return 1
    elif diff_val < 0: return -1
    return 0


def encode_window(diff_matrix: np.ndarray, t: int, window: int, n_pos: int):
    """
    编码从 t-window+1 到 t 这 window 期的方向走势。
    返回 tuple，长度 = window * n_pos
    顺序：从远到近，即 (t-window+1的P0方向, P1方向, ..., t-window+2的P0方向, ..., t的P0方向, ...)
    """
    if t - window + 1 < 0:
        return None
    parts = []
    for lag in range(window - 1, -1, -1):  # 从远到近
        for p in range(n_pos):
            parts.append(encode_direction(int(diff_matrix[t - lag, p])))
    return tuple(parts)


def run_experiment(lottery_type: str):
    print(f"\n{'='*60}")
    print(f"  E9b 走势匹配: {lottery_type}")
    print(f"{'='*60}")

    data = LotteryData(lottery_type)
    diff_matrix = build_diff_matrix(data)
    n_pos = data.red_count
    n_periods = diff_matrix.shape[0]
    stat_end = n_periods - BACKTEST_HOLD

    print(f"  总差分期数: {n_periods}, 统计期: {stat_end}, 回测期: {BACKTEST_HOLD}")
    print(f"  位置数: {n_pos}, 号码范围: 1~{data.red_range}")

    windows = [3, 4, 5, 6, 7]
    all_results = {}

    for window in windows:
        print(f"\n{'─'*50}")
        print(f"  窗口 = {window} 期")
        print(f"{'─'*50}")

        t0 = time.time()

        # 构建历史走势索引：走势编码 -> [出现的时间点t列表]
        # t 表示窗口最后一期的位置
        index = defaultdict(list)
        for t in range(window - 1, stat_end):
            code = encode_window(diff_matrix, t, window, n_pos)
            if code is not None:
                index[code].append(t)

        n_patterns = len(index)
        counts = [len(v) for v in index.values()]
        print(f"  不同走势模式数: {n_patterns}")
        print(f"  每模式出现次数: min={min(counts)}, median={int(np.median(counts))}, "
              f"max={max(counts)}, 单次占比={sum(1 for c in counts if c==1)/len(counts)*100:.1f}%")

        # 回测
        test_start = stat_end
        test_end = n_periods - 1  # 需要 t+1 存在

        dir_correct_all = []  # 每期6个位置的方向正确数
        all_dir_correct = []  # 每期是否全对
        mae_all = []
        rank_all = []
        match_counts = []
        no_match = 0
        per_pos_dir = {p: [] for p in range(n_pos)}
        per_pos_mae = {p: [] for p in range(n_pos)}
        per_pos_rank = {p: [] for p in range(n_pos)}

        for t in range(test_start, test_end):
            # 当前走势编码
            code = encode_window(diff_matrix, t, window, n_pos)
            if code is None:
                continue

            # 查找历史匹配（只在统计期内，且匹配期的下一期也在统计期内）
            matches = [h for h in index.get(code, []) if h < stat_end - 1]

            true_next = diff_matrix[t + 1]

            if not matches:
                no_match += 1
                # 无匹配时，用全局方向频率作为 fallback
                for p in range(n_pos):
                    true_d = int(true_next[p])
                    true_dir = encode_direction(true_d)
                    per_pos_dir[p].append(0)  # 算错
                    per_pos_mae[p].append(abs(true_d))
                    per_pos_rank[p].append(data.red_range)
                dir_correct_all.append(0)
                all_dir_correct.append(False)
                mae_all.append(np.mean([abs(int(true_next[p])) for p in range(n_pos)]))
                rank_all.append(data.red_range)
                match_counts.append(0)
                continue

            match_counts.append(len(matches))

            # 收集匹配期的下一期差分
            next_diffs = diff_matrix[[h + 1 for h in matches]]  # shape: (n_matches, n_pos)

            period_dir_correct = 0
            period_mae = []
            period_ranks = []

            for p in range(n_pos):
                true_d = int(true_next[p])
                true_dir = encode_direction(true_d)

                # 方向预测：多数投票
                next_dirs = [encode_direction(int(nd)) for nd in next_diffs[:, p]]
                dir_votes = {-1: 0, 0: 0, 1: 0}
                for d in next_dirs:
                    dir_votes[d] += 1
                pred_dir = max(dir_votes, key=dir_votes.get)

                is_correct = (pred_dir == true_dir)
                if is_correct:
                    period_dir_correct += 1
                per_pos_dir[p].append(1 if is_correct else 0)

                # 幅度预测：均值
                pred_diff = np.mean(next_diffs[:, p])
                mae = abs(true_d - pred_diff)
                period_mae.append(mae)
                per_pos_mae[p].append(mae)

                # 号码排名
                current_val = data.position_series[p][t + 1]
                number_counts = defaultdict(int)
                for nd in next_diffs[:, p]:
                    candidate = current_val + int(nd)
                    if 1 <= candidate <= data.red_range:
                        number_counts[candidate] += 1
                total_nc = sum(number_counts.values())

                true_val = data.position_series[p][t + 2]
                if total_nc > 0 and true_val in number_counts:
                    true_cnt = number_counts[true_val]
                    rank = sum(1 for v in number_counts.values() if v > true_cnt) + 1
                else:
                    rank = data.red_range
                period_ranks.append(rank)
                per_pos_rank[p].append(rank)

            dir_correct_all.append(period_dir_correct)
            all_dir_correct.append(period_dir_correct == n_pos)
            mae_all.append(np.mean(period_mae))
            rank_all.append(np.mean(period_ranks))

        n_test = len(dir_correct_all)
        avg_dir_acc = np.mean(dir_correct_all) / n_pos
        avg_all_correct = np.mean(all_dir_correct)
        avg_mae = np.mean(mae_all)
        avg_rank = np.mean(rank_all)
        matched = [c for c in match_counts if c > 0]

        print(f"\n  回测结果 ({n_test}期):")
        print(f"    无匹配期数: {no_match} ({no_match/n_test*100:.1f}%)")
        if matched:
            print(f"    有匹配时平均匹配数: {np.mean(matched):.1f}, "
                  f"中位数: {int(np.median(matched))}, 最大: {max(matched)}")
        print(f"    方向准确率: {avg_dir_acc:.4f}")
        print(f"    全对率: {avg_all_correct:.4f}")
        print(f"    平均MAE: {avg_mae:.2f}")
        print(f"    平均rank: {avg_rank:.2f}")

        # 分位置
        print(f"\n    分位置方向准确率:")
        for p in range(n_pos):
            acc = np.mean(per_pos_dir[p])
            mae_p = np.mean(per_pos_mae[p])
            rank_p = np.mean(per_pos_rank[p])
            print(f"      P{p}: 方向={acc:.4f}, MAE={mae_p:.2f}, rank={rank_p:.2f}")

        # 只看有匹配的期
        if no_match > 0 and no_match < n_test:
            matched_idx = [i for i, c in enumerate(match_counts) if c > 0]
            print(f"\n    仅有匹配期 ({len(matched_idx)}期):")
            m_dir = np.mean([dir_correct_all[i] for i in matched_idx]) / n_pos
            m_all = np.mean([all_dir_correct[i] for i in matched_idx])
            m_mae = np.mean([mae_all[i] for i in matched_idx])
            m_rank = np.mean([rank_all[i] for i in matched_idx])
            print(f"      方向准确率: {m_dir:.4f}")
            print(f"      全对率: {m_all:.4f}")
            print(f"      平均MAE: {m_mae:.2f}")
            print(f"      平均rank: {m_rank:.2f}")

        print(f"    耗时: {time.time()-t0:.2f}s")

        all_results[window] = {
            "n_test": n_test,
            "no_match": no_match,
            "no_match_pct": round(no_match / n_test * 100, 1),
            "avg_match_count": round(np.mean(matched), 1) if matched else 0,
            "direction_accuracy": round(avg_dir_acc, 4),
            "all_correct_rate": round(avg_all_correct, 4),
            "avg_mae": round(avg_mae, 2),
            "avg_rank": round(avg_rank, 2),
        }

    # 汇总对比
    print(f"\n{'='*60}")
    print(f"  E9b 汇总: {lottery_type}")
    print(f"{'='*60}")
    print(f"\n  {'窗口':<6} {'无匹配%':<10} {'平均匹配数':<12} {'方向准确率':<12} {'全对率':<10} {'MAE':<8} {'rank':<8}")
    print(f"  {'─'*66}")
    for w, r in sorted(all_results.items()):
        print(f"  {w:<6} {r['no_match_pct']:<10} {r['avg_match_count']:<12} "
              f"{r['direction_accuracy']:<12.4f} {r['all_correct_rate']:<10.4f} "
              f"{r['avg_mae']:<8} {r['avg_rank']:<8}")

    return all_results


def main():
    for lottery_type in ["shuangseqiu", "daletou"]:
        run_experiment(lottery_type)


if __name__ == "__main__":
    main()
