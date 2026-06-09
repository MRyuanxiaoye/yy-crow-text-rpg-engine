# -*- coding: utf-8 -*-
"""E9c: 放松匹配走势实验

思路：要预测第t期，拿最近w期的方向走势，在历史中找相似片段。
不要求所有位置都一样，而是计算"有多少个位置方向一致"作为相似度。
取最相似的top-K个历史期，用它们的下一期变化来预测。

相似度 = 在w期窗口内，所有(期, 位置)对中方向一致的比例。

用法: python3 -m src.research.experiment.e9c_relaxed_match
"""

import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData

BACKTEST_HOLD = 200


def build_diff_matrix(data: LotteryData) -> np.ndarray:
    n_pos = data.red_count
    n = data.n_draws - 1
    diff_mat = np.zeros((n, n_pos), dtype=np.int32)
    for pos in range(n_pos):
        diff_mat[:, pos] = np.diff(data.position_series[pos])
    return diff_mat


def direction_matrix(diff_matrix: np.ndarray) -> np.ndarray:
    """将差分矩阵转为方向矩阵：1=升, 0=平, -1=降"""
    return np.sign(diff_matrix).astype(np.int8)


def run_experiment(lottery_type: str):
    print(f"\n{'='*60}")
    print(f"  E9c 放松匹配: {lottery_type}")
    print(f"{'='*60}")

    data = LotteryData(lottery_type)
    diff_matrix = build_diff_matrix(data)
    dir_matrix = direction_matrix(diff_matrix)
    n_pos = data.red_count
    n_periods = diff_matrix.shape[0]
    stat_end = n_periods - BACKTEST_HOLD

    print(f"  差分期数: {n_periods}, 统计期: {stat_end}, 回测: {BACKTEST_HOLD}")
    print(f"  位置数: {n_pos}, 号码范围: 1~{data.red_range}")

    windows = [3, 5]
    top_ks = [5, 10, 20, 50]
    all_results = {}

    for window in windows:
        print(f"\n{'─'*50}")
        print(f"  窗口 = {window} 期")
        print(f"{'─'*50}")

        t0 = time.time()

        # 预计算统计期内每个时间点的方向窗口矩阵
        # hist_windows[h] = dir_matrix[h-w+1 : h+1] shape=(w, n_pos)
        # 相似度 = 两个窗口中方向一致的(期,位置)对数，总共 w*n_pos 个
        total_cells = window * n_pos

        # 回测
        test_start = stat_end
        test_end = n_periods - 1

        results_by_k = {k: [] for k in top_ks}

        for t in range(test_start, test_end):
            if t - window + 1 < 0:
                continue

            # 当前窗口 shape=(window, n_pos)
            cur_window = dir_matrix[t - window + 1: t + 1]

            # 计算和所有历史期的相似度
            # 历史期 h 的窗口 = dir_matrix[h-w+1 : h+1]
            # 向量化：构建历史窗口矩阵并批量比较
            min_h = window - 1
            max_h = stat_end - 1  # h+1 必须存在

            # 用滑动窗口向量化
            n_hist = max_h - min_h + 1
            # 构建所有历史窗口 shape=(n_hist, window, n_pos)
            hist_windows = np.lib.stride_tricks.sliding_window_view(
                dir_matrix[:stat_end], window, axis=0
            )  # shape=(stat_end - window + 1, n_pos, window)
            # 转置为 (n_hist, window, n_pos)
            hist_windows = hist_windows.transpose(0, 2, 1)

            # 比较：当前窗口 vs 所有历史窗口
            # matches shape=(n_hist, window, n_pos) bool
            matches = (hist_windows == cur_window[np.newaxis, :, :])
            # 每个历史期的相似度 = 匹配的cell数
            sim_scores = matches.reshape(n_hist, -1).sum(axis=1)  # shape=(n_hist,)

            # 排序取 top-K
            # 注意：hist_windows[i] 对应的时间点 h = i + window - 1
            sorted_idx = np.argsort(-sim_scores)

            true_next = diff_matrix[t + 1]

            for top_k in top_ks:
                top_idx = sorted_idx[:top_k]
                top_h = top_idx + window - 1  # 转回时间点
                top_sim = sim_scores[top_idx]
                max_sim = int(top_sim[0])

                # 用这些历史期的下一期差分来预测
                dir_correct = 0
                abs_errors = []
                per_pos = {}

                for p in range(n_pos):
                    true_d = int(true_next[p])
                    true_dir = 1 if true_d > 0 else (-1 if true_d < 0 else 0)

                    # 加权收集差分分布
                    diff_dist = defaultdict(float)
                    for idx, sim in zip(top_h, top_sim):
                        w = float(sim)
                        nd = int(diff_matrix[idx + 1, p])
                        diff_dist[nd] += w

                    total_w = sum(diff_dist.values())

                    # 方向
                    dir_w = {-1: 0.0, 0: 0.0, 1: 0.0}
                    for val, wt in diff_dist.items():
                        if val < 0: dir_w[-1] += wt
                        elif val == 0: dir_w[0] += wt
                        else: dir_w[1] += wt

                    pred_dir = max(dir_w, key=dir_w.get) if total_w > 0 else 0
                    is_correct = (pred_dir == true_dir)
                    if is_correct:
                        dir_correct += 1

                    # 幅度
                    pred_diff = sum(k * v for k, v in diff_dist.items()) / total_w if total_w > 0 else 0
                    mae = abs(true_d - pred_diff)
                    abs_errors.append(mae)

                    # 号码排名
                    current_val = data.position_series[p][t + 1]
                    number_counts = defaultdict(float)
                    for dv_val, dv_cnt in diff_dist.items():
                        candidate = current_val + int(dv_val)
                        if 1 <= candidate <= data.red_range:
                            number_counts[candidate] += dv_cnt
                    total_nc = sum(number_counts.values())

                    true_val = data.position_series[p][t + 2]
                    if total_nc > 0 and true_val in number_counts:
                        true_cnt = number_counts[true_val]
                        rank = sum(1 for v in number_counts.values() if v > true_cnt) + 1
                    else:
                        rank = data.red_range

                    per_pos[p] = {
                        "dir_correct": is_correct,
                        "mae": mae,
                        "rank": rank,
                    }

                results_by_k[top_k].append({
                    "t": t,
                    "max_sim": max_sim,
                    "sim_ratio": max_sim / total_cells,
                    "dir_accuracy": dir_correct / n_pos,
                    "all_dir_correct": dir_correct == n_pos,
                    "avg_mae": float(np.mean(abs_errors)),
                    "avg_rank": float(np.mean([per_pos[p]["rank"] for p in range(n_pos)])),
                    "per_position": per_pos,
                })

            # 进度
            if (t - test_start) % 50 == 0:
                print(f"    进度: {t - test_start}/{test_end - test_start}")

        # 汇总
        print(f"\n    耗时: {time.time()-t0:.1f}s")
        print(f"\n    {'top_k':<8} {'方向准确率':<12} {'全对率':<10} {'MAE':<8} {'rank':<8} {'最高相似度比':<14}")
        print(f"    {'─'*60}")

        window_results = {}
        for k in top_ks:
            periods = results_by_k[k]
            if not periods:
                continue
            s = {
                "direction_accuracy": round(np.mean([r["dir_accuracy"] for r in periods]), 4),
                "all_correct_rate": round(np.mean([r["all_dir_correct"] for r in periods]), 4),
                "avg_mae": round(np.mean([r["avg_mae"] for r in periods]), 2),
                "avg_rank": round(np.mean([r["avg_rank"] for r in periods]), 2),
                "avg_max_sim_ratio": round(np.mean([r["sim_ratio"] for r in periods]), 4),
                "per_position": {},
            }
            for p in range(n_pos):
                s["per_position"][f"P{p}"] = {
                    "dir_acc": round(np.mean([r["per_position"][p]["dir_correct"] for r in periods]), 4),
                    "mae": round(np.mean([r["per_position"][p]["mae"] for r in periods]), 2),
                    "rank": round(np.mean([r["per_position"][p]["rank"] for r in periods]), 2),
                }
            print(f"    {k:<8} {s['direction_accuracy']:<12.4f} {s['all_correct_rate']:<10.4f} "
                  f"{s['avg_mae']:<8} {s['avg_rank']:<8} {s['avg_max_sim_ratio']:<14.4f}")
            window_results[k] = s

        # 分位置详情（用最优top_k）
        best_k = min(window_results, key=lambda k: window_results[k]["avg_rank"])
        best = window_results[best_k]
        print(f"\n    最优 top_k={best_k} 分位置:")
        for p in range(n_pos):
            pp = best["per_position"][f"P{p}"]
            print(f"      P{p}: 方向={pp['dir_acc']:.4f}, MAE={pp['mae']:.2f}, rank={pp['rank']:.2f}")

        all_results[window] = window_results

    # 最终汇总
    print(f"\n{'='*60}")
    print(f"  E9c 最终汇总: {lottery_type}")
    print(f"{'='*60}")
    print(f"\n  {'窗口':<6} {'top_k':<8} {'方向准确率':<12} {'全对率':<10} {'MAE':<8} {'rank':<8}")
    print(f"  {'─'*52}")
    for w, wresults in sorted(all_results.items()):
        for k, s in sorted(wresults.items()):
            print(f"  {w:<6} {k:<8} {s['direction_accuracy']:<12.4f} "
                  f"{s['all_correct_rate']:<10.4f} {s['avg_mae']:<8} {s['avg_rank']:<8}")

    return all_results


def main():
    for lottery_type in ["shuangseqiu", "daletou"]:
        run_experiment(lottery_type)


if __name__ == "__main__":
    main()
