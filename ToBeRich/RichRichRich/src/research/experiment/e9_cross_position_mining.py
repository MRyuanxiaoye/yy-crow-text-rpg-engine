# -*- coding: utf-8 -*-
"""E9: 跨位置关联暴力穷举 (完整版 v2)

核心思路转变：
  穷举建表不是为了让每张表各自投票，而是构建一个多维度相似度索引。
  对每个回测期，计算它和每个历史期在所有维度上的匹配度（在多少张表里
  落入同一个局面），取最相似的 top-K 个历史期，直接看它们下一期的变化。

三个维度的完整穷举：
  维度一 - 同期内位置组合：P0-P5的1~6元组合
  维度二 - 多期时序窗口：回看2~3期的联合局面
  维度三 - 跨期跨位置交叉：t期P_i与t-1期P_j的交叉

粒度：G1(方向3态) / G2(方向x幅度2档=5态) / G3(方向x幅度3档=7态)

用法: python3 -m src.research.experiment.e9_cross_position_mining
"""

import sys
import time
import json
import pickle
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "e9_cross_position"
BACKTEST_HOLD = 200


# ============================================================
# 粒度编码器
# ============================================================

class GranularityEncoder:
    def __init__(self, diff_matrix: np.ndarray, granularity: str):
        self.granularity = granularity
        self.n_pos = diff_matrix.shape[1]
        self.thresholds = {}
        for pos in range(self.n_pos):
            abs_diffs = np.abs(diff_matrix[:, pos])
            abs_diffs = abs_diffs[abs_diffs > 0]
            if len(abs_diffs) == 0:
                self.thresholds[pos] = {"median": 1, "t33": 1, "t67": 2}
                continue
            self.thresholds[pos] = {
                "median": float(np.median(abs_diffs)),
                "t33": float(np.percentile(abs_diffs, 33.3)),
                "t67": float(np.percentile(abs_diffs, 66.7)),
            }

    def encode_one(self, diff_val: int, pos: int) -> int:
        if self.granularity == "G1":
            if diff_val < 0: return 0
            elif diff_val == 0: return 1
            else: return 2
        elif self.granularity == "G2":
            med = self.thresholds[pos]["median"]
            if diff_val < 0:
                return 0 if abs(diff_val) <= med else 1
            elif diff_val == 0:
                return 2
            else:
                return 3 if abs(diff_val) <= med else 4
        elif self.granularity == "G3":
            t33 = self.thresholds[pos]["t33"]
            t67 = self.thresholds[pos]["t67"]
            if diff_val == 0:
                return 3
            ad = abs(diff_val)
            if ad <= t33: amp = 0
            elif ad <= t67: amp = 1
            else: amp = 2
            if diff_val < 0: return amp
            else: return 4 + amp
        raise ValueError(f"Unknown granularity: {self.granularity}")

    def encode_positions(self, diff_row: np.ndarray, positions: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(self.encode_one(int(diff_row[p]), p) for p in positions)

    @property
    def n_states(self) -> int:
        return {"G1": 3, "G2": 5, "G3": 7}[self.granularity]


def build_diff_matrix(data: LotteryData) -> np.ndarray:
    n_pos = data.red_count
    n = data.n_draws - 1
    diff_mat = np.zeros((n, n_pos), dtype=np.int32)
    for pos in range(n_pos):
        diff_mat[:, pos] = np.diff(data.position_series[pos])
    return diff_mat


# ============================================================
# 构建所有维度的编码规格
# ============================================================

def build_encoding_specs(n_pos: int, max_d1_order: int = None,
                         max_d2_lookback: int = 3, max_d2_pos_order: int = 3,
                         max_d3_order: int = 3):
    """
    生成所有编码规格（不依赖数据，只依赖位置数）。
    每个规格描述：怎样从 diff_matrix 中提取一个局面编码。

    返回 specs 列表，每个 spec = {
        "dim": "D1"/"D2"/"D3",
        "desc": 描述字符串,
        "encode_fn": callable(diff_matrix, t, encoder) -> tuple or None
    }
    """
    if max_d1_order is None:
        max_d1_order = n_pos
    all_positions = list(range(n_pos))
    specs = []

    # 维度一：同期内位置组合（含一元组）
    for order in range(1, max_d1_order + 1):
        for combo in combinations(all_positions, order):
            def make_fn(c):
                def fn(dm, t, enc):
                    return enc.encode_positions(dm[t], c)
                return fn
            specs.append({
                "dim": "D1", "order": order, "combo": combo,
                "encode_fn": make_fn(combo),
                "min_t": 0,
            })

    # 维度二：多期时序窗口
    for pos_order in range(1, max_d2_pos_order + 1):
        for combo in combinations(all_positions, pos_order):
            for lookback in range(2, max_d2_lookback + 1):
                def make_fn(c, lb):
                    def fn(dm, t, enc):
                        if t < lb - 1:
                            return None
                        parts = []
                        for lag in range(lb):
                            parts.extend(enc.encode_positions(dm[t - lag], c))
                        return tuple(parts)
                    return fn
                specs.append({
                    "dim": "D2", "lookback": lookback, "combo": combo,
                    "encode_fn": make_fn(combo, lookback),
                    "min_t": lookback - 1,
                })

    # 维度三：跨期跨位置交叉
    for n_t in range(1, max_d3_order):
        for n_t1 in range(1, max_d3_order - n_t + 1):
            if n_t + n_t1 > max_d3_order:
                continue
            for combo_t in combinations(all_positions, n_t):
                for combo_t1 in combinations(all_positions, n_t1):
                    if combo_t == combo_t1:
                        continue
                    def make_fn(ct, ct1):
                        def fn(dm, t, enc):
                            if t < 1:
                                return None
                            return enc.encode_positions(dm[t], ct) + enc.encode_positions(dm[t - 1], ct1)
                        return fn
                    specs.append({
                        "dim": "D3", "combo_t": combo_t, "combo_t1": combo_t1,
                        "encode_fn": make_fn(combo_t, combo_t1),
                        "min_t": 1,
                    })

    return specs


# ============================================================
# 构建每期的编码指纹 + 倒排索引
# ============================================================

def build_fingerprints_and_index(
    diff_matrix: np.ndarray, stat_end: int,
    specs: List[Dict], encoder: GranularityEncoder,
) -> Tuple[List[List], Dict]:
    """
    对统计期内每一期，用所有 specs 编码，得到指纹向量。
    同时构建倒排索引：(spec_idx, encoded_state) -> [t1, t2, ...]

    Returns:
        fingerprints: fingerprints[t] = [(spec_idx, encoded_state), ...]
        inverted_index: {(spec_idx, encoded_state): set_of_t}
    """
    n_specs = len(specs)
    # 对每期，只需要记录它在哪些 spec 上编码为什么值
    # 用倒排索引来加速相似度计算
    inverted_index = defaultdict(set)

    # 也记录每期的指纹（用于调试和分析）
    fingerprints = [[] for _ in range(stat_end)]

    for si, spec in enumerate(specs):
        encode_fn = spec["encode_fn"]
        min_t = spec["min_t"]
        for t in range(min_t, stat_end):
            code = encode_fn(diff_matrix, t, encoder)
            if code is not None:
                key = (si, code)
                inverted_index[key].add(t)
                fingerprints[t].append(key)

    return fingerprints, dict(inverted_index)


# ============================================================
# 基于相似度的回测
# ============================================================

def backtest_similarity(
    diff_matrix: np.ndarray, data: LotteryData,
    specs: List[Dict], encoder: GranularityEncoder,
    fingerprints: List[List], inverted_index: Dict,
    stat_end: int, top_k_list: List[int] = None,
) -> Dict:
    """
    对每个回测期：
    1. 编码当前局面，得到指纹
    2. 通过倒排索引找到所有和当前期有至少一个维度匹配的历史期
    3. 计算每个历史期的匹配维度数（相似度分数）
    4. 取 top-K 最相似的历史期
    5. 看这 K 个历史期的下一期变化，作为预测分布
    """
    if top_k_list is None:
        top_k_list = [5, 10, 20, 50]

    n_pos = data.red_count
    all_positions = list(range(n_pos))
    n_total = diff_matrix.shape[0]
    test_start = stat_end
    test_end = n_total - 1
    n_specs = len(specs)

    print(f"\n  回测范围: 第{test_start}期 ~ 第{test_end}期 ({test_end - test_start}期)")
    print(f"  总编码维度数: {n_specs}")
    print(f"  top_k 候选: {top_k_list}")

    # 对每个 top_k 分别收集结果
    results_by_k = {k: [] for k in top_k_list}

    for t in range(test_start, test_end):
        # 1. 编码当前期的指纹
        current_keys = []
        for si, spec in enumerate(specs):
            code = spec["encode_fn"](diff_matrix, t, encoder)
            if code is not None:
                current_keys.append((si, code))

        # 2. 通过倒排索引收集候选历史期及其匹配数
        # similarity_scores[h] = 和历史期 h 匹配的维度数
        similarity_scores = defaultdict(int)
        for key in current_keys:
            if key in inverted_index:
                for h in inverted_index[key]:
                    # 排除自身（虽然回测期不在统计期内，但以防万一）
                    # 也排除最后一期（没有下一期）
                    if h < stat_end - 1:
                        similarity_scores[h] += 1

        if not similarity_scores:
            # 没有任何匹配，构造空预测（全部 rank 设为最大值）
            true_next = diff_matrix[t + 1]
            empty = {
                "t": t, "n_matches": 0, "max_sim": 0, "n_dims": len(current_keys),
                "per_position": {},
                "dir_accuracy": 0.0, "avg_mae": 0.0,
                "avg_rank": float(data.red_range), "all_dir_correct": False,
            }
            for p in all_positions:
                true_d = int(true_next[p])
                true_dir = 1 if true_d > 0 else (-1 if true_d < 0 else 0)
                empty["per_position"][p] = {
                    "true_dir": true_dir, "pred_dir": 0, "dir_correct": (true_dir == 0),
                    "true_diff": true_d, "pred_diff": 0, "mae": abs(true_d),
                    "true_val": int(data.position_series[p][t + 2]),
                    "pred_rank": data.red_range, "n_candidates": 0,
                }
            empty["avg_mae"] = round(np.mean([abs(int(true_next[p])) for p in all_positions]), 2)
            empty["dir_accuracy"] = sum(1 for p in all_positions if empty["per_position"][p]["dir_correct"]) / n_pos
            for k in top_k_list:
                results_by_k[k].append(empty)
            continue

        # 3. 排序取 top-K
        sorted_matches = sorted(similarity_scores.items(), key=lambda x: -x[1])
        max_sim = sorted_matches[0][1]

        true_next = diff_matrix[t + 1]

        for top_k in top_k_list:
            # 取 top_k 个最相似的历史期
            top_periods = sorted_matches[:top_k]

            # 4. 用这些历史期的下一期变化构建预测分布
            # 按相似度分数加权
            period = _evaluate_period(
                t, top_periods, diff_matrix, data, n_pos, all_positions,
                true_next, max_sim, len(current_keys),
            )
            results_by_k[top_k].append(period)

        # 进度
        if (t - test_start) % 50 == 0:
            print(f"    进度: {t - test_start}/{test_end - test_start}, "
                  f"候选历史期数: {len(similarity_scores)}, "
                  f"最高相似度: {max_sim}/{len(current_keys)}")

    # 汇总
    summaries = {}
    for top_k, periods in results_by_k.items():
        summaries[top_k] = _summarize_periods(periods, n_pos, all_positions, top_k)

    return summaries


def _evaluate_period(
    t, top_periods, diff_matrix, data, n_pos, all_positions,
    true_next, max_sim, n_total_dims,
):
    """评估单期预测"""
    dir_correct = 0
    abs_errors = []
    per_pos = {}

    for p in all_positions:
        true_d = int(true_next[p])
        true_dir = 1 if true_d > 0 else (-1 if true_d < 0 else 0)

        # 用相似度分数作为权重，收集差分值分布
        diff_dist = defaultdict(float)
        for h, sim_score in top_periods:
            w = sim_score  # 相似度越高权重越大
            nd = int(diff_matrix[h + 1, p])
            diff_dist[nd] += w

        total_w = sum(diff_dist.values())

        # 方向预测
        dir_w = {-1: 0.0, 0: 0.0, 1: 0.0}
        for val, w in diff_dist.items():
            if val < 0: dir_w[-1] += w
            elif val == 0: dir_w[0] += w
            else: dir_w[1] += w

        if total_w == 0:
            pred_dir = 0
        else:
            pred_dir = max(dir_w, key=dir_w.get)

        is_correct = (pred_dir == true_dir)
        if is_correct:
            dir_correct += 1

        # 幅度
        if total_w == 0:
            pred_diff = 0.0
        else:
            pred_diff = sum(k * v for k, v in diff_dist.items()) / total_w
        mae = abs(true_d - pred_diff)
        abs_errors.append(mae)

        # 号码排名
        current_val = data.position_series[p][t + 1]
        number_probs = {}
        for dv_val, dv_cnt in diff_dist.items():
            candidate = current_val + int(dv_val)
            if 1 <= candidate <= data.red_range:
                number_probs[candidate] = number_probs.get(candidate, 0) + dv_cnt
        total_np = sum(number_probs.values())
        if total_np > 0:
            number_probs = {k: v / total_np for k, v in number_probs.items()}

        true_val = data.position_series[p][t + 2]
        true_prob = number_probs.get(true_val, 0.0)
        if true_prob > 0:
            rank = sum(1 for v in number_probs.values() if v > true_prob) + 1
        else:
            rank = data.red_range

        per_pos[p] = {
            "true_dir": true_dir, "pred_dir": pred_dir,
            "dir_correct": is_correct,
            "true_diff": true_d, "pred_diff": round(pred_diff, 2),
            "mae": round(mae, 2),
            "true_val": int(true_val), "pred_rank": rank,
            "n_candidates": len(number_probs),
        }

    return {
        "t": t,
        "n_matches": len(top_periods),
        "max_sim": max_sim,
        "n_dims": n_total_dims,
        "per_position": per_pos,
        "dir_accuracy": dir_correct / n_pos,
        "avg_mae": round(np.mean(abs_errors), 2),
        "avg_rank": np.mean([per_pos[p]["pred_rank"] for p in all_positions]),
        "all_dir_correct": (dir_correct == n_pos),
    }


def _summarize_periods(periods, n_pos, all_positions, top_k):
    """汇总多期结果"""
    n_test = len(periods)
    if n_test == 0:
        return {}

    summary = {
        "top_k": top_k,
        "n_test_periods": n_test,
        "direction": {
            "avg_accuracy": round(np.mean([r["dir_accuracy"] for r in periods]), 4),
            "all_correct_rate": round(np.mean([r["all_dir_correct"] for r in periods]), 4),
            "per_position": {},
        },
        "amplitude": {
            "avg_mae": round(np.mean([r["avg_mae"] for r in periods]), 2),
            "per_position": {},
        },
        "ranking": {
            "avg_rank": round(np.mean([r["avg_rank"] for r in periods]), 2),
            "per_position": {},
        },
        "similarity": {
            "avg_max_sim": round(np.mean([r["max_sim"] for r in periods]), 1),
            "avg_n_dims": round(np.mean([r["n_dims"] for r in periods]), 1),
            "avg_sim_ratio": round(np.mean([r["max_sim"] / r["n_dims"] for r in periods if r["n_dims"] > 0]), 4),
        },
    }

    for p in all_positions:
        summary["direction"]["per_position"][f"P{p}"] = round(
            np.mean([r["per_position"][p]["dir_correct"] for r in periods]), 4)
        summary["amplitude"]["per_position"][f"P{p}"] = round(
            np.mean([r["per_position"][p]["mae"] for r in periods]), 2)
        summary["ranking"]["per_position"][f"P{p}"] = round(
            np.mean([r["per_position"][p]["pred_rank"] for r in periods]), 2)

    return summary


# ============================================================
# 主流程
# ============================================================

def run_e9(lottery_type: str):
    print(f"\n{'='*60}")
    print(f"  E9 跨位置关联暴力穷举 v2 (相似度匹配): {lottery_type}")
    print(f"{'='*60}")

    data = LotteryData(lottery_type)
    diff_matrix = build_diff_matrix(data)
    n_pos = data.red_count
    n_periods = diff_matrix.shape[0]
    stat_end = n_periods - BACKTEST_HOLD

    print(f"\n  总期数: {data.n_draws}, 差分期数: {n_periods}")
    print(f"  统计期: 0 ~ {stat_end-1} ({stat_end}期)")
    print(f"  回测期: {stat_end} ~ {n_periods-1} ({n_periods - stat_end}期)")
    print(f"  位置数: {n_pos}, 号码范围: 1~{data.red_range}")

    # 构建编码规格
    print(f"\n  构建编码规格...")
    specs = build_encoding_specs(
        n_pos,
        max_d1_order=n_pos,
        max_d2_lookback=3,
        max_d2_pos_order=3,
        max_d3_order=3,
    )
    dim_counts = defaultdict(int)
    for s in specs:
        dim_counts[s["dim"]] += 1
    print(f"  编码规格总数: {len(specs)}")
    for d, c in sorted(dim_counts.items()):
        print(f"    {d}: {c}")

    granularities = ["G1", "G2", "G3"]
    all_results = {}

    for gran in granularities:
        print(f"\n{'─'*50}")
        print(f"  粒度 {gran}")
        print(f"{'─'*50}")

        t0 = time.time()
        stat_diffs = diff_matrix[:stat_end]
        encoder = GranularityEncoder(stat_diffs, gran)

        # 构建指纹和倒排索引
        print(f"\n    构建指纹和倒排索引...")
        t1 = time.time()
        fingerprints, inverted_index = build_fingerprints_and_index(
            diff_matrix, stat_end, specs, encoder,
        )
        n_index_keys = len(inverted_index)
        avg_fp_len = np.mean([len(fp) for fp in fingerprints if fp])
        print(f"      倒排索引键数: {n_index_keys}")
        print(f"      平均指纹长度: {avg_fp_len:.1f}")
        print(f"      耗时: {time.time()-t1:.1f}s")

        # 回测
        print(f"\n    回测 (相似度匹配)...")
        t1 = time.time()
        summaries = backtest_similarity(
            diff_matrix, data, specs, encoder,
            fingerprints, inverted_index, stat_end,
            top_k_list=[5, 10, 20, 50, 100],
        )
        print(f"      回测耗时: {time.time()-t1:.1f}s")

        # 打印结果
        print(f"\n    结果:")
        print(f"    {'top_k':<8} {'方向准确率':<12} {'全对率':<10} {'MAE':<8} {'rank':<8} {'相似度比':<10}")
        print(f"    {'─'*56}")
        for k, s in sorted(summaries.items()):
            if not s:
                continue
            print(f"    {k:<8} {s['direction']['avg_accuracy']:<12.4f} "
                  f"{s['direction']['all_correct_rate']:<10.4f} "
                  f"{s['amplitude']['avg_mae']:<8} "
                  f"{s['ranking']['avg_rank']:<8} "
                  f"{s['similarity']['avg_sim_ratio']:<10.4f}")

        all_results[gran] = summaries
        print(f"\n    粒度 {gran} 总耗时: {time.time()-t0:.1f}s")

    # 保存
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  E9 v2 汇总: {lottery_type}")
    print(f"{'='*60}")

    print(f"\n  {'粒度':<6} {'top_k':<8} {'方向准确率':<12} {'全对率':<10} {'MAE':<8} {'rank':<8}")
    print(f"  {'─'*52}")
    best_rank = 999
    best_config = ""
    for gran, summaries in all_results.items():
        for k, s in sorted(summaries.items()):
            if not s:
                continue
            r = s['ranking']['avg_rank']
            print(f"  {gran:<6} {k:<8} {s['direction']['avg_accuracy']:<12.4f} "
                  f"{s['direction']['all_correct_rate']:<10.4f} "
                  f"{s['amplitude']['avg_mae']:<8} "
                  f"{r:<8}")
            if r < best_rank:
                best_rank = r
                best_config = f"{gran}/top_k={k}"

    print(f"\n  最优配置: {best_config}, rank={best_rank}")

    # 保存JSON
    save_data = {}
    for gran, summaries in all_results.items():
        save_data[gran] = {str(k): v for k, v in summaries.items()}
    summary_path = RESULTS_DIR / f"e9v2_summary_{lottery_type}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  已保存: {summary_path}")

    return all_results


def main():
    all_results = {}
    for lottery_type in ["shuangseqiu", "daletou"]:
        all_results[lottery_type] = run_e9(lottery_type)

    print(f"\n{'='*60}")
    print(f"  E9 v2 最终对比")
    print(f"{'='*60}")
    for lt, grans in all_results.items():
        print(f"\n  {lt}:")
        best_rank = 999
        best_cfg = ""
        for gran, summaries in grans.items():
            for k, s in summaries.items():
                if not s:
                    continue
                r = s['ranking']['avg_rank']
                if r < best_rank:
                    best_rank = r
                    best_cfg = f"{gran}/top_k={k}"
        print(f"    最优: {best_cfg}, rank={best_rank}")


if __name__ == "__main__":
    main()
