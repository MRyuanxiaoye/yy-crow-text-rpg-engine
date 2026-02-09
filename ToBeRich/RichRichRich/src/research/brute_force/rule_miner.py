# -*- coding: utf-8 -*-
"""
A2: 条件排除规则挖掘（并行版）

穷举所有原子条件的 1-5 组合（AND），
对每条规则计算支持度、置信度、提升度，
用卡方检验 + Bonferroni 校正筛选显著规则。

并行策略：
  - 用 Pool initializer 传递共享数据，避免重复序列化
  - 按批次分发组合，每批在子进程内循环处理
"""

import json
import time
import numpy as np
import multiprocessing as mp
from itertools import combinations, islice
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from research.config import ResearchConfig, RULES_DIR
from research.data_loader import LotteryData


# ============================================================
# 原子条件生成
# ============================================================

def build_atomic_conditions(data: LotteryData, config: ResearchConfig) -> Tuple[List[Dict], np.ndarray]:
    """
    构建所有原子条件。
    返回条件列表和对应的布尔掩码矩阵 (n_conditions, n_draws)
    """
    n = data.n_draws
    conditions = []
    masks = []

    n_bins = config.rule_value_bins
    rc = data.red_count

    # 1. 位置值域分箱
    for pos in range(rc):
        series = data.position_series[pos]
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(series, percentiles)
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1
        for b in range(actual_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < actual_bins - 1:
                mask = (series >= lo) & (series < hi)
            else:
                mask = (series >= lo) & (series <= hi)
            conditions.append({
                "type": "value_bin",
                "pos": pos,
                "bin": b,
                "range": f"[{lo:.0f},{hi:.0f}]",
                "name": f"P{pos}_val_{lo:.0f}_{hi:.0f}",
            })
            masks.append(mask.astype(np.bool_))

    # 2. 方向条件
    for pos in range(rc):
        dir_seq = data.direction_series[pos]
        for d_val, d_name in [(-1, "D"), (0, "E"), (1, "U")]:
            mask = np.zeros(n, dtype=np.bool_)
            mask[1:1 + len(dir_seq)] = (dir_seq == d_val)
            conditions.append({
                "type": "direction",
                "pos": pos,
                "dir": d_name,
                "name": f"P{pos}_dir_{d_name}",
            })
            masks.append(mask)

    # 3. 奇偶
    for pos in range(rc):
        series = data.position_series[pos]
        mask_odd = (series % 2 == 1)
        mask_even = (series % 2 == 0)
        conditions.append({"type": "parity", "pos": pos, "val": "odd",
                           "name": f"P{pos}_odd"})
        masks.append(mask_odd)
        conditions.append({"type": "parity", "pos": pos, "val": "even",
                           "name": f"P{pos}_even"})
        masks.append(mask_even)

    # 4. 大小（相对中位数）
    for pos in range(rc):
        series = data.position_series[pos]
        median_val = np.median(series)
        mask_big = series > median_val
        mask_small = series <= median_val
        conditions.append({"type": "size", "pos": pos, "val": "big",
                           "name": f"P{pos}_big"})
        masks.append(mask_big)
        conditions.append({"type": "size", "pos": pos, "val": "small",
                           "name": f"P{pos}_small"})
        masks.append(mask_small)

    # 5. 一阶差分区间
    for pos in range(rc):
        diff1 = data.get_diff_series(pos, order=1)
        percentiles = np.linspace(0, 100, 4)
        bin_edges = np.percentile(diff1, percentiles)
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1
        for b in range(actual_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = np.zeros(n, dtype=np.bool_)
            if b < actual_bins - 1:
                m = (diff1 >= lo) & (diff1 < hi)
            else:
                m = (diff1 >= lo) & (diff1 <= hi)
            mask[1:1 + len(m)] = m
            conditions.append({
                "type": "diff1_bin",
                "pos": pos,
                "bin": b,
                "range": f"[{lo:.1f},{hi:.1f}]",
                "name": f"P{pos}_diff_{lo:.0f}_{hi:.0f}",
            })
            masks.append(mask)

    # 6. 组合统计量分箱
    combo_stats = data.get_combo_stats_series()
    for stat_name, stat_series in combo_stats.items():
        percentiles = np.linspace(0, 100, 4)
        bin_edges = np.percentile(stat_series, percentiles)
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1
        for b in range(actual_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < actual_bins - 1:
                mask = (stat_series >= lo) & (stat_series < hi)
            else:
                mask = (stat_series >= lo) & (stat_series <= hi)
            conditions.append({
                "type": "combo_stat",
                "stat": stat_name,
                "bin": b,
                "range": f"[{lo:.0f},{hi:.0f}]",
                "name": f"combo_{stat_name}_{lo:.0f}_{hi:.0f}",
            })
            masks.append(mask.astype(np.bool_))

    mask_matrix = np.array(masks, dtype=np.bool_)
    print(f"[A2] 生成 {len(conditions)} 个原子条件, 掩码矩阵 {mask_matrix.shape}")
    return conditions, mask_matrix


# ============================================================
# 规则评估
# ============================================================

def evaluate_rule(
    combined_mask: np.ndarray,
    target_mask: np.ndarray,
    n_total: int,
    min_support: int,
    min_lift: float,
) -> Optional[Dict]:
    """评估一条规则：combined_mask（前件） → target_mask（后件）"""
    n_ante = int(np.sum(combined_mask))
    if n_ante < min_support:
        return None

    n_both = int(np.sum(combined_mask & target_mask))
    n_cons = int(np.sum(target_mask))

    confidence = n_both / n_ante if n_ante > 0 else 0
    expected_conf = n_cons / n_total if n_total > 0 else 0
    lift = confidence / expected_conf if expected_conf > 0 else 0

    if lift < min_lift:
        return None

    a = n_both
    b = n_ante - n_both
    c = n_cons - n_both
    d = n_total - n_ante - n_cons + n_both

    expected_a = n_ante * n_cons / n_total if n_total > 0 else 0
    if expected_a > 0:
        chi2 = (a - expected_a) ** 2 / expected_a
    else:
        chi2 = 0

    return {
        "support": n_ante,
        "n_both": n_both,
        "confidence": round(confidence, 4),
        "expected_confidence": round(expected_conf, 4),
        "lift": round(lift, 4),
        "chi2": round(chi2, 2),
        "contingency": [a, b, c, d],
    }


# ============================================================
# 目标条件
# ============================================================

def build_target_masks(data: LotteryData) -> List[Tuple[str, np.ndarray]]:
    """构建目标掩码：下一期各位置的方向"""
    targets = []
    n = data.n_draws

    for pos in range(data.red_count):
        dir_seq = data.direction_series[pos]
        for d_val, d_name in [(-1, "D"), (0, "E"), (1, "U")]:
            mask = np.zeros(n, dtype=np.bool_)
            mask[:len(dir_seq)] = (dir_seq == d_val)
            targets.append((f"next_P{pos}_{d_name}", mask))

    return targets


# ============================================================
# 并行工作函数
# ============================================================

# 全局变量，由 Pool initializer 设置
_shared_mask_matrix = None
_shared_target_names = None
_shared_target_masks = None
_shared_condition_names = None
_shared_n_total = 0
_shared_min_support = 10
_shared_min_lift = 1.2


def _init_worker(mask_matrix, target_names, target_masks, condition_names,
                 n_total, min_support, min_lift):
    """Pool 初始化函数，设置共享数据"""
    global _shared_mask_matrix, _shared_target_names, _shared_target_masks
    global _shared_condition_names, _shared_n_total, _shared_min_support, _shared_min_lift
    _shared_mask_matrix = mask_matrix
    _shared_target_names = target_names
    _shared_target_masks = target_masks
    _shared_condition_names = condition_names
    _shared_n_total = n_total
    _shared_min_support = min_support
    _shared_min_lift = min_lift


def _process_combo_batch(combo_batch: List[tuple]) -> Tuple[List[Dict], int]:
    """
    处理一批组合。在子进程中执行。
    返回 (显著规则列表, 处理的组合数)
    """
    rules = []
    count = 0

    for combo in combo_batch:
        count += 1

        # 计算组合掩码（AND）
        combined = _shared_mask_matrix[combo[0]].copy()
        for idx in combo[1:]:
            combined &= _shared_mask_matrix[idx]

        # 跳过支持度不足的
        support = int(np.sum(combined))
        if support < _shared_min_support:
            continue

        # 对每个目标评估
        for t_idx in range(len(_shared_target_names)):
            result = evaluate_rule(
                combined, _shared_target_masks[t_idx],
                _shared_n_total, _shared_min_support, _shared_min_lift,
            )
            if result is None:
                continue
            if result["chi2"] < 3.84:
                continue

            # 在子进程中直接用 Bonferroni 阈值过滤，减少传回数据量
            if result["chi2"] >= 10.83:
                rules.append({
                    "conditions": [_shared_condition_names[i] for i in combo],
                    "target": _shared_target_names[t_idx],
                    **result,
                })

    return rules, count


def _generate_combo_batches(n_conditions: int, n_comb: int, batch_size: int):
    """生成器：将组合按批次产出"""
    combo_iter = combinations(range(n_conditions), n_comb)
    while True:
        batch = list(islice(combo_iter, batch_size))
        if not batch:
            break
        yield batch


# ============================================================
# 主挖掘器
# ============================================================

class RuleMiner:
    """条件排除规则挖掘器（并行版）"""

    def __init__(self, data: LotteryData, config: ResearchConfig):
        self.data = data
        self.config = config
        self.results: List[Dict] = []

    def run(self) -> List[Dict]:
        """执行规则挖掘（并行版）"""
        start = time.time()

        # 构建原子条件
        conditions, mask_matrix = build_atomic_conditions(self.data, self.config)
        n_conditions = len(conditions)
        n_total = self.data.n_draws

        # 构建目标
        targets = build_target_masks(self.data)
        target_names = [t[0] for t in targets]
        target_masks = np.array([t[1] for t in targets], dtype=np.bool_)
        condition_names = [c["name"] for c in conditions]
        print(f"[A2] 目标条件: {len(targets)} 个")

        # 估算总组合数
        from math import comb
        total_combos = 0
        for k in range(1, self.config.rule_max_conditions + 1):
            c = comb(n_conditions, k)
            total_combos += c
            print(f"[A2] {k}条件组合: {c:,} 个")
        print(f"[A2] 总组合数: {total_combos:,}, 总评估次数: {total_combos * len(targets):,}")

        # 获取工作进程数
        import os
        n_workers = min(self.config.n_workers, max(1, (os.cpu_count() or 4) - 1))

        all_rules = []
        total_scanned = 0

        # 每批组合数：平衡序列化开销和负载均衡
        # 低阶组合批次大，高阶组合批次小（因为高阶组合更多）
        batch_sizes = {1: 50, 2: 200, 3: 1000, 4: 5000, 5: 10000}

        for n_comb in range(1, self.config.rule_max_conditions + 1):
            comb_start = time.time()
            n_this = comb(n_conditions, n_comb)
            batch_size = batch_sizes.get(n_comb, 10000)
            n_batches = (n_this + batch_size - 1) // batch_size

            print(f"[A2] 开始 {n_comb}条件组合: {n_this:,} 个, "
                  f"分 {n_batches} 批, 每批 {batch_size}")

            batch_rules = []
            batch_scanned = 0

            if n_this <= batch_size or n_workers <= 1:
                # 小规模：单进程
                for batch in _generate_combo_batches(n_conditions, n_comb, batch_size):
                    rules, cnt = _process_combo_batch_local(
                        batch, mask_matrix, target_names, target_masks,
                        condition_names, n_total,
                        self.config.rule_min_support, self.config.rule_min_lift,
                    )
                    batch_rules.extend(rules)
                    batch_scanned += cnt
            else:
                # 大规模：多进程，流式提交避免内存爆炸
                with mp.Pool(
                    processes=n_workers,
                    initializer=_init_worker,
                    initargs=(mask_matrix, target_names, target_masks,
                              condition_names, n_total,
                              self.config.rule_min_support, self.config.rule_min_lift),
                ) as pool:
                    # 用 imap_unordered + 生成器实现流式处理
                    batch_gen = _generate_combo_batches(n_conditions, n_comb, batch_size)
                    report_interval = max(1, n_batches // 20)
                    for i, (rules, cnt) in enumerate(
                        pool.imap_unordered(_process_combo_batch, batch_gen, chunksize=1)
                    ):
                        batch_rules.extend(rules)
                        batch_scanned += cnt

                        # 进度报告
                        if (i + 1) % report_interval == 0 or i == n_batches - 1:
                            elapsed = time.time() - comb_start
                            pct = batch_scanned / n_this * 100
                            speed = batch_scanned / elapsed if elapsed > 0 else 0
                            print(f"[A2] {n_comb}条件: {pct:.0f}% "
                                  f"({batch_scanned:,}/{n_this:,}), "
                                  f"发现 {len(batch_rules)} 条规则, "
                                  f"速度 {speed:,.0f} 组合/秒, "
                                  f"耗时 {elapsed:.1f}s")

            total_scanned += batch_scanned
            all_rules.extend(batch_rules)
            elapsed = time.time() - comb_start
            print(f"[A2] {n_comb}条件组合完成: "
                  f"发现 {len(batch_rules)} 条规则, "
                  f"累计 {len(all_rules)} 条, "
                  f"耗时 {elapsed:.1f}s")

            # 规则数上限检查
            if self.config.rule_max_rules > 0 and len(all_rules) > self.config.rule_max_rules:
                print(f"[A2] 规则数已超过 {self.config.rule_max_rules}，跳过更高阶组合")
                break

        # 子进程已用 Bonferroni 阈值 (chi2 >= 10.83) 过滤
        # 按提升度排序
        all_rules.sort(key=lambda x: x["lift"], reverse=True)
        self.results = all_rules

        elapsed = time.time() - start
        print(f"[A2] 完成: 扫描 {total_scanned:,} 个组合, "
              f"显著规则 {len(self.results):,}, "
              f"耗时 {elapsed:.1f}s")

        return self.results

    def save(self, filename: str = "a2_exclusion_rules.json"):
        """保存结果"""
        RULES_DIR.mkdir(parents=True, exist_ok=True)
        path = RULES_DIR / filename
        clean_results = []
        for r in self.results:
            clean = {k: v for k, v in r.items() if k != "condition_details"}
            clean_results.append(clean)

        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "lottery_type": self.data.lottery_type,
                "total_rules": len(clean_results),
                "rules": clean_results[:10000],  # 量大管饱版保存更多
            }, f, ensure_ascii=False, indent=2)
        print(f"[A2] 结果已保存: {path}")

    def summary(self) -> Dict[str, Any]:
        """生成摘要"""
        if not self.results:
            return {"total": 0}

        by_target = {}
        for r in self.results:
            t = r["target"]
            by_target.setdefault(t, []).append(r)

        return {
            "total": len(self.results),
            "by_target": {
                t: {
                    "count": len(rules),
                    "top_lift": rules[0]["lift"] if rules else 0,
                    "avg_confidence": round(
                        np.mean([r["confidence"] for r in rules]), 4
                    ),
                }
                for t, rules in sorted(by_target.items())
            },
            "top10": [r for r in self.results[:10]],
        }


def _process_combo_batch_local(combo_batch, mask_matrix, target_names, target_masks,
                                condition_names, n_total, min_support, min_lift):
    """单进程版本的批处理（不依赖全局变量）"""
    rules = []
    count = 0

    for combo in combo_batch:
        count += 1
        combined = mask_matrix[combo[0]].copy()
        for idx in combo[1:]:
            combined &= mask_matrix[idx]

        support = int(np.sum(combined))
        if support < min_support:
            continue

        for t_idx in range(len(target_names)):
            result = evaluate_rule(
                combined, target_masks[t_idx],
                n_total, min_support, min_lift,
            )
            if result is None:
                continue
            if result["chi2"] < 3.84:
                continue

            if result["chi2"] >= 10.83:
                rules.append({
                    "conditions": [condition_names[i] for i in combo],
                    "target": target_names[t_idx],
                    **result,
                })

    return rules, count
