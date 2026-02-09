# -*- coding: utf-8 -*-
"""
阶段3：组合生成器（Combination Generator）

生成50-200组高质量候选组合，4种生成策略 + 软约束检查 + 评分函数。
4种策略：
  3.1 加权随机采样
  3.2 贪心构造
  3.3 遗传算法
  3.4 蒙特卡洛树搜索（MCTS）
"""

import random
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from copy import deepcopy

from pipeline.config import LotteryConfig, get_config


# ============================================================
# 软约束检查器
# ============================================================

class ConstraintChecker:
    """基于历史统计的软约束检查"""

    def __init__(self, combo_stats: Dict[str, Any], config: LotteryConfig):
        self.config = config
        self.stats = combo_stats
        self.mid = config.red_midpoint

        # 和值的95%置信区间
        mean = combo_stats.get("sum_mean", 90)
        std = combo_stats.get("sum_std", 22)
        self.sum_low = mean - 2 * std
        self.sum_high = mean + 2 * std

        # 跨度的90%分位区间
        spans = combo_stats.get("span_values", [])
        if spans:
            self.span_low = float(np.percentile(spans, 5))
            self.span_high = float(np.percentile(spans, 95))
        else:
            self.span_low = 15.0
            self.span_high = 34.0

        # AC值的90%分位下限
        acs = combo_stats.get("ac_values", [])
        self.ac_min = float(np.percentile(acs, 10)) if acs else 3.0

    def check(self, red_balls: List[int]) -> Dict[str, Any]:
        """检查一组红球是否满足软约束，返回各项得分"""
        n = len(red_balls)
        if n == 0:
            return {"total": 0, "details": {}}

        # 奇偶比
        odd = sum(1 for b in red_balls if b % 2 == 1)
        even = n - odd
        oe_ok = 1.0 if min(odd, even) >= 1 else 0.5
        # 最优：接近均衡
        oe_score = 1.0 - abs(odd - even) / n

        # 大小比
        big = sum(1 for b in red_balls if b > self.mid)
        small = n - big
        bs_ok = 1.0 if min(big, small) >= 1 else 0.5
        bs_score = 1.0 - abs(big - small) / n

        # 和值
        s = sum(red_balls)
        if self.sum_low <= s <= self.sum_high:
            sum_score = 1.0
        else:
            dist = min(abs(s - self.sum_low), abs(s - self.sum_high))
            sum_score = max(0, 1.0 - dist / 50)

        # 跨度
        span = max(red_balls) - min(red_balls)
        if self.span_low <= span <= self.span_high:
            span_score = 1.0
        else:
            dist = min(abs(span - self.span_low), abs(span - self.span_high))
            span_score = max(0, 1.0 - dist / 10)

        # 连号组数（0-2组为佳）
        sorted_b = sorted(red_balls)
        consec = 0
        in_group = False
        for i in range(1, len(sorted_b)):
            if sorted_b[i] - sorted_b[i-1] == 1:
                if not in_group:
                    consec += 1
                    in_group = True
            else:
                in_group = False
        consec_score = 1.0 if consec <= 2 else max(0, 1.0 - (consec - 2) * 0.3)

        # AC值
        diffs = set()
        for i in range(len(red_balls)):
            for j in range(i+1, len(red_balls)):
                diffs.add(abs(red_balls[i] - red_balls[j]))
        ac = len(diffs) - (n - 1)
        ac_score = 1.0 if ac >= self.ac_min else ac / self.ac_min

        # 综合得分
        total = (oe_score * 0.15 + bs_score * 0.15 + sum_score * 0.25 +
                 span_score * 0.15 + consec_score * 0.15 + ac_score * 0.15)

        return {
            "total": total,
            "details": {
                "odd_even": oe_score,
                "big_small": bs_score,
                "sum": sum_score,
                "span": span_score,
                "consecutive": consec_score,
                "ac": ac_score,
            },
            "ac_value": ac,
        }


# ============================================================
# 评分函数
# ============================================================

def score_combination(
    red_balls: List[int],
    weights: Dict[str, float],
    checker: ConstraintChecker,
    all_candidates: List[List[int]],
) -> Dict[str, Any]:
    """
    组合评分 = 权重得分(40%) + 约束满足度(25%) + 均衡性(20%) + 历史相似度(15%)
    """
    # 权重得分：组合中号码的权重之和
    weight_score = sum(weights.get(str(b), 0) for b in red_balls)
    # 归一化到 [0,1]（理论最大值约为 top_n 个最高权重之和）
    max_possible = sum(sorted(weights.values(), reverse=True)[:len(red_balls)])
    weight_score = weight_score / max_possible if max_possible > 0 else 0

    # 约束满足度
    constraint = checker.check(red_balls)
    constraint_score = constraint["total"]

    # 均衡性：号码在区间上的分布均匀度
    n_zones = 3
    zone_size = checker.config.red_range / n_zones
    zones = [0] * n_zones
    for b in red_balls:
        z = min(int((b - 1) / zone_size), n_zones - 1)
        zones[z] += 1
    expected = len(red_balls) / n_zones
    balance_score = 1.0 - sum(abs(z - expected) for z in zones) / (2 * len(red_balls))

    # 历史相似度：与已有候选的平均差异（鼓励多样性）
    if all_candidates:
        similarities = []
        rb_set = set(red_balls)
        for other in all_candidates[-20:]:  # 只比较最近20个
            overlap = len(rb_set & set(other))
            similarities.append(overlap / len(red_balls))
        avg_sim = sum(similarities) / len(similarities)
        diversity_score = 1.0 - avg_sim  # 越不相似越好
    else:
        diversity_score = 1.0

    # 综合评分
    total = (weight_score * 0.40 + constraint_score * 0.25 +
             balance_score * 0.20 + diversity_score * 0.15)

    return {
        "score": total,
        "score_breakdown": {
            "weight": weight_score,
            "constraint": constraint_score,
            "balance": balance_score,
            "similarity": diversity_score,
        },
        "ac_value": constraint.get("ac_value", 0),
    }


# ============================================================
# 策略3.1：加权随机采样
# ============================================================

def strategy_weighted_sampling(
    remaining: List[int],
    weights: Dict[str, float],
    config: LotteryConfig,
    count: int = 2000,
) -> List[List[int]]:
    """按权重概率采样生成组合"""
    # 构建概率分布
    nums = [n for n in remaining]
    probs = np.array([weights.get(str(n), 0.001) for n in nums])
    probs = probs / probs.sum()

    results = []
    seen = set()
    for _ in range(count * 2):  # 多采样以去重
        chosen = np.random.choice(nums, size=config.red_count, replace=False, p=probs)
        combo = sorted(int(x) for x in chosen)
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            results.append(combo)
        if len(results) >= count:
            break

    return results


# ============================================================
# 策略3.2：贪心构造
# ============================================================

def strategy_greedy(
    remaining: List[int],
    weights: Dict[str, float],
    checker: ConstraintChecker,
    config: LotteryConfig,
    count: int = 50,
) -> List[List[int]]:
    """贪心逐个选入高权重号码，确保满足约束"""
    sorted_nums = sorted(remaining, key=lambda n: weights.get(str(n), 0), reverse=True)
    results = []
    seen = set()

    for start_offset in range(min(count * 2, len(sorted_nums))):
        combo = []
        used = set()

        # 从不同起点开始贪心
        start = start_offset % len(sorted_nums)
        candidates = sorted_nums[start:] + sorted_nums[:start]

        for num in candidates:
            if num in used:
                continue
            trial = sorted(combo + [num])
            if len(trial) <= config.red_count:
                check = checker.check(trial)
                if check["total"] > 0.3 or len(trial) < 3:
                    combo.append(num)
                    used.add(num)
            if len(combo) == config.red_count:
                break

        if len(combo) == config.red_count:
            key = tuple(sorted(combo))
            if key not in seen:
                seen.add(key)
                results.append(sorted(combo))

        if len(results) >= count:
            break

    return results


# ============================================================
# 策略3.3：遗传算法
# ============================================================

def strategy_genetic(
    remaining: List[int],
    weights: Dict[str, float],
    checker: ConstraintChecker,
    config: LotteryConfig,
    pop_size: int = 200,
    generations: int = 80,
    count: int = 50,
) -> List[List[int]]:
    """遗传算法：种群进化，兼顾质量和多样性"""

    def fitness(combo: List[int]) -> float:
        w_score = sum(weights.get(str(b), 0) for b in combo)
        c_score = checker.check(combo)["total"]
        return w_score * 0.6 + c_score * 0.4

    def crossover(p1: List[int], p2: List[int]) -> List[int]:
        pool = list(set(p1) | set(p2))
        if len(pool) < config.red_count:
            pool = list(set(pool) | set(remaining))
        chosen = random.sample(pool, min(config.red_count, len(pool)))
        return sorted(chosen)

    def mutate(combo: List[int], rate: float = 0.2) -> List[int]:
        result = combo[:]
        for i in range(len(result)):
            if random.random() < rate:
                candidates = [n for n in remaining if n not in result]
                if candidates:
                    # 按权重选择替换
                    cand_w = [weights.get(str(n), 0.001) for n in candidates]
                    total_w = sum(cand_w)
                    cand_p = [w / total_w for w in cand_w]
                    result[i] = int(np.random.choice(candidates, p=cand_p))
        return sorted(result)

    # 初始种群：加权随机
    population = []
    nums = list(remaining)
    probs = np.array([weights.get(str(n), 0.001) for n in nums])
    probs = probs / probs.sum()

    for _ in range(pop_size):
        chosen = np.random.choice(nums, size=config.red_count, replace=False, p=probs)
        population.append(sorted(int(x) for x in chosen))

    # 进化
    for gen in range(generations):
        # 评估适应度
        scored = [(combo, fitness(combo)) for combo in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 精英保留
        elite_count = pop_size // 5
        new_pop = [s[0] for s in scored[:elite_count]]

        # 锦标赛选择 + 交叉 + 变异
        while len(new_pop) < pop_size:
            # 锦标赛选择
            t1 = random.sample(scored, 3)
            t2 = random.sample(scored, 3)
            p1 = max(t1, key=lambda x: x[1])[0]
            p2 = max(t2, key=lambda x: x[1])[0]

            child = crossover(p1, p2)
            child = mutate(child, rate=0.15)
            new_pop.append(child)

        population = new_pop

    # 去重并返回 top
    seen = set()
    results = []
    scored = [(combo, fitness(combo)) for combo in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    for combo, _ in scored:
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            results.append(combo)
        if len(results) >= count:
            break

    return results


# ============================================================
# 策略3.4：蒙特卡洛树搜索（MCTS）
# ============================================================

def strategy_mcts(
    remaining: List[int],
    weights: Dict[str, float],
    checker: ConstraintChecker,
    config: LotteryConfig,
    simulations: int = 1000,
    count: int = 30,
) -> List[List[int]]:
    """简化版MCTS：树搜索发现高质量组合"""

    results = []
    seen = set()
    sorted_by_weight = sorted(remaining, key=lambda n: weights.get(str(n), 0), reverse=True)

    for _ in range(simulations):
        combo = []
        available = list(remaining)

        for step in range(config.red_count):
            if not available:
                break

            # UCB1-like 选择：权重 + 探索项
            scores = []
            for n in available:
                w = weights.get(str(n), 0.001)
                explore = random.gauss(0, 0.1)  # 随机探索
                scores.append(w + explore)

            # 按分数选择
            best_idx = np.argmax(scores)
            chosen = available[best_idx]
            combo.append(chosen)
            available.remove(chosen)

        if len(combo) == config.red_count:
            combo = sorted(combo)
            key = tuple(combo)
            if key not in seen:
                seen.add(key)
                results.append(combo)

        if len(results) >= count:
            break

    return results


# ============================================================
# 主函数：运行阶段3
# ============================================================

def run_stage3(
    stage0_result: Dict[str, Any],
    stage1_result: Dict[str, Any],
    stage2_result: Dict[str, Any],
    lottery_type: str,
) -> Dict[str, Any]:
    """
    运行阶段3：组合生成器

    返回:
        candidates: 候选组合列表（含评分）
        top_score: 最高评分
        generation_stats: 各策略贡献统计
    """
    config = get_config(lottery_type)
    remaining = stage1_result["remaining_numbers"]
    weights = stage2_result["number_weights"]
    combo_stats = stage0_result["combo_stats"]

    checker = ConstraintChecker(combo_stats, config)

    print(f"[阶段3] 开始组合生成，候选号码: {len(remaining)} 个")

    # 4种策略生成
    print("[阶段3] 策略3.1 加权随机采样...")
    s1 = strategy_weighted_sampling(remaining, weights, config, count=2000)

    print("[阶段3] 策略3.2 贪心构造...")
    s2 = strategy_greedy(remaining, weights, checker, config, count=50)

    print("[阶段3] 策略3.3 遗传算法...")
    s3 = strategy_genetic(remaining, weights, checker, config, count=50)

    print("[阶段3] 策略3.4 蒙特卡洛树搜索...")
    s4 = strategy_mcts(remaining, weights, checker, config, count=30)

    stats = {
        "total_generated": len(s1) + len(s2) + len(s3) + len(s4),
        "strategy_1": len(s1),
        "strategy_2": len(s2),
        "strategy_3": len(s3),
        "strategy_4": len(s4),
    }

    # 合并去重
    all_combos = []
    seen = set()
    for combo in s1 + s2 + s3 + s4:
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            all_combos.append(combo)

    print(f"[阶段3] 合并去重后: {len(all_combos)} 组")

    # 评分并排序
    print("[阶段3] 评分中...")
    scored_list = []
    scored_combos = []
    for combo in all_combos:
        result = score_combination(combo, weights, checker, scored_combos)
        scored_combos.append(combo)

        # 随机生成蓝球
        blue_balls = sorted(random.sample(config.blue_numbers, config.blue_count))

        scored_list.append({
            "red_balls": combo,
            "blue_balls": blue_balls,
            "score": result["score"],
            "score_breakdown": result["score_breakdown"],
            "ac_value": result["ac_value"],
        })

    # 按评分降序排列
    scored_list.sort(key=lambda x: x["score"], reverse=True)

    # 取 top N
    target = config.combo_candidate_count
    candidates = scored_list[:target]

    stats["after_filter"] = len(candidates)
    top_score = candidates[0]["score"] if candidates else 0

    print(f"[阶段3] 完成，保留 {len(candidates)} 组，最高评分: {top_score:.4f}")

    return {
        "candidates": candidates,
        "top_score": top_score,
        "generation_stats": stats,
    }
