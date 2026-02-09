# -*- coding: utf-8 -*-
"""
阶段4：购买优化器（Purchase Optimizer）

在预算内选出最优购买方案，4种优化策略。
4种策略：
  4.1 贪心集合覆盖
  4.2 分层覆盖策略
  4.3 整数线性规划（ILP）简化版
  4.4 复式/胆拖投注优化
"""

import math
import random
from itertools import combinations
from typing import Dict, List, Any, Tuple, Set

from pipeline.config import LotteryConfig, get_config


# ============================================================
# 工具函数
# ============================================================

def calc_dantuo_combinations(
    dan_red: List[int], tuo_red: List[int],
    dan_blue: List[int], tuo_blue: List[int],
    config: LotteryConfig,
) -> int:
    """计算胆拖注数"""
    from math import comb
    need_tuo_red = config.red_count - len(dan_red)
    if need_tuo_red < 0 or need_tuo_red > len(tuo_red):
        return 0
    red_combos = comb(len(tuo_red), need_tuo_red)

    if config.blue_count == 1:
        # 双色球：蓝球单选
        blue_combos = max(len(dan_blue) + len(tuo_blue), 1)
    else:
        # 大乐透：蓝球也可胆拖
        need_tuo_blue = config.blue_count - len(dan_blue)
        if len(tuo_blue) >= need_tuo_blue > 0:
            blue_combos = comb(len(tuo_blue), need_tuo_blue)
        else:
            blue_combos = 1

    return red_combos * blue_combos


def evaluate_plan(
    tickets: List[Dict],
    weights: Dict[str, float],
    config: LotteryConfig,
) -> Dict[str, Any]:
    """评估购买方案的覆盖率和性价比"""
    all_red = set()
    all_blue = set()
    total_cost = 0
    total_combos = 0

    for t in tickets:
        total_cost += t.get("cost", 0)
        total_combos += t.get("combinations", 0)

        if t["type"] == "胆拖":
            all_red.update(t.get("dan_red", []))
            all_red.update(t.get("tuo_red", []))
            all_blue.update(t.get("dan_blue", []))
            all_blue.update(t.get("tuo_blue", []))
        else:
            all_red.update(t.get("red_balls", []))
            all_blue.update(t.get("blue_balls", []))

    # 权重覆盖
    weight_covered = sum(weights.get(str(n), 0) for n in all_red)

    # Top10 覆盖
    sorted_by_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top10 = {int(k) for k, _ in sorted_by_w[:10]}
    top10_covered = len(all_red & top10)

    coverage_pct = len(all_red) / config.red_range if config.red_range > 0 else 0

    return {
        "all_red_numbers": sorted(all_red),
        "all_blue_numbers": sorted(all_blue),
        "total_cost": total_cost,
        "total_combinations": total_combos,
        "coverage_pct": coverage_pct,
        "weight_coverage": weight_covered,
        "top10_red_covered": top10_covered,
        "efficiency": {
            "weight_per_yuan": weight_covered / total_cost if total_cost > 0 else 0,
            "cost_per_combination": total_cost / total_combos if total_combos > 0 else 0,
        },
    }


# ============================================================
# 策略4.1：贪心集合覆盖
# ============================================================

def strategy_greedy_cover(
    candidates: List[Dict],
    weights: Dict[str, float],
    config: LotteryConfig,
    budget: int,
) -> List[Dict]:
    """
    经典贪心集合覆盖：每次选择覆盖最多未覆盖高权重号码的组合。
    """
    price = config.ticket_price
    max_tickets = budget // price
    covered_red: Set[int] = set()
    tickets = []

    # 按评分排序的候选
    sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)

    for _ in range(max_tickets):
        best = None
        best_value = -1

        for cand in sorted_cands:
            red = set(cand["red_balls"])
            new_covered = red - covered_red
            # 价值 = 新覆盖号码的权重之和
            value = sum(weights.get(str(n), 0) for n in new_covered)
            # 加上评分加成
            value += cand["score"] * 0.1
            if value > best_value:
                best_value = value
                best = cand

        if best is None or best_value <= 0:
            break

        covered_red.update(best["red_balls"])
        tickets.append({
            "type": "单式",
            "red_balls": best["red_balls"],
            "blue_balls": best["blue_balls"],
            "combinations": 1,
            "cost": price,
        })
        sorted_cands.remove(best)

    return tickets


# ============================================================
# 策略4.2：分层覆盖策略
# ============================================================

def strategy_layered_cover(
    candidates: List[Dict],
    weights: Dict[str, float],
    config: LotteryConfig,
    budget: int,
) -> List[Dict]:
    """
    按号码重要性分层，优先覆盖高权重层。
    第一层：Top5号码必须覆盖
    第二层：Top6-15号码尽量覆盖
    第三层：其余号码随机覆盖
    """
    price = config.ticket_price
    max_tickets = budget // price

    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    tier1 = {int(k) for k, _ in sorted_w[:5]}
    tier2 = {int(k) for k, _ in sorted_w[5:15]}

    tickets = []
    covered = set()

    # 优先选包含 tier1 号码最多的组合
    sorted_cands = sorted(candidates, key=lambda c: (
        len(set(c["red_balls"]) & tier1),
        len(set(c["red_balls"]) & tier2),
        c["score"],
    ), reverse=True)

    for cand in sorted_cands:
        if len(tickets) >= max_tickets:
            break
        red_set = set(cand["red_balls"])
        # 至少覆盖一个新的 tier1 或 tier2 号码
        new_t1 = red_set & tier1 - covered
        new_t2 = red_set & tier2 - covered
        if new_t1 or new_t2 or len(tickets) < 2:
            covered.update(red_set)
            tickets.append({
                "type": "单式",
                "red_balls": cand["red_balls"],
                "blue_balls": cand["blue_balls"],
                "combinations": 1,
                "cost": price,
            })

    return tickets


# ============================================================
# 策略4.3：ILP简化版（穷举最优子集）
# ============================================================

def strategy_ilp_simple(
    candidates: List[Dict],
    weights: Dict[str, float],
    config: LotteryConfig,
    budget: int,
) -> List[Dict]:
    """
    简化版ILP：在 top 候选中穷举最优子集。
    由于完整ILP需要额外依赖，这里用穷举 top20 中选 N 注的方式近似。
    """
    price = config.ticket_price
    max_tickets = min(budget // price, 10)

    # 只在 top20 中搜索
    top_cands = candidates[:20]
    best_tickets = []
    best_value = -1

    # 穷举组合（限制搜索空间）
    search_size = min(len(top_cands), 12)
    search_cands = top_cands[:search_size]

    for r in range(min(max_tickets, search_size), 0, -1):
        for combo in combinations(range(search_size), r):
            selected = [search_cands[i] for i in combo]
            cost = r * price
            if cost > budget:
                continue

            # 计算覆盖价值
            covered = set()
            for s in selected:
                covered.update(s["red_balls"])
            value = sum(weights.get(str(n), 0) for n in covered)
            value += sum(s["score"] for s in selected) * 0.05

            if value > best_value:
                best_value = value
                best_tickets = [{
                    "type": "单式",
                    "red_balls": s["red_balls"],
                    "blue_balls": s["blue_balls"],
                    "combinations": 1,
                    "cost": price,
                } for s in selected]

        if best_tickets:
            break  # 找到最大注数的最优解就停

    return best_tickets


# ============================================================
# 策略4.4：复式/胆拖投注优化
# ============================================================

def strategy_dantuo(
    candidates: List[Dict],
    weights: Dict[str, float],
    config: LotteryConfig,
    budget: int,
) -> List[Dict]:
    """
    利用胆拖玩法降低成本，同时覆盖更多号码。
    胆码：权重最高的2个号码（必选）
    拖码：权重次高的若干号码（选N个）
    """
    price = config.ticket_price
    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    tickets = []
    remaining_budget = budget

    # 胆拖票：胆码2个 + 拖码若干
    dan_red = [int(k) for k, _ in sorted_w[:2]]

    # 计算拖码数量使得注数在预算内
    need_tuo = config.red_count - len(dan_red)
    best_tuo_red = []
    best_cost = 0

    for tuo_count in range(need_tuo, need_tuo + 6):
        tuo_red = [int(k) for k, _ in sorted_w[2:2+tuo_count]]
        if len(tuo_red) < need_tuo:
            continue

        # 蓝球胆拖
        blue_sorted = []
        # 从候选组合中统计蓝球频率
        blue_freq: Dict[int, int] = {}
        for c in candidates[:20]:
            for b in c.get("blue_balls", []):
                blue_freq[b] = blue_freq.get(b, 0) + 1
        blue_sorted = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)

        if config.blue_count == 2:
            dan_blue = [blue_sorted[0][0]] if blue_sorted else [1]
            tuo_blue = [blue_sorted[i][0] for i in range(1, min(3, len(blue_sorted)))]
        else:
            dan_blue = [blue_sorted[0][0]] if blue_sorted else [1]
            tuo_blue = []

        combos = calc_dantuo_combinations(dan_red, tuo_red, dan_blue, tuo_blue, config)
        cost = combos * price

        if cost <= remaining_budget and cost > best_cost:
            best_tuo_red = tuo_red
            best_cost = cost
            best_dan_blue = dan_blue
            best_tuo_blue = tuo_blue
            best_combos = combos

    if best_tuo_red:
        tickets.append({
            "type": "胆拖",
            "dan_red": sorted(dan_red),
            "tuo_red": sorted(best_tuo_red),
            "dan_blue": sorted(best_dan_blue),
            "tuo_blue": sorted(best_tuo_blue),
            "combinations": best_combos,
            "cost": best_cost,
        })
        remaining_budget -= best_cost

    # 剩余预算用单式票补充
    if remaining_budget >= price:
        covered = set(dan_red + best_tuo_red) if best_tuo_red else set()
        for cand in candidates:
            if remaining_budget < price:
                break
            red_set = set(cand["red_balls"])
            if red_set - covered:  # 有新号码
                tickets.append({
                    "type": "单式",
                    "red_balls": cand["red_balls"],
                    "blue_balls": cand["blue_balls"],
                    "combinations": 1,
                    "cost": price,
                })
                covered.update(red_set)
                remaining_budget -= price

    return tickets


# ============================================================
# 主函数：运行阶段4
# ============================================================

def run_stage4(
    stage0_result: Dict[str, Any],
    stage2_result: Dict[str, Any],
    stage3_result: Dict[str, Any],
    lottery_type: str,
    budget: int = 20,
) -> Dict[str, Any]:
    """
    运行阶段4：购买优化器

    返回:
        budget, total_cost, total_combinations, tickets,
        coverage, coverage_pct, weight_coverage, efficiency,
        strategy_comparison
    """
    config = get_config(lottery_type)
    candidates = stage3_result["candidates"]
    weights = stage2_result["number_weights"]

    print(f"[阶段4] 开始购买优化，预算: {budget}元，候选: {len(candidates)} 组")

    # 4种策略
    print("[阶段4] 策略4.1 贪心集合覆盖...")
    t1 = strategy_greedy_cover(candidates, weights, config, budget)
    e1 = evaluate_plan(t1, weights, config)

    print("[阶段4] 策略4.2 分层覆盖...")
    t2 = strategy_layered_cover(candidates, weights, config, budget)
    e2 = evaluate_plan(t2, weights, config)

    print("[阶段4] 策略4.3 ILP简化版...")
    t3 = strategy_ilp_simple(candidates, weights, config, budget)
    e3 = evaluate_plan(t3, weights, config)

    print("[阶段4] 策略4.4 复式/胆拖...")
    t4 = strategy_dantuo(candidates, weights, config, budget)
    e4 = evaluate_plan(t4, weights, config)

    # 策略对比
    strategies = [
        {"name": "贪心覆盖", "tickets": t1, "eval": e1},
        {"name": "分层覆盖", "tickets": t2, "eval": e2},
        {"name": "ILP精确解", "tickets": t3, "eval": e3},
        {"name": "复式/胆拖", "tickets": t4, "eval": e4},
    ]

    # 综合评分选最优
    for s in strategies:
        ev = s["eval"]
        s["score"] = (
            ev["weight_coverage"] * 0.35 +
            ev["top10_red_covered"] / 10 * 0.30 +
            ev["coverage_pct"] * 0.20 +
            (1 - ev["total_cost"] / budget) * 0.15
        )

    strategies.sort(key=lambda s: s["score"], reverse=True)
    best = strategies[0]

    print(f"[阶段4] 最优策略: {best['name']}，评分: {best['score']:.4f}")

    # 构建对比表
    comparison = []
    for s in strategies:
        ev = s["eval"]
        comparison.append({
            "name": s["name"],
            "score": round(s["score"], 4),
            "cost": ev["total_cost"],
            "combinations": ev["total_combinations"],
            "weight_coverage": round(ev["weight_coverage"], 4),
            "top10_covered": ev["top10_red_covered"],
        })

    best_eval = best["eval"]
    result = {
        "budget": budget,
        "total_cost": best_eval["total_cost"],
        "total_combinations": best_eval["total_combinations"],
        "tickets": best["tickets"],
        "coverage": {
            "all_red_numbers": best_eval["all_red_numbers"],
            "all_blue_numbers": best_eval["all_blue_numbers"],
            "top10_red_covered": best_eval["top10_red_covered"],
        },
        "coverage_pct": best_eval["coverage_pct"],
        "weight_coverage": best_eval["weight_coverage"],
        "efficiency": best_eval["efficiency"],
        "strategy_comparison": comparison,
    }

    print(f"[阶段4] 完成，总花费: {best_eval['total_cost']}元，"
          f"总注数: {best_eval['total_combinations']}注")

    return result
