# -*- coding: utf-8 -*-
"""E0-Step12：胆拖/复式购买策略优化

基于 Step10~11 的预测结果，生成最终购买策略：
  1. 胆码选择：高置信度位置的高概率号码作为胆码
  2. 拖码选择：中等置信度位置的候选号码作为拖码
  3. 复式方案：根据预算约束优化复式注数
  4. 回测评估：历史回测计算期望收益率

策略类型：
  A: 纯胆拖（胆码+拖码）
  B: 复式（每位置选N个号码的全组合）
  C: 混合（胆码固定+部分位置复式）

用法: python3 -m src.research.experiment.e0_step12_purchase_strategy
"""

import sys
from pathlib import Path
from itertools import combinations
from math import comb

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, Timer, save_json, load_json, RESULTS_DIR,
)
from research.experiment.e0_step10c_combo_rerank import (
    generate_amplitude_constrained_candidates,
)
from research.experiment.e0_step11_confidence import (
    compute_confidence,
)

STEP10_DIR = RESULTS_DIR / "e0_step10"
STEP11_DIR = RESULTS_DIR / "e0_step11"
STEP12_DIR = RESULTS_DIR / "e0_step12"
STEP12_DIR.mkdir(parents=True, exist_ok=True)

# 奖金表（简化）
PRIZE_TABLE = {
    'daletou': {
        # (红球命中数, 蓝球命中数): 奖金
        (5, 2): 10_000_000,  # 一等奖
        (5, 1): 800_000,     # 二等奖（浮动，取近似值）
        (5, 0): 10_000,      # 三等奖
        (4, 2): 3_000,       # 四等奖
        (4, 1): 300,         # 五等奖
        (3, 2): 200,         # 六等奖
        (4, 0): 100,         # 七等奖
        (3, 1): 15,          # 八等奖
        (2, 2): 15,          # 八等奖
        (3, 0): 5,           # 九等奖
        (2, 1): 5,           # 九等奖
        (1, 2): 5,           # 九等奖
        (0, 2): 5,           # 九等奖
    },
    'shuangseqiu': {
        (6, 1): 5_000_000,   # 一等奖
        (6, 0): 250_000,     # 二等奖（浮动）
        (5, 1): 3_000,       # 三等奖
        (5, 0): 200,         # 四等奖
        (4, 1): 200,         # 四等奖
        (4, 0): 10,          # 五等奖
        (3, 1): 10,          # 五等奖
        (2, 1): 5,           # 六等奖
        (1, 1): 5,           # 六等奖
        (0, 1): 5,           # 六等奖
    },
}

TICKET_PRICE = 2  # 每注2元


# ============================================================
#  胆码/拖码选择
# ============================================================

def select_dan_tuo(data, t, scan_results, confidence_threshold=0.6):
    """选择胆码和拖码

    Args:
        data: LotteryData
        t: 当前期索引
        scan_results: Step10a 结果
        confidence_threshold: 胆码置信度阈值

    Returns:
        dan_codes: list of (pos, number) 胆码
        tuo_codes: list of (pos, set_of_numbers) 拖码
        all_candidates: list of sets 每位置候选集
    """
    n_pos = data.red_count
    dan_codes = []
    tuo_codes = []
    all_candidates = []

    for pos in range(n_pos):
        conf, components = compute_confidence(data, pos, t, scan_results)
        candidates, info = generate_amplitude_constrained_candidates(
            data, pos, t, scan_results, confidence_level='medium')

        all_candidates.append(candidates)

        if conf >= confidence_threshold and len(candidates) <= 5:
            # 高置信度 + 候选集小 → 胆码（取中位数附近的号码）
            sorted_cands = sorted(candidates)
            mid_idx = len(sorted_cands) // 2
            dan_codes.append((pos, sorted_cands[mid_idx]))
        else:
            tuo_codes.append((pos, candidates))

    return dan_codes, tuo_codes, all_candidates


# ============================================================
#  策略A：纯胆拖
# ============================================================

def strategy_dan_tuo(data, t, scan_results, max_cost=20):
    """纯胆拖策略

    Returns:
        strategy: dict with dan, tuo, n_bets, cost
    """
    dan_codes, tuo_codes, all_candidates = select_dan_tuo(
        data, t, scan_results, confidence_threshold=0.6)

    n_pos = data.red_count

    # 胆码数量
    n_dan = len(dan_codes)

    # 拖码：每位置取 top-K 个候选
    # 计算允许的拖码数量（受预算约束）
    if n_dan >= n_pos:
        # 所有位置都有胆码，只需1注
        return {
            'type': 'dan_tuo',
            'dan': [(pos, int(num)) for pos, num in dan_codes],
            'tuo': [],
            'n_bets': 1,
            'cost': TICKET_PRICE,
        }

    # 拖码位置数
    n_tuo_pos = n_pos - n_dan
    # 每位置拖码数量，使总注数 <= max_cost / TICKET_PRICE
    max_bets = max_cost // TICKET_PRICE

    # 二分搜索每位置拖码数
    best_k = 1
    for k in range(2, 20):
        n_bets = k ** n_tuo_pos
        if n_bets <= max_bets:
            best_k = k
        else:
            break

    # 每位置取 best_k 个拖码
    tuo_selected = []
    for pos, candidates in tuo_codes:
        sorted_cands = sorted(candidates)
        # 取中间的 best_k 个
        if len(sorted_cands) <= best_k:
            selected = sorted_cands
        else:
            mid = len(sorted_cands) // 2
            half = best_k // 2
            start = max(0, mid - half)
            end = min(len(sorted_cands), start + best_k)
            start = max(0, end - best_k)
            selected = sorted_cands[start:end]
        tuo_selected.append((pos, [int(x) for x in selected]))

    # 计算实际注数
    n_bets = 1
    for _, nums in tuo_selected:
        n_bets *= len(nums)

    return {
        'type': 'dan_tuo',
        'dan': [(pos, int(num)) for pos, num in dan_codes],
        'tuo': tuo_selected,
        'n_bets': n_bets,
        'cost': n_bets * TICKET_PRICE,
    }


# ============================================================
#  策略B：复式
# ============================================================

def strategy_fushi(data, t, scan_results, max_cost=100):
    """复式策略：每位置选N个号码

    Returns:
        strategy: dict
    """
    n_pos = data.red_count
    max_bets = max_cost // TICKET_PRICE

    # 获取每位置候选集
    candidates_list = []
    for pos in range(n_pos):
        candidates, _ = generate_amplitude_constrained_candidates(
            data, pos, t, scan_results, confidence_level='medium')
        candidates_list.append(sorted(candidates))

    # 贪心缩减：从最大候选集开始缩减
    current_sizes = [len(c) for c in candidates_list]

    while True:
        n_bets = 1
        for s in current_sizes:
            n_bets *= s
        if n_bets <= max_bets:
            break

        # 缩减最大的位置
        max_idx = np.argmax(current_sizes)
        current_sizes[max_idx] = max(1, current_sizes[max_idx] - 1)

    # 取每位置的中间 K 个号码
    selected = []
    for pos in range(n_pos):
        k = current_sizes[pos]
        cands = candidates_list[pos]
        if len(cands) <= k:
            selected.append([int(x) for x in cands])
        else:
            mid = len(cands) // 2
            half = k // 2
            start = max(0, mid - half)
            end = min(len(cands), start + k)
            start = max(0, end - k)
            selected.append([int(x) for x in cands[start:end]])

    n_bets = 1
    for s in selected:
        n_bets *= len(s)

    return {
        'type': 'fushi',
        'positions': {f'P{i}': nums for i, nums in enumerate(selected)},
        'n_bets': n_bets,
        'cost': n_bets * TICKET_PRICE,
    }


# ============================================================
#  策略C：混合（胆码固定 + 部分复式）
# ============================================================

def strategy_hybrid(data, t, scan_results, max_cost=50):
    """混合策略"""
    dan_codes, tuo_codes, all_candidates = select_dan_tuo(
        data, t, scan_results, confidence_threshold=0.55)

    n_pos = data.red_count
    max_bets = max_cost // TICKET_PRICE

    # 胆码位置固定
    fixed = {}
    for pos, num in dan_codes:
        fixed[pos] = [int(num)]

    # 拖码位置复式
    variable = {}
    for pos, candidates in tuo_codes:
        variable[pos] = sorted([int(x) for x in candidates])

    # 缩减拖码使总注数 <= max_bets
    var_sizes = {pos: len(nums) for pos, nums in variable.items()}

    while True:
        n_bets = 1
        for s in var_sizes.values():
            n_bets *= s
        if n_bets <= max_bets:
            break

        max_pos = max(var_sizes, key=var_sizes.get)
        var_sizes[max_pos] = max(1, var_sizes[max_pos] - 1)

    # 取中间 K 个
    for pos in variable:
        k = var_sizes[pos]
        cands = variable[pos]
        if len(cands) > k:
            mid = len(cands) // 2
            half = k // 2
            start = max(0, mid - half)
            end = min(len(cands), start + k)
            start = max(0, end - k)
            variable[pos] = cands[start:end]

    n_bets = 1
    for nums in variable.values():
        n_bets *= len(nums)

    all_positions = {}
    for pos in range(n_pos):
        if pos in fixed:
            all_positions[f'P{pos}'] = {'type': 'dan', 'numbers': fixed[pos]}
        elif pos in variable:
            all_positions[f'P{pos}'] = {'type': 'tuo', 'numbers': variable[pos]}

    return {
        'type': 'hybrid',
        'positions': all_positions,
        'n_dan': len(dan_codes),
        'n_tuo': len(tuo_codes),
        'n_bets': n_bets,
        'cost': n_bets * TICKET_PRICE,
    }


# ============================================================
#  奖金计算
# ============================================================

def calculate_prize(true_red, true_blue, pred_red_set, pred_blue_set, lottery_type):
    """计算单注奖金

    Args:
        true_red: set of true red numbers
        true_blue: set of true blue numbers
        pred_red_set: set of predicted red numbers
        pred_blue_set: set of predicted blue numbers
        lottery_type: 'daletou' or 'shuangseqiu'

    Returns:
        prize: int
    """
    red_hit = len(true_red & pred_red_set)
    blue_hit = len(true_blue & pred_blue_set)

    prize_table = PRIZE_TABLE.get(lottery_type, {})
    return prize_table.get((red_hit, blue_hit), 0)


# ============================================================
#  回测评估
# ============================================================

def backtest_strategy(data, scan_results, test_start, test_end, lottery_type,
                      strategy_fn, max_cost=20):
    """回测策略

    Returns:
        dict with backtest results
    """
    series_all = data.position_series
    n_pos = data.red_count

    total_cost = 0
    total_prize = 0
    n_periods = 0
    prize_history = []
    hit_history = []

    for t in range(test_start + 5, test_end - 1):
        strategy = strategy_fn(data, t, scan_results, max_cost=max_cost)
        cost = strategy['cost']
        total_cost += cost

        # 真实下一期号码
        true_red = set()
        for pos in range(n_pos):
            true_red.add(int(series_all[pos][t + 1]))

        # 蓝球（简化：不预测蓝球，假设随机）
        true_blue = set()
        if hasattr(data, 'blue_position_series') and data.blue_position_series is not None:
            for bp in range(len(data.blue_position_series)):
                true_blue.add(int(data.blue_position_series[bp][t + 1]))

        # 生成所有注的号码组合并计算奖金
        period_prize = 0

        if strategy['type'] == 'dan_tuo':
            # 胆码 + 拖码的所有组合
            dan_nums = {pos: num for pos, num in strategy['dan']}
            tuo_lists = [(pos, nums) for pos, nums in strategy['tuo']]

            # 生成组合
            if tuo_lists:
                from itertools import product as iter_product
                tuo_combos = list(iter_product(*[nums for _, nums in tuo_lists]))
                for combo in tuo_combos:
                    pred_red = set()
                    for pos, num in dan_nums.items():
                        pred_red.add(num)
                    for i, (pos, _) in enumerate(tuo_lists):
                        pred_red.add(combo[i])

                    # 简化：蓝球随机，不计入
                    red_hit = len(true_red & pred_red)
                    # 用红球命中数估算奖金（蓝球假设0命中）
                    prize = PRIZE_TABLE.get(lottery_type, {}).get((red_hit, 0), 0)
                    period_prize += prize
            else:
                pred_red = {num for _, num in strategy['dan']}
                red_hit = len(true_red & pred_red)
                prize = PRIZE_TABLE.get(lottery_type, {}).get((red_hit, 0), 0)
                period_prize += prize

        elif strategy['type'] in ('fushi', 'hybrid'):
            # 复式：所有位置号码的笛卡尔积
            if strategy['type'] == 'fushi':
                pos_nums = [strategy['positions'][f'P{i}'] for i in range(n_pos)]
            else:
                pos_nums = []
                for i in range(n_pos):
                    key = f'P{i}'
                    if key in strategy['positions']:
                        pos_nums.append(strategy['positions'][key]['numbers'])
                    else:
                        pos_nums.append([int(series_all[i][t])])  # 默认当前值

            from itertools import product as iter_product
            # 限制组合数避免爆炸
            n_combos = 1
            for nums in pos_nums:
                n_combos *= len(nums)

            if n_combos <= 10000:
                for combo in iter_product(*pos_nums):
                    pred_red = set(combo)
                    red_hit = len(true_red & pred_red)
                    prize = PRIZE_TABLE.get(lottery_type, {}).get((red_hit, 0), 0)
                    period_prize += prize

        total_prize += period_prize
        n_periods += 1
        prize_history.append(period_prize)
        hit_history.append(period_prize > 0)

    # 统计
    roi = (total_prize - total_cost) / total_cost if total_cost > 0 else 0
    win_rate = sum(hit_history) / len(hit_history) if hit_history else 0

    return {
        'n_periods': n_periods,
        'total_cost': total_cost,
        'total_prize': total_prize,
        'net_profit': total_prize - total_cost,
        'roi': float(roi),
        'win_rate': float(win_rate),
        'avg_cost_per_period': float(total_cost / n_periods) if n_periods > 0 else 0,
        'avg_prize_per_period': float(total_prize / n_periods) if n_periods > 0 else 0,
        'max_prize': max(prize_history) if prize_history else 0,
        'prize_std': float(np.std(prize_history)) if prize_history else 0,
    }


# ============================================================
#  主入口
# ============================================================

def run_step12(lottery_type):
    """运行 Step12 购买策略优化"""
    log(f"\n{'═'*60}")
    log(f"  E0-Step12: 购买策略优化 [{lottery_type}]")
    log(f"{'═'*60}")

    data = LotteryData(lottery_type)
    n_draws = data.n_draws

    # 加载 Step10a 结果
    scan_path = STEP10_DIR / f"step10a_amplitude_scan_{lottery_type}.json"
    if not scan_path.exists():
        log(f"  Step10a 结果不存在，请先运行 Step10a")
        return None

    scan_results = load_json(scan_path)

    test_start = int(n_draws * 0.6)
    test_end = n_draws

    log(f"  总期数: {n_draws}, 测试期: [{test_start}, {test_end})")

    all_results = {
        'lottery_type': lottery_type,
        'strategies': {},
    }

    # 策略A：胆拖（预算20元）
    log(f"\n  --- 策略A: 胆拖 (预算20元) ---")
    with Timer("策略A 回测"):
        bt_a = backtest_strategy(
            data, scan_results, test_start, test_end, lottery_type,
            strategy_dan_tuo, max_cost=20)
    all_results['strategies']['dan_tuo_20'] = bt_a
    log(f"    总投入: {bt_a['total_cost']}元, 总奖金: {bt_a['total_prize']}元")
    log(f"    ROI: {bt_a['roi']:.2%}, 中奖率: {bt_a['win_rate']:.2%}")

    # 策略B：复式（预算100元）
    log(f"\n  --- 策略B: 复式 (预算100元) ---")
    with Timer("策略B 回测"):
        bt_b = backtest_strategy(
            data, scan_results, test_start, test_end, lottery_type,
            strategy_fushi, max_cost=100)
    all_results['strategies']['fushi_100'] = bt_b
    log(f"    总投入: {bt_b['total_cost']}元, 总奖金: {bt_b['total_prize']}元")
    log(f"    ROI: {bt_b['roi']:.2%}, 中奖率: {bt_b['win_rate']:.2%}")

    # 策略C：混合（预算50元）
    log(f"\n  --- 策略C: 混合 (预算50元) ---")
    with Timer("策略C 回测"):
        bt_c = backtest_strategy(
            data, scan_results, test_start, test_end, lottery_type,
            strategy_hybrid, max_cost=50)
    all_results['strategies']['hybrid_50'] = bt_c
    log(f"    总投入: {bt_c['total_cost']}元, 总奖金: {bt_c['total_prize']}元")
    log(f"    ROI: {bt_c['roi']:.2%}, 中奖率: {bt_c['win_rate']:.2%}")

    # 最优策略
    best_strategy = max(
        all_results['strategies'].keys(),
        key=lambda k: all_results['strategies'][k]['roi']
    )
    all_results['best_strategy'] = best_strategy
    all_results['best_roi'] = all_results['strategies'][best_strategy]['roi']

    log(f"\n  最优策略: {best_strategy}, ROI: {all_results['best_roi']:.2%}")

    # 生成下一期推荐
    t_latest = n_draws - 2  # 最新可用期
    log(f"\n  下一期推荐 (基于第{t_latest}期):")

    rec_dan_tuo = strategy_dan_tuo(data, t_latest, scan_results, max_cost=20)
    rec_fushi = strategy_fushi(data, t_latest, scan_results, max_cost=100)

    all_results['next_recommendation'] = {
        'dan_tuo': rec_dan_tuo,
        'fushi': rec_fushi,
    }

    log(f"    胆拖: 胆码={rec_dan_tuo['dan']}, 注数={rec_dan_tuo['n_bets']}")
    if rec_dan_tuo['tuo']:
        for pos, nums in rec_dan_tuo['tuo']:
            log(f"      P{pos} 拖码: {nums}")

    log(f"    复式: 注数={rec_fushi['n_bets']}, 费用={rec_fushi['cost']}元")
    for pos_key, nums in rec_fushi['positions'].items():
        log(f"      {pos_key}: {nums}")

    save_json(all_results, STEP12_DIR / f"step12_strategy_{lottery_type}.json")
    return all_results


def main():
    setup_logging()
    log("=" * 60)
    log("  E0-Step12: 胆拖/复式购买策略优化")
    log("=" * 60)

    results = {}
    for lt in ['daletou', 'shuangseqiu']:
        with Timer(f"Step12 [{lt}]"):
            results[lt] = run_step12(lt)

    # 保存汇总
    summary = {}
    for lt, r in results.items():
        if r:
            summary[lt] = {
                'best_strategy': r['best_strategy'],
                'best_roi': r['best_roi'],
            }
    save_json(summary, STEP12_DIR / "step12_summary.json")

    log("\n  Step12 全部完成!")


if __name__ == '__main__':
    main()
