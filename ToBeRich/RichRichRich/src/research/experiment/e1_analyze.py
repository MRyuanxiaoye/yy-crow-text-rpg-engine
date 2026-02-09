# -*- coding: utf-8 -*-
"""E1 分析脚本：规律统计、与 Step10a 对比、覆盖率分析、冲突分析

用法: python3 -m src.research.experiment.e1_analyze
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.data_loader import LotteryData
from research.experiment.utils import (
    setup_logging, log, save_json, load_json, RESULTS_DIR,
)

E1_DIR = RESULTS_DIR / "e1_search"
TASK_DIR = E1_DIR / "tasks"
LIBRARY_DIR = E1_DIR / "library"
STEP10_DIR = RESULTS_DIR / "e0_step10"


def analyze_distribution(lottery):
    """规律分布统计：按维度、按位置"""
    path = LIBRARY_DIR / f"{lottery}_rules.json"
    if not path.exists():
        log(f"  {lottery} 规律库不存在，跳过")
        return None

    lib = load_json(path)
    rules = lib['rules']
    log(f"\n{'─' * 40}")
    log(f"  {lottery}: {len(rules)} 条规律")

    # 按维度统计
    by_dim = defaultdict(lambda: {'count': 0, 'avg_score': [], 'avg_support': []})
    # 按位置统计
    by_pos = defaultdict(lambda: {'count': 0, 'avg_score': []})
    # 按维度数统计
    by_ndim = defaultdict(int)

    for r in rules:
        dim_detail = r.get('dimension_detail', '')
        # 提取基础维度
        if '_x_' in dim_detail:
            parts = dim_detail.split('_x_')
            base_dims = []
            for p in parts:
                bd = p.split('_')[0]
                base_dims.append(bd)
            dim_key = ' × '.join(base_dims)
            by_ndim[len(parts)] += 1
        else:
            dim_key = dim_detail.split('_')[0]
            by_ndim[1] += 1

        by_dim[dim_key]['count'] += 1
        by_dim[dim_key]['avg_score'].append(r['quality_score'])
        by_dim[dim_key]['avg_support'].append(r.get('support_train', 0))

        # 从 rule_id 提取位置
        rid = r.get('rule_id', '')
        if '_P' in rid:
            pos_str = rid.split('_P')[1].split('_')[0]
            by_pos[f'P{pos_str}']['count'] += 1
            by_pos[f'P{pos_str}']['avg_score'].append(r['quality_score'])

    log(f"\n  按维度:")
    for dim_key in sorted(by_dim.keys()):
        info = by_dim[dim_key]
        avg_s = np.mean(info['avg_score']) if info['avg_score'] else 0
        avg_sup = np.mean(info['avg_support']) if info['avg_support'] else 0
        log(f"    {dim_key:20s}: {info['count']:4d} 条, "
            f"平均质量={avg_s:.3f}, 平均支持度={avg_sup:.0f}")

    log(f"\n  按位置:")
    for pos_key in sorted(by_pos.keys()):
        info = by_pos[pos_key]
        avg_s = np.mean(info['avg_score']) if info['avg_score'] else 0
        log(f"    {pos_key}: {info['count']:4d} 条, 平均质量={avg_s:.3f}")

    log(f"\n  按维度数: {dict(sorted(by_ndim.items()))}")

    return {
        'lottery': lottery,
        'total_rules': len(rules),
        'by_dimension': {k: v['count'] for k, v in by_dim.items()},
        'by_position': {k: v['count'] for k, v in by_pos.items()},
        'by_ndim': dict(by_ndim),
    }


def compare_with_step10a(lottery):
    """与 Step10a 结果对比：新增维度贡献了多少新规律"""
    step10a_path = STEP10_DIR / f"step10a_amplitude_scan_{lottery}.json"
    lib_path = LIBRARY_DIR / f"{lottery}_rules.json"

    if not step10a_path.exists() or not lib_path.exists():
        log(f"  {lottery} 缺少对比文件，跳过")
        return None

    step10a = load_json(step10a_path)
    lib = load_json(lib_path)

    # Step10a 的维度集合
    old_dims = {'D1', 'D2', 'D3', 'D4', 'D5', 'D6'}
    old_cross = {
        'D1_x_D3', 'D1_x_D4', 'D2_x_D3', 'D4_x_D6', 'D5_x_D3',
    }

    new_only = 0
    old_only = 0
    mixed_cross = 0

    for r in lib['rules']:
        dim_detail = r.get('dimension_detail', '')
        if '_x_' in dim_detail:
            parts = dim_detail.split('_x_')
            bases = set()
            for p in parts:
                bd = p.split('_')[0]
                # 去掉子参数如 D1n3 → D1
                base = ''
                for c in bd:
                    if c.isdigit() and base.startswith('D'):
                        base += c
                        break
                    base += c
                if not base[-1].isdigit():
                    base = base.rstrip('n')
                bases.add(base)

            if bases.issubset(old_dims):
                old_only += 1
            elif bases.isdisjoint(old_dims):
                new_only += 1
            else:
                mixed_cross += 1
        else:
            bd = dim_detail.split('_')[0]
            base = bd.rstrip('0123456789').rstrip('n') if not bd[1:].isdigit() else bd[:2]
            # 简化：D1-D6 为旧，D7-D12 为新
            dim_num = int(''.join(c for c in bd if c.isdigit())[:2])
            if dim_num <= 6:
                old_only += 1
            else:
                new_only += 1

    # Step10a 验证通过数
    step10a_validated = step10a.get('summary', {}).get('total_validated', 0)

    log(f"\n  与 Step10a 对比 [{lottery}]:")
    log(f"    Step10a 验证通过: {step10a_validated}")
    log(f"    E1 总规律: {lib['total_rules']}")
    log(f"    其中旧维度(D1-D6): {old_only}")
    log(f"    其中新维度(D7-D12): {new_only}")
    log(f"    混合交叉: {mixed_cross}")
    log(f"    新增贡献: {new_only + mixed_cross} 条 "
        f"({(new_only + mixed_cross) / max(lib['total_rules'], 1) * 100:.1f}%)")

    return {
        'lottery': lottery,
        'step10a_validated': step10a_validated,
        'e1_total': lib['total_rules'],
        'old_dims_only': old_only,
        'new_dims_only': new_only,
        'mixed_cross': mixed_cross,
        'new_contribution_pct': round(
            (new_only + mixed_cross) / max(lib['total_rules'], 1) * 100, 1),
    }


def analyze_coverage(lottery):
    """覆盖率分析：历史数据中有多少期能匹配到至少一条规律"""
    lib_path = LIBRARY_DIR / f"{lottery}_rules.json"
    if not lib_path.exists():
        log(f"  {lottery} 规律库不存在，跳过")
        return None

    lib = load_json(lib_path)
    rules = lib['rules']
    if not rules:
        return None

    data = LotteryData(lottery)
    n_draws = data.n_draws
    test_start = int(n_draws * 0.6)

    # 统计测试期每期匹配的规律数（简化：只统计 top-100 规律）
    top_rules = rules[:100]

    # 按位置分组
    rules_by_pos = defaultdict(list)
    for r in top_rules:
        rid = r.get('rule_id', '')
        if '_P' in rid:
            pos_str = rid.split('_P')[1].split('_')[0]
            rules_by_pos[int(pos_str)].append(r)

    n_test = n_draws - test_start
    # 每期是否至少有一个位置匹配到规律（简化统计）
    # 这里只统计规律数量分布，不做实际匹配（实际匹配需要重建标签，开销大）
    total_rules_per_pos = {pos: len(rs) for pos, rs in rules_by_pos.items()}

    log(f"\n  覆盖率分析 [{lottery}]:")
    log(f"    Top-100 规律按位置分布: {total_rules_per_pos}")
    log(f"    测试期: {n_test} 期")
    log(f"    注意: 精确覆盖率需要重建标签匹配，此处仅统计规律分布")

    return {
        'lottery': lottery,
        'top100_by_position': total_rules_per_pos,
        'test_periods': n_test,
    }


def analyze_conflicts(lottery):
    """冲突分析：同一维度+条件下是否有矛盾的幅度预测"""
    lib_path = LIBRARY_DIR / f"{lottery}_rules.json"
    if not lib_path.exists():
        return None

    lib = load_json(lib_path)
    rules = lib['rules']

    # 按 (dimension_detail, condition) 分组（理论上去重后不应有重复）
    # 但不同位置的同一条件可能有不同的幅度分布
    by_cond = defaultdict(list)
    for r in rules:
        key = r.get('dimension_detail', '') + '|' + r.get('condition', '')
        by_cond[key].append(r)

    conflicts = 0
    high_variance = 0
    for key, group in by_cond.items():
        if len(group) < 2:
            continue
        means = [r['amplitude_distribution']['mean'] for r in group]
        if max(means) - min(means) > 3.0:
            conflicts += 1
        if np.std(means) > 1.5:
            high_variance += 1

    log(f"\n  冲突分析 [{lottery}]:")
    log(f"    多位置共享条件组: {sum(1 for g in by_cond.values() if len(g) > 1)}")
    log(f"    均值差>3的冲突: {conflicts}")
    log(f"    高方差组(std>1.5): {high_variance}")

    return {
        'lottery': lottery,
        'shared_conditions': sum(1 for g in by_cond.values() if len(g) > 1),
        'mean_conflicts': conflicts,
        'high_variance_groups': high_variance,
    }


def main():
    setup_logging()
    log("=" * 50)
    log("  E1 规律分析")
    log("=" * 50)

    report = {}

    for lottery in ['daletou', 'shuangseqiu']:
        dist = analyze_distribution(lottery)
        comp = compare_with_step10a(lottery)
        cov = analyze_coverage(lottery)
        conf = analyze_conflicts(lottery)

        report[lottery] = {
            'distribution': dist,
            'step10a_comparison': comp,
            'coverage': cov,
            'conflicts': conf,
        }

    save_json(report, E1_DIR / "analysis_report.json")
    log("\n分析报告已保存")


if __name__ == '__main__':
    main()
