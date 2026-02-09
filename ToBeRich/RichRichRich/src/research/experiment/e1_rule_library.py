# -*- coding: utf-8 -*-
"""E1 Layer 3: 规律库汇总

扫描所有已完成的原子任务结果，合并验证通过的规律，
计算质量评分，去重排序，输出统一规律库。

用法: python3 -m src.research.experiment.e1_rule_library
"""

import sys
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research.experiment.utils import (
    setup_logging, log, save_json, load_json, RESULTS_DIR,
)

# === 路径 ===
E1_DIR = RESULTS_DIR / "e1_search"
TASK_DIR = E1_DIR / "tasks"
LIBRARY_DIR = E1_DIR / "library"
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

# === 质量评分权重 ===
W1 = 0.2   # 测试期 KS 显著性
W2 = 0.3   # JS 散度
W3 = 0.2   # 集中度（越低越好 → 1-concentration）
W4 = 0.1   # 支持度（log）
W5 = 0.2   # 训练/测试均值一致性


def compute_quality_score(pattern, uncond_std):
    """计算规律质量评分

    quality = w1*(1-ks_p_test) + w2*js_div + w3*(1-concentration)
            + w4*log(support_test) + w5*(1 - |mean_diff|/uncond_std)
    """
    val = pattern.get('validation', {})
    if val.get('status') != 'valid':
        return 0.0

    ks_p_test = val.get('ks_p', 1.0)
    js_div = pattern.get('js_divergence', 0)
    concentration = pattern.get('concentration', 1.0)
    support_test = val.get('n_test', 1)
    mean_diff = val.get('train_test_mean_diff', 999)

    # 归一化各项到 [0, 1] 范围
    s1 = 1.0 - min(ks_p_test, 1.0)
    s2 = min(js_div / 0.3, 1.0)  # JS 散度上限约 0.3
    s3 = max(0, 1.0 - concentration)
    s4 = min(math.log(max(support_test, 1)) / math.log(200), 1.0)  # log(200)≈5.3
    s5 = max(0, 1.0 - mean_diff / max(uncond_std, 0.01))

    score = W1 * s1 + W2 * s2 + W3 * s3 + W4 * s4 + W5 * s5
    return round(score, 4)


def generate_condition_human(dimension, condition):
    """生成条件的人类可读描述"""
    dim = dimension
    cond = condition

    descriptions = {
        'D1': '方向序列',
        'D2': '幅度区间',
        'D3': '值域位置',
        'D4': '差分符号+幅度',
        'D5': '差分变化模式',
        'D6': '遗漏期数',
        'D7': '跨位置方向联合',
        'D8': '组合统计量',
        'D9': '奇偶大小格局',
        'D10': '长期趋势',
        'D11': '连号状态',
        'D12': '冷热度',
    }

    # 提取主维度名
    base_dim = dim.split('_')[0]
    if base_dim.startswith('D') and base_dim[1:].split('n')[0].isdigit():
        # 如 D1_n3, D2_n2, D8_sum, D10_w10
        base_dim = 'D' + base_dim[1:].split('n')[0].split('_')[0]

    desc = descriptions.get(base_dim, dim)

    # 交叉维度
    if '_x_' in dim:
        parts = dim.split('_x_')
        descs = []
        for p in parts:
            bd = p.split('_')[0]
            if bd.startswith('D'):
                key = 'D' + bd[1:].split('n')[0]
                descs.append(descriptions.get(key, p))
            else:
                descs.append(p)
        desc = ' × '.join(descs)

    return f"{desc}: {cond}"


def build_library():
    """扫描所有任务结果，构建规律库"""
    log("开始构建规律库...")

    # 按 (lottery, position) 分组收集
    rules_by_lp = defaultdict(list)
    task_count = 0
    skip_count = 0

    for f in sorted(TASK_DIR.glob("*.json")):
        try:
            result = load_json(f)
        except Exception as e:
            log(f"  跳过损坏文件: {f.name} ({e})")
            skip_count += 1
            continue

        if result.get('status') != 'completed':
            skip_count += 1
            continue

        task_count += 1
        lottery = result.get('lottery', '')
        pos = result.get('position', 0)
        uncond_std = result.get('uncond_std', 1.0)
        dims = result.get('dimensions', [])

        for pat in result.get('patterns', []):
            val = pat.get('validation', {})
            if val.get('status') != 'valid':
                continue

            # 计算质量评分
            score = compute_quality_score(pat, uncond_std)

            rule = {
                'dimensions': dims,
                'dimension_detail': pat['dimension'],
                'condition': pat['condition'],
                'condition_human': generate_condition_human(
                    pat['dimension'], pat['condition']),
                'support_train': pat.get('support', 0),
                'support_test': val.get('n_test', 0),
                'amplitude_distribution': {
                    'mean': pat.get('mean', 0),
                    'median': pat.get('median', 0),
                    'q10': pat.get('q10', 0),
                    'q25': pat.get('q25', 0),
                    'q50': pat.get('q50', 0),
                    'q75': pat.get('q75', 0),
                    'q90': pat.get('q90', 0),
                    'iqr': pat.get('iqr', 0),
                    'concentration': pat.get('concentration', 1),
                },
                'significance': {
                    'ks_stat': pat.get('ks_stat', 0),
                    'ks_p': pat.get('ks_p', 1),
                    'js_divergence': pat.get('js_divergence', 0),
                    'mean_shift_sigma': pat.get('mean_shift_sigma', 0),
                },
                'validation': {
                    'status': val.get('status'),
                    'ks_p_test': val.get('ks_p', 1),
                    'mean_test': val.get('test_mean', 0),
                    'mean_diff': val.get('train_test_mean_diff', 0),
                },
                'quality_score': score,
            }

            rules_by_lp[(lottery, pos)].append(rule)

    log(f"  扫描完成: {task_count} 个任务, {skip_count} 个跳过")

    # 去重 + 排序 + 输出
    libraries = {}
    for (lottery, pos), rules in rules_by_lp.items():
        # 去重：相同 (dimension_detail, condition) 只保留质量最高的
        seen = {}
        for r in rules:
            key = (r['dimension_detail'], r['condition'])
            if key not in seen or r['quality_score'] > seen[key]['quality_score']:
                seen[key] = r

        unique_rules = list(seen.values())
        unique_rules.sort(key=lambda r: r['quality_score'], reverse=True)

        # 分配 rule_id
        short = 'dlt' if lottery == 'daletou' else 'ssq'
        for i, r in enumerate(unique_rules):
            r['rule_id'] = f"{short}_P{pos}_{i:04d}"

        lp_key = f"{lottery}_P{pos}"
        libraries[lp_key] = {
            'lottery': lottery,
            'position': pos,
            'total_rules': len(unique_rules),
            'rules': unique_rules,
        }

    # 按彩种汇总输出
    for lottery in ['daletou', 'shuangseqiu']:
        all_rules = []
        pos_summary = {}
        for key, lib in libraries.items():
            if lib['lottery'] != lottery:
                continue
            all_rules.extend(lib['rules'])
            pos_summary[f"P{lib['position']}"] = lib['total_rules']

        if not all_rules:
            continue

        all_rules.sort(key=lambda r: r['quality_score'], reverse=True)

        output = {
            'version': '1.0',
            'lottery': lottery,
            'total_rules': len(all_rules),
            'by_position': pos_summary,
            'by_dimension': _count_by_dimension(all_rules),
            'quality_stats': _quality_stats(all_rules),
            'rules': all_rules,
        }

        path = LIBRARY_DIR / f"{lottery}_rules.json"
        save_json(output, path)
        log(f"  {lottery}: {len(all_rules)} 条规律, 保存到 {path.name}")

    # 汇总报告
    summary = {
        'total_tasks_scanned': task_count,
        'libraries': {},
    }
    for lottery in ['daletou', 'shuangseqiu']:
        path = LIBRARY_DIR / f"{lottery}_rules.json"
        if path.exists():
            lib = load_json(path)
            summary['libraries'][lottery] = {
                'total_rules': lib['total_rules'],
                'by_position': lib['by_position'],
                'by_dimension': lib['by_dimension'],
                'quality_stats': lib['quality_stats'],
            }

    save_json(summary, LIBRARY_DIR / "summary.json")
    log("规律库构建完成")
    return summary


def _count_by_dimension(rules):
    """按维度统计规律数量"""
    counts = defaultdict(int)
    for r in rules:
        # 提取主维度
        dim = r.get('dimension_detail', '')
        if '_x_' in dim:
            counts['cross'] += 1
        else:
            base = dim.split('_')[0]
            counts[base] += 1
    return dict(counts)


def _quality_stats(rules):
    """质量评分统计"""
    if not rules:
        return {}
    scores = [r['quality_score'] for r in rules]
    return {
        'mean': round(float(np.mean(scores)), 4),
        'median': round(float(np.median(scores)), 4),
        'min': round(float(np.min(scores)), 4),
        'max': round(float(np.max(scores)), 4),
        'q25': round(float(np.percentile(scores, 25)), 4),
        'q75': round(float(np.percentile(scores, 75)), 4),
    }


def main():
    setup_logging()
    log("=" * 50)
    log("  E1 规律库汇总")
    log("=" * 50)
    summary = build_library()

    for lottery, info in summary.get('libraries', {}).items():
        log(f"\n  {lottery}:")
        log(f"    总规律数: {info['total_rules']}")
        log(f"    按位置: {info['by_position']}")
        log(f"    按维度: {info['by_dimension']}")
        qs = info['quality_stats']
        log(f"    质量评分: mean={qs['mean']}, median={qs['median']}, "
            f"range=[{qs['min']}, {qs['max']}]")


if __name__ == '__main__':
    main()
