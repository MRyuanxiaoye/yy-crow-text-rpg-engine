"""阶段四：结果分析

对比3A/3B/3C实验结果，分析feature importance，生成最终报告。
"""

import numpy as np

from .utils import (
    log, Timer, save_json, load_json,
    EXPERIMENT_DIR,
)


def analyze_comparison(results_3a, results_3b, results_3c):
    """对比三个实验的结果"""
    comparison = {'methods': []}

    for name, res in [('3A_statistical', results_3a), ('3B_xgboost', results_3b), ('3C_mlp', results_3c)]:
        if res is None:
            continue
        entry = {
            'method': name,
            'direction_accuracy': res['overall']['direction_accuracy'],
            'avg_reduction': res['overall']['avg_reduction'],
            'avg_survival': res['overall']['avg_survival'],
            'efficiency': res['overall']['efficiency'],
            'per_position': {},
        }
        for pos_key, pos_data in res['per_position'].items():
            entry['per_position'][pos_key] = {
                'direction_accuracy': pos_data['direction_accuracy'],
                'avg_reduction': pos_data['avg_reduction'],
                'avg_survival': pos_data['avg_survival'],
                'efficiency': pos_data['efficiency'],
            }
        comparison['methods'].append(entry)

    # 排名
    if comparison['methods']:
        comparison['methods'].sort(key=lambda x: x['efficiency'], reverse=True)
        comparison['best_method'] = comparison['methods'][0]['method']
        comparison['best_efficiency'] = comparison['methods'][0]['efficiency']

    return comparison


def analyze_features(results_3b):
    """分析XGBoost的feature importance"""
    if results_3b is None or 'feature_importances' not in results_3b:
        return None

    analysis = {
        'per_position': {},
        'aggregated_top20': [],
        'category_importance': {},
    }

    # 汇总所有位置的importance
    all_imp = {}
    for pos_key, top_features in results_3b['feature_importances'].items():
        analysis['per_position'][pos_key] = top_features
        for fname, imp in top_features:
            if fname not in all_imp:
                all_imp[fname] = 0.0
            all_imp[fname] += imp

    # 全局top20
    sorted_imp = sorted(all_imp.items(), key=lambda x: x[1], reverse=True)[:20]
    analysis['aggregated_top20'] = sorted_imp

    # 按类别汇总
    categories = {
        'val_lag': 0.0,    # 位置值
        'dir_lag': 0.0,    # 方向
        'diff_lag': 0.0,   # 差分
        'stat_lag': 0.0,   # 统计量
        'a2_c': 0.0,       # A2规则簇
        'a1_p': 0.0,       # A1模式
        'n_triggered': 0.0, # 触发元信息
        'avg_conf': 0.0,
        'avg_lift': 0.0,
        'n_matched': 0.0,
    }
    for fname, imp in all_imp.items():
        matched = False
        for prefix in categories:
            if fname.startswith(prefix):
                categories[prefix] += imp
                matched = True
                break
        if not matched:
            categories.setdefault('other', 0.0)
            categories['other'] += imp

    # 归一化
    total = sum(categories.values())
    if total > 0:
        analysis['category_importance'] = {k: round(v / total, 4) for k, v in
                                            sorted(categories.items(), key=lambda x: x[1], reverse=True)
                                            if v > 0}

    # 规则特征 vs 局面特征占比
    rule_imp = categories.get('a2_c', 0) + categories.get('a1_p', 0)
    context_imp = categories.get('val_lag', 0) + categories.get('dir_lag', 0) + \
                  categories.get('diff_lag', 0) + categories.get('stat_lag', 0)
    meta_imp = categories.get('n_triggered', 0) + categories.get('avg_conf', 0) + \
               categories.get('avg_lift', 0) + categories.get('n_matched', 0)

    if total > 0:
        analysis['importance_split'] = {
            'rule_features': round(rule_imp / total, 4),
            'context_features': round(context_imp / total, 4),
            'meta_features': round(meta_imp / total, 4),
        }

    return analysis


def analyze_failures(results_3b, lottery_type):
    """分析存活率低的位置和失败模式"""
    if results_3b is None:
        return None

    failures = {'low_survival_positions': [], 'observations': []}

    for pos_key, pos_data in results_3b['per_position'].items():
        if pos_data['avg_survival'] < 0.85:
            failures['low_survival_positions'].append({
                'position': pos_key,
                'survival': pos_data['avg_survival'],
                'reduction': pos_data['avg_reduction'],
                'efficiency': pos_data['efficiency'],
            })

    if not failures['low_survival_positions']:
        failures['observations'].append("所有位置存活率均 >= 85%，无明显失败模式")
    else:
        failures['observations'].append(
            f"{len(failures['low_survival_positions'])} 个位置存活率 < 85%，"
            "可能需要降低排除阈值或增加规则覆盖"
        )

    # 缩减率 vs 存活率的权衡分析
    if results_3b['overall']['avg_reduction'] < 0.1:
        failures['observations'].append("缩减率过低（<10%），规则触发不足或方向预测不够确定")
    if results_3b['overall']['avg_survival'] < 0.7:
        failures['observations'].append("存活率过低（<70%），排除过于激进，需要放宽阈值")

    return failures


def generate_final_report(lottery_type, comparison, feature_analysis, failure_analysis,
                          results_3a, results_3b, results_3c):
    """生成最终报告"""
    report = {
        'lottery_type': lottery_type,
        'comparison': comparison,
        'feature_analysis': feature_analysis,
        'failure_analysis': failure_analysis,
        'conclusion': {},
        'recommendations': [],
    }

    # 结论
    if comparison and comparison.get('best_method'):
        best = comparison['methods'][0]
        report['conclusion'] = {
            'best_method': best['method'],
            'best_efficiency': best['efficiency'],
            'direction_accuracy': best['direction_accuracy'],
            'avg_reduction': best['avg_reduction'],
            'avg_survival': best['avg_survival'],
        }

        # 判断实验是否成功
        eff = best['efficiency']
        if eff >= 0.25:
            report['conclusion']['verdict'] = 'SUCCESS'
            report['conclusion']['message'] = f"综合效率 {eff:.4f} >= 0.25，规则适配方案可行"
        elif eff >= 0.15:
            report['conclusion']['verdict'] = 'PARTIAL'
            report['conclusion']['message'] = f"综合效率 {eff:.4f}，有一定效果但未达目标，需优化"
        else:
            report['conclusion']['verdict'] = 'INSUFFICIENT'
            report['conclusion']['message'] = f"综合效率 {eff:.4f} < 0.15，当前方案效果不足"

    # 建议
    if comparison and comparison.get('methods'):
        best = comparison['methods'][0]

        if best['avg_reduction'] < 0.2:
            report['recommendations'].append(
                "缩减率不足：考虑降低方向概率排除阈值（当前0.15），或增加更多规则特征")
        if best['avg_survival'] < 0.85:
            report['recommendations'].append(
                "存活率不足：考虑提高方向概率排除阈值，减少误排除")

        if feature_analysis and feature_analysis.get('importance_split'):
            split = feature_analysis['importance_split']
            if split.get('rule_features', 0) < 0.1:
                report['recommendations'].append(
                    "规则特征贡献低（<10%）：规则簇可能压缩过度，考虑增加簇数或使用原始规则")
            if split.get('context_features', 0) < 0.1:
                report['recommendations'].append(
                    "局面特征贡献低（<10%）：考虑增加更多局面特征（如更长的lag、二阶差分等）")

        # 是否需要严格时间切分
        report['recommendations'].append(
            "当前规则用全量数据挖掘，存在数据泄露风险。"
            "如果实验结果有价值，建议用时间切分后的数据重新挖掘规则做严格验证")

        # v2 建议
        if best['efficiency'] >= 0.15:
            report['recommendations'].append(
                f"v2 Step 3 建议采用 {best['method']} 方案，"
                f"方向预测准确率 {best['direction_accuracy']:.4f}，"
                f"可在此基础上叠加约束传播进一步缩号")

    return report


def run_phase4(lottery_type, results_3a, results_3b, results_3c):
    """执行阶段四：结果分析

    Returns:
        final_report: 最终报告字典
    """
    log(f"\n{'─'*40}")
    log(f"阶段四：结果分析 [{lottery_type}]")
    log(f"{'─'*40}")

    with Timer("方案对比"):
        comparison = analyze_comparison(results_3a, results_3b, results_3c)
        save_json(comparison, EXPERIMENT_DIR / f"phase4_comparison_{lottery_type}.json")
        if comparison.get('best_method'):
            log(f"    最优方案: {comparison['best_method']}, 效率={comparison['best_efficiency']:.4f}")

    with Timer("特征分析"):
        feature_analysis = analyze_features(results_3b)
        if feature_analysis:
            save_json(feature_analysis, EXPERIMENT_DIR / f"phase4_feature_analysis_{lottery_type}.json")
            if feature_analysis.get('importance_split'):
                split = feature_analysis['importance_split']
                log(f"    特征贡献: 规则={split.get('rule_features', 0):.1%}, "
                    f"局面={split.get('context_features', 0):.1%}, "
                    f"元信息={split.get('meta_features', 0):.1%}")

    with Timer("失败分析"):
        failure_analysis = analyze_failures(results_3b, lottery_type)
        if failure_analysis:
            save_json(failure_analysis, EXPERIMENT_DIR / f"phase4_failure_analysis_{lottery_type}.json")
            for obs in failure_analysis.get('observations', []):
                log(f"    {obs}")

    with Timer("生成最终报告"):
        final_report = generate_final_report(
            lottery_type, comparison, feature_analysis, failure_analysis,
            results_3a, results_3b, results_3c)
        save_json(final_report, EXPERIMENT_DIR / f"phase4_final_report_{lottery_type}.json")

        if final_report.get('conclusion'):
            c = final_report['conclusion']
            log(f"\n  ═══ 结论 ═══")
            log(f"  {c.get('verdict', 'N/A')}: {c.get('message', '')}")
            log(f"  最优方案: {c.get('best_method', 'N/A')}")
            log(f"  方向准确率: {c.get('direction_accuracy', 0):.4f}")
            log(f"  缩减率: {c.get('avg_reduction', 0):.4f}")
            log(f"  存活率: {c.get('avg_survival', 0):.4f}")
            log(f"  综合效率: {c.get('best_efficiency', 0):.4f}")

        if final_report.get('recommendations'):
            log(f"\n  ═══ 建议 ═══")
            for i, rec in enumerate(final_report['recommendations'], 1):
                log(f"  {i}. {rec}")

    return final_report
