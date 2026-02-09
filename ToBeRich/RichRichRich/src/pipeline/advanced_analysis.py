# -*- coding: utf-8 -*-
"""
高级分析主入口

统一调用三个分析器，返回合并结果：
  - 方向1：位置序列模式挖掘 + 差分分析
  - 方向2：序列模式挖掘（PrefixSpan）
  - 方向3：信息论评估
"""

from typing import Dict, List, Any

from pipeline.advanced.position_pattern import PositionPatternAnalyzer
from pipeline.advanced.sequence_mining import SequencePatternMiner
from pipeline.advanced.information_theory import InformationTheoryAnalyzer


def run_advanced_analysis(
    draws: List[Dict],
    red_count: int,
    red_range: int,
) -> Dict[str, Any]:
    """
    运行所有高级分析模块。

    参数:
        draws: 历史开奖数据
        red_count: 每期红球个数
        red_range: 红球号码范围上限

    返回:
        包含三个方向分析结果的字典
    """
    print("[高级分析] 方向1：位置序列模式挖掘...")
    pos_analyzer = PositionPatternAnalyzer(draws, red_count, red_range)
    position_result = pos_analyzer.analyze()

    print("[高级分析] 方向2：序列模式挖掘（PrefixSpan）...")
    seq_miner = SequencePatternMiner(draws, red_count, red_range)
    sequence_result = seq_miner.analyze()

    print("[高级分析] 方向3：信息论评估...")
    info_analyzer = InformationTheoryAnalyzer(draws, red_count, red_range)
    info_result = info_analyzer.analyze()

    # 汇总摘要
    summary = _build_summary(position_result, sequence_result, info_result)

    print(f"[高级分析] 完成，可利用信息评级: "
          f"{info_result.get('exploitable_information', {}).get('level', '?')}")

    return {
        "position_pattern": position_result,
        "sequence_mining": sequence_result,
        "information_theory": info_result,
        "summary": summary,
    }


def _build_summary(
    pos_result: Dict, seq_result: Dict, info_result: Dict
) -> Dict[str, Any]:
    """构建高级分析摘要"""
    # 位置预测摘要
    predictions = pos_result.get("position_predictions", {})
    pred_summary = []
    for pos, pred in predictions.items():
        if pred.get("confidence", 0) > 0.3:
            pred_summary.append({
                "position": pos,
                "direction": pred.get("direction", "?"),
                "confidence": pred.get("confidence", 0),
                "value_range": pred.get("value_range", []),
            })

    # 活跃规则摘要
    active_rules = seq_result.get("active_rules", [])
    active_summary = []
    for rule in active_rules[:10]:
        active_summary.append({
            "prediction": rule.get("prediction", ""),
            "confidence": rule.get("confidence", 0),
            "lift": rule.get("lift", 0),
            "description": rule.get("decoded", {}).get("description", ""),
        })

    # 信息论摘要
    exploit = info_result.get("exploitable_information", {})

    return {
        "position_predictions": pred_summary,
        "active_rules_count": len(active_rules),
        "top_active_rules": active_summary,
        "exploitable_info_level": exploit.get("level", "?"),
        "exploitable_info_score": exploit.get("score", 0),
    }
