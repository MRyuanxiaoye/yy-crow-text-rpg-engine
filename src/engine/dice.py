"""3d6 通用骰子判定系统。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from src.engine.dimension import DimensionReference, DimensionState, calculate_action_modifier


DC_TRIVIAL = 4
DC_EASY = 6
DC_MODERATE = 8
DC_HARD = 10
DC_VERY_HARD = 12
DC_NEARLY_IMPOSSIBLE = 14

RESULT_CRITICAL_SUCCESS = "大成功"
RESULT_SUCCESS = "成功"
RESULT_PARTIAL_SUCCESS = "部分成功"
RESULT_FAILURE = "失败"
RESULT_CRITICAL_FAILURE = "大失败"


@dataclass
class JudgmentResult:
    """一次 3d6 判定的完整结果。"""

    dice_values: list[int]
    total: int
    modifier: int
    dc: int
    final_value: int
    result_tier: str
    is_critical: bool


def roll_3d6() -> tuple[list[int], int]:
    """投掷三颗六面骰，返回每颗骰子的值和总和。"""

    dice_values = [random.randint(1, 6) for _ in range(3)]
    return dice_values, sum(dice_values)


def judge(roll_total: int, modifier: int, dc: int) -> str:
    """根据 3d6 总和、修正值和 DC 返回五档判定结果。

    大成功/大失败仅由天然 18/3 触发，并优先于普通数值判定。
    """

    if roll_total == 18:
        return RESULT_CRITICAL_SUCCESS
    if roll_total == 3:
        return RESULT_CRITICAL_FAILURE

    final_value = roll_total + modifier
    if final_value >= dc:
        return RESULT_SUCCESS
    if final_value >= dc - 2:
        return RESULT_PARTIAL_SUCCESS
    return RESULT_FAILURE


def quick_judge(
    state: DimensionState,
    main_dim: DimensionReference | str,
    aux_dims: Sequence[DimensionReference | str] | None = None,
    tags: Sequence[DimensionReference | str] | None = None,
    dc: int = DC_MODERATE,
    npc_id: str = "",
) -> JudgmentResult:
    """自动计算修正值、投掷 3d6，并返回完整判定结果。"""

    modifier = calculate_action_modifier(
        state=state,
        main_dimension=main_dim,
        auxiliary_dimensions=aux_dims,
        tags=tags,
        npc_id=npc_id,
    )
    dice_values, total = roll_3d6()
    result_tier = judge(total, modifier, dc)

    return JudgmentResult(
        dice_values=dice_values,
        total=total,
        modifier=modifier,
        dc=dc,
        final_value=total + modifier,
        result_tier=result_tier,
        is_critical=result_tier in {RESULT_CRITICAL_SUCCESS, RESULT_CRITICAL_FAILURE},
    )


__all__ = [
    "DC_EASY",
    "DC_HARD",
    "DC_MODERATE",
    "DC_NEARLY_IMPOSSIBLE",
    "DC_TRIVIAL",
    "DC_VERY_HARD",
    "JudgmentResult",
    "RESULT_CRITICAL_FAILURE",
    "RESULT_CRITICAL_SUCCESS",
    "RESULT_FAILURE",
    "RESULT_PARTIAL_SUCCESS",
    "RESULT_SUCCESS",
    "judge",
    "quick_judge",
    "roll_3d6",
]
