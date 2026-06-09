"""行动识别、两步确认与被动响应流程。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Any, Mapping, Sequence

from src.engine.dice import (
    RESULT_CRITICAL_FAILURE,
    RESULT_CRITICAL_SUCCESS,
    RESULT_FAILURE,
    RESULT_PARTIAL_SUCCESS,
    RESULT_SUCCESS,
    JudgmentResult,
    judge,
    quick_judge,
)
from src.engine.dimension import DimensionReference, DimensionState, calculate_action_modifier
from src.engine.state import GameState


class ActionType(str, Enum):
    """玩家与系统可识别的四种动作类型。"""

    DIALOGUE = "dialogue"
    DECISION = "decision"
    ACTION = "action"
    PASSIVE_RESPONSE = "passive_response"


@dataclass
class ActionProposal:
    """行动确认卡片所需的结构化行动提案。"""

    action_type: ActionType
    description: str
    dc: int
    modifier: int
    main_dimension: str
    auxiliary_dimensions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    time_cost: int = 0
    success_rate: float = 0.0
    npc_id: str = ""


@dataclass
class ActionResult:
    """一次行动确认并掷骰后的结算结果。"""

    proposal: ActionProposal
    judgment: JudgmentResult
    narrative_hint: str
    effects: dict[str, Any] = field(default_factory=dict)


# 3d6 总可能数固定为 216，预计算每个点数出现次数，避免每次重复枚举。
_3D6_TOTAL_COUNTS: dict[int, int] = {}
for dice_values in product(range(1, 7), repeat=3):
    total = sum(dice_values)
    _3D6_TOTAL_COUNTS[total] = _3D6_TOTAL_COUNTS.get(total, 0) + 1
_3D6_TOTAL_OUTCOMES = 6**3


_NARRATIVE_HINTS: dict[str, str] = {
    RESULT_CRITICAL_SUCCESS: "大成功：除达成原目标外，应考虑追加机制性影响，如解锁新行动路径、额外提升维度、获得意外资源或关键情报。",
    RESULT_SUCCESS: "成功：行动按预期达成，叙事应明确玩家目标如何落实，并给出相应状态变化。",
    RESULT_PARTIAL_SUCCESS: "部分成功：目标基本推进，但需要让玩家或系统选择一项代价，如消耗资源、引发怀疑、牺牲时间或留下隐患。",
    RESULT_FAILURE: "失败：行动未能达成，叙事应体现条件变化，如机会流失、对手警觉、局势恶化或难度上升。",
    RESULT_CRITICAL_FAILURE: "大失败：行动造成不可逆后果，如关键人物死亡/决裂、长期压力源生成、重要资源永久损失或强制触发危机事件。",
}


def calculate_success_rate(modifier: int, dc: int) -> float:
    """按 3d6 概率分布计算“成功及以上”的预估概率。"""

    successful_outcomes = 0
    for roll_total, count in _3D6_TOTAL_COUNTS.items():
        result_tier = judge(roll_total, int(modifier), int(dc))
        if result_tier in {RESULT_SUCCESS, RESULT_CRITICAL_SUCCESS}:
            successful_outcomes += count
    return successful_outcomes / _3D6_TOTAL_OUTCOMES


def create_action_proposal(
    description: str,
    dc: int,
    state: GameState | DimensionState,
    main_dim: DimensionReference | str,
    aux_dims: Sequence[DimensionReference | str] | None = None,
    tags: Sequence[DimensionReference | str] | None = None,
    time_cost: int = 0,
    npc_id: str = "",
) -> ActionProposal:
    """组装主动行动的两步确认卡片数据。"""

    dimension_state = _get_dimension_state(state)
    normalized_aux_dims = list(aux_dims or [])
    normalized_tags = list(tags or [])
    modifier = calculate_action_modifier(
        state=dimension_state,
        main_dimension=main_dim,
        auxiliary_dimensions=normalized_aux_dims,
        tags=normalized_tags,
        npc_id=npc_id,
    )

    return ActionProposal(
        action_type=ActionType.ACTION,
        description=str(description).strip(),
        dc=int(dc),
        modifier=modifier,
        main_dimension=_reference_name(main_dim),
        auxiliary_dimensions=[_reference_name(item) for item in normalized_aux_dims],
        tags=[_reference_name(item) for item in normalized_tags],
        time_cost=max(0, int(time_cost)),
        success_rate=calculate_success_rate(modifier, int(dc)),
        npc_id=str(npc_id).strip(),
    )


def execute_action(proposal: ActionProposal, state: GameState | DimensionState) -> ActionResult:
    """玩家确认行动后，调用骰子系统执行判定并返回叙事提示。"""

    dimension_state = _get_dimension_state(state)
    judgment = quick_judge(
        state=dimension_state,
        main_dim=proposal.main_dimension,
        aux_dims=proposal.auxiliary_dimensions,
        tags=proposal.tags,
        dc=proposal.dc,
        npc_id=proposal.npc_id,
    )
    return ActionResult(
        proposal=proposal,
        judgment=judgment,
        narrative_hint=_NARRATIVE_HINTS.get(judgment.result_tier, "根据判定档位生成对应叙事与状态影响。"),
        effects=_default_effects_for_result(proposal, judgment),
    )


def create_passive_response(
    event_description: str,
    suggested_responses: list[dict],
    state: GameState | DimensionState,
) -> dict[str, Any]:
    """组装被动响应事件卡片信息，建议选项复用行动确认提案结构。"""

    response_proposals: list[ActionProposal] = []
    for response in suggested_responses:
        if not isinstance(response, dict):
            continue
        proposal = create_action_proposal(
            description=str(response.get("description", "")).strip(),
            dc=int(response.get("dc", 8)),
            state=state,
            main_dim=response.get("main_dim", response.get("main_dimension", "意志")),
            aux_dims=response.get("aux_dims", response.get("auxiliary_dimensions", [])),
            tags=response.get("tags", []),
            time_cost=int(response.get("time_cost", 0)),
            npc_id=str(response.get("npc_id", "")).strip(),
        )
        proposal.action_type = ActionType.PASSIVE_RESPONSE
        response_proposals.append(proposal)

    return {
        "action_type": ActionType.PASSIVE_RESPONSE,
        "event_description": str(event_description).strip(),
        "suggested_responses": response_proposals,
        "allow_custom_response": True,
        "confirm_label": "确认执行",
        "change_label": "换个方式",
        "can_abandon": False,
    }


def resolve_partial_success(proposal: ActionProposal, cost_options: list[dict]) -> dict[str, Any]:
    """处理部分成功的代价选择，默认采用列表中的第一项代价。"""

    normalized_options = [_normalize_cost_option(option) for option in cost_options if isinstance(option, dict)]
    if not normalized_options:
        return {
            "selected_cost": None,
            "effects": {},
            "narrative_hint": "部分成功但未提供代价选项，请由叙事层补充一个合理代价。",
        }

    selected_cost = normalized_options[0]
    return {
        "proposal_description": proposal.description,
        "selected_cost": selected_cost,
        "effects": dict(selected_cost.get("effects", {})),
        "narrative_hint": str(selected_cost.get("narrative_hint", selected_cost.get("description", ""))).strip(),
    }


def _get_dimension_state(state: GameState | DimensionState) -> DimensionState:
    """从 GameState 或 DimensionState 中取得维度运行态。"""

    if isinstance(state, DimensionState):
        return state
    dimensions = getattr(state, "dimensions", None)
    if isinstance(dimensions, DimensionState):
        return dimensions
    raise TypeError("state 必须是 GameState 或 DimensionState")


def _reference_name(reference: DimensionReference | str) -> str:
    """提取维度或标签的展示名称。"""

    if isinstance(reference, DimensionReference):
        return reference.name
    return str(reference).strip()


def _default_effects_for_result(proposal: ActionProposal, judgment: JudgmentResult) -> dict[str, Any]:
    """给叙事/状态层的默认效果骨架，具体数值由上层或AI补全。"""

    effects: dict[str, Any] = {
        "action_description": proposal.description,
        "result_tier": judgment.result_tier,
    }
    if proposal.time_cost > 0:
        effects["time_cost"] = proposal.time_cost
    if judgment.result_tier == RESULT_PARTIAL_SUCCESS:
        effects["requires_cost_choice"] = True
    if judgment.result_tier == RESULT_FAILURE:
        effects["condition_changed"] = True
    if judgment.result_tier == RESULT_CRITICAL_FAILURE:
        effects["irreversible_consequence"] = True
    if judgment.result_tier == RESULT_CRITICAL_SUCCESS:
        effects["mechanical_bonus"] = True
    return effects


def _normalize_cost_option(option: Mapping[str, Any]) -> dict[str, Any]:
    """清洗代价选项，确保返回结构稳定。"""

    effects = option.get("effects", {})
    return {
        "id": str(option.get("id", "")).strip(),
        "description": str(option.get("description", "")).strip(),
        "effects": dict(effects) if isinstance(effects, dict) else {},
        "narrative_hint": str(option.get("narrative_hint", "")).strip(),
    }


__all__ = [
    "ActionProposal",
    "ActionResult",
    "ActionType",
    "calculate_success_rate",
    "create_action_proposal",
    "create_passive_response",
    "execute_action",
    "resolve_partial_success",
]
