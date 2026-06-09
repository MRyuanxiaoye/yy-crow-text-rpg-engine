"""新结算系统：底线检查与结局叙事。"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from src.engine.dimension import DimensionReference, get_dimension_value
from src.engine.state import GameState, get_state_manager
from src.llm.client import get_llm_client

logger = logging.getLogger(__name__)

ENDING_SYSTEM_PROMPT = """
你是文字RPG结算旁白。请根据当前目标、维度状态、事件历史与结局定义生成最终叙事。
要求：
1. 明确说明为什么进入该结局。
2. 回收玩家关键行动、压力源和底线击穿原因。
3. 不使用旧指标概念，统一称为维度或局势。
4. 语气克制，有因果感，篇幅200-450字。
""".strip()


def _to_int(value: Any, default: int = 0) -> int:
    """安全转换整数。"""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _compare(left: Any, operator: str, right: Any) -> bool:
    """执行通用条件比较。"""

    op = str(operator or "==").strip()
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        left_value = _to_int(left)
        right_value = _to_int(right)
    else:
        left_value = str(left)
        right_value = str(right)

    if op in {"<", "below", "lt"}:
        return left_value < right_value
    if op in {"<=", "at_most", "lte"}:
        return left_value <= right_value
    if op in {">", "above", "gt"}:
        return left_value > right_value
    if op in {">=", "at_least", "gte"}:
        return left_value >= right_value
    if op in {"==", "=", "equals", "eq"}:
        return left_value == right_value
    if op in {"!=", "not_equals", "neq"}:
        return left_value != right_value
    return False


def _read_dimension(state: GameState, name: str, npc_id: str = "") -> int | None:
    """读取维度值，支持自动分类与关系维度。"""

    if not name:
        return None
    try:
        return get_dimension_value(state.dimensions, DimensionReference(name=name, category="auto", npc_id=npc_id))
    except (KeyError, TypeError, ValueError):
        return None


def _latest_decision_choice(state: GameState, decision_id: str) -> str | None:
    """从决策历史中取最近一次选择。"""

    for decision in reversed(state.decisions):
        if not isinstance(decision, Mapping):
            continue
        if str(decision.get("decision_id", "")) != decision_id and str(decision.get("id", "")) != decision_id:
            continue
        choice = decision.get("choice") or decision.get("choice_key") or decision.get("decision")
        if choice:
            return str(choice)
    return None


def _condition_met(state: GameState, condition: Any) -> bool:
    """检查结局/底线条件。"""

    if isinstance(condition, list):
        return all(_condition_met(state, item) for item in condition)
    if not isinstance(condition, Mapping):
        return False
    if "all" in condition:
        return all(_condition_met(state, item) for item in _as_list(condition.get("all")))
    if "any" in condition:
        return any(_condition_met(state, item) for item in _as_list(condition.get("any")))
    if "not" in condition:
        return not _condition_met(state, condition.get("not"))

    if "bottom_line_triggered" in condition:
        triggered = state.storylines.get("bottom_line_triggered")
        return str(triggered) == str(condition.get("bottom_line_triggered"))
    if "event_triggered" in condition:
        return str(condition.get("event_triggered")) in state.event_history
    if "event_not_triggered" in condition:
        return str(condition.get("event_not_triggered")) not in state.event_history
    if "decision_id" in condition:
        actual = _latest_decision_choice(state, str(condition.get("decision_id", "")))
        return _compare(actual, str(condition.get("op", condition.get("operator", "=="))), condition.get("choice"))

    dimension_name = str(condition.get("dimension", condition.get("metric", ""))).strip()
    operator = str(condition.get("operator", condition.get("op", ""))).strip()
    expected = condition.get("value", condition.get("threshold"))
    npc_id = str(condition.get("npc_id", "") or "")
    current_value = _read_dimension(state, dimension_name, npc_id=npc_id)
    if current_value is None or not isinstance(expected, (int, float)):
        return False
    return _compare(current_value, operator, expected)


def _bottom_lines(state: GameState) -> list[dict[str, Any]]:
    """从玩家目标和角色层结局配置读取底线。"""

    goal = state.player_goal if isinstance(state.player_goal, dict) else {}
    endings = goal.get("endings") if isinstance(goal.get("endings"), dict) else {}
    raw = goal.get("bottom_lines") or endings.get("bottom_lines") or goal.get("failure_conditions") or []
    return [dict(item) for item in _as_list(raw) if isinstance(item, Mapping)]


async def check_bottom_lines(state: GameState) -> dict[str, Any] | None:
    """检查底线是否击穿，命中则返回底线定义。"""

    for bottom_line in _bottom_lines(state):
        condition = bottom_line.get("condition", bottom_line.get("conditions"))
        if _condition_met(state, condition):
            state.storylines["bottom_line_triggered"] = str(bottom_line.get("bottom_line_id", bottom_line.get("id", "")))
            logger.info("底线击穿: %s", bottom_line)
            return bottom_line

    # 缺省兜底：任一已启用维度归零视为底线风险。
    values = [*state.dimensions.character.values(), *state.dimensions.world.values(), *state.dimensions.extensions.values()]
    if values and any(value <= 0 for value in values):
        fallback = {
            "bottom_line_id": "BL_dimension_zero",
            "name": "关键维度归零",
            "ending_id": "END_bottom_line_failure",
            "description": "至少一项关键维度已经降至零。",
        }
        state.storylines["bottom_line_triggered"] = fallback["bottom_line_id"]
        return fallback
    return None


async def check_ending_conditions(state: GameState) -> str | None:
    """兼容入口：返回第一个满足条件的结局ID。"""

    goal = state.player_goal if isinstance(state.player_goal, dict) else {}
    endings = goal.get("endings") if isinstance(goal.get("endings"), dict) else {}
    catalog = endings.get("catalog") if isinstance(endings.get("catalog"), list) else []
    for ending in catalog:
        if not isinstance(ending, Mapping):
            continue
        ending_id = str(ending.get("ending_id", "")).strip()
        condition = ending.get("conditions")
        if ending_id and _condition_met(state, condition):
            return ending_id
    broken = await check_bottom_lines(state)
    if broken:
        return str(broken.get("ending_id") or broken.get("bottom_line_id") or "bottom_line_failure")
    return None


async def generate_ending_narration(state: GameState, ending_id: str) -> str:
    """基于新结局目录与当前状态生成最终叙事。"""

    ending_info = _find_ending_info(state, ending_id)
    state_brief = {
        "script_id": state.script_id,
        "player_role": state.player_role,
        "game_date": state.game_date,
        "game_time": state.game_time,
        "ending_id": ending_id,
        "ending_info": ending_info,
        "dimension_desc": get_state_manager().get_dimension_description(state),
        "player_goal": state.player_goal,
        "recent_decisions": state.decisions[-8:],
        "event_history": state.event_history[-12:],
        "growth_log": state.growth_log[-8:],
    }
    try:
        narration = await get_llm_client().chat(
            ENDING_SYSTEM_PROMPT,
            json.dumps(state_brief, ensure_ascii=False),
            temperature=0.7,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("结局叙事生成失败，使用兜底文本 ending_id=%s err=%s", ending_id, exc)
        narration = "风云至此收束。你的选择与迟疑一并沉入局势的回声，此局已终。"
    return narration.strip() or str(ending_info.get("description") or "此局已终。")


def _find_ending_info(state: GameState, ending_id: str) -> dict[str, Any]:
    """从角色层结局目录中查找结局定义。"""

    goal = state.player_goal if isinstance(state.player_goal, dict) else {}
    endings = goal.get("endings") if isinstance(goal.get("endings"), dict) else {}
    for item in _as_list(endings.get("catalog")):
        if isinstance(item, Mapping) and str(item.get("ending_id", "")) == ending_id:
            return dict(item)
    for item in _bottom_lines(state):
        if str(item.get("ending_id", "")) == ending_id or str(item.get("bottom_line_id", "")) == ending_id:
            return dict(item)
    return {"ending_id": ending_id, "name": ending_id, "description": "局势抵达终点。"}


def _as_list(value: Any) -> list[Any]:
    """安全转换列表。"""

    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


__all__ = ["check_bottom_lines", "check_ending_conditions", "generate_ending_narration"]
