"""压力源系统：衰减、时间里程碑与状态响应。"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from src.engine.dimension import DimensionReference, get_dimension_value, update_dimension_value
from src.engine.state import GameState

logger = logging.getLogger(__name__)

_PRESSURE_TYPES = ("decay", "milestones", "reactions")
_OPERATOR_ALIASES = {
    "<": "<",
    "below": "<",
    "lt": "<",
    "<=": "<=",
    "at_most": "<=",
    "lte": "<=",
    ">": ">",
    "above": ">",
    "gt": ">",
    ">=": ">=",
    "at_least": ">=",
    "gte": ">=",
    "==": "==",
    "=": "==",
    "equals": "==",
    "eq": "==",
    "!=": "!=",
    "not_equals": "!=",
    "neq": "!=",
}


def apply_decay(state: GameState, pressure_sources: Any) -> dict[str, int]:
    """对所有衰减型压力源执行一次扣减，返回各维度实际变化量。"""

    changes: dict[str, int] = {}
    for source in _iter_pressure_sources(pressure_sources, "decay"):
        dimension_name = str(source.get("dimension", source.get("metric", ""))).strip()
        if not dimension_name:
            continue

        rate = _safe_int(source.get("rate", source.get("delta", 0)))
        if rate == 0:
            continue

        reference = _dimension_reference(source, dimension_name)
        try:
            current_value = get_dimension_value(state.dimensions, reference)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("跳过无法读取的衰减维度 dimension=%s err=%s", dimension_name, exc)
            continue

        delta = _clamp_decay_delta(current_value, rate, source)
        if delta == 0:
            continue

        try:
            new_value = update_dimension_value(state.dimensions, reference, delta)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("跳过无法更新的衰减维度 dimension=%s delta=%s err=%s", dimension_name, delta, exc)
            continue

        changes[dimension_name] = changes.get(dimension_name, 0) + new_value - current_value

    return changes


def check_milestones(state: GameState, pressure_sources: Any) -> list[dict[str, Any]]:
    """检查当前时间是否命中推进型压力节点，返回触发事件列表。"""

    triggered: list[dict[str, Any]] = []
    for source in _iter_pressure_sources(pressure_sources, "milestones"):
        event_key = _event_key(source)
        if event_key and event_key in state.event_history:
            continue
        if not _time_matches(state, source.get("at", source.get("time", source.get("time_anchor")))):
            continue

        event = _pressure_event(source, "milestone")
        triggered.append(event)
        if event_key:
            state.event_history.append(event_key)

    return triggered


def check_reactions(state: GameState, pressure_sources: Any) -> list[dict[str, Any]]:
    """检查维度阈值与响应条件，返回触发事件列表。"""

    triggered: list[dict[str, Any]] = []
    for source in _iter_pressure_sources(pressure_sources, "reactions"):
        cooldown_key = _cooldown_key(source)
        if cooldown_key and state.event_cooldowns.get(cooldown_key, 0) > 0:
            continue

        trigger = source.get("trigger", source.get("condition", source.get("conditions")))
        if not _condition_met(state, trigger):
            continue

        triggered.append(_pressure_event(source, "reaction"))
        cooldown_units = _safe_int(source.get("cooldown_units", source.get("cooldown_time_units", 0)))
        if cooldown_key and cooldown_units > 0:
            state.event_cooldowns[cooldown_key] = cooldown_units

    return triggered


def tick_pressure(state: GameState, pressure_sources: Any) -> list[dict[str, Any]]:
    """执行一次压力源推进，返回本轮触发的世界事件。"""

    apply_decay(state, pressure_sources)
    return [
        *check_milestones(state, pressure_sources),
        *check_reactions(state, pressure_sources),
    ]


def _iter_pressure_sources(pressure_sources: Any, source_type: str) -> Iterable[dict[str, Any]]:
    """兼容分组式与扁平式压力源声明。"""

    if source_type not in _PRESSURE_TYPES:
        return []

    if isinstance(pressure_sources, dict):
        items = pressure_sources.get(source_type, [])
        if source_type == "milestones" and not items:
            items = pressure_sources.get("milestone", [])
        if source_type == "reactions" and not items:
            items = pressure_sources.get("reaction", [])
        return [dict(item) for item in _as_list(items) if isinstance(item, dict)]

    result: list[dict[str, Any]] = []
    for item in _as_list(pressure_sources):
        if not isinstance(item, dict):
            continue
        if source_type in item and isinstance(item[source_type], list):
            result.extend(dict(child) for child in item[source_type] if isinstance(child, dict))
            continue
        raw_type = str(item.get("type", item.get("source_type", item.get("kind", item.get("form", item.get("pressure_type", "")))))).strip()
        normalized_type = "milestones" if raw_type == "milestone" else "reactions" if raw_type == "reaction" else raw_type
        if normalized_type == source_type:
            result.append(dict(item))
    return result


def _dimension_reference(source: dict[str, Any], dimension_name: str) -> DimensionReference:
    """从压力源条目构造维度引用。"""

    category = str(source.get("category", source.get("dimension_category", "auto")) or "auto")
    npc_id = str(source.get("npc_id", source.get("target_npc", "")) or "")
    return DimensionReference(name=dimension_name, category=category, npc_id=npc_id)


def _clamp_decay_delta(current_value: int, rate: int, source: dict[str, Any]) -> int:
    """按 floor/ceiling 限制衰减的实际增量。"""

    delta = rate
    if rate < 0 and "floor" in source:
        floor = _safe_int(source.get("floor"))
        delta = max(rate, floor - current_value)
    if rate > 0 and "ceiling" in source:
        ceiling = _safe_int(source.get("ceiling"))
        delta = min(rate, ceiling - current_value)
    return delta


def _time_matches(state: GameState, anchor: Any) -> bool:
    """判断时间锚点是否与当前游戏时间完全命中。"""

    if isinstance(anchor, dict):
        for key, value in anchor.items():
            if key not in state.game_time:
                continue
            if state.game_time.get(key) != _safe_int(value):
                return False
        return bool(anchor)
    if isinstance(anchor, int):
        return state.turn == anchor
    if isinstance(anchor, str):
        return state.game_date == anchor.strip()
    return False


def _condition_met(state: GameState, condition: Any) -> bool:
    """递归检查 all/any/not 组合条件或单条维度条件。"""

    if isinstance(condition, str):
        parsed = _parse_condition_text(condition)
        return _condition_met(state, parsed) if parsed else False
    if isinstance(condition, list):
        return all(_condition_met(state, item) for item in condition)
    if not isinstance(condition, dict):
        return False

    if "all" in condition:
        items = condition.get("all")
        return all(_condition_met(state, item) for item in _as_list(items))
    if "any" in condition:
        items = condition.get("any")
        return any(_condition_met(state, item) for item in _as_list(items))
    if "not" in condition:
        return not _condition_met(state, condition.get("not"))

    dimension_name = str(condition.get("dimension", condition.get("metric", ""))).strip()
    operator = _normalize_operator(str(condition.get("operator", condition.get("op", ""))).strip())
    expected = condition.get("value", condition.get("threshold"))
    if not dimension_name or not operator or not isinstance(expected, (int, float)):
        return False

    reference = _dimension_reference(condition, dimension_name)
    try:
        current_value = get_dimension_value(state.dimensions, reference)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("跳过无法读取的响应条件维度 dimension=%s err=%s", dimension_name, exc)
        return False

    return _compare(current_value, operator, expected)


def _parse_condition_text(condition: str) -> dict[str, Any] | None:
    """解析“维度 < 3”形式的简单条件文本。"""

    match = re.fullmatch(r"\s*(.+?)\s*(<=|>=|==|!=|<|>|=)\s*(-?\d+(?:\.\d+)?)\s*", condition)
    if not match:
        return None
    value_text = match.group(3)
    value: int | float = float(value_text) if "." in value_text else int(value_text)
    return {"dimension": match.group(1).strip(), "operator": match.group(2), "value": value}


def _normalize_operator(operator: str) -> str:
    """统一条件操作符。"""

    return _OPERATOR_ALIASES.get(operator, "")


def _compare(current_value: int, operator: str, expected: int | float) -> bool:
    """执行数值比较。"""

    if operator == "<":
        return current_value < expected
    if operator == "<=":
        return current_value <= expected
    if operator == ">":
        return current_value > expected
    if operator == ">=":
        return current_value >= expected
    if operator == "==":
        return current_value == expected
    if operator == "!=":
        return current_value != expected
    return False


def _pressure_event(source: dict[str, Any], source_type: str) -> dict[str, Any]:
    """将压力源条目规范化为事件对象。"""

    event_id = str(source.get("event_id", source.get("event", source.get("id", ""))) or "").strip()
    pressure_id = str(source.get("pressure_id", "") or "").strip()
    return {
        "id": event_id or pressure_id,
        "event_id": event_id,
        "pressure_id": pressure_id,
        "type": source_type,
        "source_type": "pressure",
        "name": str(source.get("name", source.get("event", event_id or pressure_id)) or ""),
        "description": str(source.get("description", source.get("narrative", "")) or ""),
        "payload": dict(source),
    }


def _event_key(source: dict[str, Any]) -> str:
    """取得用于一次性事件去重的键。"""

    return str(source.get("event_id", source.get("pressure_id", source.get("id", ""))) or "").strip()


def _cooldown_key(source: dict[str, Any]) -> str:
    """取得响应型压力冷却键。"""

    key = _event_key(source)
    return f"pressure:{key}" if key else ""


def _as_list(value: Any) -> list[Any]:
    """安全转换列表。"""

    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _safe_int(value: Any, default: int = 0) -> int:
    """安全转换整数。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default
