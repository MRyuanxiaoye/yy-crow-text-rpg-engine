"""时间系统：玩家主动推进、压力源结算与行动队列处理。"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from src.engine.dimension import DimensionReference, get_dimension_value
from src.engine.pressure import apply_decay as apply_pressure_decay
from src.engine.pressure import check_milestones, check_reactions
from src.engine.state import GameState

logger = logging.getLogger(__name__)


@dataclass
class AdvanceResult:
    """一次主动推进的结算结果。"""

    elapsed_time: int = 0
    triggered_events: list[dict[str, Any]] = field(default_factory=list)
    due_actions: list[dict[str, Any]] = field(default_factory=list)
    bottom_line_broken: bool = False
    decay_changes: dict[str, int] = field(default_factory=dict)
    old_time: dict[str, int] = field(default_factory=dict)
    new_time: dict[str, int] = field(default_factory=dict)


def get_time_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """从 manifest 提取时间配置，缺省给合理默认值。"""

    time_block = manifest.get("time")
    if not isinstance(time_block, dict):
        return {
            "unit": "月",
            "display_format": "崇祯{year}年{month}月",
            "start": {"year": 1, "month": 8},
            "turns_per_unit": 3,
        }

    return {
        "unit": str(time_block.get("unit", "月")),
        "display_format": str(time_block.get("display_format", "崇祯{year}年{month}月")),
        "start": time_block.get("start", {"year": 1, "month": 8}),
        "turns_per_unit": max(1, int(time_block.get("turns_per_unit", 3))),
    }


def format_game_time(game_time: dict[str, int], display_format: str) -> str:
    """将 game_time 格式化为可读字符串。"""

    year = game_time.get("year", 1)
    month = game_time.get("month", 1)
    try:
        return display_format.format(year=year, month=month)
    except (KeyError, ValueError):
        return f"第{year}年第{month}月"


def advance_time(game_time: dict[str, int], units: int = 1) -> dict[str, int]:
    """推进游戏时间 N 个单位（月制：12 月进 1 年）。"""

    year = game_time.get("year", 1)
    month = game_time.get("month", 1)

    month += units
    while month > 12:
        month -= 12
        year += 1

    return {"year": year, "month": month}


def apply_decay(state: GameState, pressure_sources: Any) -> dict[str, int]:
    """兼容旧入口：按压力源执行自然衰减。"""

    return apply_pressure_decay(state, pressure_sources)


def check_thresholds(state: GameState, thresholds: list[dict[str, Any]]) -> list[str]:
    """兼容旧事件阈值接口，返回触发的 event_tag 列表。"""

    if not isinstance(thresholds, list):
        return []

    triggered_tags: list[str] = []
    for threshold in thresholds:
        if not isinstance(threshold, dict):
            continue
        dimension_name = str(threshold.get("dimension", threshold.get("metric", ""))).strip()
        if not dimension_name:
            continue

        tag = str(threshold.get("trigger_tag", "")).strip()
        if not tag:
            continue

        current_value = _read_dimension_value(state, dimension_name)
        if current_value is None:
            continue

        if "below" in threshold and isinstance(threshold["below"], int) and current_value < threshold["below"]:
            triggered_tags.append(tag)
        if "above" in threshold and isinstance(threshold["above"], int) and current_value > threshold["above"]:
            triggered_tags.append(tag)

    return triggered_tags


def tick_time(state: GameState, time_config: dict[str, Any]) -> dict[str, Any] | None:
    """推进一个时间单位，并按 state.pressure_sources 结算压力源。"""

    old_time = dict(state.game_time)
    state.game_time = advance_time(state.game_time, units=1)
    state.turn += 1
    _tick_event_cooldowns(state)

    decay_changes = apply_pressure_decay(state, state.pressure_sources)
    triggered_events = [
        *check_milestones(state, state.pressure_sources),
        *check_reactions(state, state.pressure_sources),
        *_check_random_events(state),
        *_check_consequence_seeds(state),
    ]

    display_format = str(time_config.get("display_format", "崇祯{year}年{month}月"))
    new_time_str = format_game_time(state.game_time, display_format)
    state.game_date = new_time_str

    logger.info(
        "时间推进 %s → %s decay=%s events=%d",
        format_game_time(old_time, display_format),
        new_time_str,
        decay_changes,
        len(triggered_events),
    )

    return {
        "old_time": old_time,
        "new_time": dict(state.game_time),
        "new_time_str": new_time_str,
        "decay_changes": decay_changes,
        "triggered_events": triggered_events,
    }


def process_advance_queue(state: GameState) -> AdvanceResult:
    """按玩家行动队列主动推进时间，遇到行动到期、事件或底线击穿即停。"""

    result = AdvanceResult(old_time=dict(state.game_time), new_time=dict(state.game_time))
    _prepare_advance_queue(state)
    if not state.advance_queue:
        return result

    immediate_due = _pop_due_actions(state)
    if immediate_due:
        result.due_actions = immediate_due
        return result

    while state.advance_queue:
        state.game_time = advance_time(state.game_time, units=1)
        _sync_game_date(state)
        state.turn += 1
        result.elapsed_time += 1
        _tick_event_cooldowns(state)
        _decrement_action_times(state)

        decay_changes = apply_pressure_decay(state, state.pressure_sources)
        _merge_changes(result.decay_changes, decay_changes)

        triggered_events = [
            *check_milestones(state, state.pressure_sources),
            *check_reactions(state, state.pressure_sources),
            *_check_random_events(state),
            *_check_consequence_seeds(state),
        ]
        result.triggered_events.extend(triggered_events)
        result.due_actions = _pop_due_actions(state)
        result.bottom_line_broken = _is_bottom_line_broken(state)
        result.new_time = dict(state.game_time)

        if triggered_events or result.due_actions or result.bottom_line_broken:
            break

    return result


def _prepare_advance_queue(state: GameState) -> None:
    """初始化行动剩余时间，并按到期时间排序。"""

    normalized: list[dict[str, Any]] = []
    for action in state.advance_queue:
        if not isinstance(action, dict):
            continue
        item = dict(action)
        time_cost = _safe_int(item.get("time_cost", item.get("time_cost_units", item.get("cost", 0))))
        item.setdefault("original_time_cost", max(0, time_cost))
        item["remaining_time"] = max(0, _safe_int(item.get("remaining_time", time_cost)))
        normalized.append(item)
    normalized.sort(key=lambda item: item.get("remaining_time", 0))
    state.advance_queue = normalized


def _decrement_action_times(state: GameState) -> None:
    """每推进一个时间单位，所有并行行动剩余时间减一。"""

    for action in state.advance_queue:
        action["remaining_time"] = max(0, _safe_int(action.get("remaining_time", 0)) - 1)
    state.advance_queue.sort(key=lambda item: item.get("remaining_time", 0))


def _pop_due_actions(state: GameState) -> list[dict[str, Any]]:
    """取出已经到期、需要掷骰结算的行动。"""

    due_actions: list[dict[str, Any]] = []
    remaining_actions: list[dict[str, Any]] = []
    for action in state.advance_queue:
        if _safe_int(action.get("remaining_time", action.get("time_cost", 0))) <= 0:
            due_actions.append(action)
        else:
            remaining_actions.append(action)
    state.advance_queue = remaining_actions
    return due_actions


def _check_consequence_seeds(state: GameState) -> list[dict[str, Any]]:
    """检查后果种子，命中候选后随机选择一颗触发。"""

    candidates: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    for seed in state.consequence_seeds:
        if not isinstance(seed, dict):
            continue
        item = dict(seed)
        if _is_seed_candidate(state, item):
            candidates.append(item)
        else:
            survivors.append(item)

    if not candidates:
        state.consequence_seeds = survivors
        return []

    chosen = random.choice(candidates)
    for seed in candidates:
        if seed is chosen:
            continue
        seed["remaining_chances"] = max(0, _safe_int(seed.get("remaining_chances", 1)) - 1)
        if seed["remaining_chances"] > 0:
            survivors.append(seed)

    state.consequence_seeds = survivors
    return [_seed_event(chosen)]


def _is_seed_candidate(state: GameState, seed: dict[str, Any]) -> bool:
    """判断后果种子本轮是否进入候选池。"""

    if _safe_int(seed.get("remaining_chances", 1)) <= 0:
        return False

    trigger = seed.get("trigger", seed.get("condition", seed.get("conditions")))
    if trigger is None:
        return bool(seed.get("ready", seed.get("always_check", False)))
    return _condition_met(state, trigger)


def _seed_event(seed: dict[str, Any]) -> dict[str, Any]:
    """将后果种子转换为突发事件。"""

    source = str(seed.get("source", "未知后果种子") or "未知后果种子")
    return {
        "id": str(seed.get("event_id", seed.get("seed_id", f"seed:{source}")) or f"seed:{source}"),
        "type": "consequence_seed",
        "source_type": "consequence_seed",
        "name": f"延迟后果：{source}",
        "severity": seed.get("severity", 1),
        "directions": list(seed.get("directions", [])) if isinstance(seed.get("directions"), list) else [],
        "payload": dict(seed),
    }


def _check_random_events(state: GameState) -> list[dict[str, Any]]:
    """检查状态或压力源中声明的随机事件。"""

    triggered: list[dict[str, Any]] = []
    for event in _random_event_sources(state):
        event_id = str(event.get("event_id", event.get("id", "")) or "").strip()
        cooldown_key = f"random:{event_id}" if event_id else ""
        if event_id and event_id in state.event_history:
            continue
        if cooldown_key and state.event_cooldowns.get(cooldown_key, 0) > 0:
            continue

        trigger = event.get("trigger", event.get("condition", event.get("conditions")))
        if trigger is not None and not _condition_met(state, trigger):
            continue

        probability = event.get("probability", event.get("chance", 0))
        if not isinstance(probability, (int, float)) or probability <= 0:
            continue
        if random.random() > min(1.0, float(probability)):
            continue

        triggered.append(_random_event(event))
        if event_id:
            state.event_history.append(event_id)
        cooldown_units = _safe_int(event.get("cooldown_units", event.get("cooldown_time_units", 0)))
        if cooldown_key and cooldown_units > 0:
            state.event_cooldowns[cooldown_key] = cooldown_units
    return triggered


def _random_event_sources(state: GameState) -> list[dict[str, Any]]:
    """兼容从 state 或 pressure_sources 获取随机事件池。"""

    sources: list[Any] = []
    for attr_name in ("random_events", "random_pool"):
        sources.extend(_as_list(getattr(state, attr_name, [])))
    if isinstance(state.pressure_sources, dict):
        sources.extend(_as_list(state.pressure_sources.get("random", [])))
    else:
        for item in _as_list(state.pressure_sources):
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("type", item.get("source_type", ""))).strip()
            if raw_type == "random":
                sources.append(item)
    return [dict(item) for item in sources if isinstance(item, dict)]


def _random_event(event: dict[str, Any]) -> dict[str, Any]:
    """将随机事件声明转换为推进结果事件。"""

    event_id = str(event.get("event_id", event.get("id", "")) or "").strip()
    return {
        "id": event_id,
        "event_id": event_id,
        "type": "random",
        "source_type": "random",
        "name": str(event.get("name", event.get("event", event_id)) or ""),
        "description": str(event.get("description", event.get("narrative", "")) or ""),
        "payload": dict(event),
    }


def _condition_met(state: GameState, condition: Any) -> bool:
    """时间系统内的轻量条件检查，用于后果种子和底线。"""

    if isinstance(condition, list):
        return all(_condition_met(state, item) for item in condition)
    if not isinstance(condition, dict):
        return False
    if "all" in condition:
        return all(_condition_met(state, item) for item in _as_list(condition.get("all")))
    if "any" in condition:
        return any(_condition_met(state, item) for item in _as_list(condition.get("any")))
    if "not" in condition:
        return not _condition_met(state, condition.get("not"))

    dimension_name = str(condition.get("dimension", condition.get("metric", ""))).strip()
    operator = str(condition.get("operator", condition.get("op", ""))).strip()
    expected = condition.get("value", condition.get("threshold"))
    current_value = _read_dimension_value(state, dimension_name)
    if current_value is None or not isinstance(expected, (int, float)):
        return False
    return _compare(current_value, operator, expected)


def _is_bottom_line_broken(state: GameState) -> bool:
    """检查显式底线条件，缺省时以任一维度归零作为击穿。"""

    for condition in _bottom_line_conditions(state):
        if _condition_met(state, condition):
            return True

    values = [
        *state.dimensions.character.values(),
        *state.dimensions.world.values(),
        *state.dimensions.extensions.values(),
    ]
    return any(value <= 0 for value in values)


def _bottom_line_conditions(state: GameState) -> list[Any]:
    """从玩家目标中提取底线条件。"""

    goal = state.player_goal if isinstance(state.player_goal, dict) else {}
    raw_items = goal.get("bottom_lines", goal.get("failure_conditions", goal.get("bottom_line_conditions", [])))
    conditions: list[Any] = []
    for item in _as_list(raw_items):
        if isinstance(item, dict) and "condition" in item:
            conditions.append(item["condition"])
        else:
            conditions.append(item)
    return conditions


def _tick_event_cooldowns(state: GameState) -> None:
    """递减事件冷却。"""

    expired_cooldowns: list[str] = []
    for event_id in list(state.event_cooldowns.keys()):
        state.event_cooldowns[event_id] -= 1
        if state.event_cooldowns[event_id] <= 0:
            expired_cooldowns.append(event_id)
    for event_id in expired_cooldowns:
        del state.event_cooldowns[event_id]


def _sync_game_date(state: GameState) -> None:
    """用默认格式同步可读日期。"""

    state.game_date = format_game_time(state.game_time, "崇祯{year}年{month}月")


def _read_dimension_value(state: GameState, dimension_name: str) -> int | None:
    """读取维度值，失败时返回 None。"""

    if not dimension_name:
        return None
    try:
        return get_dimension_value(state.dimensions, DimensionReference(name=dimension_name, category="auto"))
    except (KeyError, TypeError, ValueError):
        return None


def _compare(current_value: int, operator: str, expected: int | float) -> bool:
    """执行数值比较。"""

    if operator in {"<", "below", "lt"}:
        return current_value < expected
    if operator in {"<=", "at_most", "lte"}:
        return current_value <= expected
    if operator in {">", "above", "gt"}:
        return current_value > expected
    if operator in {">=", "at_least", "gte"}:
        return current_value >= expected
    if operator in {"==", "=", "equals", "eq"}:
        return current_value == expected
    if operator in {"!=", "not_equals", "neq"}:
        return current_value != expected
    return False


def _merge_changes(target: dict[str, int], changes: dict[str, int]) -> None:
    """合并多轮衰减变化。"""

    for key, value in changes.items():
        target[key] = target.get(key, 0) + value


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
