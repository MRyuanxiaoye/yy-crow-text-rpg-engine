"""事件调度器：条件匹配、优先级、冷却、倒计时。"""

from __future__ import annotations

import logging
import random
from typing import Any

from src.engine.state import GameState

logger = logging.getLogger(__name__)

MAX_ACTIVE_EVENTS = 3

SEVERITY_PRIORITY = {"critical": 0, "moderate": 1, "minor": 2}


def _time_reached(game_time: dict[str, int], anchor: dict[str, Any]) -> bool:
    """判断游戏时间是否已到达或超过锚点。"""

    if not isinstance(anchor, dict):
        return False
    anchor_year = int(anchor.get("year", 9999))
    anchor_month = int(anchor.get("month", 1))
    cur_year = game_time.get("year", 1)
    cur_month = game_time.get("month", 1)

    if cur_year > anchor_year:
        return True
    if cur_year == anchor_year and cur_month >= anchor_month:
        return True
    return False


def _conditions_met(state: GameState, conditions: list[dict[str, Any]]) -> bool:
    """检查数值条件是否满足（全部满足才返回True）。"""

    if not isinstance(conditions, list) or not conditions:
        return False

    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        metric = str(cond.get("metric", "")).strip()
        operator = str(cond.get("operator", "")).strip()
        threshold = cond.get("value")
        if not metric or not operator or not isinstance(threshold, (int, float)):
            continue

        current = state.metrics.get(metric, 50)
        if operator == "below" and current >= threshold:
            return False
        if operator == "above" and current <= threshold:
            return False
        if operator == "equals" and current != threshold:
            return False

    return True


def _prerequisites_met(state: GameState, prerequisites: list[str]) -> bool:
    """检查前置事件是否都已触发过。"""

    if not isinstance(prerequisites, list) or not prerequisites:
        return True
    for event_id in prerequisites:
        if str(event_id).strip() not in state.event_history:
            return False
    return True


def _is_on_cooldown(state: GameState, event_id: str) -> bool:
    """检查事件是否在冷却中。"""

    return event_id in state.event_cooldowns and state.event_cooldowns[event_id] > 0


def _sort_by_severity(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按严重程度排序，critical优先。"""

    return sorted(
        events,
        key=lambda e: SEVERITY_PRIORITY.get(str(e.get("severity", "minor")), 2),
    )


def check_timeline_events(
    state: GameState,
    timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """检查时间线事件是否到达触发时间。"""

    triggered: list[dict[str, Any]] = []
    for event in timeline:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue
        if event_id in state.event_history:
            continue
        if _is_on_cooldown(state, event_id):
            continue

        time_anchor = event.get("time") or event.get("time_anchor")
        if not isinstance(time_anchor, dict):
            continue

        if _time_reached(state.game_time, time_anchor):
            prereqs = event.get("prerequisite_events", [])
            if _prerequisites_met(state, prereqs if isinstance(prereqs, list) else []):
                triggered.append(event)

    return _sort_by_severity(triggered)


def check_random_events(
    state: GameState,
    random_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """对随机事件池掷骰，返回命中的事件。"""

    triggered: list[dict[str, Any]] = []
    for event in random_pool:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue
        if event_id in state.event_history:
            continue
        if _is_on_cooldown(state, event_id):
            continue

        probability = event.get("probability", 0)
        if not isinstance(probability, (int, float)) or probability <= 0:
            continue

        # 条件过滤
        conditions = event.get("conditions") or event.get("trigger", {}).get("conditions")
        if isinstance(conditions, list) and conditions:
            if not _conditions_met(state, conditions):
                continue

        if random.random() < probability:
            triggered.append(event)

    return _sort_by_severity(triggered)


def check_disaster_events(
    state: GameState,
    disasters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """检查天灾事件（时间锚点或概率触发）。"""

    triggered: list[dict[str, Any]] = []
    for event in disasters:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue
        if event_id in state.event_history:
            continue
        if _is_on_cooldown(state, event_id):
            continue

        # 时间锚点触发
        time_anchor = event.get("time") or event.get("time_anchor")
        if isinstance(time_anchor, dict) and _time_reached(state.game_time, time_anchor):
            triggered.append(event)
            continue

        # 概率触发
        probability = event.get("probability", 0)
        if isinstance(probability, (int, float)) and probability > 0:
            conditions = event.get("conditions")
            if isinstance(conditions, list) and conditions:
                if not _conditions_met(state, conditions):
                    continue
            if random.random() < probability:
                triggered.append(event)

    return _sort_by_severity(triggered)


def check_condition_events(
    state: GameState,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """检查条件触发事件（npc_action / chain 等）。"""

    triggered: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue
        if event_id in state.event_history:
            continue
        if _is_on_cooldown(state, event_id):
            continue

        # 前置事件检查
        prereqs = event.get("prerequisite_events", [])
        if isinstance(prereqs, list) and prereqs:
            if not _prerequisites_met(state, prereqs):
                continue

        # 数值条件检查
        conditions = event.get("conditions")
        if isinstance(conditions, list) and conditions:
            if not _conditions_met(state, conditions):
                continue
            triggered.append(event)

    return _sort_by_severity(triggered)


def check_threshold_events(
    state: GameState,
    all_events: list[dict[str, Any]],
    triggered_tags: list[str],
) -> list[dict[str, Any]]:
    """根据阈值触发的tag匹配事件。"""

    if not triggered_tags:
        return []

    tag_set = set(triggered_tags)
    matched: list[dict[str, Any]] = []
    for event in all_events:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue
        if event_id in state.event_history:
            continue
        if _is_on_cooldown(state, event_id):
            continue

        event_tag = str(event.get("trigger_tag", "")).strip()
        if event_tag and event_tag in tag_set:
            matched.append(event)

    return _sort_by_severity(matched)


def activate_event(state: GameState, event: dict[str, Any]) -> None:
    """将事件加入活跃队列。"""

    event_id = str(event.get("id", "")).strip()
    if not event_id:
        return

    # 检查是否已在活跃列表
    for active in state.active_events:
        if isinstance(active, dict) and active.get("id") == event_id:
            return

    active_entry = {
        "id": event_id,
        "name": str(event.get("name", event_id)),
        "severity": str(event.get("severity", "moderate")),
        "description": str(event.get("description", "")),
        "narrative_hint": str(event.get("narrative_hint", "")),
        "forced_npcs": event.get("forced_npcs", []),
        "player_options_hint": event.get("player_options_hint", []),
        "turns_remaining": _get_delay(event),
        "default_consequence": event.get("default_consequence", {}),
    }
    state.active_events.append(active_entry)

    # 记录到历史，防止重复触发
    if event_id not in state.event_history:
        state.event_history.append(event_id)

    # 设置冷却
    cooldown = event.get("cooldown_time_units", 0)
    if isinstance(cooldown, int) and cooldown > 0:
        state.event_cooldowns[event_id] = cooldown

    logger.info("事件激活 id=%s name=%s severity=%s", event_id, active_entry["name"], active_entry["severity"])


def _get_delay(event: dict[str, Any]) -> int:
    """获取事件的默认后果延迟时间单位。"""

    consequence = event.get("default_consequence", {})
    if isinstance(consequence, dict):
        delay = consequence.get("delay_time_units", 3)
        if isinstance(delay, int):
            return max(1, delay)
    return 3


def tick_active_events(state: GameState) -> list[dict[str, Any]]:
    """每次时间推进时，递减活跃事件倒计时，返回到期事件。"""

    expired: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []

    for event in state.active_events:
        if not isinstance(event, dict):
            continue
        turns_left = event.get("turns_remaining", 0)
        if isinstance(turns_left, int):
            turns_left -= 1
            event["turns_remaining"] = turns_left

        if turns_left <= 0:
            expired.append(event)
        else:
            remaining.append(event)

    state.active_events = remaining
    return expired


def apply_default_consequence(state: GameState, event: dict[str, Any]) -> dict[str, Any]:
    """应用到期事件的默认后果，返回效果摘要。"""

    from src.engine.state import get_state_manager

    consequence = event.get("default_consequence", {})
    if not isinstance(consequence, dict):
        consequence = {}

    effects = consequence.get("effects", {})
    if not isinstance(effects, dict):
        effects = {}

    narrative = str(consequence.get("narrative", "")).strip()
    event_name = str(event.get("name", event.get("id", "未知事件")))

    # 应用数值效果
    metrics_delta = effects.get("metrics", {})
    if isinstance(metrics_delta, dict):
        state_manager = get_state_manager()
        state_manager.apply_effects(state, {"metrics": metrics_delta})

    # 触发连锁事件（标记到历史，下次调度时会被check_condition_events捡起）
    chain_events = consequence.get("chain_events", [])

    logger.info("事件到期执行默认后果 id=%s name=%s effects=%s", event.get("id"), event_name, metrics_delta)

    return {
        "event_id": str(event.get("id", "")),
        "event_name": event_name,
        "narrative": narrative or f"「{event_name}」久未处置，局势恶化。",
        "metrics_delta": metrics_delta if isinstance(metrics_delta, dict) else {},
        "chain_events": chain_events if isinstance(chain_events, list) else [],
    }


def resolve_active_event(state: GameState, event_id: str) -> bool:
    """玩家处理了某个活跃事件，从队列移除。"""

    event_id = event_id.strip()
    before_count = len(state.active_events)
    state.active_events = [
        e for e in state.active_events
        if not (isinstance(e, dict) and str(e.get("id", "")).strip() == event_id)
    ]
    removed = len(state.active_events) < before_count
    if removed:
        logger.info("活跃事件已解决 id=%s", event_id)
    return removed


def run_event_scheduler(
    state: GameState,
    timeline: list[dict[str, Any]],
    random_pool: list[dict[str, Any]],
    disasters: list[dict[str, Any]],
    npc_events: list[dict[str, Any]],
    threshold_tags: list[str],
) -> list[dict[str, Any]]:
    """主调度入口：检查所有事件源，激活新事件（受上限约束）。

    返回本次新激活的事件列表。
    """

    available_slots = MAX_ACTIVE_EVENTS - len(state.active_events)
    if available_slots <= 0:
        logger.info("活跃事件已满(%d)，跳过调度", MAX_ACTIVE_EVENTS)
        return []

    candidates: list[dict[str, Any]] = []

    # 按优先级顺序收集
    candidates.extend(check_timeline_events(state, timeline))
    candidates.extend(check_disaster_events(state, disasters))
    candidates.extend(check_threshold_events(state, npc_events + random_pool + disasters, threshold_tags))
    candidates.extend(check_random_events(state, random_pool))
    candidates.extend(check_condition_events(state, npc_events))

    # 去重
    seen_ids: set[str] = set()
    unique_candidates: list[dict[str, Any]] = []
    for event in candidates:
        eid = str(event.get("id", "")).strip()
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            unique_candidates.append(event)

    # 按severity排序后取可用槽位数
    sorted_candidates = _sort_by_severity(unique_candidates)
    to_activate = sorted_candidates[:available_slots]

    for event in to_activate:
        activate_event(state, event)

    if to_activate:
        logger.info("本次调度激活 %d 个事件: %s", len(to_activate), [e.get("id") for e in to_activate])

    return to_activate
