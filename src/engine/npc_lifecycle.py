"""NPC生命周期管理。"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.engine.state import GameState
from src.scripts.loader import load_npc_events

logger = logging.getLogger(__name__)


def _to_int(value: Any, default: int = 0) -> int:
    """安全转整数。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compare(left: Any, op: str, right: Any) -> bool:
    """通用比较器。"""

    if op in {"<=", ">=", "<", ">"}:
        left_num = _to_int(left, default=0)
        right_num = _to_int(right, default=0)
        if op == "<=":
            return left_num <= right_num
        if op == ">=":
            return left_num >= right_num
        if op == "<":
            return left_num < right_num
        return left_num > right_num

    left_text = "" if left is None else str(left)
    right_text = "" if right is None else str(right)
    if op == "!=":
        return left_text != right_text
    return left_text == right_text


def _get_value(state: GameState, npc_name: str, key: str) -> Any:
    """从状态读取条件比较值。"""

    if key in {"turn", "回合"}:
        return state.turn
    if key in state.metrics:
        return state.metrics.get(key)
    if key in {"npc_status", "npc_statuses"}:
        return state.npc_statuses.get(npc_name)
    if key.startswith("npc_status."):
        target = key.split(".", 1)[1].strip()
        return state.npc_statuses.get(target)
    if key == "chapter_id":
        return state.chapter_id
    return None


def _eval_string_condition(state: GameState, npc_name: str, condition: str) -> bool:
    """解析字符串条件。"""

    normalized = condition.replace(" ", "")
    match = re.match(r"^(.+?)(<=|>=|==|!=|<|>)(.+)$", normalized)
    if not match:
        logger.warning("无法解析NPC事件条件 npc=%s condition=%s", npc_name, condition)
        return False

    left_key, op, right_raw = match.groups()
    left_val = _get_value(state, npc_name, left_key)
    return _compare(left_val, op, right_raw)


def _eval_condition_item(state: GameState, npc_name: str, condition: Any) -> bool:
    """判定单条条件。"""

    if isinstance(condition, str):
        return _eval_string_condition(state, npc_name, condition)

    if not isinstance(condition, dict):
        return False

    ctype = str(condition.get("type", "metric")).strip().lower()
    if ctype == "metric":
        name = str(condition.get("name", "")).strip()
        op = str(condition.get("op", "<="))
        value = condition.get("value", 0)
        return _compare(state.metrics.get(name, 0), op, value)
    if ctype == "turn":
        op = str(condition.get("op", ">="))
        value = condition.get("value", 0)
        return _compare(state.turn, op, value)
    if ctype == "npc_status":
        target_npc = str(condition.get("npc", npc_name)).strip() or npc_name
        status = str(condition.get("status", "")).strip()
        op = str(condition.get("op", "=="))
        return _compare(state.npc_statuses.get(target_npc), op, status)
    if ctype == "decision_completed":
        decision_id = str(condition.get("decision_id", "")).strip()
        return bool(decision_id) and decision_id in state.decisions_completed

    logger.warning("未知NPC条件类型 npc=%s type=%s", npc_name, ctype)
    return False


def _is_event_already_triggered(state: GameState, event_id: str) -> bool:
    """检查事件是否在本章已触发。"""

    return event_id in state.current_chapter_events_triggered


def _check_event_trigger(state: GameState, npc_name: str, event: dict[str, Any]) -> bool:
    """判定事件是否触发。"""

    trigger = event.get("trigger", {})
    if not isinstance(trigger, dict):
        return False

    ttype = str(trigger.get("type", "condition")).strip().lower()

    if ttype == "chapter_enter":
        event_chapter = str(event.get("chapter", "")).strip()
        # 章节进入事件只在新章首次处理时触发，依赖章节切换后事件列表调用。
        return not event_chapter or event_chapter == state.chapter_id

    if ttype == "condition":
        conditions = trigger.get("conditions", [])
        if not isinstance(conditions, list) or not conditions:
            return False
        mode = str(trigger.get("mode", "all")).strip().lower()
        results = [_eval_condition_item(state, npc_name, cond) for cond in conditions]
        return all(results) if mode == "all" else any(results)

    if ttype == "decision":
        decision_id = str(trigger.get("decision_id", "")).strip()
        if not decision_id:
            return False
        if decision_id not in state.decisions_completed:
            return False

        expected_choice = str(trigger.get("choice", "")).strip().lower()
        if not expected_choice:
            return True

        # 从最近决策记录中匹配选项关键词。
        for record in reversed(state.decisions):
            if not isinstance(record, dict):
                continue
            record_decision_id = str(record.get("decision_id", "")).strip()
            if record_decision_id and record_decision_id != decision_id:
                continue
            # 优先使用结构化choice字段，避免自然语言匹配偏差。
            record_choice = str(record.get("choice", "")).strip().lower()
            if record_choice and record_choice == expected_choice:
                return True
            decision_text = str(record.get("decision", "")).strip().lower()
            if expected_choice and expected_choice in decision_text:
                return True
        return False

    if ttype == "npc_interaction":
        target_text = str(trigger.get("contains", "")).strip()
        target_npc = str(trigger.get("npc", npc_name)).strip() or npc_name
        if target_npc != npc_name:
            return False
        if not target_text:
            return False

        for message in reversed(state.conversation_history[-10:]):
            if not isinstance(message, dict):
                continue
            if str(message.get("speaker", "")) != "player":
                continue
            content = str(message.get("content", ""))
            if target_text in content:
                return True
        return False

    logger.warning("未知事件触发类型 npc=%s type=%s", npc_name, ttype)
    return False


async def check_npc_events(
    state: GameState,
    trigger_types: set[str] | None = None,
) -> list[dict]:
    """扫描活跃NPC事件，返回本轮触发事件列表。"""

    triggered: list[dict] = []

    for npc_name in list(state.active_npcs.keys()):
        try:
            events = await load_npc_events(state.script_id, npc_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载NPC事件失败 npc=%s err=%s", npc_name, exc)
            continue

        for event in events:
            event_id = str(event.get("id", "")).strip()
            if not event_id:
                continue
            if _is_event_already_triggered(state, event_id):
                continue

            trigger = event.get("trigger", {})
            trigger_type = str(trigger.get("type", "condition")).strip().lower() if isinstance(trigger, dict) else ""
            if trigger_types is not None and trigger_type not in trigger_types:
                continue

            event_chapter = str(event.get("chapter", "")).strip()
            if event_chapter and event_chapter != state.chapter_id:
                continue

            if _check_event_trigger(state, npc_name, event):
                enriched = dict(event)
                enriched["npc_name"] = npc_name
                triggered.append(enriched)

    if triggered:
        logger.info("检测到NPC事件触发 chapter=%s event_ids=%s", state.chapter_id, [e.get("id") for e in triggered])
    return triggered


async def apply_event(state: GameState, event: dict[str, Any]) -> None:
    """执行NPC事件效果。"""

    event_id = str(event.get("id", "")).strip()
    npc_name = str(event.get("npc_name", event.get("npc_name", ""))).strip()
    effect = event.get("effect", {})
    if not isinstance(effect, dict):
        effect = {}

    # 支持 effect.metrics 与平铺指标写法。
    metrics_patch = effect.get("metrics", {}) if isinstance(effect.get("metrics"), dict) else {}
    if not metrics_patch:
        metrics_patch = {key: value for key, value in effect.items() if key in state.metrics and isinstance(value, (int, float))}

    for metric_name, delta in metrics_patch.items():
        if not isinstance(delta, (int, float)):
            continue
        current = int(state.metrics.get(metric_name, 0))
        state.metrics[metric_name] = max(0, min(100, current + int(delta)))

    npc_status = effect.get("npc_status")
    if npc_status is not None and npc_name:
        state.npc_statuses[npc_name] = str(npc_status)

    # 支持直接事件迁移NPC是否在场。
    active_npcs_patch = effect.get("active_npcs")
    if isinstance(active_npcs_patch, dict):
        for target_npc, patch in active_npcs_patch.items():
            target = str(target_npc).strip()
            if not target:
                continue
            if not isinstance(patch, dict):
                continue
            current = dict(state.active_npcs.get(target, {}))
            current.update(patch)
            state.active_npcs[target] = current

    if event_id and event_id not in state.current_chapter_events_triggered:
        state.current_chapter_events_triggered.append(event_id)

    logger.info("应用NPC事件完成 event_id=%s npc=%s metrics_patch=%s npc_status=%s", event_id, npc_name, metrics_patch, npc_status)
