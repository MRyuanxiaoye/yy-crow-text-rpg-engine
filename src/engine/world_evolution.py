"""世界自动演化：处理逾期未决策的默认后果。"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.engine.state import GameState, get_state_manager
from src.llm.client import get_claude_client
from src.scripts.loader import load_chapter, load_decision_point

logger = logging.getLogger(__name__)

# 未处理决策超过该章节回合数后，触发自动演化。
UNRESOLVED_DECISION_TTL = 3


def _get_chapter_turn_elapsed(state: GameState) -> int:
    """计算当前章节已推进回合。"""

    progression = state.storylines.get("progression", {})
    if not isinstance(progression, dict):
        return max(0, state.turn)
    start_turn = progression.get("chapter_start_turn")
    if not isinstance(start_turn, int):
        return max(0, state.turn)
    return max(0, state.turn - start_turn)


def _get_progression_bucket(state: GameState) -> dict[str, Any]:
    """读取并初始化推进元数据。"""

    progression = state.storylines.get("progression")
    if not isinstance(progression, dict):
        progression = {}
        state.storylines["progression"] = progression

    if "decision_first_seen" not in progression or not isinstance(progression.get("decision_first_seen"), dict):
        progression["decision_first_seen"] = {}
    if "auto_evolved_decisions" not in progression or not isinstance(progression.get("auto_evolved_decisions"), dict):
        progression["auto_evolved_decisions"] = {}
    if "chapter_start_turn" not in progression or not isinstance(progression.get("chapter_start_turn"), int):
        progression["chapter_start_turn"] = max(0, state.turn)
    return progression


def _normalize_effects(raw_effects: Any) -> dict[str, Any]:
    """清洗Claude返回的effects，保证可被StateManager应用。"""

    if not isinstance(raw_effects, dict):
        return {}

    effects: dict[str, Any] = {}
    metrics = raw_effects.get("metrics", {})
    if isinstance(metrics, dict):
        clean_metrics: dict[str, int] = {}
        for key, value in metrics.items():
            if isinstance(value, int):
                clean_metrics[str(key)] = value
            elif isinstance(value, float):
                clean_metrics[str(key)] = int(value)
        if clean_metrics:
            effects["metrics"] = clean_metrics

    if isinstance(raw_effects.get("npc"), dict):
        effects["npc"] = raw_effects["npc"]
    if isinstance(raw_effects.get("storyline"), dict):
        effects["storyline"] = raw_effects["storyline"]
    if isinstance(raw_effects.get("npc_statuses"), dict):
        effects["npc_statuses"] = raw_effects["npc_statuses"]
    return effects


def _build_fallback_consequence(decision_id: str) -> dict[str, Any]:
    """Claude失败时的兜底后果，不阻塞主流程。"""

    return {
        "consequence": f"关于「{decision_id}」久议不决，朝野自行其是，局势在迟疑中恶化。",
        "effects": {"metrics": {"朝廷稳定": -4, "民心": -3}},
        "npc_impact": "相关官员观望加剧，执行效率下降。",
    }


async def _simulate_unresolved_consequence(
    state: GameState,
    decision_id: str,
    decision_def: dict[str, Any],
) -> dict[str, Any]:
    """调用Claude推演未决策后果。"""

    system_prompt = (
        "你是历史文字RPG的世界演化器。\n"
        "当关键决策久未处理时，请推演“放任不管”的自然后果。\n"
        "输出JSON："
        "{\"consequence\":\"叙事描述\",\"effects\":{\"metrics\":{\"国库\":-3}},\"npc_impact\":\"NPC状态变化\"}\n"
        "要求：effects 只给必要变化，数值变化建议在[-15,+15]。"
    )
    payload = {
        "chapter_id": state.chapter_id,
        "turn": state.turn,
        "decision_id": decision_id,
        "decision_context": decision_def.get("context", ""),
        "decision_options": decision_def.get("options", []),
        "metrics": state.metrics,
        "recent_history": state.conversation_history[-5:],
    }
    user_content = f"状态摘要：{json.dumps(payload, ensure_ascii=False)}"

    try:
        result = await get_claude_client().chat_json(system_prompt, user_content, temperature=0.3)
        consequence = str(result.get("consequence", "")).strip()
        npc_impact = str(result.get("npc_impact", "")).strip()
        effects = _normalize_effects(result.get("effects", {}))
        if not consequence:
            raise ValueError("missing consequence")
        return {
            "consequence": consequence,
            "effects": effects,
            "npc_impact": npc_impact or "局中人物被迫调整立场。",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Claude推演未决策后果失败 decision_id=%s err=%s，使用默认后果", decision_id, exc)
        return _build_fallback_consequence(decision_id)


async def evolve_unresolved_decisions(state: GameState) -> list[dict[str, Any]]:
    """处理逾期未处理的关键决策，并返回已触发后果。"""

    try:
        chapter = await load_chapter(state.script_id, state.chapter_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("世界演化读取章节失败 chapter_id=%s err=%s", state.chapter_id, exc)
        return []

    raw_key_decisions = chapter.get("key_decisions", [])
    if not isinstance(raw_key_decisions, list):
        return []

    key_decisions = [str(item).strip() for item in raw_key_decisions if str(item).strip()]
    if not key_decisions:
        return []

    progression = _get_progression_bucket(state)
    first_seen = progression["decision_first_seen"]
    auto_evolved = progression["auto_evolved_decisions"]
    turn_elapsed = _get_chapter_turn_elapsed(state)

    triggered: list[dict[str, Any]] = []
    state_manager = get_state_manager()

    # 清理已完成决策，避免遗留tracking造成误判。
    for decision_id in list(first_seen.keys()):
        if decision_id in state.decisions_completed:
            first_seen.pop(decision_id, None)

    for decision_id in key_decisions:
        if decision_id in state.decisions_completed:
            continue
        if decision_id in auto_evolved:
            continue

        if decision_id not in first_seen:
            first_seen[decision_id] = turn_elapsed
            continue

        pending_turns = turn_elapsed - int(first_seen.get(decision_id, turn_elapsed))
        if pending_turns < UNRESOLVED_DECISION_TTL:
            continue

        try:
            decision_def = await load_decision_point(state.script_id, decision_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载决策定义失败 decision_id=%s err=%s，走兜底后果", decision_id, exc)
            decision_def = {}

        simulated = await _simulate_unresolved_consequence(state, decision_id, decision_def)
        effects = _normalize_effects(simulated.get("effects", {}))
        if effects:
            state_manager.apply_effects(state, effects)

        # 标记为已处理，避免重复触发。
        if decision_id not in state.decisions_completed:
            state.decisions_completed.append(decision_id)
        auto_evolved[decision_id] = {
            "trigger_turn": state.turn,
            "chapter_turn_elapsed": turn_elapsed,
        }
        first_seen.pop(decision_id, None)

        decision_record = {
            "decision_id": decision_id,
            "decision": f"[系统自动演化] {decision_id}",
            "valid": True,
            "auto_evolved": True,
            "narrative_hint": str(simulated.get("consequence", "")),
            "triggered_events": [],
        }
        state.decisions.append(decision_record)

        outcome = {
            "decision_id": decision_id,
            "consequence": str(simulated.get("consequence", "")).strip(),
            "effects": effects,
            "npc_impact": str(simulated.get("npc_impact", "")).strip(),
            "pending_turns": pending_turns,
        }
        triggered.append(outcome)
        logger.info("未决策自动演化已触发 decision_id=%s pending_turns=%s effects=%s", decision_id, pending_turns, effects)

    return triggered

