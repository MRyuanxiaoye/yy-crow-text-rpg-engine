"""决策判定器。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.engine.state import GameState, get_state_manager
from src.llm.client import get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """决策判定结果。"""

    valid: bool
    rejection_reason: str
    effects: dict[str, Any]
    narrative_hint: str
    triggered_events: list[str]


JUDGE_SYSTEM_PROMPT = """
你是历史文字RPG的“决策合理性判定器”。

任务：根据当前国力、制度约束、社会背景，判断玩家决策是否可执行。

判定原则：
1. 以明末政治军事财政现实为基线。
2. 对合理创新保持开放，允许玩家改变历史。
3. 对明显荒谬或条件不足的决策，给出可理解的拒绝原因。
4. 若决策可执行，给出结构化后果effects（维度变化/NPC态度/剧情线变化）。
5. 维度变化建议范围在[-2, +2]，避免过激。

输出JSON结构：
{
  "valid": true/false,
  "rejection_reason": "若valid=false则必填",
  "effects": {
    "dimensions": {"world": {"财政": -1, "兵力": 1}, "character": {}, "relations": {}},
    "npc": {"袁崇焕": {"trust": 5}},
    "storyline": {"line_id": {"status": "推进", "summary": "..."}},
    "phase": "free_dialogue/decision/transition(可选)",
    "game_date": "可选",
    "turn_increment": 1
  },
  "narrative_hint": "给旁白的后果提要",
  "triggered_events": ["event_id_1"]
}
""".strip()


def _normalize_effects(effects: Any) -> dict[str, Any]:
    """清洗effects结构，保证与StateManager兼容。"""

    if not isinstance(effects, dict):
        return {}

    normalized: dict[str, Any] = {}

    dimensions = effects.get("dimensions") or effects.get("dimension_delta") or effects.get("dimension_deltas")
    if isinstance(dimensions, (dict, list)):
        normalized["dimensions"] = dimensions

    npc = effects.get("npc", {})
    if isinstance(npc, dict):
        normalized_npc: dict[str, dict[str, Any]] = {}
        for npc_name, patch in npc.items():
            if isinstance(patch, dict):
                normalized_npc[str(npc_name)] = patch
        if normalized_npc:
            normalized["npc"] = normalized_npc

    storyline = effects.get("storyline", {})
    if isinstance(storyline, dict):
        normalized_storyline: dict[str, dict[str, Any]] = {}
        for line_id, patch in storyline.items():
            if isinstance(patch, dict):
                normalized_storyline[str(line_id)] = patch
        if normalized_storyline:
            normalized["storyline"] = normalized_storyline

    for key in ["phase", "game_date"]:
        if isinstance(effects.get(key), str):
            normalized[key] = effects[key]

    turn_inc = effects.get("turn_increment")
    if isinstance(turn_inc, int):
        normalized["turn_increment"] = turn_inc
    else:
        normalized["turn_increment"] = 1

    if isinstance(effects.get("current_scene"), dict):
        normalized["current_scene"] = effects["current_scene"]

    if isinstance(effects.get("cast"), dict):
        normalized["cast"] = effects["cast"]

    return normalized


def _build_decision_record(
    user_decision: str,
    valid: bool,
    narrative_hint: str,
    triggered_events: list[str],
) -> dict[str, Any]:
    """构建决策历史记录。"""

    return {
        "decision": user_decision,
        "valid": valid,
        "narrative_hint": narrative_hint,
        "triggered_events": triggered_events,
    }


def _match_preset_choice(user_decision: str, decision_point: dict[str, Any]) -> dict[str, Any] | None:
    """在预设选项中匹配用户输入。"""

    choices = decision_point.get("preset_choices", [])
    if not isinstance(choices, list):
        return None

    normalized_user = user_decision.strip().lower()
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        label = str(choice.get("label", "")).strip()
        if not label:
            continue

        # 精确匹配或包含匹配。
        label_lower = label.lower()
        if normalized_user == label_lower or label_lower in normalized_user or normalized_user in label_lower:
            return choice

    return None


async def judge_decision(
    state: GameState,
    user_decision: str,
    decision_point: dict[str, Any] | None = None,
) -> JudgeResult:
    """判定玩家决策可行性并返回结构化后果。"""

    # 步骤1：预设选项优先。
    if decision_point:
        matched = _match_preset_choice(user_decision, decision_point)
        if matched is not None:
            logger.info("命中预设选项: %s", matched.get("label"))
            effects = _normalize_effects(matched.get("effects", {}))
            hint = str(matched.get("narrative_hint", "圣断既下，群臣随之而动，局势应声生变。"))
            events = matched.get("triggered_events", [])
            if not isinstance(events, list):
                events = []

            effects["decision_record"] = _build_decision_record(
                user_decision=user_decision,
                valid=True,
                narrative_hint=hint,
                triggered_events=[str(item) for item in events],
            )

            return JudgeResult(
                valid=True,
                rejection_reason="",
                effects=effects,
                narrative_hint=hint,
                triggered_events=[str(item) for item in events],
            )

    # 步骤2：自由输入走LLM判定。
    llm = get_llm_client()
    dimension_desc = get_state_manager().get_dimension_description(state)

    user_content = (
        "请判定下列决策是否可执行，并给出结构化后果。\n"
        f"玩家决策：{user_decision}\n"
        f"当前阶段：{state.phase}\n"
        f"当前日期：{state.game_date}\n"
        f"当前场景：{json.dumps(state.current_scene, ensure_ascii=False)}\n"
        f"维度状态：\n{dimension_desc}\n"
        f"活跃NPC：{json.dumps(state.active_npcs, ensure_ascii=False)}\n"
        f"近期决策历史：{json.dumps(state.decisions[-6:], ensure_ascii=False)}\n"
        f"决策点信息：{json.dumps(decision_point or {}, ensure_ascii=False)}"
    )

    logger.info("进入LLM自由判定 user_decision=%s", user_decision)
    result_json = await llm.chat_json(JUDGE_SYSTEM_PROMPT, user_content, temperature=0.2)

    valid = bool(result_json.get("valid", False))
    rejection_reason = str(result_json.get("rejection_reason", "")).strip()
    narrative_hint = str(result_json.get("narrative_hint", "局势正随圣断而变。"))
    triggered_events_raw = result_json.get("triggered_events", [])
    triggered_events = [str(item) for item in triggered_events_raw] if isinstance(triggered_events_raw, list) else []

    effects = _normalize_effects(result_json.get("effects", {}))

    if valid:
        effects["decision_record"] = _build_decision_record(
            user_decision=user_decision,
            valid=True,
            narrative_hint=narrative_hint,
            triggered_events=triggered_events,
        )
    else:
        # 无效决策不应用实质后果，但保留记录。
        effects = {
            "decision_record": _build_decision_record(
                user_decision=user_decision,
                valid=False,
                narrative_hint=narrative_hint,
                triggered_events=triggered_events,
            )
        }
        if not rejection_reason:
            rejection_reason = "群议未允，此令暂难施行。"

    logger.info("决策判定完成 valid=%s triggered_events=%s", valid, triggered_events)

    return JudgeResult(
        valid=valid,
        rejection_reason=rejection_reason,
        effects=effects,
        narrative_hint=narrative_hint,
        triggered_events=triggered_events,
    )
