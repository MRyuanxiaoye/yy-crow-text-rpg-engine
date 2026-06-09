"""章节推进引擎。"""

# 已废弃：章节推进机制已被目标驱动的压力源系统替代。
# 保留本文件仅供旧剧本兼容与历史实现参考，请勿在新流程中继续扩展。

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.engine.state import GameState
from src.llm.client import get_claude_client
from src.scripts.loader import load_chapter, load_npc_profile, load_timeline

logger = logging.getLogger(__name__)


def _to_int(value: Any, default: int = 0) -> int:
    """安全转整数。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compare(left: Any, op: str, right: Any) -> bool:
    """通用比较器，支持数值和字符串比较。"""

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


def _get_condition_value(state: GameState, key: str) -> Any:
    """从状态中读取条件对应值。"""

    if key == "turn" or key == "回合":
        return state.turn
    if key in {"decision_completed", "decisions_completed"}:
        return state.decisions_completed
    if key in {"chapter_id", "章节"}:
        return state.chapter_id

    if key in state.metrics:
        return state.metrics.get(key)

    # 兼容“npc_status.袁崇焕”写法。
    if key.startswith("npc_status."):
        npc_name = key.split(".", 1)[1].strip()
        return state.npc_statuses.get(npc_name)

    return None


def _eval_string_condition(state: GameState, condition: str) -> bool:
    """解析字符串条件，如“边防<=40”“回合 >= 6”。"""

    cond = condition.strip()
    if not cond:
        return False

    # 兼容“完成决策:D_xxx”语法。
    complete_match = re.match(r"^完成决策[:：]\s*(.+)$", cond)
    if complete_match:
        decision_id = complete_match.group(1).strip()
        return bool(decision_id) and decision_id in state.decisions_completed

    # 兼容 "decision_completed contains D_xxx" 语法。
    contains_match = re.match(
        r"^(decision_completed|decisions_completed)\s+contains\s+(.+)$",
        cond,
        re.IGNORECASE,
    )
    if contains_match:
        _, decision_id = contains_match.groups()
        return decision_id.strip() in state.decisions_completed

    normalized = cond.replace(" ", "")
    match = re.match(r"^(.+?)(<=|>=|==|!=|<|>)(.+)$", normalized)
    if not match:
        logger.warning("无法解析章节条件: %s", condition)
        return False

    left_key, op, right_raw = match.groups()
    left_val = _get_condition_value(state, left_key)

    # decision_completed contains D_xxx
    if left_key in {"decision_completed", "decisions_completed"} and op in {"==", "!="}:
        target = right_raw.strip()
        exists = target in state.decisions_completed
        return exists if op == "==" else not exists

    return _compare(left_val, op, right_raw)


def _eval_dict_condition(state: GameState, condition: dict[str, Any]) -> bool:
    """解析字典条件。"""

    ctype = str(condition.get("type", "metric")).strip().lower()

    if ctype == "metric":
        metric_name = str(condition.get("name", "")).strip()
        op = str(condition.get("op", "<="))
        value = condition.get("value", 0)
        current = state.metrics.get(metric_name, 0)
        return _compare(current, op, value)

    if ctype == "turn":
        op = str(condition.get("op", ">="))
        value = condition.get("value", 0)
        return _compare(state.turn, op, value)

    if ctype == "decision_completed":
        decision_id = str(condition.get("decision_id", "")).strip()
        if not decision_id:
            return False
        return decision_id in state.decisions_completed

    if ctype == "npc_status":
        npc_name = str(condition.get("npc", "")).strip()
        status = str(condition.get("status", "")).strip()
        op = str(condition.get("op", "=="))
        current = state.npc_statuses.get(npc_name)
        return _compare(current, op, status)

    logger.warning("未知章节条件类型: %s", ctype)
    return False


def _eval_condition(state: GameState, condition: Any) -> bool:
    """统一条件判定入口。"""

    if isinstance(condition, dict):
        return _eval_dict_condition(state, condition)
    if isinstance(condition, str):
        return _eval_string_condition(state, condition)
    return False


def _extract_advance_conditions(chapter: dict[str, Any]) -> list[Any]:
    """提取章节推进条件。"""

    for key in ["advance_conditions", "progression_conditions", "推进条件", "conditions"]:
        value = chapter.get(key)
        if isinstance(value, list):
            return value
        # 兼容：
        # advance_conditions:
        #   mode: any/all
        #   conditions: [...]
        if isinstance(value, dict):
            conditions = value.get("conditions")
            if isinstance(conditions, list):
                return conditions
    return []


def _extract_hybrid_advance_config(chapter: dict[str, Any]) -> dict[str, Any]:
    """提取推进配置：min_turns 后每回合 Claude 判定，max_turns 强制推进。"""

    block = chapter.get("advance_conditions")
    if not isinstance(block, dict):
        return {}

    max_turns_raw = block.get("max_turns") or block.get("turn_threshold")
    if max_turns_raw is None:
        return {}

    max_turns = max(3, _to_int(max_turns_raw, default=8))
    min_turns = max(1, _to_int(block.get("min_turns", 2), default=2))

    acceleration_raw = block.get("acceleration")
    acceleration = acceleration_raw if isinstance(acceleration_raw, dict) else {}
    accel_enabled = bool(acceleration.get("enabled", True))
    accel_turns_reduction = max(0, min(3, _to_int(acceleration.get("turns_reduction"), default=1)))
    accel_decisions = acceleration.get("key_decisions")
    if not isinstance(accel_decisions, list):
        accel_decisions = chapter.get("key_decisions", [])
    accel_decision_ids = [str(item).strip() for item in accel_decisions if str(item).strip()]

    return {
        "min_turns": min_turns,
        "max_turns": max_turns,
        "acceleration_enabled": accel_enabled,
        "acceleration_turns_reduction": accel_turns_reduction,
        "acceleration_key_decisions": accel_decision_ids,
    }


def _get_chapter_turn_elapsed(state: GameState) -> int:
    """计算当前章节已推进的玩家回合数。"""

    progression = state.storylines.get("progression")
    if not isinstance(progression, dict):
        progression = {}
        state.storylines["progression"] = progression
    start_turn = progression.get("chapter_start_turn")
    if not isinstance(start_turn, int):
        # 兼容旧存档：首次缺失时从当前回合开始计，不瞬间触发推进。
        progression["chapter_start_turn"] = int(state.turn)
        return 0
    return max(0, state.turn - start_turn)


def _get_acceleration_bonus(state: GameState, key_decisions: list[str], turns_reduction: int) -> int:
    """根据关键决策完成情况计算阈值减免。"""

    if turns_reduction <= 0 or not key_decisions:
        return 0
    completed = [decision_id for decision_id in key_decisions if decision_id in state.decisions_completed]
    return turns_reduction if completed else 0


def _build_narrative_maturity_payload(
    state: GameState,
    chapter: dict[str, Any],
    turn_elapsed: int,
    min_turns: int,
    max_turns: int,
) -> dict[str, Any]:
    """构建叙事成熟度判定输入。"""

    recent_history: list[dict[str, Any]] = []
    for item in state.conversation_history[-8:]:
        if not isinstance(item, dict):
            continue
        recent_history.append(
            {
                "role": str(item.get("role", "")),
                "speaker": str(item.get("speaker", "")),
                "content": str(item.get("content", ""))[:200],
            }
        )

    return {
        "chapter_id": state.chapter_id,
        "chapter_title": str(chapter.get("title", chapter.get("name", state.chapter_id))),
        "chapter_theme": str(chapter.get("theme", chapter.get("opening_narration", "")))[:200],
        "turn": state.turn,
        "turn_elapsed_in_chapter": turn_elapsed,
        "min_turns": min_turns,
        "max_turns": max_turns,
        "decisions_completed": state.decisions_completed,
        "key_decisions": chapter.get("key_decisions", []),
        "metrics": state.metrics,
        "recent_history": recent_history,
    }


async def _check_narrative_maturity(
    state: GameState,
    chapter: dict[str, Any],
    turn_elapsed: int,
    min_turns: int,
    max_turns: int,
) -> bool:
    """调用Claude判断叙事成熟度——当前章节是否已讲完该讲的故事。"""

    payload = _build_narrative_maturity_payload(
        state=state,
        chapter=chapter,
        turn_elapsed=turn_elapsed,
        min_turns=min_turns,
        max_turns=max_turns,
    )
    system_prompt = (
        "你是明末历史文字RPG的叙事节奏评估器。\n"
        "你的任务是判断当前章节的核心戏剧冲突是否已经充分展开并达到了一个自然段落。\n"
        "判断标准：\n"
        "- 玩家是否已经做出了影响局势的关键选择（不要求所有决策都完成）\n"
        "- 当前章节的核心矛盾是否已经有了明确走向\n"
        "- 对话是否已经从探索/铺垫进入了收束/总结阶段\n"
        "- 如果玩家行动果断、推动力强，即使回合数少也可以判定ready\n"
        "只返回JSON：{\"ready\": true/false, \"reason\": \"一句话原因\"}"
    )
    user_content = f"状态摘要：{json.dumps(payload, ensure_ascii=False)}"

    try:
        result = await get_claude_client().chat_json(system_prompt, user_content, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Claude叙事成熟度判定失败 chapter_id=%s err=%s", state.chapter_id, exc)
        return False

    ready = bool(result.get("ready", False))
    reason = str(result.get("reason", "")).strip()
    logger.info("Claude叙事成熟度判定 chapter_id=%s turn_elapsed=%s ready=%s reason=%s", state.chapter_id, turn_elapsed, ready, reason)
    return ready


def _find_next_chapter_id(chapter: dict[str, Any], timeline: list[dict], current_chapter_id: str) -> str | None:
    """确定下一章节ID。"""

    explicit = chapter.get("next_chapter") or chapter.get("next_chapter_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    # 从timeline顺序推断。
    if timeline:
        chapter_ids: list[str] = []
        for item in timeline:
            if not isinstance(item, dict):
                continue
            chapter_id = str(item.get("chapter_id") or item.get("id") or "").strip()
            if chapter_id:
                chapter_ids.append(chapter_id)
        if current_chapter_id in chapter_ids:
            idx = chapter_ids.index(current_chapter_id)
            if idx + 1 < len(chapter_ids):
                return chapter_ids[idx + 1]

    # 最后尝试兼容章节文件中定义 chapters: [a,b,c]
    chapters = chapter.get("chapters")
    if isinstance(chapters, list):
        chapter_ids = [str(item).strip() for item in chapters if str(item).strip()]
        if current_chapter_id in chapter_ids:
            idx = chapter_ids.index(current_chapter_id)
            if idx + 1 < len(chapter_ids):
                return chapter_ids[idx + 1]

    return None


async def check_advance_conditions(state: GameState) -> bool:
    """检查当前章节推进条件是否满足。

    核心逻辑：min_turns 后每回合 Claude 判定叙事成熟度，max_turns 强制推进。
    """

    try:
        chapter = await load_chapter(state.script_id, state.chapter_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("加载章节失败，无法检查推进条件 chapter_id=%s err=%s", state.chapter_id, exc)
        return False

    hybrid = _extract_hybrid_advance_config(chapter)
    if hybrid:
        min_turns = int(hybrid["min_turns"])
        max_turns = int(hybrid["max_turns"])
        key_decisions = list(hybrid["acceleration_key_decisions"])
        turn_elapsed = _get_chapter_turn_elapsed(state)

        # 决策加速：完成关键决策可减少 min_turns。
        accel_bonus = 0
        if bool(hybrid["acceleration_enabled"]):
            accel_bonus = _get_acceleration_bonus(
                state=state,
                key_decisions=key_decisions,
                turns_reduction=int(hybrid["acceleration_turns_reduction"]),
            )
        effective_min = max(1, min_turns - accel_bonus)

        # 强制推进：到达 max_turns 无条件推进。
        if turn_elapsed >= max_turns:
            logger.info(
                "章节强制推进(max_turns) chapter_id=%s turn_elapsed=%s max_turns=%s",
                state.chapter_id, turn_elapsed, max_turns,
            )
            return True

        # 未达 min_turns：不推进。
        if turn_elapsed < effective_min:
            logger.info(
                "章节推进未满足(未达min) chapter_id=%s turn_elapsed=%s effective_min=%s max=%s",
                state.chapter_id, turn_elapsed, effective_min, max_turns,
            )
            return False

        # min_turns ~ max_turns 之间：每回合 Claude 判定叙事成熟度。
        ready = await _check_narrative_maturity(
            state=state,
            chapter=chapter,
            turn_elapsed=turn_elapsed,
            min_turns=effective_min,
            max_turns=max_turns,
        )
        if ready:
            logger.info(
                "章节推进命中叙事成熟度 chapter_id=%s turn_elapsed=%s min=%s max=%s",
                state.chapter_id, turn_elapsed, effective_min, max_turns,
            )
            return True

        logger.info(
            "Claude判定未成熟，继续当前章节 chapter_id=%s turn_elapsed=%s min=%s max=%s",
            state.chapter_id, turn_elapsed, effective_min, max_turns,
        )
        return False

    # 旧格式兜底兼容。
    conditions = _extract_advance_conditions(chapter)
    if not conditions:
        logger.info("章节未配置推进条件 chapter_id=%s", state.chapter_id)
        return False

    mode = str(chapter.get("advance_mode", "")).strip().lower()
    if not mode:
        advance_block = chapter.get("advance_conditions")
        if isinstance(advance_block, dict):
            mode = str(advance_block.get("mode", "")).strip().lower()
    if not mode:
        mode = "any"
    result_list = [_eval_condition(state, condition) for condition in conditions]
    ok = all(result_list) if mode == "all" else any(result_list)

    logger.info("章节推进条件检查(旧格式) chapter_id=%s mode=%s results=%s ok=%s", state.chapter_id, mode, result_list, ok)
    return ok


async def advance_chapter(state: GameState) -> dict[str, Any]:
    """执行章节切换并返回新章节信息。"""

    current_chapter_id = state.chapter_id
    current_chapter = await load_chapter(state.script_id, current_chapter_id)
    timeline = await load_timeline(state.script_id)

    next_chapter_id = _find_next_chapter_id(current_chapter, timeline, current_chapter_id)
    if not next_chapter_id:
        logger.info("未找到下一章节，保持当前章节 chapter_id=%s", current_chapter_id)
        return {
            "changed": False,
            "chapter_id": state.chapter_id,
            "reason": "no_next_chapter",
        }

    next_chapter = await load_chapter(state.script_id, next_chapter_id)
    initial_scene = next_chapter.get("initial_scene", {})
    if not isinstance(initial_scene, dict):
        initial_scene = {}

    raw_present_npcs = initial_scene.get("present_npcs", [])
    present_npcs = [str(item).strip() for item in raw_present_npcs if str(item).strip()] if isinstance(raw_present_npcs, list) else []

    next_active_npcs: dict[str, dict[str, Any]] = {}
    for npc_name in present_npcs:
        try:
            profile = await load_npc_profile(state.script_id, npc_name)
            next_active_npcs[npc_name] = dict(profile) if isinstance(profile, dict) else {}
        except FileNotFoundError:
            logger.warning("章节切换时缺少NPC人格卡 script_id=%s npc=%s", state.script_id, npc_name)
            next_active_npcs[npc_name] = {"name": npc_name}

    # 章节切换时重置”本章已触发事件”和施压状态。
    state.npc_pressure_count = {}
    state.settlements_since_last_objective = 0
    state.chapter_id = next_chapter_id
    state.game_date = str(next_chapter.get("game_date_start", state.game_date))
    state.current_scene = initial_scene
    state.active_npcs = next_active_npcs
    state.current_chapter_events_triggered = []
    for npc_name in next_active_npcs:
        if npc_name not in state.npc_statuses:
            state.npc_statuses[npc_name] = "在场"

    # 更新推进元数据，记录新章起始回合并清空未决策跟踪。
    progression = state.storylines.get("progression", {})
    if not isinstance(progression, dict):
        progression = {}
    progression["chapter_start_turn"] = int(state.turn)
    progression["decision_first_seen"] = {}
    progression["auto_evolved_decisions"] = {}
    state.storylines["progression"] = progression

    chapter_enter_events = next_chapter.get("chapter_enter_events", [])
    if not isinstance(chapter_enter_events, list):
        chapter_enter_events = []

    transition_text = str(next_chapter.get("transition_narration", "")).strip()
    if not transition_text:
        transition_text = f"时局推演至新章《{next_chapter.get('title', next_chapter_id)}》。"

    logger.info(
        "章节推进完成 from=%s to=%s game_date=%s npcs=%s",
        current_chapter_id,
        next_chapter_id,
        state.game_date,
        present_npcs,
    )

    return {
        "changed": True,
        "from_chapter_id": current_chapter_id,
        "chapter_id": next_chapter_id,
        "title": str(next_chapter.get("title", next_chapter_id)),
        "game_date": state.game_date,
        "current_scene": state.current_scene,
        "active_npcs": list(state.active_npcs.keys()),
        "chapter_enter_events": chapter_enter_events,
        "transition_narration": transition_text,
    }
