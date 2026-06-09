"""旁白引擎，负责生成叙事内容。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.engine.state import GameState, get_state_manager
from src.engine.tracer import get_tracer, GameTracer
from src.llm.client import get_llm_client
from src.scripts.loader import is_new_format_script, load_npc_profile_at_time, load_role

logger = logging.getLogger(__name__)

DEFAULT_NARRATION_STYLE: dict[str, Any] = {
    "person": "second_person",
    "player_address": "你",
    "narrator_voice": "古典但不晦涩，冷静、克制、强调因果",
    "information_delivery": {
        "fallback_enabled": True,
        "fallback_after_units": 3,
        "fallback_style": "以急报、传闻或环境变化自然补足信息。",
    },
    "taboo": ["不要直接剧透隐藏真相", "不要替玩家做价值判断"],
}

NARRATOR_SYSTEM_PROMPT = """
你是一个文字RPG的旁白机器人，职责是推进世界、汇报局势、推送事件，而不是替NPC说话。

写作总则：
1. 严格遵守本次用户消息中的旁白风格配置，包括人称、玩家称呼、叙事声音、信息投递方式和禁忌。
2. 叙述默认使用状态趋势词；只有玩家明确询问数值，或任务要求汇报结构化变化时，才可给出具体数字。
3. 保持世界因果清晰，有画面感，句式有节奏；不要输出标题、标签或解释性前后缀。
4. 你知道设定中的世界压力，但不得剧透隐藏真相，不得替玩家判断下一步该怎么做。
5. 不要描述NPC的性格特征、内心想法或行为动机。只描述外在可观察到的言行举止和公开身份。
""".strip()

_LOW_DIMENSION_WARNING = 3
_DIFFICULTY_LOW_DIMENSION_COUNT = 2
_ODD_ACTION_PATTERNS = [
    r"随便|乱来|无所谓|都杀|全杀|烧光|毁掉|投降|自尽|摆烂|睡觉|什么都不做",
    r"把.*(国库|军队|百姓|朝廷).*(扔|送|全给|烧|毁)",
]
_PRIORITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "low": 3}


@dataclass(frozen=True)
class NarrationStyle:
    """旁白风格配置。"""

    person: str = "second_person"
    player_address: str = "你"
    narrator_voice: str = "古典但不晦涩，冷静、克制、强调因果"
    information_delivery: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_NARRATION_STYLE["information_delivery"]))
    taboo: list[str] = field(default_factory=lambda: list(DEFAULT_NARRATION_STYLE["taboo"]))

    @property
    def fallback_enabled(self) -> bool:
        """是否启用旁白信息兜底。"""

        return bool(self.information_delivery.get("fallback_enabled", True))

    @property
    def fallback_after_units(self) -> int:
        """读取兜底等待时间单位，至少为一。"""

        raw_value = self.information_delivery.get("fallback_after_units", 3)
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 3

    @property
    def fallback_style(self) -> str:
        """读取兜底信息投递风格。"""

        return str(self.information_delivery.get("fallback_style", "以急报、传闻或环境变化自然补足信息。"))


def _merge_style_payload(payload: Mapping[str, Any] | None) -> NarrationStyle:
    """合并剧本配置与默认旁白风格。"""

    merged = dict(DEFAULT_NARRATION_STYLE)
    if isinstance(payload, Mapping):
        for key in ("person", "player_address", "narrator_voice"):
            if payload.get(key):
                merged[key] = str(payload[key])
        if isinstance(payload.get("information_delivery"), Mapping):
            info_delivery = dict(DEFAULT_NARRATION_STYLE["information_delivery"])
            info_delivery.update(dict(payload["information_delivery"]))
            merged["information_delivery"] = info_delivery
        if isinstance(payload.get("taboo"), list):
            merged["taboo"] = [str(item) for item in payload["taboo"] if str(item).strip()]
    return NarrationStyle(
        person=str(merged["person"]),
        player_address=str(merged["player_address"]),
        narrator_voice=str(merged["narrator_voice"]),
        information_delivery=dict(merged["information_delivery"]),
        taboo=list(merged["taboo"]),
    )


async def _load_narration_style(state: GameState) -> NarrationStyle:
    """从角色层剧本读取旁白风格，旧剧本或缺失时降级为默认配置。"""

    if not is_new_format_script(state.script_id):
        return _merge_style_payload(None)
    try:
        role = await load_role(state.script_id, state.player_role)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        logger.warning("读取旁白风格失败，使用默认配置 script_id=%s role=%s err=%s", state.script_id, state.player_role, exc)
        return _merge_style_payload(None)
    payload = role.get("narration_style") if isinstance(role, dict) else None
    return _merge_style_payload(payload if isinstance(payload, Mapping) else None)


def _public_npc_summary(state: GameState) -> list[str]:
    """生成旁白可见的在场NPC公开摘要。"""

    scene = state.current_scene if isinstance(state.current_scene, dict) else {}
    present_npcs = scene.get("present_npcs", [])
    if not isinstance(present_npcs, list):
        return []

    summaries: list[str] = []
    for raw_name in present_npcs:
        name = str(raw_name).strip()
        if not name:
            continue
        fallback_profile = state.active_npcs.get(name, {}) if isinstance(state.active_npcs.get(name), dict) else {}
        display_name = str(fallback_profile.get("name", "")).strip() or name
        try:
            profile = load_npc_profile_at_time(state.script_id, display_name, state.game_time, state)
        except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
            logger.warning("读取NPC公开摘要失败，降级使用旧格式 script_id=%s npc=%s err=%s", state.script_id, display_name, exc)
            title = str(fallback_profile.get("title") or fallback_profile.get("position") or "未知")
            summaries.append(f"{display_name}（{title}）")
            continue

        if profile.get("is_new_format") is True:
            current_state = profile.get("current_state") if isinstance(profile.get("current_state"), dict) else {}
            title = str(current_state.get("title") or "未知")
            faction = str(current_state.get("faction") or "未知")
            status = str(current_state.get("status") or "未知")
            summaries.append(f"{display_name}（{title}，{faction}，{status}）")
        else:
            logger.warning("NPC公开摘要遇到旧格式，降级使用title script_id=%s npc=%s", state.script_id, display_name)
            title = str(profile.get("title") or profile.get("position") or fallback_profile.get("title") or fallback_profile.get("position") or "未知")
            summaries.append(f"{display_name}（{title}）")
    return summaries


def _state_brief(state: GameState) -> dict[str, Any]:
    """提炼用于旁白生成的状态摘要。"""

    manager = get_state_manager()
    scene = state.current_scene if isinstance(state.current_scene, dict) else {}
    return {
        "script_id": state.script_id,
        "player_role": state.player_role,
        "phase": state.phase,
        "turn": state.turn,
        "game_date": state.game_date,
        "game_time": state.game_time,
        "scene": {
            "location": scene.get("location", "未明"),
            "present_npcs": _public_npc_summary(state),
            "context": scene.get("context", ""),
        },
        "dimension_desc": manager.get_dimension_description(state),
        "pressure_summary": _pressure_summary(state),
        "low_dimensions": _low_dimensions(state),
        "world_memory": state.world_memory,
        "recent_history": state.conversation_history[-5:],
    }


def _pressure_summary(state: GameState) -> dict[str, Any]:
    """压缩压力源摘要，供旁白理解世界压力方向。"""

    sources = state.pressure_sources
    if isinstance(sources, Mapping):
        return {
            "decay": _compact_pressure_items(sources.get("decay", [])),
            "milestones": _compact_pressure_items(sources.get("milestones", sources.get("milestone", []))),
            "reactions": _compact_pressure_items(sources.get("reactions", sources.get("reaction", []))),
        }
    return {"items": _compact_pressure_items(sources)}


def _compact_pressure_items(value: Any, limit: int = 6) -> list[dict[str, Any]]:
    """保留压力源中适合旁白参考的字段。"""

    items = value if isinstance(value, list) else []
    result: list[dict[str, Any]] = []
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            continue
        item = {key: raw_item.get(key) for key in ("pressure_id", "id", "type", "dimension", "rate", "event", "name", "description", "condition", "at") if key in raw_item}
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _low_dimensions(state: GameState, threshold: int = _LOW_DIMENSION_WARNING) -> list[dict[str, Any]]:
    """找出低于警戒线的维度，用于动态立场判断。"""

    result: list[dict[str, Any]] = []
    dimension_groups: list[tuple[str, Mapping[str, int]]] = [
        ("character", state.dimensions.character),
        ("world", state.dimensions.world),
        ("extensions", state.dimensions.extensions),
    ]
    for category, values in dimension_groups:
        for name, value in values.items():
            if value <= threshold:
                result.append({"category": category, "name": name, "value": value})
    for npc_id, relation in state.dimensions.relations.items():
        for name, value in relation.values.items():
            if value <= threshold:
                result.append({"category": "relation", "npc_id": npc_id, "name": name, "value": value})
    return sorted(result, key=lambda item: int(item.get("value", 0)))


def _detect_stance(state: GameState, player_input: str = "") -> dict[str, Any]:
    """根据局势和玩家输入决定旁白动态立场。"""

    low_dimensions = _low_dimensions(state)
    odd_action = _is_odd_player_action(player_input)
    if odd_action:
        return {
            "stance": "讽刺或质疑",
            "reason": "玩家行动明显反常、鲁莽或自毁",
            "guidance": "允许轻微冷嘲，但仍需客观呈现后果，不羞辱玩家。",
            "low_dimensions": low_dimensions,
        }
    if len(low_dimensions) >= _DIFFICULTY_LOW_DIMENSION_COUNT:
        return {
            "stance": "同情并提供更多提示",
            "reason": "多个维度已低于警戒线",
            "guidance": "语气可更体恤，点出可观察线索和风险来源，但不替玩家决策。",
            "low_dimensions": low_dimensions,
        }
    return {
        "stance": "中立客观",
        "reason": "局势未达到困难兜底阈值，玩家行为未见明显反常",
        "guidance": "冷静汇报事实、趋势和后果。",
        "low_dimensions": low_dimensions,
    }


def _is_odd_player_action(player_input: str) -> bool:
    """用轻量规则辅助判断玩家行为是否反常，最终仍交由模型把握语气。"""

    if not player_input.strip():
        return False
    return any(re.search(pattern, player_input) for pattern in _ODD_ACTION_PATTERNS)


def _style_instructions(style: NarrationStyle, stance: dict[str, Any]) -> str:
    """将风格和动态立场转换成提示词。"""

    payload = {
        "person": style.person,
        "player_address": style.player_address,
        "narrator_voice": style.narrator_voice,
        "information_delivery": style.information_delivery,
        "taboo": style.taboo,
        "dynamic_stance": stance,
    }
    return json.dumps(payload, ensure_ascii=False)


async def _chat_narrator(
    state: GameState,
    task: str,
    *,
    player_input: str = "",
    temperature: float = 0.7,
) -> str:
    """统一调用旁白模型，注入剧本风格和动态立场。"""

    llm = get_llm_client()
    style = await _load_narration_style(state)
    stance = _detect_stance(state, player_input)
    user_content = (
        f"旁白风格与动态立场：{_style_instructions(style, stance)}\n"
        "动态立场补充：若玩家输入本身明显荒诞、鲁莽或自毁，模型可判断并改用讽刺或质疑语气，但不得羞辱玩家。\n"
        f"任务：{task}\n"
        f"状态摘要：{json.dumps(_state_brief(state), ensure_ascii=False)}"
    )
    result = await llm.chat(NARRATOR_SYSTEM_PROMPT, user_content, temperature=temperature)
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        _t = get_tracer(_cid, enabled=state.trace_enabled)
        _t.record("llm_call", "_chat_narrator", {
            "task": task[:500],
            "user_content": user_content[:1000],
            "result": result[:500],
        }, GameTracer.state_summary(state))
    return result


async def narrate_scene(state: GameState, scene_setup: str) -> str:
    """生成章节开场或场景切换叙事。"""

    task = (
        "请写一段场景叙事（不超过120字），用于开场或场景切换。\n"
        f"场景设定：{scene_setup}"
    )
    return await _chat_narrator(state, task, temperature=0.75)


async def narrate_action(state: GameState, action_desc: str) -> str:
    """生成玩家行动过程与即时反馈叙事。"""

    task = (
        "玩家发起了一个行动，请叙述行动的推进过程与当下反馈。\n"
        "要求：给出结果倾向，但不要替玩家下后续指令；不超过120字。\n"
        f"玩家行动：{action_desc}"
    )
    return await _chat_narrator(state, task, player_input=action_desc, temperature=0.75)


async def narrate_decision_result(state: GameState, decision: str, effects: dict[str, Any]) -> str:
    """生成决策后果叙事。"""

    task = (
        "玩家已经做出决策，请叙述该决策带来的后果。\n"
        "要求：先写朝堂/局势反应，再写短期影响；不超过120字。\n"
        f"玩家决策：{decision}\n"
        f"结构化后果：{json.dumps(effects, ensure_ascii=False)}"
    )
    return await _chat_narrator(state, task, player_input=decision, temperature=0.7)


async def narrate_transition(state: GameState, time_skip: str, events_during: str) -> str:
    """生成时间推进过渡叙事。"""

    task = (
        "请生成一段时间推进叙事。\n"
        "要求：体现时局流转和关键变化，不超过120字。\n"
        f"时间跨度：{time_skip}\n"
        f"期间事件：{events_during}"
    )
    return await _chat_narrator(state, task, temperature=0.7)


async def narrate_query(state: GameState, query: str) -> str:
    """生成信息查询回复。"""

    need_numbers = any(token in query for token in ["多少", "几", "数值", "具体", "国库", "军力", "民心", "边防", "稳定", "维度"])
    answer_rule = "玩家在查阅信息，请给出准确维度数值并附一句形势解读。" if need_numbers else "玩家在查阅信息，请给出简明局势说明。"
    task = (
        f"{answer_rule}\n"
        "要求：条理清晰，避免空泛；使用当前维度系统，不要提旧metrics。\n"
        f"玩家问题：{query}"
    )
    return await _chat_narrator(state, task, player_input=query, temperature=0.45)


async def narrate_opening_briefing(state: GameState, briefing_label: str) -> str:
    """生成开局简报——分部门/来源的结构化信息汇览。"""

    role_data = state.storylines.get("role", {})
    narration_style = role_data.get("narration_style", {}) if isinstance(role_data, dict) else {}
    player_address = str(narration_style.get("player_address", "你"))
    role_info = role_data.get("role", {}) if isinstance(role_data, dict) else {}
    start_location = str(role_info.get("start_location", ""))

    agenda_lines = []
    for item in (state.turn_agenda if hasattr(state, "turn_agenda") else []):
        if isinstance(item, dict):
            desc = str(item.get("description", ""))
            urgency = str(item.get("urgency", "normal"))
            npcs = item.get("relevant_npcs", [])
            if desc:
                agenda_lines.append(f"- [{urgency}] {desc}（相关：{', '.join(str(n) for n in npcs)}）" if npcs else f"- [{urgency}] {desc}")

    info_lines = []
    for item in (state.information_pool if hasattr(state, "information_pool") else []):
        if isinstance(item, dict):
            content = str(item.get("content", ""))
            known_by = item.get("known_by", [])
            priority = str(item.get("delivery_priority", "medium"))
            if content:
                info_lines.append(f"- [{priority}] {content}（知情者：{', '.join(str(n) for n in known_by)}）" if known_by else f"- [{priority}] {content}")

    identity = str(role_info.get("identity", ""))
    perspective = str(role_info.get("perspective", ""))

    task = (
        f'玩家刚选好角色，这是游戏开局的第一次信息获取。玩家选择了”{briefing_label}”。\n'
        f'角色身份：{identity}\n'
        f'角色视角：{perspective}\n'
        f'当前地点：{start_location}\n'
        f'当前时间：{state.game_date}\n\n'
        '你的任务是生成一份简报（briefing），不是文档转抄。\n'
        '简报 = 替玩家筛选、转述、提炼后的信息摘要。\n'
        '段落标题中的渠道名称、文书类型必须匹配角色身份和世界观，由你推导。\n\n'
        f'**{start_location}，{state.game_date}·{briefing_label}**\n\n'
        '**一、[正式渠道]·[常规文书类型]若干件**\n'
        '  3-4条，每条格式：”**[部门/机构简称][文书简称]：**”+转述内容。\n'
        '  来源标签用部门级（如”户部奏”），不要写人名全称。\n\n'
        '**二、[机密/紧急渠道]·[机密文书类型]**\n'
        '  2-3条，来源不同于第一段。\n\n'
        '**三、[非正式渠道]·[口述/传闻类]**\n'
        '  1-2条，简短转述。\n\n'
        f'**四、局势研判·{player_address}可察之关键**\n'
        '  3-4点，每点一句话，不加粗整句。\n\n'
        f'最后恰好一句话收尾，制造紧迫感。\n\n'
        '语态规则（最重要）：\n'
        '- 用旁白转述语态，不要模拟原始文件的语气\n'
        '- 禁止直接引用原文（禁止出现引号内的角色原话或文件原文）\n'
        '- 每条80-120字，一个关键数字或事实+一个核心判断，一口气能读完\n'
        '- 来源标签只写机构/部门简称，不写人物全名和官职\n'
        f'- 全程用”{player_address}”称呼玩家\n'
        '- 内容必须基于状态摘要中的真实数据，不编造\n'
        '- 段落标题的方括号是占位符，替换为符合世界观的具体名称\n'
        '- 禁止文学修辞、排比句、四字成语堆砌\n\n'
        f'当前议程：\n{"".join(agenda_lines) or "无"}\n\n'
        f'信息池：\n{"".join(info_lines) or "无"}\n'
    )
    return await _chat_narrator(state, task, temperature=0.5)


async def present_decision(state: GameState, decision_point: dict[str, Any]) -> str:
    """呈现决策点与可选行动。"""

    setup = str(decision_point.get("narrator_setup", "御前有一事待决。"))
    preset_choices = decision_point.get("preset_choices", [])
    labels = [str(item.get("label", "")).strip() for item in preset_choices if isinstance(item, dict)]
    labels = [item for item in labels if item]

    task = (
        "请生成一段引导玩家做决策的旁白。\n"
        "要求：简述问题核心，不替玩家定夺，不超过120字。\n"
        f"决策背景：{setup}\n"
        f"可选项：{labels if labels else '无预设选项'}"
    )
    opening = await _chat_narrator(state, task, temperature=0.6)

    if not labels:
        return opening

    options_text = "\n".join([f"▸ {label}" for label in labels])
    return f"{opening}\n\n可选行动：\n{options_text}"


async def narrate_information_fallback(state: GameState) -> str:
    """检查并用旁白兜底传递长期未由NPC传达的信息。"""

    style = await _load_narration_style(state)
    state.information_pool = _ensure_information_seen(state)
    due_infos = _collect_fallback_information(state, style.fallback_after_units) if style.fallback_enabled else []
    if not due_infos:
        return ""

    llm = get_llm_client()
    stance = _detect_stance(state)
    task_payload = {
        "fallback_style": style.fallback_style,
        "fallback_after_units": style.fallback_after_units,
        "due_information": [_public_information_payload(item) for item in due_infos],
        "rules": [
            "以急报、传闻、环境描写或文书转呈等方式自然补足信息。",
            "只传递信息内容本身，不暴露未公开来源和隐藏机制。",
            "不超过120字；信息紧急时可更短促。",
        ],
    }
    user_content = (
        f"旁白风格与动态立场：{_style_instructions(style, stance)}\n"
        "任务：执行信息兜底，传递长期未被NPC传达的重要信息。\n"
        f"兜底信息：{json.dumps(task_payload, ensure_ascii=False)}\n"
        f"状态摘要：{json.dumps(_state_brief(state), ensure_ascii=False)}"
    )
    narration = await llm.chat(NARRATOR_SYSTEM_PROMPT, user_content, temperature=0.65)
    _mark_information_delivered_by_narrator(state, due_infos)
    return narration


def _collect_fallback_information(state: GameState, fallback_after_units: int, limit: int = 3) -> list[dict[str, Any]]:
    """找出超过兜底等待时间仍未传达的信息。"""

    due_infos: list[dict[str, Any]] = []
    for info in state.information_pool:
        if not isinstance(info, dict) or _is_information_delivered(info):
            continue
        if _is_information_expired(state, info):
            continue
        age = _information_age_units(state, info)
        if age < fallback_after_units:
            continue
        due_infos.append(info)

    due_infos.sort(key=lambda item: (_PRIORITY_ORDER.get(str(item.get("delivery_priority", "medium")), 2), -_information_age_units(state, item)))
    return due_infos[:limit]


def _is_information_delivered(info: Mapping[str, Any]) -> bool:
    """判断信息是否已经完成对玩家传递。"""

    info_type = str(info.get("type", "fact"))
    if info_type == "fact":
        return bool(info.get("delivered", False))
    if bool(info.get("resolved", False)):
        return True

    delivered_by = info.get("delivered_by")
    if not isinstance(delivered_by, list):
        delivered_by = []
    npc_delivered = any(str(source) != "narrator" for source in delivered_by)
    if info_type == "persistent":
        return npc_delivered or bool(info.get("fallback_delivered", False))

    if info_type == "escalating":
        current_severity = _safe_int(info.get("severity", 1), 1)
        last_delivered_severity = info.get("last_delivered_severity", info.get("last_fallback_severity"))
        if last_delivered_severity is not None:
            return _safe_int(last_delivered_severity, 0) >= current_severity
        return npc_delivered or bool(info.get("fallback_delivered", False))

    if bool(info.get("fallback_delivered", False)):
        return True
    return False


def _is_information_expired(state: GameState, info: Mapping[str, Any]) -> bool:
    """判断信息是否已经过期，过期信息不再兜底播报。"""

    expires_after = info.get("expires_after_units")
    if expires_after is None:
        return False
    try:
        return _information_age_units(state, info) > int(expires_after)
    except (TypeError, ValueError):
        return False


def _information_age_units(state: GameState, info: Mapping[str, Any]) -> int:
    """计算信息在池中停留的时间单位，兼容多种创建字段。"""

    for key in ("undelivered_units", "age_units"):
        if key in info:
            return max(0, _safe_int(info.get(key), 0))
    for key in ("created_at_turn", "created_turn", "turn_created", "introduced_turn"):
        if key in info:
            return max(0, state.turn - _safe_int(info.get(key), state.turn))
    for key in ("created_at_unit", "created_time_unit", "introduced_time_unit"):
        if key in info:
            return max(0, _time_index(state.game_time) - _safe_int(info.get(key), _time_index(state.game_time)))
    if isinstance(info.get("created_at_time"), Mapping):
        return max(0, _time_index(state.game_time) - _time_index(info["created_at_time"]))

    # 旧数据没有创建时间时，视为开局即在信息池中，按当前回合估算年龄。
    first_seen = info.get("fallback_first_seen_turn")
    if first_seen is None:
        return max(0, state.turn)
    return max(0, state.turn - _safe_int(first_seen, state.turn))


def _time_index(game_time: Mapping[str, Any]) -> int:
    """将年月式 game_time 转为可比较的单位序号。"""

    year = _safe_int(game_time.get("year"), 1)
    month = _safe_int(game_time.get("month"), 1)
    return (year - 1) * 12 + month


def _public_information_payload(info: Mapping[str, Any]) -> dict[str, Any]:
    """提取可交给模型的公开信息字段。"""

    payload = {
        "info_id": info.get("info_id"),
        "type": info.get("type", "fact"),
        "delivery_priority": info.get("delivery_priority", "medium"),
        "content": _information_content(info),
    }
    if "severity" in info:
        payload["severity"] = info.get("severity")
    return payload


def _information_content(info: Mapping[str, Any]) -> str:
    """根据升级等级取信息文本。"""

    if str(info.get("type", "fact")) == "escalating" and isinstance(info.get("escalation"), Mapping):
        severity = str(info.get("severity", 1))
        escalation = info["escalation"]
        return str(escalation.get(severity, escalation.get(int(severity) if severity.isdigit() else severity, info.get("content", ""))))
    return str(info.get("content", ""))


def _mark_information_delivered_by_narrator(state: GameState, delivered_infos: list[dict[str, Any]]) -> None:
    """兜底播报后更新信息状态。"""

    delivered_ids = {str(info.get("info_id", "")) for info in delivered_infos}
    state.information_pool = _ensure_information_seen(state)
    for info in state.information_pool:
        if str(info.get("info_id", "")) not in delivered_ids:
            continue
        info_type = str(info.get("type", "fact"))
        info["last_delivered_turn"] = state.turn
        info["last_delivered_by"] = "narrator"
        info["fallback_delivered"] = True
        delivered_by = info.get("delivered_by")
        if not isinstance(delivered_by, list):
            delivered_by = []
        if "narrator" not in delivered_by:
            delivered_by.append("narrator")
        info["delivered_by"] = delivered_by
        if info_type == "fact":
            info["delivered"] = True
        elif info_type == "escalating":
            info["last_fallback_severity"] = info.get("severity", 1)
            info["last_delivered_severity"] = info.get("severity", 1)


def _ensure_information_seen(state: GameState) -> list[dict[str, Any]]:
    """清理信息池中的非对象项，保留既有时间字段。"""

    normalized: list[dict[str, Any]] = []
    for raw_info in state.information_pool:
        if not isinstance(raw_info, dict):
            continue
        normalized.append(raw_info)
    return normalized


def inspect_information_fallback(state: GameState, fallback_after_units: int = 3) -> list[dict[str, Any]]:
    """供调试或测试使用：检查当前应由旁白兜底的信息，并补齐首次检查时间。"""

    state.information_pool = _ensure_information_seen(state)
    return [_public_information_payload(info) for info in _collect_fallback_information(state, fallback_after_units)]


async def narrate_time_advance(state: GameState, advance_result: Any) -> str:
    """汇报主动推进期间的世界变化。"""

    result = _advance_result_payload(advance_result)
    if _is_empty_advance_result(result):
        return "时辰尚未真正推移，局势仍停在原处。"

    style = await _load_narration_style(state)
    stance = _detect_stance(state)
    state.information_pool = _ensure_information_seen(state)
    fallback_infos = _collect_fallback_information(state, style.fallback_after_units) if style.fallback_enabled else []
    payload = {
        "advance_result": result,
        "fallback_information": [_public_information_payload(item) for item in fallback_infos],
        "reporting_rules": [
            "必须说明经过的时间。",
            "必须包含衰减影响、触发事件、到期行动；没有则简明说明无重大变故。",
            "自动跳过纯衰减无事件的时间段，只用一句概括，不逐单位流水账。",
            "如有兜底信息，将其自然并入急报、传闻或环境变化。",
            "不超过120字；事件多时只写最关键的变化。",
        ],
    }
    task = (
        "请汇报一次时间推进期间的世界变化。\n"
        f"推进结构化结果：{json.dumps(payload, ensure_ascii=False)}"
    )
    narration = await _chat_narrator(state, task, temperature=0.65)
    if fallback_infos:
        _mark_information_delivered_by_narrator(state, fallback_infos)
    return narration


def _advance_result_payload(advance_result: Any) -> dict[str, Any]:
    """兼容 dataclass、dict 与旧 tick_time 结果。"""

    if hasattr(advance_result, "__dataclass_fields__"):
        payload = asdict(advance_result)
    elif isinstance(advance_result, Mapping):
        payload = dict(advance_result)
    else:
        payload = {}

    elapsed_time = _safe_int(payload.get("elapsed_time", payload.get("elapsed_units", 0)), 0)
    if elapsed_time == 0 and payload.get("old_time") and payload.get("new_time"):
        elapsed_time = max(0, _time_index(payload["new_time"]) - _time_index(payload["old_time"]))

    triggered_events = payload.get("triggered_events", payload.get("events", []))
    due_actions = payload.get("due_actions", payload.get("expired_actions", payload.get("completed_actions", [])))
    decay_changes = payload.get("decay_changes", payload.get("decay", {}))

    return {
        "elapsed_time": elapsed_time,
        "old_time": payload.get("old_time", {}),
        "new_time": payload.get("new_time", {}),
        "new_time_str": payload.get("new_time_str", ""),
        "decay_changes": decay_changes if isinstance(decay_changes, Mapping) else {},
        "triggered_events": _compact_events(triggered_events),
        "due_actions": _compact_actions(due_actions),
        "bottom_line_broken": bool(payload.get("bottom_line_broken", False)),
        "pure_decay_only": bool(decay_changes) and not triggered_events and not due_actions and not payload.get("bottom_line_broken", False),
    }


def _is_empty_advance_result(result: Mapping[str, Any]) -> bool:
    """判断推进结果是否没有任何可汇报变化。"""

    return (
        _safe_int(result.get("elapsed_time"), 0) <= 0
        and not result.get("decay_changes")
        and not result.get("triggered_events")
        and not result.get("due_actions")
        and not result.get("bottom_line_broken")
    )


def _compact_events(events: Any) -> list[dict[str, Any]]:
    """压缩触发事件字段，避免把隐藏结构完整喂给旁白。"""

    result: list[dict[str, Any]] = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, Mapping):
            continue
        result.append({
            "id": event.get("event_id", event.get("id", "")),
            "type": event.get("type", event.get("source_type", "event")),
            "name": event.get("name", event.get("event", "")),
            "description": event.get("description", event.get("narrative_hint", "")),
        })
    return result


def _compact_actions(actions: Any) -> list[dict[str, Any]]:
    """压缩到期行动字段。"""

    result: list[dict[str, Any]] = []
    for action in actions if isinstance(actions, list) else []:
        if not isinstance(action, Mapping):
            continue
        result.append({
            "id": action.get("action_id", action.get("id", "")),
            "summary": action.get("summary", action.get("description", action.get("action", ""))),
            "actor": action.get("actor", action.get("npc_id", "")),
            "original_time_cost": action.get("original_time_cost", action.get("time_cost", "")),
        })
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    """安全转整数。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default
