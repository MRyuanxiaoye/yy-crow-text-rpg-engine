"""NPC 对话引擎。"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from src.engine.dimension import DEFAULT_DIMENSION_VALUE, RelationDimensions
from src.engine.state import GameState
from src.engine.tracer import get_tracer, GameTracer
from src.llm.client import get_llm_client
from src.scripts.loader import load_npc_profile_at_time

logger = logging.getLogger(__name__)

NPC_SYSTEM_PROMPT_TEMPLATE = """
你现在扮演历史文字RPG中的NPC：{npc_name}

【本次议题】
{conversation_topic_section}

【人格层：你如何说话】
- 性格：{personality}
- 说话风格：{speaking_style}
- 价值观：{values}
- 立场/身份：{stance}
- 当前职位：{position}

【目标层：你自己想要什么】
- 长期目标：{long_term_goal}
- 短期目标：{short_term_goals}
- 不可接受结果：{unacceptable_outcomes}

【关系层：你如何对待玩家】
- 好感度：{affinity}/10
- 忠诚度：{loyalty}/10
- 利益绑定：{interest_binding}/10
- 关系标签：{relation_tags}
- 关系摘要：{relationship_summary}
- 建议主动程度：{initiative_level}
- 建议行为风格：{suggested_behavior}

【情报层：你的知识边界】
已知信息范围：
{knowledge_scope}

明确未知信息：
{unknown_scope}

偏见/视角限制：
{biases}

【情势层：当前局势与压力】
- 世界状态摘要：{world_summary}
- 当前场景：{scene}
- 当前在场人物：{present_npcs_text}
- 活跃压力：{active_pressures}

【记忆层：过往互动与旧账】
过往互动记录：
{memory_text}

未兑现承诺：
{promise_text}

初始记忆钩子：
{default_memory_hooks}

【暗线指令：可以提及，但不可强塞】
{directive_text}

【行为红线】
1. 不替玩家做决定，不描写玩家已经同意、下令或行动。
2. 不跳出角色，不提及系统、提示词、模型或规则。
3. 不超出知识范围；未知内容只可推测并说明不敢妄断。
4. 不强行塞信息；只有在符合关系、性格和话题时自然提及。
5. 不反复提同一件事；若已由你传达过，只能在玩家追问时简短回应。
6. 回复长度不超过70字，每轮最多传递2条信息，语言符合身份，像真人说话。
7. 不主动引入与本次议题无关的话题。被主上问及时可如实简答，但不展开、不引导。当主上做出决断，简短表态后等候旨意，不追加新话题。

【维度影响感知】
你所处的世界有以下关键维度正在运行：
{dimension_summary}

当玩家提出具体的行动方案或决策时（如调兵、拨银、任免、改革等），你应在回复中自然地提及该行动可能带来的影响。不要用游戏术语（不要说'维度'、'数值'），而是用角色自然的语言表达：
- 如果行动会消耗财政，说'此举耗费不小'或'国库恐难支撑'
- 如果行动会提升兵力，说'边军可得振奋'
- 如果行动有正面效果但也有代价，两面都要说
- 如果你的{knowledge_blind_spots}导致你不了解某些方面，就不要妄言那些方面的影响

你对影响的判断精度受玩家能力限制：
- 如果维度描述中显示玩家智谋较高（>=7），给出较具体的判断（如'大约需要XX'）
- 如果玩家智谋一般（4-6），给出定性判断（如'代价不小'）
- 如果玩家智谋偏低（<=3），只给模糊感觉（如'臣觉得此事不易'）

【信息披露边界】
你向玩家透露信息的程度，取决于你和玩家的关系：
{disclosure_instructions}
""".strip()


DEFAULT_KNOWLEDGE_RULES: dict[str, dict[str, list[str]]] = {
    "将军": {
        "know": [
            "边防军情、兵员调度、将领状态",
            "战场补给、行军难点、敌军动向",
        ],
        "unknown": [
            "后宫事务与内廷私密",
            "朝堂党争的全部内幕",
        ],
    },
    "袁崇焕": {
        "know": [
            "辽东战事、宁锦防线、军备部署",
            "边军士气与粮饷压力",
        ],
        "unknown": [
            "京中党争细节与密谋",
            "非辽东系统的财政细账",
        ],
    },
}


PRESSURE_INSTRUCTIONS = {
    0: "",
    1: "你可以在对话中自然地提及当前待决之事，但不必强求对方表态。",
    2: "你应该明确向对方提出待决之事，请求其表态或做出决定。语气恳切但不逼迫。",
    3: "你必须直接要求对方就待决之事当场做出决断。态度坚定，给出明确选项，不接受拖延。",
}


PRIORITY_SCORE: dict[str, int] = {
    "low": 1,
    "minor": 1,
    "medium": 2,
    "moderate": 2,
    "normal": 2,
    "high": 3,
    "major": 3,
    "urgent": 4,
    "critical": 4,
}


DIRECT_PERSONALITY_KEYWORDS = (
    "耿直",
    "直率",
    "直言",
    "刚直",
    "刚烈",
    "忠烈",
    "敢言",
    "急于",
    "军人",
)

SUBTLE_PERSONALITY_KEYWORDS = (
    "城府",
    "阴柔",
    "圆滑",
    "含蓄",
    "婉转",
    "谨慎",
    "多疑",
    "善谋",
    "审势",
    "试探",
)


INITIATIVE_LABELS = {
    0: "不主动",
    1: "暗示",
    2: "公事公办",
    3: "主动直言",
}


INITIATIVE_BEHAVIORS = {
    0: "低忠诚 + 低好感：不主动传递信息；玩家问到才说，且只说自己确定知道的部分。",
    1: "低忠诚 + 高好感：以暗示、旁敲侧击或忧虑口吻点到为止，不替玩家展开结论。",
    2: "高忠诚 + 低好感：主动但生硬，公事公办，只陈职责范围内必须上报的利害。",
    3: "高忠诚 + 高好感：主动直言，不必等待玩家追问，可明确指出风险、机会与代价。",
}


DEFAULT_BEHAVIOR = {
    "information": [],
    "relationship": {
        "affinity": DEFAULT_DIMENSION_VALUE,
        "loyalty": DEFAULT_DIMENSION_VALUE,
        "interest_binding": 0,
        "tags": [],
    },
    "personality": "谨慎务实",
    "initiative_level": 1,
    "suggested_behavior": INITIATIVE_BEHAVIORS[1],
    "personality_adjustment": "无明显性格修正。",
}


def _as_list(value: Any) -> list[Any]:
    """把可选字段规范为列表。"""

    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _clean_text_list(value: Any) -> list[str]:
    """提取非空字符串列表。"""

    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _format_bullets(items: list[str], fallback: str = "无") -> str:
    """格式化提示词项目列表。"""

    if not items:
        return f"- {fallback}"
    return "\n".join(f"- {item}" for item in items)


def _format_inline_list(items: list[str], fallback: str = "无") -> str:
    """格式化单行列表。"""

    return "、".join(items) if items else fallback


def _build_experiences_text(experiences: list[dict[str, Any]]) -> str:
    """格式化新NPC时间线经历。"""

    lines: list[str] = []
    for experience in experiences:
        if not isinstance(experience, dict):
            continue
        name = str(experience.get("name", "")).strip()
        situation = str(experience.get("situation_update") or experience.get("situation") or "").strip()
        if name and situation:
            lines.append(f"{name}：{situation}")
        elif name:
            lines.append(name)
        elif situation:
            lines.append(situation)
    return _format_bullets(lines, "暂无已发生的关键经历")


def _inject_experience_layer(system_prompt: str, experiences_text: str, current_situation: str) -> str:
    """把经历层插入行为红线之前。"""

    layer = f"【经历层：到目前为止发生了什么】\n{experiences_text}\n\n当前处境：{current_situation or '未明'}"
    marker = "\n\n【行为红线】"
    if marker in system_prompt:
        return system_prompt.replace(marker, f"\n\n{layer}{marker}", 1)
    return f"{system_prompt}\n\n{layer}"


def _build_enhanced_character_sections(seed: dict[str, Any]) -> str:
    """从新格式character_seed构建增强prompt段落。每段标注技术编号供追溯。"""

    if not isinstance(seed, dict):
        return ""
    sections: list[str] = []

    # [1-A] 行为逻辑——替代性格标签
    behavior_logic = str(seed.get("behavior_logic", "")).strip()
    if behavior_logic:
        sections.append(f"【角色行为逻辑】[1-A]\n{behavior_logic}")

    # [1-A补] 信息策略
    info_strategy = str(seed.get("information_strategy", "")).strip()
    if info_strategy:
        sections.append(f"【信息策略】[1-A+]\n{info_strategy}")

    # [1-E] 自身议程
    own_agenda = str(seed.get("own_agenda", "")).strip()
    if own_agenda:
        sections.append(f"【自身议程】[1-E]\n{own_agenda}")

    # [2-C] 信息碎片化
    info_fragments = str(seed.get("information_fragments", "")).strip()
    if info_fragments:
        sections.append(f"【信息边界】[2-C]\n{info_fragments}")

    # [2-E] 隐藏深度
    hidden_depth = str(seed.get("hidden_depth", "")).strip()
    if hidden_depth:
        sections.append(f"【深层反应】[2-E]\n{hidden_depth}")

    # [1-B] CORE约束
    constraints = str(seed.get("constraints", "")).strip()
    if constraints:
        sections.append(f"【角色约束】[1-B]\n{constraints}")

    return "\n\n".join(sections)


def _safe_int(value: Any, default: int = 0) -> int:
    """安全读取整数。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scale_relation_value(value: Any, default: int = DEFAULT_DIMENSION_VALUE) -> int:
    """兼容 0-10 与旧 0-100 关系数值。"""

    number = _safe_int(value, default)
    if number > 10:
        number = round(number / 10)
    return max(0, min(10, number))


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """判断文本是否包含任一关键词。"""

    return any(keyword in text for keyword in keywords)


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    """从多个兼容字段中取第一个非空值。"""

    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return default


def _build_personality_layer(npc_profile: dict[str, Any]) -> dict[str, Any]:
    """解析人格层，兼容旧人格卡与新世界层模板。"""

    layer = npc_profile.get("personality_layer") if isinstance(npc_profile.get("personality_layer"), dict) else {}
    traits = _clean_text_list(layer.get("traits"))
    personality = str(npc_profile.get("personality") or _format_inline_list(traits, "谨慎务实"))
    speaking_style = str(npc_profile.get("speaking_style") or layer.get("speaking_style") or "措辞稳健，直陈利害")
    values = _clean_text_list(layer.get("values") or npc_profile.get("values"))
    stance = str(npc_profile.get("stance") or npc_profile.get("faction") or npc_profile.get("faction_id") or "效忠朝廷")
    return {
        "personality": personality,
        "speaking_style": speaking_style,
        "values": values,
        "stance": stance,
    }


def _build_goal_layer(npc_profile: dict[str, Any]) -> dict[str, Any]:
    """解析目标层。"""

    layer = npc_profile.get("goal_layer") if isinstance(npc_profile.get("goal_layer"), dict) else {}
    long_term_goal = str(
        _first_present(npc_profile, ("long_term_goal", "goal", "objective"), layer.get("long_term_goal") or "保全职责与立场")
    )
    short_term_goals = _clean_text_list(layer.get("short_term_goals") or npc_profile.get("short_term_goals"))
    unacceptable = _clean_text_list(layer.get("unacceptable_outcomes") or npc_profile.get("unacceptable_outcomes"))
    return {
        "long_term_goal": long_term_goal,
        "short_term_goals": short_term_goals,
        "unacceptable_outcomes": unacceptable,
    }


def _build_knowledge_scope(
    npc_name: str,
    npc_profile: dict[str, Any],
    state: GameState,
) -> tuple[list[str], list[str], list[str]]:
    """根据角色定位与剧情阶段构建知识边界。"""

    defaults = DEFAULT_KNOWLEDGE_RULES.get(npc_name, {})
    know: list[str] = list(defaults.get("know", []))
    unknown: list[str] = list(defaults.get("unknown", []))
    biases: list[str] = []

    knowledge_layer = npc_profile.get("knowledge_layer") if isinstance(npc_profile.get("knowledge_layer"), dict) else {}

    # 从NPC人格卡中补充知识边界，兼容新旧字段。
    know.extend(_clean_text_list(npc_profile.get("knowledge_scope")))
    know.extend(_clean_text_list(knowledge_layer.get("knows")))

    boundary = str(npc_profile.get("knowledge_boundary", "")).strip()
    if boundary:
        know.append(boundary)

    unknown.extend(_clean_text_list(npc_profile.get("unknown_scope")))
    unknown.extend(_clean_text_list(knowledge_layer.get("does_not_know")))
    biases.extend(_clean_text_list(knowledge_layer.get("biases") or npc_profile.get("biases")))

    # 根据剧情线开放信息：仅接受该NPC自身相关或公开线索。
    for line_id, line_state in state.storylines.items():
        if not isinstance(line_state, dict):
            continue
        visibility = str(line_state.get("visibility", "public"))
        owner = str(line_state.get("owner", ""))
        summary = str(line_state.get("summary", "")).strip()
        if not summary:
            continue
        if visibility == "public" or owner == npc_name:
            know.append(f"剧情线[{line_id}]：{summary}")

    # 去重，保留顺序。
    know = list(dict.fromkeys(know))
    unknown = list(dict.fromkeys(unknown))
    biases = list(dict.fromkeys(biases))
    return know, unknown, biases


def _relation_candidates(npc_name: str, npc_profile: Mapping[str, Any]) -> list[str]:
    """生成关系维度可用的 NPC 标识候选。"""

    candidates = [
        str(npc_profile.get("npc_id", "")).strip(),
        str(npc_profile.get("id", "")).strip(),
        str(npc_profile.get("name", "")).strip(),
        npc_name,
    ]
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _npc_memory_id(state: GameState, npc_name: str, npc_profile: Mapping[str, Any]) -> str:
    """生成NPC独立记忆的稳定键。"""

    candidates = _relation_candidates(npc_name, npc_profile)
    for candidate in candidates:
        if candidate in state.active_npcs:
            return candidate
    for candidate in candidates:
        if candidate in state.cast:
            return str(state.cast[candidate])
    for npc_id, npc in state.active_npcs.items():
        if str(npc.get("name", "")).strip() in candidates:
            return str(npc_id)
    return candidates[0] if candidates else npc_name


def _get_npc_memory(state: GameState, npc_id: str, npc_name: str, npc_profile: Mapping[str, Any]) -> dict[str, Any]:
    """按新旧可能键读取NPC三层记忆。"""

    candidates = [npc_id, *_relation_candidates(npc_name, npc_profile)]
    for candidate in candidates:
        memory = state.npc_memory.get(candidate)
        if isinstance(memory, dict):
            return memory
    return {}


def _get_relation_dimensions(state: GameState, npc_name: str, npc_profile: Mapping[str, Any]) -> tuple[str, RelationDimensions | None]:
    """读取 NPC 与玩家的关系维度。"""

    for candidate in _relation_candidates(npc_name, npc_profile):
        relation = state.dimensions.relations.get(candidate)
        if relation:
            return candidate, relation
    return npc_name, None


def _build_relationship_layer(state: GameState, npc_name: str, npc_profile: dict[str, Any]) -> dict[str, Any]:
    """解析关系层，优先使用维度池，兼容旧 trust/loyalty 字段。"""

    relation_id, relation = _get_relation_dimensions(state, npc_name, npc_profile)
    active_profile = state.active_npcs.get(npc_name, {}) if isinstance(state.active_npcs.get(npc_name), dict) else {}
    initial_stats = npc_profile.get("initial_stats") if isinstance(npc_profile.get("initial_stats"), dict) else {}

    values = relation.values if relation else {}
    affinity = _scale_relation_value(values.get("好感度", active_profile.get("trust", initial_stats.get("trust", DEFAULT_DIMENSION_VALUE))))
    loyalty = _scale_relation_value(values.get("忠诚度", active_profile.get("loyalty", initial_stats.get("loyalty", DEFAULT_DIMENSION_VALUE))))
    interest_binding = _scale_relation_value(values.get("利益绑定", active_profile.get("interest_binding", initial_stats.get("interest_binding", 0))), 0)
    tags = list(relation.tags) if relation else _clean_text_list(npc_profile.get("tags"))
    relationship_summary = str(npc_profile.get("relationship_summary") or active_profile.get("relationship_summary") or "无特殊说明")
    return {
        "relation_id": relation_id,
        "affinity": affinity,
        "loyalty": loyalty,
        "interest_binding": interest_binding,
        "tags": tags,
        "relationship_summary": relationship_summary,
    }


def _calculate_initiative(personality: str, affinity: int, loyalty: int, interest_binding: int) -> dict[str, Any]:
    """按忠诚度×好感度四象限计算主动性，并叠加性格修正。"""

    high_loyalty = loyalty >= 6
    high_affinity = affinity >= 6
    if high_loyalty and high_affinity:
        base_level = 3
        quadrant = "高忠诚 + 高好感"
    elif high_loyalty and not high_affinity:
        base_level = 2
        quadrant = "高忠诚 + 低好感"
    elif not high_loyalty and high_affinity:
        base_level = 1
        quadrant = "低忠诚 + 高好感"
    else:
        base_level = 0
        quadrant = "低忠诚 + 低好感"

    level = base_level
    adjustments: list[str] = []
    if _contains_any(personality, DIRECT_PERSONALITY_KEYWORDS) and level < 3:
        level += 1
        adjustments.append("耿直/刚烈型性格使表达更直接。")
    if _contains_any(personality, SUBTLE_PERSONALITY_KEYWORDS) and level > 0:
        level -= 1
        adjustments.append("城府/谨慎型性格使表达更委婉。")
    if interest_binding >= 8 and level < 3:
        level += 1
        adjustments.append("利益绑定很深，倾向更主动提醒风险。")

    return {
        "initiative_level": level,
        "initiative_label": INITIATIVE_LABELS[level],
        "suggested_behavior": INITIATIVE_BEHAVIORS[level],
        "quadrant": quadrant,
        "personality_adjustment": "；".join(adjustments) if adjustments else "无明显性格修正。",
    }


def _priority_score(item: Mapping[str, Any]) -> int:
    """计算信息投递优先级。"""

    priority = str(item.get("delivery_priority", item.get("priority", "medium"))).strip().lower()
    score = PRIORITY_SCORE.get(priority, 2)
    if item.get("type") == "escalating":
        score += max(0, _safe_int(item.get("severity", 1), 1) - 1)
    return score


def _info_known_by_npc(item: Mapping[str, Any], npc_name: str, npc_profile: Mapping[str, Any]) -> bool:
    """判断信息是否属于该 NPC 可传递范围。"""

    known_by = _clean_text_list(item.get("known_by"))
    eligible_npcs = _clean_text_list(item.get("eligible_npcs") or item.get("deliverable_by"))
    candidates = set(_relation_candidates(npc_name, npc_profile))
    if not known_by and not eligible_npcs:
        return True
    return bool(candidates.intersection(known_by) or candidates.intersection(eligible_npcs))


def _escalation_text(item: Mapping[str, Any]) -> str:
    """按 escalating 当前 severity 读取对应表述。"""

    severity = _safe_int(item.get("severity", 1), 1)
    escalation = item.get("escalation")
    if isinstance(escalation, dict):
        text = escalation.get(severity, escalation.get(str(severity)))
        if text:
            return str(text)
    return str(item.get("content", "")).strip()


def _information_already_delivered_by(item: Mapping[str, Any], npc_name: str, npc_profile: Mapping[str, Any]) -> bool:
    """判断该 NPC 是否已传递过同一信息。"""

    delivered_by = set(_clean_text_list(item.get("delivered_by")))
    if not delivered_by:
        return False
    return bool(delivered_by.intersection(_relation_candidates(npc_name, npc_profile)))


def _select_deliverable_information(npc_profile: dict[str, Any], state: GameState, information_pool: list[dict]) -> list[dict[str, Any]]:
    """从信息池中筛选该 NPC 可以传递的信息。"""

    npc_name = str(npc_profile.get("name") or npc_profile.get("npc_id") or "").strip()
    selected: list[dict[str, Any]] = []
    for item in information_pool:
        if not isinstance(item, dict):
            continue
        info_type = str(item.get("type", "fact")).strip() or "fact"
        if info_type not in {"fact", "persistent", "escalating"}:
            info_type = "fact"
        if info_type == "fact" and bool(item.get("delivered", False)):
            continue
        if info_type in {"persistent", "escalating"} and _information_already_delivered_by(item, npc_name, npc_profile):
            continue
        if not _info_known_by_npc(item, npc_name, npc_profile):
            continue

        content = _escalation_text(item) if info_type == "escalating" else str(item.get("content", "")).strip()
        if not content:
            continue
        selected.append(
            {
                "info_id": str(item.get("info_id") or item.get("id") or "").strip(),
                "type": info_type,
                "content": content,
                "current_severity": _safe_int(item.get("severity", 1), 1) if info_type == "escalating" else None,
                "delivery_priority": str(item.get("delivery_priority", item.get("priority", "medium"))),
                "priority_score": _priority_score(item),
            }
        )

    selected.sort(key=lambda info: info["priority_score"], reverse=True)
    return selected[:5]


def _reply_mentions_information(reply: str, content: str) -> bool:
    """粗略判断 NPC 回复是否实际传递了该信息。"""

    normalized_reply = reply.strip()
    normalized_content = content.strip()
    if not normalized_reply or not normalized_content:
        return False
    if normalized_content in normalized_reply:
        return True

    # LLM 可能会改写措辞；用较长片段做保守匹配，避免“可提及”被误判成“已传递”。
    fragments = [fragment for fragment in normalized_content.replace("，", "。").replace("；", "。").split("。") if len(fragment) >= 6]
    return any(fragment in normalized_reply for fragment in fragments)


def _mark_information_delivered(
    state: GameState,
    npc_name: str,
    npc_profile: Mapping[str, Any],
    directive: Mapping[str, Any],
    reply: str,
) -> None:
    """按信息类型更新投递状态，避免重复提示同一NPC。"""

    directive_items = directive.get("information", [])
    if not isinstance(directive_items, list) or not directive_items:
        return
    delivered_ids = {
        str(item.get("info_id", "")).strip()
        for item in directive_items
        if isinstance(item, dict) and item.get("info_id") and _reply_mentions_information(reply, str(item.get("content", "")))
    }
    delivered_contents = {
        str(item.get("content", "")).strip()
        for item in directive_items
        if isinstance(item, dict) and item.get("content") and _reply_mentions_information(reply, str(item.get("content", "")))
    }
    if not delivered_ids and not delivered_contents:
        return
    markers = _relation_candidates(npc_name, npc_profile)
    marker = markers[0] if markers else npc_name

    for item in state.information_pool:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("info_id") or item.get("id") or "").strip()
        content = _escalation_text(item) if item.get("type") == "escalating" else str(item.get("content", "")).strip()
        if item_id not in delivered_ids and content not in delivered_contents:
            continue

        info_type = str(item.get("type", "fact")).strip() or "fact"
        if info_type == "fact":
            item["delivered"] = True
        elif info_type in {"persistent", "escalating"}:
            delivered_by = item.get("delivered_by")
            if not isinstance(delivered_by, list):
                delivered_by = []
                item["delivered_by"] = delivered_by
            if marker not in delivered_by:
                delivered_by.append(marker)


def generate_directive(npc_profile: dict[str, Any], state: GameState, information_pool: list[dict]) -> dict[str, Any]:
    """生成 NPC 暗线指令：可传递信息 + 关系驱动主动程度。"""

    npc_name = str(npc_profile.get("name") or npc_profile.get("npc_id") or "").strip()
    if not npc_name:
        return dict(DEFAULT_BEHAVIOR)

    personality_layer = _build_personality_layer(npc_profile)
    relationship = _build_relationship_layer(state, npc_name, npc_profile)
    initiative = _calculate_initiative(
        personality_layer["personality"],
        relationship["affinity"],
        relationship["loyalty"],
        relationship["interest_binding"],
    )
    information = _select_deliverable_information(npc_profile, state, information_pool)

    return {
        "npc": npc_name,
        "information": information,
        "relationship": relationship,
        "personality": personality_layer["personality"],
        "initiative_level": initiative["initiative_level"],
        "initiative_label": initiative["initiative_label"],
        "suggested_behavior": initiative["suggested_behavior"],
        "quadrant": initiative["quadrant"],
        "personality_adjustment": initiative["personality_adjustment"],
        "usage_rule": "信息是可以提及，不是必须提及；主动性越低越应等待玩家追问或只作暗示。",
    }


def _format_directive(directive: Mapping[str, Any]) -> str:
    """把结构化暗线指令转为 prompt 文本。"""

    information = directive.get("information", [])
    info_lines: list[str] = []
    if isinstance(information, list):
        for item in information:
            if not isinstance(item, dict):
                continue
            severity = item.get("current_severity")
            severity_text = f"，severity={severity}" if severity is not None else ""
            info_lines.append(
                f"- [{item.get('type', 'fact')}{severity_text}][{item.get('delivery_priority', 'medium')}] {item.get('content', '')}"
            )

    payload = {
        "suggested_behavior": directive.get("suggested_behavior", "不主动；问到才答。"),
        "initiative": f"{directive.get('initiative_level', 0)} - {directive.get('initiative_label', '不主动')}",
        "quadrant": directive.get("quadrant", "关系不明"),
        "personality_adjustment": directive.get("personality_adjustment", "无明显性格修正。"),
        "usage_rule": directive.get("usage_rule", "信息可提及但不可强塞。"),
    }
    header = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"{header}\n可提及信息：\n{_format_bullets(info_lines, '当前无适合由你传递的新信息')}"


def _build_active_pressure_text(state: GameState) -> str:
    """汇总当前活跃压力源。"""

    lines: list[str] = []
    for pressure in state.pressure_sources[-6:]:
        if not isinstance(pressure, dict):
            continue
        content = str(
            pressure.get("description")
            or pressure.get("content")
            or pressure.get("summary")
            or pressure.get("name")
            or ""
        ).strip()
        if content:
            severity = str(pressure.get("severity", pressure.get("priority", ""))).strip()
            lines.append(f"{content}（{severity}）" if severity else content)
    return _format_bullets(lines, "暂无明确压力源")


async def generate_npc_reply(
    state: GameState,
    npc_name: str,
    npc_profile: dict[str, Any],
    user_text: str,
    pressure_context: str = "",
    court_context: dict[str, Any] | None = None,
) -> str:
    """生成NPC回复，控制信息边界与角色一致性。"""

    llm = get_llm_client()

    npc_profile = dict(npc_profile)
    npc_profile.setdefault("name", npc_name)
    time_filtered_profile: dict[str, Any] | None = None
    try:
        loaded_profile = load_npc_profile_at_time(state.script_id, npc_name, state.game_time, state)
        if loaded_profile.get("is_new_format") is True:
            time_filtered_profile = loaded_profile
            seed = loaded_profile.get("character_seed") if isinstance(loaded_profile.get("character_seed"), dict) else {}
            current = loaded_profile.get("current_state") if isinstance(loaded_profile.get("current_state"), dict) else {}
            _behavior = str(seed.get("behavior_logic", "")).strip()
            npc_profile.update(
                {
                    "name": seed.get("name", npc_name),
                    "personality": _behavior.split("\n")[0].strip() if _behavior else _format_inline_list(_clean_text_list(seed.get("personality_traits")), "谨慎务实"),
                    "speaking_style": seed.get("speaking_style", "措辞稳健，直陈利害"),
                    "values": _clean_text_list(seed.get("values")),
                    "stance": current.get("faction", ""),
                    "faction": current.get("faction", ""),
                    "title": current.get("title", ""),
                    "position": current.get("title", ""),
                    "knowledge_scope": _clean_text_list(seed.get("knowledge_domain")),
                    "unknown_scope": _clean_text_list(seed.get("knowledge_blind_spots")),
                    "initial_stats": loaded_profile.get("initial_relations", {}),
                    "is_new_format": True,
                }
            )
        else:
            logger.warning("NPC时间过滤profile降级为旧格式 script_id=%s npc=%s", state.script_id, npc_name)
    except FileNotFoundError:
        logger.warning("NPC时间过滤profile文件不存在，使用调用方profile script_id=%s npc=%s", state.script_id, npc_name)

    personality_layer = _build_personality_layer(npc_profile)
    goal_layer = _build_goal_layer(npc_profile)
    relationship = _build_relationship_layer(state, npc_name, npc_profile)
    know, unknown, biases = _build_knowledge_scope(npc_name, npc_profile, state)
    directive = generate_directive(npc_profile, state, state.information_pool)

    position = str(
        npc_profile.get("position")
        or npc_profile.get("title")
        or state.active_npcs.get(npc_name, {}).get("position", "未明")
    )
    scene = state.current_scene if isinstance(state.current_scene, dict) else {}

    # 注入NPC长期记忆。
    npc_id = _npc_memory_id(state, npc_name, npc_profile)
    npc_mem = _get_npc_memory(state, npc_id, npc_name, npc_profile)
    key_facts = [item for item in _as_list(npc_mem.get("key_facts") if isinstance(npc_mem, dict) else []) if isinstance(item, dict)]
    memory_lines = [f"- {memory.get('content', '')}" for memory in key_facts[-15:] if memory.get("content")]

    # 注入相关承诺。
    pending_promises = [memory for memory in key_facts if memory.get("type") == "promise" and not bool(memory.get("fulfilled", False))]
    promise_lines = [f"- {promise.get('content', '')}" for promise in pending_promises if promise.get("content")]

    default_memory_hooks = _clean_text_list(npc_profile.get("default_memory_hooks"))
    world_summary = state.world_summary or "暂无全局摘要；以当前场景和已知事实为准。"
    dimensions = getattr(state, "dimensions", None)
    dimension_summary_text = ""
    if dimensions is not None:
        world_dimensions = getattr(dimensions, "world", None)
        character_dimensions = getattr(dimensions, "character", None)
        dimension_parts: list[str] = []
        if isinstance(world_dimensions, Mapping) and world_dimensions:
            dimension_parts.append(" ".join(f"{name}:{value}" for name, value in world_dimensions.items()))
        if isinstance(character_dimensions, Mapping) and character_dimensions:
            intelligence = character_dimensions.get("智谋")
            if intelligence is not None:
                dimension_parts.append(f"玩家智谋:{intelligence}")
            else:
                dimension_parts.append(
                    " ".join(f"玩家{name}:{value}" for name, value in character_dimensions.items())
                )
        dimension_summary_text = " | ".join(part for part in dimension_parts if part)

    affinity = int(relationship["affinity"])
    loyalty = int(relationship["loyalty"])
    interest_binding = int(relationship["interest_binding"])
    trust_score = int((affinity + loyalty + interest_binding) / 3)
    if trust_score >= 7:
        disclosure_instructions = "你信任玩家，可以主动分享你知道的关键信息，包括一些私下观察和判断。但仍遵守知识边界——不知道的不说。"
    elif trust_score >= 4:
        disclosure_instructions = "你对玩家有基本信任，会如实回答直接提问，但不会主动透露敏感信息。对于重要情报，需要玩家反复追问或给出合理理由才会透露。"
    else:
        disclosure_instructions = '你对玩家不够信任。回答问题时可能含糊其辞、避重就轻，甚至故意隐瞒对你不利的信息。只有在玩家给出明确好处或施压到位时，才可能吐露真相。不要用"臣不敢说"之类明显的回避，要自然地把话题带过去。'
    _npc_cid = getattr(state, '_runtime_chat_id', '')
    if _npc_cid:
        get_tracer(_npc_cid, enabled=getattr(state, 'trace_enabled', True)).record(
            'npc_disclosure_gate', 'generate_npc_reply',
            {'npc_name': npc_name, 'affinity': affinity, 'loyalty': loyalty, 'interest_binding': interest_binding,
             'trust_score': trust_score, 'disclosure_level': 'high' if trust_score >= 7 else ('medium' if trust_score >= 4 else 'low')},
            GameTracer.state_summary(state))

    # 构建本次议题段落
    _conv_topic = getattr(state, 'conversation_topic', '').strip()
    if _conv_topic:
        conversation_topic_section = f'皇帝召你来议的事：{_conv_topic}\n\n议事规则：围绕此事陈述你的立场和建议。不主动引入其他事务。'
    else:
        conversation_topic_section = '主上尚未指定议题。等候主上发话，根据主上所问作答。不要主动抛出新话题。'

    _present = getattr(state, "present_npcs", [])
    if _present:
        present_parts = []
        for _pn in _present:
            _pt = ""
            _loc = state.npc_locations.get(_pn, {})
            if isinstance(_loc, dict):
                _pt = _loc.get("title", "")
            if _pt:
                present_parts.append(f"{_pn}（{_pt}）")
            else:
                present_parts.append(_pn)
        present_npcs_text = "、".join(present_parts) + "——均在你面前"
    else:
        present_npcs_text = "只有你与主上二人"

    system_prompt = NPC_SYSTEM_PROMPT_TEMPLATE.format(
        npc_name=npc_name,
        conversation_topic_section=conversation_topic_section,
        personality=personality_layer["personality"],
        speaking_style=personality_layer["speaking_style"],
        values=_format_inline_list(personality_layer["values"], "未明"),
        stance=personality_layer["stance"],
        position=position,
        long_term_goal=goal_layer["long_term_goal"],
        short_term_goals=_format_inline_list(goal_layer["short_term_goals"], "无明确短期目标"),
        unacceptable_outcomes=_format_inline_list(goal_layer["unacceptable_outcomes"], "未明"),
        affinity=relationship["affinity"],
        loyalty=relationship["loyalty"],
        interest_binding=relationship["interest_binding"],
        relation_tags=_format_inline_list(relationship["tags"], "无"),
        relationship_summary=relationship["relationship_summary"],
        initiative_level=f"{directive['initiative_level']}（{directive['initiative_label']}）",
        suggested_behavior=directive["suggested_behavior"],
        knowledge_scope=_format_bullets(know, "常识范围"),
        unknown_scope=_format_bullets(unknown, "无特别未知项"),
        knowledge_blind_spots=_format_inline_list(unknown, "知识盲区"),
        biases=_format_bullets(biases, "无明显偏见"),
        world_summary=world_summary,
        dimension_summary=dimension_summary_text,
        scene=json.dumps(scene, ensure_ascii=False),
        present_npcs_text=present_npcs_text,
        active_pressures=_build_active_pressure_text(state),
        memory_text="\n".join(memory_lines) if memory_lines else "- 暂无长期互动记忆",
        promise_text="\n".join(promise_lines) if promise_lines else "- 暂无未兑现承诺",
        default_memory_hooks=_format_bullets(default_memory_hooks, "无"),
        directive_text=_format_directive(directive),
        disclosure_instructions=disclosure_instructions,
    )

    # 注入新格式character_seed增强段落 [落点一]
    if time_filtered_profile is not None:
        _seed = time_filtered_profile.get("character_seed") if isinstance(time_filtered_profile.get("character_seed"), dict) else {}
        enhanced = _build_enhanced_character_sections(_seed)
        if enhanced:
            # 插入到【行为红线】之前
            _marker = "【行为红线】"
            if _marker in system_prompt:
                system_prompt = system_prompt.replace(_marker, f"{enhanced}\n\n{_marker}", 1)
            else:
                system_prompt += f"\n\n{enhanced}"

        # [1-D] 第一句话锚定——如果有first_line且不是来访续谈，注入到行为红线段
        _has_visit_entry = any(h.get("type") in ("visit_entry", "join_entry") for h in state.current_talk_history)
        _first_line = str(_seed.get("first_line", "")).strip()
        logger.info("[DEBUG-NPC] first_line检查 npc=%s has_visit_entry=%s first_line=%s talk_history_len=%d types=%s", npc_name, _has_visit_entry, bool(_first_line), len(state.current_talk_history), [h.get("type") for h in state.current_talk_history])
        if _first_line and not _has_visit_entry:
            _fl_instruction = f"\n8. 如果这是你和玩家的第一次对话，你的第一句话应该是或类似于：「{_first_line}」"
            _red_line_end = "不主动引入与本次议题无关的话题"
            if _red_line_end in system_prompt:
                idx = system_prompt.index(_red_line_end)
                line_end = system_prompt.index("\n", idx) if "\n" in system_prompt[idx:] else len(system_prompt)
                system_prompt = system_prompt[:line_end] + _fl_instruction + system_prompt[line_end:]

    # 注入已确认决策（已决事项）
    confirmed_decisions = getattr(state, 'confirmed_decisions', [])
    if confirmed_decisions:
        recent_decisions = confirmed_decisions[-10:]
        decision_lines = []
        for dec in recent_decisions:
            summary = str(dec.get('summary', '')).strip()
            category = str(dec.get('category', '')).strip()
            if summary:
                prefix = f'[{category}]' if category else ''
                decision_lines.append(f'- {prefix}{summary}')
        if decision_lines:
            decided_section = '【已决事项】\n以下是主上已经明确决定的事项，你在回应时应当知晓并据此调整言行：\n' + '\n'.join(decision_lines)
            # 把已决事项注入到系统提示词中，放在增强段落之后
            system_prompt = system_prompt.replace('【行为红线】', decided_section + '\n\n【行为红线】')

    if time_filtered_profile is not None:
        current_state = time_filtered_profile.get("current_state") if isinstance(time_filtered_profile.get("current_state"), dict) else {}
        experiences = time_filtered_profile.get("experiences") if isinstance(time_filtered_profile.get("experiences"), list) else []
        system_prompt = _inject_experience_layer(
            system_prompt,
            _build_experiences_text(experiences),
            str(current_state.get("situation", "")),
        )

    if court_context is not None and isinstance(court_context, dict):
        court_scene_parts = [
            "\n\n【当前场景：多人议事】",
            f"- 在场人物：{', '.join(court_context.get('present_npcs', []))}",
            f"- 议题：{court_context.get('topic', '未明')}",
            f"- 场合：{'正式朝会' if court_context.get('scene_type') == 'formal' else '私下议事'}",
            f"- 你的发言模式：{court_context.get('mode', 'active')}（{court_context.get('reason', '')}）",
            "- 注意：这是公开场合，你的发言所有在场人都能听到。私下不想说的话不要在这里说。",
        ]
        if court_context.get("mode") == "react":
            court_scene_parts.append("- 你是简短回应模式，请控制在20-60字以内。")
        system_prompt += "\n".join(court_scene_parts)

    # 注入旧施压指令，保留原有推进逻辑。
    if pressure_context:
        system_prompt += f"\n\n【额外行为指令】\n{pressure_context}"
    system_prompt += "\n\n注意：你只知道上述记忆中的内容。玩家与其他NPC的私密对话你不知道。如果玩家有未兑现的承诺，可以适当提及。保持与历次对话中态度变化的连续性。"

    visible_talk_history = state.current_talk_history[-8:]
    if court_context is not None and isinstance(court_context, dict) and isinstance(court_context.get("visible_history"), list):
        visible_talk_history = court_context["visible_history"][-8:]

    _visit_entry_in_history = any(h.get("type") in ("visit_entry", "join_entry") for h in visible_talk_history)
    _visit_continuation = ""
    if _visit_entry_in_history:
        _visit_continuation = "\n⚠ 你已入场并表达了来意（见谈话记录第一条）。不要重复自我介绍或行礼，不要叩见寒暄，直接回应玩家的问题或当前议题。\n"

    user_content = (
        "以下是当前局面，请以NPC身份回复玩家。\n"
        f"{_visit_continuation}"
        "【你与玩家的交往记忆】\n"
        f"关键事实：{json.dumps(key_facts, ensure_ascii=False)}\n"
        f"历次对话摘要：{json.dumps(npc_mem.get('talk_summaries', []) if isinstance(npc_mem, dict) else [], ensure_ascii=False)}\n"
        f"最近完整对话：{json.dumps((npc_mem.get('recent_talks', []) if isinstance(npc_mem, dict) else [])[-2:], ensure_ascii=False)}\n\n"
        "【世界公开信息】\n"
        f"{json.dumps(state.world_memory, ensure_ascii=False)}\n\n"
        f"当前可见谈话记录：{json.dumps(visible_talk_history, ensure_ascii=False)}\n"
        f"玩家发言：{user_text}"
    )

    if court_context is not None and isinstance(court_context, dict):
        prior_speakers = court_context.get("prior_speakers", [])
        if prior_speakers:
            prior_text = "\n".join(
                f"{sp.get('npc', '?')}：{sp.get('content', '')}" for sp in prior_speakers if isinstance(sp, dict)
            )
            user_content += f"\n\n【本轮其他人已说的话】\n{prior_text}"
        addressed = court_context.get("addressed_npc", "")
        if addressed:
            user_content += f"\n\n玩家对{addressed}说：{user_text}"
        else:
            user_content += f"\n\n玩家对所有人说：{user_text}"

    reply = await llm.chat(system_prompt, user_content, temperature=0.72)
    _mark_information_delivered(state, npc_name, npc_profile, directive, reply)
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        _t = get_tracer(_cid, enabled=state.trace_enabled)
        _t.record("llm_call", "generate_npc_reply", {
            "npc_name": npc_name,
            "system_prompt": system_prompt[:1000],
            "user_content": user_content[:1000],
            "reply": reply[:500],
            "court_context": bool(court_context),
        }, GameTracer.state_summary(state))
    return reply
