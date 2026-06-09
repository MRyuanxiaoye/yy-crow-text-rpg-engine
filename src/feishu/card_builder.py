"""飞书卡片构建器。"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _base_card(title: str, template: str, elements: list[dict[str, Any]]) -> dict[str, Any]:
    """生成统一飞书卡片骨架。"""

    return {
        "config": {
            "wide_screen_mode": True,
            "enable_forward": True,
        },
        "header": {
            "template": template,
            "title": {
                "tag": "plain_text",
                "content": title,
            },
        },
        "elements": elements,
    }


def _to_dict(value: Any) -> dict[str, Any]:
    """兼容dict和dataclass输入。"""

    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    return {}


def _safe_int(value: Any, default: int = 0) -> int:
    """安全整数转换。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_modifier(value: Any) -> str:
    """格式化判定修正值。"""

    modifier = _safe_int(value)
    return f"+{modifier}" if modifier > 0 else str(modifier)


def _format_success_rate(value: Any) -> str:
    """格式化预估成功率，兼容0-1和0-100输入。"""

    try:
        rate = float(value)
    except (TypeError, ValueError):
        return "未知"
    if rate <= 1:
        rate *= 100
    return f"{rate:.0f}%"


def _button(label: str, action: str, button_type: str = "default", **value: Any) -> dict[str, Any]:
    """生成飞书按钮。"""

    payload = {"action": action}
    payload.update(value)
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": button_type,
        "value": payload,
    }


def _compact_payload(payload: dict[str, Any]) -> str:
    """将结构化按钮值压缩为JSON字符串，便于回调恢复。"""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _option_label(option: dict[str, Any], fallback: str) -> str:
    """从选项中提取按钮文案。"""

    label = str(option.get("label") or option.get("title") or option.get("description") or fallback).strip()
    return label[:24] + "…" if len(label) > 25 else label


def build_advance_button_card() -> dict[str, Any]:
    """构建单独推进时间按钮卡片。"""

    return _base_card(
        "时间推进",
        "orange",
        [
            {"tag": "markdown", "content": "若当前部署已经完毕，可以主动推进时间，结算压力源、到期行动和突发事件。"},
            {"tag": "action", "actions": [_button("推进时间", "advance_time", "primary")]},
        ],
    )


def build_narration_card(
    title: str,
    content: str,
    choices: list[str] | None = None,
) -> dict[str, Any]:
    """构建叙事卡片，包含标题、正文和可选选项列表。"""

    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": content,
        }
    ]

    if choices:
        choice_lines = "\n".join([f"▸ {choice}" for choice in choices])
        elements.append(
            {
                "tag": "hr",
            }
        )
        elements.append(
            {
                "tag": "markdown",
                "content": f"**可选行动**\n{choice_lines}",
            }
        )

    return {
        "config": {
            "wide_screen_mode": True,
            "enable_forward": True,
        },
        "header": {
            "template": "blue",
            "title": {
                "tag": "plain_text",
                "content": title,
            },
        },
        "elements": elements,
    }


def build_prologue_card(
    narration: str,
    dimensions: dict,
    pressure_warnings: list[str],
    main_quest: str,
    turn_agenda: list[dict] = None,
) -> dict[str, Any]:
    """构建开局序幕卡片：叙事 + 压力 + 议程 + 主线（国势只在行动面板展示）。"""

    warning_lines = [f"⚠️ {str(item).strip()}" for item in pressure_warnings if str(item).strip()]
    pressure_text = "\n".join(warning_lines) if warning_lines else ""

    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": narration},
    ]
    if pressure_text:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": pressure_text})
    if turn_agenda:
        agenda_lines = []
        for item in turn_agenda:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "")
            if title:
                agenda_lines.append(f"• {title}")
        if agenda_lines:
            agenda_text = "\n".join(agenda_lines)
            elements.append({"tag": "hr"})
            elements.append({"tag": "markdown", "content": f"**━━━ 本回合议程 ━━━**\n{agenda_text}"})
    elements.append({"tag": "markdown", "content": f"📋 {main_quest or '暂无明确目标'}"})
    return _base_card("序幕", "blue", elements)


def build_opening_card(
    icebreaker: str,
    action_modes: list[dict[str, Any]],
    player_address: str = "你",
) -> dict[str, Any]:
    """构建GM主持开局卡片：破冰语 + 行为模式按钮。"""

    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": icebreaker},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**{player_address}打算先做什么？**"},
    ]

    buttons: list[dict[str, Any]] = []
    for mode in action_modes:
        if not isinstance(mode, dict):
            continue
        icon = str(mode.get("icon", "")).strip()
        label = str(mode.get("label", "选项")).strip()
        btn_text = f"{icon} {label}" if icon else label
        description = str(mode.get("description", "")).strip()
        if description:
            elements.append({
                "tag": "markdown",
                "content": f"{icon} **{label}**：{description}",
            })
        buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": btn_text},
            "type": "primary" if len(buttons) == 0 else "default",
            "value": {
                "action": "opening_select",
                "mode_id": str(mode.get("mode_id", "")),
                "route": str(mode.get("route", "")),
                "route_params": mode.get("route_params") or {},
                "label": label,
            },
        })

    if buttons:
        elements.append({"tag": "action", "actions": buttons})

    return _base_card("开局", "blue", elements)


def build_report_card(title: str, dimension_text: str) -> dict[str, Any]:
    """构建状态汇报卡片。"""

    return {
        "config": {
            "wide_screen_mode": True,
            "enable_forward": True,
        },
        "header": {
            "template": "turquoise",
            "title": {
                "tag": "plain_text",
                "content": title,
            },
        },
        "elements": [
            {
                "tag": "markdown",
                "content": dimension_text,
            }
        ],
    }


def build_transition_card(text: str) -> dict[str, Any]:
    """构建时间推进类过渡卡片。"""

    return {
        "config": {
            "wide_screen_mode": True,
            "enable_forward": True,
        },
        "header": {
            "template": "orange",
            "title": {
                "tag": "plain_text",
                "content": "时局推演",
            },
        },
        "elements": [
            {
                "tag": "markdown",
                "content": text,
            }
        ],
    }


def build_action_panel(
    time_display: str,
    dimension_summary: str,
    situation: str,
    npc_buttons: list[str],
    objectives: list[dict[str, Any]] | None = None,
    related_npcs: set[str] | None = None,
    backlog_count: int = 0,
    agenda: list[dict[str, Any]] | None = None,
    campaign_goal: str = "",
) -> dict[str, Any]:
    """构建行动面板卡片：时间 + 剧本目标 + 国势 + 议程 + 按钮。"""

    elements: list[dict[str, Any]] = []

    if campaign_goal:
        elements.append({
            "tag": "markdown",
            "content": f"**🎯 {campaign_goal}**",
        })

    elements.append({
        "tag": "markdown",
        "content": f"📅 当前时间：{time_display}",
    })

    elements.append({"tag": "hr"})

    elements.append({
        "tag": "markdown",
        "content": f"**—— 国势 ——**\n{dimension_summary}",
    })

    if agenda:
        elements.append({"tag": "hr"})
        agenda_lines = []
        for item in agenda:
            urgency = str(item.get("urgency", "normal"))
            prefix = "🔴" if urgency == "urgent" else "🟡"
            desc = str(item.get("description", ""))
            consequence = str(item.get("consequence", ""))
            line = f"{prefix} **{desc}**" if urgency == "urgent" else f"{prefix} {desc}"
            if consequence:
                line += f"\n   └ 不处理后果：{consequence}"
            agenda_lines.append(line)
        elements.append({
            "tag": "markdown",
            "content": "**📋 本回合议程**\n" + "\n".join(agenda_lines),
        })

    if situation and situation.strip():
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": situation,
        })

    elements.append({"tag": "hr"})

    related = related_npcs or set()
    buttons: list[dict[str, Any]] = []
    for npc_name in npc_buttons:
        is_related = npc_name in related
        label = f"⭐{npc_name}" if is_related else f"对话{npc_name}"
        buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": label},
            "type": "primary" if is_related else "default",
            "value": {"action": "talk", "npc": npc_name},
        })

    buttons.append({
        "tag": "button",
        "text": {"tag": "plain_text", "content": "查看局势"},
        "type": "default",
        "value": {"action": "query_status"},
    })

    buttons.append({
        "tag": "button",
        "text": {"tag": "plain_text", "content": f"推进时间（{max(0, backlog_count)}项待办）"},
        "type": "primary",
        "value": {"action": "advance_time"},
    })

    elements.append({
        "tag": "action",
        "actions": buttons,
    })

    return _base_card("朝堂行动", "indigo", elements)


def build_npc_reply_card(
    npc_name: str,
    reply_text: str,
    suggestions: list[str] | None = None,
    frame_options: list[str] | None = None,
    npc_title: str = "",
    gm_note: str = "",
) -> dict[str, Any]:
    """构建NPC回复卡片：对话内容 + 可选方向按钮 + 结束对话按钮 + 可选决策多选。"""

    title_text = f"【{npc_name}·{npc_title}】" if npc_title else f"【{npc_name}】"
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": reply_text,
        },
    ]

    if gm_note:
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": gm_note,
        })

    if suggestions:
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": "**可执行建议：**",
        })

        sug_buttons: list[dict[str, Any]] = []
        for idx, sug in enumerate(suggestions):
            sug_buttons.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": sug},
                "type": "primary",
                "value": {"action": "pick_decision", "npc": npc_name, "decision": sug, "idx": idx},
            })

        sug_buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "全部采纳"},
            "type": "danger",
            "value": {"action": "confirm_all_decisions", "npc": npc_name, "decisions": json.dumps(suggestions, ensure_ascii=False)},
        })

        elements.append({
            "tag": "action",
            "actions": sug_buttons,
        })

    if frame_options:
        frame_buttons: list[dict[str, Any]] = []
        for option in frame_options[:3]:
            clean_option = str(option).strip()
            if not clean_option:
                continue
            frame_buttons.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": clean_option},
                "type": "default",
                "value": {"action": "gm_frame_input", "text": clean_option},
            })
        if frame_buttons:
            elements.append({"tag": "hr"})
            elements.append({
                "tag": "action",
                "actions": frame_buttons,
            })

    elements.append({"tag": "hr"})
    elements.append({
        "tag": "markdown",
        "content": "💬 或直接输入你想说的",
    })
    elements.append({
        "tag": "action",
        "actions": [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "结束对话"},
                "type": "default",
                "value": {"action": "end_talk", "npc": npc_name},
            },
        ],
    })

    return {
        "config": {
            "wide_screen_mode": True,
            "enable_forward": True,
        },
        "header": {
            "template": "violet",
            "title": {
                "tag": "plain_text",
                "content": title_text,
            },
        },
        "elements": elements,
    }


def build_dialogue_summary_card(npc_name: str, summary_text: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """构建对话汇总卡片。"""

    elements: list[dict[str, Any]] = [{"tag": "markdown", "content": summary_text or "此次对话暂未整理出明确行动。"}]
    has_pending = False
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        item = candidate.get("item")
        if not isinstance(item, dict):
            continue
        status = str(candidate.get("status", "pending"))
        description = str(item.get("description", f"候选事项{index}")).strip()
        if status == "added":
            elements.append({"tag": "markdown", "content": f"✓ 已加入 | {description}"})
        elif status == "skipped":
            elements.append({"tag": "markdown", "content": f"— 已跳过 | {description}"})
        else:
            has_pending = True
            elements.append({"tag": "markdown", "content": description})
            elements.append({
                "tag": "action",
                "actions": [
                    _button("＋加入", "add_to_backlog", "primary", item_index=index),
                    _button("跳过", "skip_backlog_item", "default", item_index=index),
                ],
            })
    if candidates:
        button_type = "default" if has_pending else "primary"
        elements.append({"tag": "action", "actions": [_button("确认完毕", "confirm_dialogue_summary", button_type)]})
    else:
        elements.append({"tag": "markdown", "content": "没有候选待办"})
    return _base_card(f"与{npc_name}的谈话汇总", "blue", elements)


def build_empty_advance_confirmation_card() -> dict[str, Any]:
    """构建空待办二次确认卡。"""

    return _base_card(
        "确认推进",
        "orange",
        [
            {"tag": "markdown", "content": "本回合未安排任何行动，确定推进吗？"},
            {"tag": "action", "actions": [_button("确认", "confirm_empty_advance", "primary"), _button("返回修改", "return_to_modify", "default")]},
        ],
    )


def build_settlement_confirmation_card(
    backlog_items: list[dict[str, Any]],
    world_state: dict[str, Any],
    delayed_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建结算确认预览卡。"""

    elements: list[dict[str, Any]] = []
    elements.append({"tag": "markdown", "content": f"**世界状态**\n{json.dumps(world_state, ensure_ascii=False)}"})
    deterministic_totals = _sum_deterministic_world_effects(backlog_items)
    insufficient = _find_insufficient_world_values(world_state, deterministic_totals)
    if deterministic_totals:
        elements.append({"tag": "markdown", "content": "**确定性净值汇总**\n" + "\n".join(f"- {name}: {delta:+d}" for name, delta in deterministic_totals.items())})
    if insufficient:
        elements.append({"tag": "markdown", "content": "🔴 **维度不足**\n" + "\n".join(insufficient)})
    if backlog_items:
        for index, item in enumerate(backlog_items, start=1):
            description = str(item.get("description", f"待办{index}")).strip()
            elements.append({"tag": "markdown", "content": f"**待办 {index}**\n{description}"})
            if item.get("deterministic"):
                deterministic_effects = item.get("deterministic_effects", {}) if isinstance(item.get("deterministic_effects"), dict) else {}
                elements.append({"tag": "markdown", "content": f"确定性效果：{json.dumps(deterministic_effects, ensure_ascii=False)}"})
            else:
                success_rate = item.get("predicted_effects", {}).get("success_rate", item.get("success_rate", 0)) if isinstance(item.get("predicted_effects"), dict) else item.get("success_rate", 0)
                elements.append({"tag": "markdown", "content": f"DC {item.get('dc', 0)} / 修正 { _format_modifier(item.get('modifier', 0)) } / 预计成功率 {_format_success_rate(success_rate)}"})
            if item.get("duration") == "delayed":
                elements.append({"tag": "markdown", "content": f"延迟结算：{max(0, int(item.get('delay_count', 0)))} 次推进后"})
            if item.get("severity_warning"):
                elements.append({"tag": "markdown", "content": f"⚠ {item.get('severity_warning')}"})
        for index, item in enumerate(backlog_items, start=1):
            elements.append({"tag": "action", "actions": [_button(f"移除待办{index}", "remove_backlog_item", "default", item=item)]})
    else:
        elements.append({"tag": "markdown", "content": "当前无待办事项。"})

    if delayed_items:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**进行中延迟项**\n" + "\n".join(f"- {str(item.get('description', '未命名'))}（剩余 {item.get('remaining_advances', 0)} 次推进）" for item in delayed_items)})

    elements.append({"tag": "hr"})
    elements.append({"tag": "action", "actions": [_button("确认结算", "confirm_settlement", "primary"), _button("返回修改", "return_to_modify", "default")]})
    return _base_card("结算确认", "orange", elements)


def _sum_deterministic_world_effects(backlog_items: list[dict[str, Any]]) -> dict[str, int]:
    """汇总确定性待办的世界维度净值。"""

    totals: dict[str, int] = {}
    for item in backlog_items:
        if not item.get("deterministic"):
            continue
        effects = item.get("deterministic_effects", {}) if isinstance(item.get("deterministic_effects"), dict) else {}
        dimensions = effects.get("dimensions") if isinstance(effects.get("dimensions"), dict) else {}
        world = dimensions.get("world") if isinstance(dimensions.get("world"), dict) else {}
        for name, delta in world.items():
            if isinstance(delta, int):
                totals[str(name)] = totals.get(str(name), 0) + delta
    return totals


def _find_insufficient_world_values(world_state: dict[str, Any], totals: dict[str, int]) -> list[str]:
    """找出确定性净值会击穿的世界维度。"""

    world_values = world_state.get("world") if isinstance(world_state.get("world"), dict) else {}
    lines: list[str] = []
    for name, delta in totals.items():
        current = world_values.get(name)
        if isinstance(current, int) and current + delta < 0:
            lines.append(f"{name}: {current} {delta:+d} → {current + delta}")
    return lines


def build_settlement_result_card(item: dict[str, Any], dice_result: dict[str, Any], outcome: str, narrative: str, interference_info: dict[str, Any] | None) -> dict[str, Any]:
    """构建单项结算结果卡。"""

    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": f"**{item.get('description', '未命名待办')}**\n结果：{outcome}\n掷骰：{json.dumps(dice_result, ensure_ascii=False)}"},
    ]
    if narrative:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": narrative})
    if interference_info:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": f"**NPC干扰**\n{json.dumps(interference_info, ensure_ascii=False)}"})
    return _base_card("待办结算", "purple", elements)


def build_situation_card(world_state: dict[str, Any], forces_summary: str, backlog: list[dict[str, Any]], delayed_queue: list[dict[str, Any]]) -> dict[str, Any]:
    """构建局势卡片。"""

    time_str = world_state.get("time", "未知时间")
    world_dims = world_state.get("world", {})
    char_dims = world_state.get("character", {})

    world_lines = []
    for name, value in world_dims.items():
        clamped = max(0, min(10, int(value)))
        bar = "█" * clamped + "░" * (10 - clamped)
        world_lines.append(f"{name} {bar} {value}/10")
    world_text = "\n".join(world_lines) if world_lines else "暂无数据"

    char_lines = []
    for name, value in char_dims.items():
        clamped = max(0, min(10, int(value)))
        bar = "█" * clamped + "░" * (10 - clamped)
        char_lines.append(f"{name} {bar} {value}/10")
    char_text = "\n".join(char_lines) if char_lines else "暂无数据"

    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": f"**{time_str}**"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**天下大势**\n{world_text}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**天子素养**\n{char_text}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": forces_summary or "暂无势力动向。"},
    ]
    if backlog:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**当前待办**\n" + "\n".join(f"- {str(item.get('description', '未命名'))}" for item in backlog)})
    if delayed_queue:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**进行中**\n" + "\n".join(f"- {str(item.get('description', '未命名'))}（剩余 {item.get('remaining_advances', 0)} 次推进）" for item in delayed_queue)})
    return _base_card("查看局势", "turquoise", elements)


def build_board_card(
    time_str: str,
    dimension_summary: str,
    board_regions: dict[str, Any],
    delayed_queue: list[dict[str, Any]],
    turn_agenda: list[dict[str, Any]],
    discovered_clues: list[dict[str, Any]],
    first_time: bool = False,
    campaign_goal: str = "",
) -> dict[str, Any]:
    """构建棋盘迷雾卡片：国势 + 区域 + 迷雾 + 议程。"""

    elements: list[dict[str, Any]] = []

    if campaign_goal:
        elements.append({"tag": "markdown", "content": f"**{campaign_goal}**"})

    elements.append({"tag": "markdown", "content": f"**{time_str}**"})

    if first_time:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": (
            "这是你的**天下棋盘**。已知信息直接显示，"
            "❓标记的是尚未掌握的情报——提升情报能力或执行特定行动可揭开迷雾。"
        )})

    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": f"**—— 国势 ——**\n{dimension_summary}"})

    for region_id, region_value in board_regions.items():
        region = region_value if isinstance(region_value, dict) else {}
        region_name = region.get("name", region_id)
        lines: list[str] = []

        for layer_value in region.get("layers", []):
            layer = layer_value if isinstance(layer_value, dict) else {}
            label = layer.get("label", "")
            known = layer.get("known_text", "")
            fog = layer.get("fog")
            events = layer.get("events", [])

            if known:
                lines.append(f"  {label}：{known}")
            if isinstance(fog, dict):
                if fog.get("unlocked"):
                    unlock_text = fog.get("unlock_text") or fog.get("revealed_text", "")
                    if unlock_text:
                        lines.append(f"  {label}：🔓 {unlock_text}")
                else:
                    hint = fog.get("hint", "未知")
                    cond = fog.get("unlock_condition", {})
                    cond_text = ""
                    if isinstance(cond, dict):
                        if cond.get("dimension"):
                            cond_text = f'需{cond["dimension"]}>={cond.get("threshold", "?")}'
                        elif cond.get("action"):
                            cond_text = str(cond["action"])
                    if cond_text:
                        lines.append(f"  ❓ {hint}（{cond_text}）")
                    else:
                        lines.append(f"  ❓ {hint}")
            for evt in events:
                lines.append(f"  ⚡ {evt}")

        for news_item in region.get("news", []):
            lines.append(f"  {news_item}")

        region_text = "\n".join(lines) if lines else "  ❓❓❓ 缺乏了解"
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": f"**【{region_name}】**\n{region_text}"})

    if delayed_queue:
        elements.append({"tag": "hr"})
        dq_lines = [f'· {item.get("description", "未命名")}（剩余 {item.get("remaining_advances", 0)} 回合）' for item in delayed_queue]
        elements.append({"tag": "markdown", "content": "**【进行中】**\n" + "\n".join(dq_lines)})

    if turn_agenda:
        elements.append({"tag": "hr"})
        agenda_lines: list[str] = []
        for ag in turn_agenda:
            urgency = ag.get("urgency", "normal")
            icon = "🔴" if urgency == "urgent" else "🟡"
            desc = ag.get("description", "未命名")
            consequence = ag.get("consequence", "")
            line = f"{icon} {desc}"
            if consequence:
                line += f"——{consequence}"
            agenda_lines.append(line)
        elements.append({"tag": "markdown", "content": "**【本回合议程】**\n" + "\n".join(agenda_lines)})

    if discovered_clues:
        elements.append({"tag": "hr"})
        clue_lines = [f'· {c.get("content", "")}' for c in discovered_clues[-5:]]
        clue_title = "**【已知线索】**"
        if len(discovered_clues) > 5:
            clue_title += f"（共{len(discovered_clues)}条线索，显示最近5条）"
        elements.append({"tag": "markdown", "content": clue_title + "\n" + "\n".join(clue_lines)})

    elements.append({
        "tag": "action",
        "actions": [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "返回行动"},
                "type": "primary",
                "value": {"action": "back_to_action"},
            },
        ],
    })

    return _base_card("天下棋盘", "turquoise", elements)


def build_settlement_card(
    narration: str,
    dimension_changes: dict[str, tuple[int, int]] | None = None,
    time_change: str = "",
    new_events: list[str] | None = None,
    npc_buttons: list[str] | None = None,
) -> dict[str, Any]:
    """构建结算卡片：叙事 + 维度变化 + 时间推进 + 新事件 + 行动面板。"""

    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": narration,
        },
    ]

    if dimension_changes:
        lines = []
        for dimension, (old_val, new_val) in dimension_changes.items():
            delta = new_val - old_val
            sign = "+" if delta > 0 else ""
            lines.append(f"{dimension} {old_val} → {new_val} ({sign}{delta})")
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": "**局势变化**\n" + "\n".join(lines),
        })

    if time_change:
        elements.append({
            "tag": "markdown",
            "content": f"⏳ {time_change}",
        })

    if new_events:
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": "**新事件**\n" + "\n".join(f"• {e}" for e in new_events),
        })

    if npc_buttons:
        elements.append({"tag": "hr"})
        buttons: list[dict[str, Any]] = []
        for npc_name in npc_buttons:
            buttons.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": f"对话{npc_name}"},
                "type": "primary",
                "value": {"action": "talk", "npc": npc_name},
            })
        buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "查看局势"},
            "type": "default",
            "value": {"action": "query_status"},
        })
        elements.append({
            "tag": "action",
            "actions": buttons,
        })

    return {
        "config": {
            "wide_screen_mode": True,
            "enable_forward": True,
        },
        "header": {
            "template": "orange",
            "title": {
                "tag": "plain_text",
                "content": "圣断既行",
            },
        },
        "elements": elements,
    }


def build_role_selection_card(title: str, description: str, roles: list[dict[str, Any]]) -> dict[str, Any]:
    """构建新格式剧本的角色选择卡片。"""

    elements: list[dict[str, Any]] = [{"tag": "markdown", "content": description or "请选择本局扮演的角色。"}]
    buttons: list[dict[str, Any]] = []
    for index, role in enumerate(roles, start=1):
        role_id = str(role.get("role_id", "")).strip()
        name = str(role.get("name", role_id)).strip() or f"角色{index}"
        detail = str(role.get("core_play") or role.get("description") or role.get("teaser_goal") or "前路自定。")
        elements.append({"tag": "markdown", "content": f"**{index}. {name}**\n{detail}"})
        if role_id:
            buttons.append(_button(f"选择{name}", "select_role", "primary" if index == 1 else "default", role_id=role_id))
    if buttons:
        elements.append({"tag": "action", "actions": buttons})
    return _base_card(title or "选择角色", "blue", elements)


def _proposal_card_payload(payload: Any) -> dict[str, Any]:
    """清洗提案载荷，确保可放入飞书按钮 value。"""

    data = _to_dict(payload) or (getattr(payload, "__dict__", {}) if payload is not None else {})
    action_type = data.get("action_type", "action")
    if hasattr(action_type, "value"):
        action_type = action_type.value
    return {
        "action_type": str(action_type),
        "description": str(data.get("description", "")),
        "dc": _safe_int(data.get("dc", 8), 8),
        "modifier": _safe_int(data.get("modifier", 0), 0),
        "main_dimension": str(data.get("main_dimension", "意志")),
        "auxiliary_dimensions": list(data.get("auxiliary_dimensions", [])) if isinstance(data.get("auxiliary_dimensions", []), list) else [],
        "tags": list(data.get("tags", [])) if isinstance(data.get("tags", []), list) else [],
        "time_cost": _safe_int(data.get("time_cost", 0), 0),
        "success_rate": float(data.get("success_rate", 0.0) or 0.0),
        "npc_id": str(data.get("npc_id", "") or ""),
        "proposal_id": str(data.get("proposal_id", "")),
    }


def build_action_confirmation_card(proposal: Any) -> dict[str, Any]:
    """构建行动两步确认卡片。"""

    data = _proposal_card_payload(proposal)
    elements = [
        {
            "tag": "markdown",
            "content": (
                f"**行动描述**\n{data['description']}\n\n"
                f"**判定信息**\n"
                f"主维度：{data['main_dimension']}\n"
                f"DC：{data['dc']}\n"
                f"当前修正值：{_format_modifier(data['modifier'])}\n"
                f"预估成功率：{_format_success_rate(data['success_rate'])}\n"
                f"时间成本：{data['time_cost']}单位"
            ),
        },
        {"tag": "hr"},
        {
            "tag": "action",
            "actions": [
                _button("确认执行", "confirm_action", "primary", proposal=data),
                _button("放弃行动", "abandon_action", "default", proposal_id=data.get("proposal_id", "")),
            ],
        },
    ]
    return _base_card("行动确认", "indigo", elements)


def build_dice_result_card(action_result: Any, narration: str = "", growth: dict[str, Any] | None = None) -> dict[str, Any]:
    """构建掷骰结果卡片，兼容ActionResult和裸JudgmentResult。"""

    proposal = getattr(action_result, "proposal", None)
    judgment = getattr(action_result, "judgment", action_result)
    result = _to_dict(judgment)
    dice_values = result.get("dice_values", result.get("dice", []))
    if not isinstance(dice_values, list):
        dice_values = []
    dice_text = " + ".join(str(_safe_int(item)) for item in dice_values) if dice_values else "未记录"
    tier = str(result.get("result_tier", result.get("tier", "未知")))
    template_map = {"大成功": "yellow", "成功": "green", "部分成功": "orange", "失败": "red", "大失败": "purple"}
    description = getattr(proposal, "description", "")
    elements = [{
        "tag": "markdown",
        "content": (
            f"**行动**：{description}\n"
            f"**结果档位**：{tier}\n"
            f"🎲 三颗骰子：{dice_text}\n"
            f"点数总和：{_safe_int(result.get('total', 0))}\n"
            f"修正值：{_format_modifier(result.get('modifier', 0))}\n"
            f"DC：{_safe_int(result.get('dc', 0))}\n"
            f"最终值：{_safe_int(result.get('final_value', 0))}"
        ),
    }]
    if narration:
        elements.extend([{"tag": "hr"}, {"tag": "markdown", "content": narration}])
    if growth:
        elements.extend([{"tag": "hr"}, {"tag": "markdown", "content": f"**成长**：{growth.get('dimension')} +{growth.get('delta', 0)}"}])
    return _base_card("行动判定", template_map.get(tier, "grey"), elements)


def build_cost_choice_card(proposal: Any, cost_options: list[dict[str, Any]]) -> dict[str, Any]:
    """构建部分成功代价选择卡片。"""

    data = _proposal_card_payload(proposal)
    elements: list[dict[str, Any]] = [{"tag": "markdown", "content": f"**{data['description']}**取得部分成功，请选择一项代价。"}]
    buttons: list[dict[str, Any]] = []
    for index, option in enumerate(cost_options[:3], start=1):
        if not isinstance(option, dict):
            continue
        cost_id = str(option.get("id") or f"cost_{index}")
        desc = str(option.get("description") or f"代价{index}")
        elements.append({"tag": "markdown", "content": f"{index}. {desc}"})
        buttons.append(_button(f"选择{index}", "partial_success_cost", "primary" if index == 1 else "default", cost_id=cost_id, proposal=data))
    elements.append({"tag": "action", "actions": buttons})
    return _base_card("部分成功", "yellow", elements)


def build_passive_response_card(passive_payload: dict[str, Any]) -> dict[str, Any]:
    """构建被动响应选择卡片。"""

    event_description = str(passive_payload.get("event_description", "突发事件需要回应。"))
    elements: list[dict[str, Any]] = [{"tag": "markdown", "content": event_description}]
    buttons: list[dict[str, Any]] = []
    for index, proposal in enumerate(passive_payload.get("suggested_responses", [])[:3], start=1):
        data = _proposal_card_payload(proposal)
        label = data.get("description") or f"回应{index}"
        buttons.append(_button(label[:20], "passive_response", "primary" if index == 1 else "default", proposal=data))
    if buttons:
        elements.append({"tag": "action", "actions": buttons})
    return _base_card("被动响应", "red", elements)


def build_gm_interpretation_card(interpretation: dict[str, Any]) -> dict[str, Any]:
    """构建GM解读确认卡片，展示玩家意图解读与可选方案。"""
    elements: list[dict[str, Any]] = []
    interp_text = str(interpretation.get("interpretation", "")).strip()
    if interp_text:
        elements.append({"tag": "markdown", "content": f"**GM解读：**{interp_text}"})
    gm_note = str(interpretation.get("gm_note", "")).strip()
    if gm_note:
        elements.append({"tag": "markdown", "content": gm_note})
    elements.append({"tag": "hr"})
    options = interpretation.get("options", [])
    buttons: list[dict[str, Any]] = []
    for i, option in enumerate(options):
        if not isinstance(option, dict):
            continue
        desc = str(option.get("description", "选项")).strip()
        time_cost = option.get("time_cost", 0)
        prerequisites = str(option.get("prerequisites", "")).strip()
        risk = str(option.get("risk_hint", "")).strip()
        detail_parts: list[str] = []
        if time_cost:
            detail_parts.append(f"耗时{time_cost}回合")
        if prerequisites:
            detail_parts.append(prerequisites)
        if risk:
            detail_parts.append(f"风险：{risk}")
        option_text = f"**方案{i + 1}：**{desc}"
        if detail_parts:
            option_text += "\n" + " | ".join(detail_parts)
        elements.append({"tag": "markdown", "content": option_text})
        label = desc[:20] + "…" if len(desc) > 20 else desc
        buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": label},
            "type": "primary" if i == 0 else "default",
            "value": {"action": "gm_select", "option_index": i},
        })
    if not buttons:
        elements.append({"tag": "markdown", "content": "暂无可行方案。"})
    else:
        elements.append({"tag": "action", "actions": buttons})
    feasibility = str(interpretation.get("feasibility", "可行")).strip()
    template = "green" if feasibility == "可行" else "orange" if feasibility == "有条件" else "red"
    return _base_card("GM解读", template, elements)


def build_court_session_card(
    topic: str,
    round_num: int,
    responses: list[dict[str, Any]],
    present_npcs: list[str],
    pending_npcs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构建朝会议事组合卡片，包含所有NPC当轮发言和按钮。"""
    elements: list[dict[str, Any]] = []
    for resp in responses:
        if not isinstance(resp, dict):
            continue
        npc = str(resp.get("npc", "?"))
        mode = str(resp.get("mode", "active"))
        content = str(resp.get("content", "")).strip()
        expression = str(resp.get("expression", "")).strip()
        title = str(resp.get("title", "")).strip()
        header = f"**【{npc}】**" + (f"（{title}）" if title else "")
        if mode == "silent":
            if expression:
                elements.append({"tag": "markdown", "content": f"{header}\n*{expression}*"})
        elif content:
            elements.append({"tag": "markdown", "content": f"{header}\n\"{content}\""})
        elements.append({"tag": "hr"})
    if elements and elements[-1].get("tag") == "hr":
        elements.pop()
    footer_parts = [f"在场：{', '.join(present_npcs)}"]
    if pending_npcs:
        for pnpc in pending_npcs:
            if isinstance(pnpc, dict):
                footer_parts.append(f"{pnpc.get('name', '?')}（召回中，{pnpc.get('eta', '?')}回合后到达）")
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": " | ".join(footer_parts)})
    buttons: list[dict[str, Any]] = []
    buttons.append({
        "tag": "button",
        "text": {"tag": "plain_text", "content": "结束议事"},
        "type": "danger",
        "value": {"action": "court_end"},
    })
    elements.append({"tag": "action", "actions": buttons})
    return _base_card(f"朝会议事 · {topic} · 第{round_num}轮", "purple", elements)


def build_decision_card(decisions: list[dict[str, Any]], npc_name: str) -> dict[str, Any]:
    """构建决策确认卡片，玩家可逐条确认或撤回检测到的决策。"""
    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": f"**📜 GM检测到以下决策：**"},
    ]
    buttons: list[dict[str, Any]] = []
    for idx, decision in enumerate(decisions):
        summary = str(decision.get("summary", "未知决策")).strip()
        category = str(decision.get("category", "")).strip()
        prefix = f"[{category}] " if category else ""
        elements.append({"tag": "markdown", "content": f"{idx + 1}. {prefix}{summary}"})
        dec_id = str(decision.get("id", f"dec_{idx}"))
        buttons.append(_button(f"✓ 确认{idx + 1}", "confirm_decision", "primary", decision_id=dec_id, decision=decision))
        buttons.append(_button(f"✗ 撤回{idx + 1}", "revoke_decision", "default", decision_id=dec_id))
    if buttons:
        elements.append({"tag": "action", "actions": buttons})
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": "💬 继续对话不受影响"})
    return _base_card(f"决策确认 · {npc_name}", "yellow", elements)
