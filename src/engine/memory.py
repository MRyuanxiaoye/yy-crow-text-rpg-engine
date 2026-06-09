"""记忆系统：事实提取、世界状态、承诺追踪、记忆压缩。"""

from __future__ import annotations

import logging
from typing import Any

from src.engine.state import GameState
from src.llm.client import get_claude_client

logger = logging.getLogger(__name__)


async def extract_facts(
    state: GameState,
    npc_name: str,
    talk_history: list[dict],
) -> dict[str, Any]:
    """从一段对话中提取关键事实和承诺。返回 {"facts": [...], "promises": [...]}"""

    if not talk_history:
        return {"facts": [], "promises": []}

    dialogue_text = "\n".join(
        f"{'玩家' if msg.get('role') == 'user' else npc_name}: {msg.get('content', '')}"
        for msg in talk_history
    )

    system_prompt = (
        "你是一个信息提取器。从以下对话中提取：\n"
        "1. 关键事实（玩家做了什么决定、表达了什么态度、透露了什么信息）\n"
        "2. 玩家对NPC做出的承诺（答应做某事、许诺某条件）\n\n"
        "返回JSON格式：\n"
        '{"facts": [{"fact": "描述", "sentiment": "grateful/fearful/neutral/hostile/loyal"}], '
        '"promises": [{"content": "承诺内容", "made_to": "NPC名"}]}\n\n'
        "规则：\n"
        "- 只提取有实质意义的事实，忽略寒暄\n"
        "- sentiment表示这件事让NPC对玩家的态度\n"
        "- 如果没有关键事实或承诺，对应数组为空\n"
        "- 最多提取5条事实、3条承诺"
    )

    claude = get_claude_client()
    result = await claude.chat_json(system_prompt, dialogue_text, temperature=0.2)

    facts = result.get("facts", [])
    promises = result.get("promises", [])

    if not isinstance(facts, list):
        facts = []
    if not isinstance(promises, list):
        promises = []

    return {"facts": facts, "promises": promises}


def save_npc_facts(
    state: GameState,
    npc_name: str,
    facts: list[dict],
) -> None:
    """将提取的事实写入NPC记忆。"""

    if not facts:
        return

    if npc_name not in state.npc_memories:
        state.npc_memories[npc_name] = []

    for fact_item in facts:
        if not isinstance(fact_item, dict):
            continue
        fact_text = str(fact_item.get("fact", "")).strip()
        if not fact_text:
            continue
        state.npc_memories[npc_name].append({
            "fact": fact_text,
            "sentiment": str(fact_item.get("sentiment", "neutral")),
            "turn": state.turn,
            "chapter": state.chapter_id,
        })

    logger.info("NPC记忆更新 npc=%s 新增%d条 总计%d条", npc_name, len(facts), len(state.npc_memories[npc_name]))


def save_promises(state: GameState, promises: list[dict], npc_name: str) -> None:
    """将提取的承诺写入承诺追踪。"""

    if not promises:
        return

    max_id = 0
    for p in state.promises:
        pid = str(p.get("id", ""))
        if pid.startswith("p") and pid[1:].isdigit():
            max_id = max(max_id, int(pid[1:]))

    for promise_item in promises:
        if not isinstance(promise_item, dict):
            continue
        content = str(promise_item.get("content", "")).strip()
        if not content:
            continue
        max_id += 1
        state.promises.append({
            "id": f"p{max_id:03d}",
            "content": content,
            "made_to": npc_name,
            "turn": state.turn,
            "chapter": state.chapter_id,
            "status": "pending",
        })

    logger.info("承诺更新 新增%d条 总pending=%d", len(promises), sum(1 for p in state.promises if p.get("status") == "pending"))


async def update_world_summary(state: GameState) -> str:
    """根据当前状态重新生成世界状态描述。"""

    metrics_text = ", ".join(f"{k}:{v}" for k, v in state.metrics.items())
    pending_promises = [p for p in state.promises if p.get("status") == "pending"]
    promises_text = "; ".join(p.get("content", "") for p in pending_promises[:5]) if pending_promises else "无"

    recent_decisions = state.decisions[-5:] if state.decisions else []
    decisions_text = "; ".join(str(d.get("summary", d.get("choice", ""))) for d in recent_decisions) if recent_decisions else "无"

    system_prompt = (
        "你是一个游戏世界状态总结器。根据以下信息，生成一段200字以内的世界状态描述。\n"
        "要求：\n"
        "- 描述当前政治格局、各方势力态势\n"
        "- 提及关键NPC的当前状态和态度\n"
        "- 提及玩家未兑现的承诺（如有）\n"
        "- 用第三人称叙事口吻\n"
        "- 不超过200字"
    )

    user_content = (
        f"当前数值：{metrics_text}\n"
        f"当前章节：{state.chapter_id}\n"
        f"当前时间：{state.game_date}\n"
        f"近期决策：{decisions_text}\n"
        f"未兑现承诺：{promises_text}\n"
        f"NPC状态：{state.npc_statuses}"
    )

    claude = get_claude_client()
    summary = await claude.chat(system_prompt, user_content, temperature=0.4)
    state.world_summary = summary
    logger.info("世界状态更新 len=%d", len(summary))
    return summary


async def generate_chapter_summary(state: GameState) -> dict:
    """生成当前章节的压缩摘要。"""

    metrics_text = ", ".join(f"{k}:{v}" for k, v in state.metrics.items())
    decisions_text = "; ".join(
        str(d.get("summary", d.get("choice", "")))
        for d in state.decisions[-10:]
    ) if state.decisions else "无"

    npc_memories_text = ""
    for npc, mems in state.npc_memories.items():
        chapter_mems = [m for m in mems if m.get("chapter") == state.chapter_id]
        if chapter_mems:
            facts = "; ".join(m.get("fact", "") for m in chapter_mems[:5])
            npc_memories_text += f"{npc}: {facts}\n"

    system_prompt = (
        "你是一个章节摘要生成器。将本章发生的事压缩为结构化摘要。\n"
        "返回JSON：\n"
        '{"summary": "一段话概括本章", "key_decisions": ["决策1", "决策2"], '
        '"npc_attitude_shifts": {"NPC名": "态度变化描述"}}'
    )

    user_content = (
        f"章节：{state.chapter_id}\n"
        f"最终数值：{metrics_text}\n"
        f"本章决策：{decisions_text}\n"
        f"NPC互动：\n{npc_memories_text}"
    )

    claude = get_claude_client()
    result = await claude.chat_json(system_prompt, user_content, temperature=0.3)
    state.chapter_summaries[state.chapter_id] = result
    logger.info("章节摘要生成 chapter=%s", state.chapter_id)
    return result


async def compress_npc_memory(state: GameState, npc_name: str) -> None:
    """当NPC记忆超过15条时，压缩早期记忆。"""

    memories = state.npc_memories.get(npc_name, [])
    if len(memories) <= 15:
        return

    to_compress = memories[:-10]
    to_keep = memories[-10:]

    facts_text = "\n".join(
        f"- {m.get('fact', '')} (态度:{m.get('sentiment', 'neutral')}, 回合:{m.get('turn', '?')})"
        for m in to_compress
    )

    system_prompt = (
        "将以下NPC记忆条目压缩合并为3-5条核心事实。\n"
        "保留最重要的信息（重大决策、态度转变、关键承诺）。\n"
        "返回JSON：\n"
        '{"compressed": [{"fact": "描述", "sentiment": "态度", "turn": 最早回合数, "chapter": "章节"}]}'
    )

    claude = get_claude_client()
    result = await claude.chat_json(system_prompt, f"NPC: {npc_name}\n记忆:\n{facts_text}", temperature=0.2)

    compressed = result.get("compressed", [])
    if isinstance(compressed, list) and compressed:
        state.npc_memories[npc_name] = compressed + to_keep
        logger.info("NPC记忆压缩 npc=%s %d条→%d条", npc_name, len(memories), len(state.npc_memories[npc_name]))


async def check_promises(state: GameState) -> list[dict]:
    """检查是否有承诺被兑现或超时。返回状态变化的承诺列表。"""

    pending = [p for p in state.promises if p.get("status") == "pending"]
    if not pending:
        return []

    recent_decisions = state.decisions[-5:] if state.decisions else []
    decisions_text = "; ".join(str(d.get("summary", d.get("choice", ""))) for d in recent_decisions)

    promises_text = "\n".join(
        f"- [{p['id']}] {p['content']} (对{p.get('made_to', '?')}，第{p.get('turn', '?')}回合)"
        for p in pending
    )

    system_prompt = (
        "检查以下承诺是否已被近期决策兑现。\n"
        "返回JSON：\n"
        '{"fulfilled": ["承诺id1", "承诺id2"], "broken": []}\n'
        "只有明确被兑现的才标记fulfilled，不确定的保持pending。"
    )

    user_content = f"待检查承诺：\n{promises_text}\n\n近期决策：{decisions_text}"

    claude = get_claude_client()
    result = await claude.chat_json(system_prompt, user_content, temperature=0.2)

    fulfilled_ids = result.get("fulfilled", [])
    broken_ids = result.get("broken", [])
    changed: list[dict] = []

    for p in state.promises:
        if p.get("id") in fulfilled_ids:
            p["status"] = "fulfilled"
            changed.append(p)
        elif p.get("id") in broken_ids:
            p["status"] = "broken"
            changed.append(p)

    if changed:
        logger.info("承诺状态变化 %s", [(c["id"], c["status"]) for c in changed])

    return changed
