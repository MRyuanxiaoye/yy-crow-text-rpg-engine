"""调试毕自严出场重复问题：拦截所有发送的卡片/文本，记录调用栈。"""
import asyncio
import os
import json
import traceback

os.environ.setdefault("NARRATOR_APP_ID", "fake")
os.environ.setdefault("NARRATOR_APP_SECRET", "fake")
os.environ.setdefault("NPC_APP_ID", "fake")
os.environ.setdefault("NPC_APP_SECRET", "fake")
os.environ.setdefault("FEISHU_VERIFICATION_TOKEN", "fake")

send_log = []

async def mock_send_narrator(chat_id, card_json):
    n = len(send_log) + 1
    title = card_json.get("header", {}).get("title", {}).get("content", "")
    elements_text = []
    for e in card_json.get("elements", []):
        if e.get("tag") == "markdown":
            elements_text.append(e.get("content", ""))
    body = "\n".join(elements_text)
    stack = "".join(traceback.format_stack()[-6:-1])
    entry = {"n": n, "type": "NARRATOR_CARD", "title": title, "body": body, "stack": stack}
    send_log.append(entry)
    print(f"\n{'='*60}")
    print(f"[SEND#{n}] NARRATOR_CARD title={title}")
    print(f"  body={body[:300]}")
    print(f"  调用栈:")
    for line in stack.strip().split("\n"):
        if "game_master" in line or "npc_engine" in line or "debug_bzy" in line:
            print(f"    {line.strip()}")

async def mock_send_npc(chat_id, card_json):
    n = len(send_log) + 1
    title = card_json.get("header", {}).get("title", {}).get("content", "")
    elements_text = []
    for e in card_json.get("elements", []):
        if e.get("tag") == "markdown":
            elements_text.append(e.get("content", ""))
    body = "\n".join(elements_text)
    stack = "".join(traceback.format_stack()[-6:-1])
    entry = {"n": n, "type": "NPC_CARD", "title": title, "body": body, "stack": stack}
    send_log.append(entry)
    print(f"\n{'='*60}")
    print(f"[SEND#{n}] NPC_CARD title={title}")
    print(f"  body={body[:500]}")
    print(f"  调用栈:")
    for line in stack.strip().split("\n"):
        if "game_master" in line or "npc_engine" in line or "debug_bzy" in line:
            print(f"    {line.strip()}")

async def mock_send_text(chat_id, role, text):
    n = len(send_log) + 1
    stack = "".join(traceback.format_stack()[-6:-1])
    entry = {"n": n, "type": "TEXT", "role": role, "body": text, "stack": stack}
    send_log.append(entry)
    print(f"\n{'='*60}")
    print(f"[SEND#{n}] TEXT role={role}")
    print(f"  body={text[:300]}")

import src.feishu.sender as sender_mod
sender_mod.send_narrator_card = mock_send_narrator
sender_mod.send_npc_card = mock_send_npc
sender_mod.send_npc_text = mock_send_text

import src.engine.game_master as gm_mod
gm_mod.send_narrator_card = mock_send_narrator
gm_mod.send_npc_card = mock_send_npc
gm_mod.send_npc_text = mock_send_text

# 拦截 generate_npc_reply，记录入参和结果
from src.engine.npc_engine import generate_npc_reply as _orig_npc_reply
async def _debug_npc_reply(state, npc_name, npc_profile, user_text, **kw):
    has_visit = any(h.get("type") == "visit_entry" for h in state.current_talk_history)
    print(f"\n>>> generate_npc_reply 调用: npc={npc_name}")
    print(f"    has_visit_entry={has_visit}")
    print(f"    talk_history_len={len(state.current_talk_history)}")
    for i, h in enumerate(state.current_talk_history):
        print(f"    [{i}] type={h.get('type','NONE')} speaker={h.get('speaker')} content={h.get('content','')[:80]}")
    reply = await _orig_npc_reply(state, npc_name, npc_profile, user_text, **kw)
    print(f"    reply={reply[:300]}")
    return reply

import src.engine.npc_engine as npc_mod
npc_mod.generate_npc_reply = _debug_npc_reply
gm_mod.generate_npc_reply = _debug_npc_reply

CHAT_ID = "debug_bzy_repeat"

async def main():
    save_path = f"data/saves/{CHAT_ID}.json"
    if os.path.exists(save_path):
        os.remove(save_path)

    from src.engine.game_master import handle_message, handle_card_action
    from src.engine.state import get_state_manager

    print("========== 开始游戏 ==========")
    await handle_message(CHAT_ID, "开始游戏")

    print("\n========== 选择皇帝 ==========")
    await handle_card_action(CHAT_ID, {"action": "select_role", "role_id": "emperor"})

    state = get_state_manager().load(CHAT_ID)
    print(f"\n选角后: phase={state.phase} talking_to={state.talking_to}")
    print(f"  visit_queue({len(state.visit_queue)}): {[v.get('npc_id') for v in state.visit_queue]}")
    print(f"  talk_history({len(state.current_talk_history)}):")
    for i, h in enumerate(state.current_talk_history):
        print(f"    [{i}] type={h.get('type','NONE')} speaker={h.get('speaker')} content={h.get('content','')[:80]}")

    # 循环dismiss直到遇到毕自严
    for rnd in range(20):
        state = get_state_manager().load(CHAT_ID)
        current = state.talking_to or ""
        print(f"\n{'#'*60}")
        print(f"ROUND {rnd}: talking_to='{current}' phase={state.phase}")

        if not current:
            print("  无对话目标，结束")
            break

        if "毕自严" in current:
            print(f"\n{'*'*60}")
            print(f"*** 找到毕自严！当前状态 ***")
            print(f"  phase={state.phase}")
            print(f"  talk_history({len(state.current_talk_history)}):")
            for i, h in enumerate(state.current_talk_history):
                print(f"    [{i}] type={h.get('type','NONE')} speaker={h.get('speaker')} content={h.get('content','')[:120]}")

            print(f"\n*** 向毕自严发消息 ***")
            await handle_message(CHAT_ID, "户部情况如何？")

            state = get_state_manager().load(CHAT_ID)
            print(f"\n*** 毕自严回复后 ***")
            print(f"  talk_history({len(state.current_talk_history)}):")
            for i, h in enumerate(state.current_talk_history):
                print(f"    [{i}] type={h.get('type','NONE')} speaker={h.get('speaker')} content={h.get('content','')[:120]}")
            break

        print(f"  dismiss {current}")
        await handle_message(CHAT_ID, "退下")

    # 打印完整发送记录
    print(f"\n\n{'='*60}")
    print(f"=== 完整发送记录 ({len(send_log)} 条) ===")
    print(f"{'='*60}")
    bzy_sends = []
    for entry in send_log:
        is_bzy = "毕自严" in entry.get("body", "") or "毕自严" in entry.get("title", "")
        marker = " *** 毕自严相关 ***" if is_bzy else ""
        print(f"\n[SEND#{entry['n']}] {entry['type']} {entry.get('title','')}{marker}")
        print(f"  body前200字: {entry.get('body','')[:200]}")
        if is_bzy:
            bzy_sends.append(entry)

    print(f"\n\n{'='*60}")
    print(f"=== 毕自严相关发送 ({len(bzy_sends)} 条) ===")
    for entry in bzy_sends:
        print(f"\n[SEND#{entry['n']}] {entry['type']} title={entry.get('title','')}")
        print(f"  完整body:\n{entry.get('body','')}")
        print(f"  关键调用栈:")
        for line in entry.get("stack", "").strip().split("\n"):
            if "game_master" in line or "npc_engine" in line:
                print(f"    {line.strip()}")

asyncio.run(main())
