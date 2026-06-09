"""调试来访重复问题：跑到毕自严出场。"""
import asyncio, os, json

os.environ.setdefault("NARRATOR_APP_ID", "fake")
os.environ.setdefault("NARRATOR_APP_SECRET", "fake")
os.environ.setdefault("NPC_APP_ID", "fake")
os.environ.setdefault("NPC_APP_SECRET", "fake")
os.environ.setdefault("FEISHU_VERIFICATION_TOKEN", "fake")

card_counter = [0]

async def mock_send_narrator(chat_id, card_json):
    card_counter[0] += 1
    n = card_counter[0]
    title = card_json.get("header", {}).get("title", {}).get("content", "")
    print(f"\n[CARD#{n} NARRATOR] {title}")
    for e in card_json.get("elements", []):
        if e.get("tag") == "markdown":
            print(f"  {e.get('content', '')[:200]}")

async def mock_send_npc(chat_id, card_json):
    card_counter[0] += 1
    n = card_counter[0]
    title = card_json.get("header", {}).get("title", {}).get("content", "")
    print(f"\n[CARD#{n} NPC] {title}")
    for e in card_json.get("elements", []):
        if e.get("tag") == "markdown":
            print(f"  {e.get('content', '')[:200]}")

async def mock_send_text(chat_id, role, text):
    card_counter[0] += 1
    print(f"\n[CARD#{card_counter[0]} TEXT] {role}: {text[:200]}")

import src.feishu.sender as sender_mod
sender_mod.send_narrator_card = mock_send_narrator
sender_mod.send_npc_card = mock_send_npc
sender_mod.send_npc_text = mock_send_text

import src.engine.game_master as gm_mod
gm_mod.send_narrator_card = mock_send_narrator
gm_mod.send_npc_card = mock_send_npc
gm_mod.send_npc_text = mock_send_text

from src.engine.npc_engine import generate_npc_reply as _orig_npc_reply
async def _debug_npc_reply(state, npc_name, npc_profile, user_text, **kw):
    has_visit = any(h.get("type") == "visit_entry" for h in state.current_talk_history)
    print(f"\n>>> generate_npc_reply: npc={npc_name} has_visit_entry={has_visit} talk_len={len(state.current_talk_history)}")
    reply = await _orig_npc_reply(state, npc_name, npc_profile, user_text, **kw)
    print(f">>> reply[:150]={reply[:150]}")
    return reply

import src.engine.npc_engine as npc_mod
npc_mod.generate_npc_reply = _debug_npc_reply
gm_mod.generate_npc_reply = _debug_npc_reply

CHAT_ID = "debug_bzy"

async def main():
    save_path = f"data/saves/{CHAT_ID}.json"
    if os.path.exists(save_path):
        os.remove(save_path)

    from src.engine.game_master import handle_message, handle_card_action
    from src.engine.state import get_state_manager

    print("=== 开始游戏 ===")
    await handle_message(CHAT_ID, "开始游戏")

    print("\n=== 选择皇帝 ===")
    await handle_card_action(CHAT_ID, {"role_id": "emperor"})

    state = get_state_manager().load(CHAT_ID)
    print(f"\nphase={state.phase} talking_to={state.talking_to} visit_queue={len(state.visit_queue)}")
    visit_npcs = [v.get("npc_id") for v in state.visit_queue]
    print(f"visit_queue: {visit_npcs}")

    for rnd in range(15):
        state = get_state_manager().load(CHAT_ID)
        current = state.talking_to
        if not current:
            print(f"\nROUND {rnd}: 没有对话目标 phase={state.phase}")
            break

        print(f"\n{'='*50}")
        print(f"ROUND {rnd}: talking_to={current} talk_history_len={len(state.current_talk_history)}")
        for i, h in enumerate(state.current_talk_history):
            print(f"  [{i}] type={h.get('type','NONE')} speaker={h.get('speaker')} content={h.get('content','')[:60]}")

        if "毕自严" in current:
            print("\n*** 毕自严！发消息测试 ***")
            await handle_message(CHAT_ID, "户部情况如何？")
            state = get_state_manager().load(CHAT_ID)
            print(f"\n--- 毕自严对话后 talk_history ---")
            for i, h in enumerate(state.current_talk_history):
                print(f"  [{i}] type={h.get('type','NONE')} speaker={h.get('speaker')} content={h.get('content','')[:100]}")
            break

        print(f"  dismiss {current}")
        await handle_message(CHAT_ID, "退下")

asyncio.run(main())
