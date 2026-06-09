"""纯Python CLI入口，替代飞书交互层，零额外token消耗。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

CHAT_ID = "cli_test"

# 全局按钮队列：sender每次发卡片时把按钮追加到这里，主循环从这里读取
_pending_buttons: list[dict[str, Any]] = []
# 最后一批按钮（用于区分哪些是最新卡片的按钮）
_last_card_buttons: list[dict[str, Any]] = []

# ── 终端颜色 ──────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"

HEADER_COLORS = {
    "indigo": BLUE,
    "violet": MAGENTA,
    "turquoise": CYAN,
    "green": GREEN,
    "red": RED,
    "orange": YELLOW,
    "blue": BLUE,
    "yellow": YELLOW,
    "grey": DIM,
    "purple": MAGENTA,
}


def _separator():
    print(f"{DIM}{'─' * 60}{RESET}")


# ── 卡片渲染 ──────────────────────────────────────────

def render_card(card_json: dict[str, Any]) -> list[dict[str, Any]]:
    """将飞书卡片JSON渲染为终端文本，返回收集到的按钮列表。"""

    buttons: list[dict[str, Any]] = []

    header = card_json.get("header", {})
    title = header.get("title", {}).get("content", "")
    template = header.get("template", "")
    color = HEADER_COLORS.get(template, CYAN)

    if title:
        print(f"\n{color}{BOLD}{'═' * 4} {title} {'═' * (54 - len(title))}{RESET}")

    for elem in card_json.get("elements", []):
        tag = elem.get("tag", "")

        if tag == "markdown":
            content = elem.get("content", "")
            print(content)

        elif tag == "hr":
            _separator()

        elif tag == "action":
            for action_item in elem.get("actions", []):
                if action_item.get("tag") == "button":
                    text = action_item.get("text", {}).get("content", "?")
                    value = action_item.get("value", {})
                    buttons.append({"text": text, "value": value})

        elif tag == "div":
            text = elem.get("text", {})
            if isinstance(text, dict):
                print(text.get("content", ""))
            elif isinstance(text, str):
                print(text)

        elif tag == "note":
            note_elems = elem.get("elements", [])
            for ne in note_elems:
                if isinstance(ne, dict):
                    print(f"{DIM}{ne.get('content', '')}{RESET}")

    return buttons


def display_buttons(buttons: list[dict[str, Any]]) -> None:
    """在终端展示按钮菜单。"""

    if not buttons:
        return
    print(f"\n{YELLOW}可选操作：{RESET}")
    for i, btn in enumerate(buttons, 1):
        print(f"  {YELLOW}[{i}]{RESET} {btn['text']}")
    print(f"  {DIM}输入数字选择，或直接输入文字{RESET}")


# ── Sender 替换 ──────────────────────────────────────

async def cli_send_narrator_card(chat_id: str, card_json: dict[str, Any]) -> None:
    buttons = render_card(card_json)
    if buttons:
        _pending_buttons.clear()
        _pending_buttons.extend(buttons)
        _last_card_buttons.clear()
        _last_card_buttons.extend(buttons)


async def cli_send_npc_card(chat_id: str, card_json: dict[str, Any]) -> None:
    buttons = render_card(card_json)
    if buttons:
        _pending_buttons.clear()
        _pending_buttons.extend(buttons)
        _last_card_buttons.clear()
        _last_card_buttons.extend(buttons)


async def cli_send_npc_text(chat_id: str, role_name: str, text: str) -> None:
    print(f"\n{MAGENTA}{BOLD}【{role_name}】{RESET}{text}")


def _patch_sender():
    """Monkey-patch sender模块，替换为CLI版本。"""

    import src.feishu.sender as sender_mod
    sender_mod.send_narrator_card = cli_send_narrator_card
    sender_mod.send_npc_card = cli_send_npc_card
    sender_mod.send_npc_text = cli_send_npc_text

    import src.engine.game_master as gm_mod
    gm_mod.send_narrator_card = cli_send_narrator_card
    gm_mod.send_npc_card = cli_send_npc_card
    gm_mod.send_npc_text = cli_send_npc_text


def _patch_env():
    """填充飞书相关环境变量的假值，避免Settings校验失败。"""

    fake_vars = {
        "NARRATOR_APP_ID": "cli_fake",
        "NARRATOR_APP_SECRET": "cli_fake",
        "NPC_APP_ID": "cli_fake",
        "NPC_APP_SECRET": "cli_fake",
        "FEISHU_VERIFICATION_TOKEN": "cli_fake",
    }
    for k, v in fake_vars.items():
        if not os.environ.get(k):
            os.environ[k] = v


# ── 主循环 ────────────────────────────────────────────

async def main():
    _patch_env()
    _patch_sender()

    from src.engine.game_master import handle_message, handle_card_action

    print(f"{CYAN}{BOLD}")
    print("╔══════════════════════════════════════╗")
    print("║     文字RPG引擎 · CLI测试模式        ║")
    print("╚══════════════════════════════════════╝")
    print(f"{RESET}")
    print(f"{DIM}命令：/quit 退出 | /restart 重开 | /trace on/off 追踪{RESET}")
    _separator()

    # 发送初始消息触发新游戏
    await handle_message(CHAT_ID, "开始游戏")
    if _pending_buttons:
        display_buttons(_pending_buttons)

    while True:
        try:
            user_input = input(f"\n{GREEN}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}再见。{RESET}")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print(f"{DIM}退出游戏。{RESET}")
            break

        if user_input == "/restart":
            save_path = f"data/saves/{CHAT_ID}.json"
            if os.path.exists(save_path):
                os.remove(save_path)
                print(f"{DIM}存档已删除。{RESET}")
            _pending_buttons.clear()
            await handle_message(CHAT_ID, "重新开始")
            if _pending_buttons:
                display_buttons(_pending_buttons)
            continue

        # 数字选择 → 按钮点击
        if user_input.isdigit() and _pending_buttons:
            idx = int(user_input) - 1
            if 0 <= idx < len(_pending_buttons):
                btn = _pending_buttons[idx]
                print(f"{DIM}→ 选择了: {btn['text']}{RESET}")
                _pending_buttons.clear()
                result = await handle_card_action(CHAT_ID, btn["value"])
                if isinstance(result, dict) and "card" in result:
                    new_buttons = render_card(result["card"])
                    if new_buttons:
                        _pending_buttons.extend(new_buttons)
                if _pending_buttons:
                    display_buttons(_pending_buttons)
                continue
            else:
                print(f"{RED}无效选择，请输入 1-{len(_pending_buttons)}{RESET}")
                display_buttons(_pending_buttons)
                continue

        # 文本输入 → handle_message
        _pending_buttons.clear()
        await handle_message(CHAT_ID, user_input)
        if _pending_buttons:
            display_buttons(_pending_buttons)


if __name__ == "__main__":
    asyncio.run(main())
