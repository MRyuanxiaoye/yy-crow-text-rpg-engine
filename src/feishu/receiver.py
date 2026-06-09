"""飞书事件接收模块。"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import hashlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi import APIRouter, HTTPException, Request

from src.config import get_settings
from src.feishu.sender import get_sender

logger = logging.getLogger(__name__)

router = APIRouter()

EVENT_CACHE_TTL_SECONDS = 600

_message_handler: Callable[[str, str], Awaitable[None]] | None = None
_card_action_handler: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
_event_cache: dict[str, float] = {}
_cache_lock = asyncio.Lock()


def register_message_handler(handler: Callable[[str, str], Awaitable[None]]) -> None:
    """注册消息处理器（通常为GameMaster.handle_message）。"""

    global _message_handler
    _message_handler = handler


def register_card_action_handler(handler: Callable[[str, dict[str, Any]], Awaitable[None]]) -> None:
    """注册卡片按钮回调处理器。"""

    global _card_action_handler
    _card_action_handler = handler


def _extract_verification_token(payload: dict[str, Any]) -> str | None:
    """从请求体提取verification token。"""

    header = payload.get("header", {})
    if isinstance(header, dict) and header.get("token"):
        return str(header.get("token"))
    if payload.get("token"):
        return str(payload.get("token"))
    return None


async def _is_duplicate_event(event_id: str) -> bool:
    """判断事件是否重复，并在缓存中登记。"""

    now = time.time()
    async with _cache_lock:
        expired_keys = [key for key, ts in _event_cache.items() if now - ts > EVENT_CACHE_TTL_SECONDS]
        for key in expired_keys:
            _event_cache.pop(key, None)

        if event_id in _event_cache:
            return True

        _event_cache[event_id] = now
        return False


def _extract_message_text(message_type: str, content: str) -> str:
    """从飞书消息内容中提取可用文本。"""

    if not content:
        return ""

    try:
        content_obj = json.loads(content)
    except json.JSONDecodeError:
        return content.strip()

    if message_type == "text":
        return str(content_obj.get("text", "")).strip()

    if message_type == "post":
        return json.dumps(content_obj, ensure_ascii=False)

    return json.dumps(content_obj, ensure_ascii=False)


def _extract_sender_open_id(event_payload: dict[str, Any]) -> str | None:
    """提取发送者open_id。"""

    sender = event_payload.get("sender", {})
    sender_id = sender.get("sender_id", {}) if isinstance(sender, dict) else {}
    if isinstance(sender_id, dict):
        open_id = sender_id.get("open_id")
        if open_id:
            return str(open_id)
    return None


async def _dispatch_to_game_master(chat_id: str, user_text: str) -> None:
    """转发消息给GameMaster。"""

    if await _handle_slash_command(chat_id, user_text):
        return

    handler = _message_handler
    if handler is None:
        from src.engine.game_master import handle_message  # 延迟导入，避免循环依赖

        handler = handle_message

    await handler(chat_id, user_text)


async def _handle_slash_command(chat_id: str, user_text: str) -> bool:
    """处理不进入正常游戏路由的文本命令。"""

    command = user_text.strip()
    if command not in {"/支线任务", "/主线任务"}:
        return False

    from src.engine.state import get_state_manager
    from src.feishu.card_builder import build_narration_card
    from src.feishu.sender import send_narrator_card

    state = get_state_manager().load(chat_id)
    if command == "/支线任务":
        content = _format_side_quests(state.side_quests)
        await send_narrator_card(chat_id, build_narration_card("支线任务", content))
        return True

    content = _format_main_objectives(state)
    await send_narrator_card(chat_id, build_narration_card("主线任务", content))
    return True


def _format_side_quests(side_quests: list[dict[str, Any]]) -> str:
    if not side_quests:
        return "暂无支线任务。"

    icons = {
        "active": "⬜",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "➖",
    }
    lines: list[str] = []
    for quest in side_quests:
        if not isinstance(quest, dict):
            continue
        status = str(quest.get("status") or "active")
        icon = icons.get(status, "⬜")
        name = str(quest.get("name") or quest.get("id") or "未命名支线").strip()
        description = str(quest.get("description") or "").strip()
        rewards = str(quest.get("rewards") or "").strip()
        penalties = str(quest.get("penalties") or "").strip()
        lines.append(f"{icon} **{name}**")
        if description:
            lines.append(f"· {description}")
        if rewards:
            lines.append(f"· 成功：{rewards}")
        if penalties:
            lines.append(f"· 风险：{penalties}")
    return "\n".join(lines) if lines else "暂无支线任务。"


def _format_main_objectives(state: Any) -> str:
    objectives = getattr(state, "objectives", None)
    if not isinstance(objectives, list) or not objectives:
        player_goal = getattr(state, "player_goal", {})
        if isinstance(player_goal, dict):
            objectives = []
            core_goal = str(player_goal.get("core_goal") or "").strip()
            if core_goal:
                objectives.append({"name": core_goal, "completed": False})
            raw_items = player_goal.get("milestones") or player_goal.get("objectives") or []
            if isinstance(raw_items, list):
                objectives.extend(raw_items)

    if not isinstance(objectives, list) or not objectives:
        return "暂无主线任务。"

    lines: list[str] = []
    for item in objectives:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("description") or item.get("id") or "目标").strip()
            completed = bool(item.get("completed"))
        else:
            name = str(item).strip()
            completed = False
        if name:
            lines.append(f"{'✅' if completed else '⬜'} {name}")
    return "\n".join(lines) if lines else "暂无主线任务。"


def _decrypt_feishu_payload(encrypt_text: str, encrypt_key: str) -> dict[str, Any]:
    """解密飞书加密事件并返回原始JSON。"""

    if not encrypt_key:
        raise ValueError("missing FEISHU_ENCRYPT_KEY")

    # 飞书加密密钥先做SHA256，得到32字节AES-256密钥
    aes_key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()

    # 密文为base64编码：前16字节IV + 其余AES-CBC密文
    raw_cipher = base64.b64decode(encrypt_text)
    if len(raw_cipher) <= 16:
        raise ValueError("invalid encrypted payload")

    iv = raw_cipher[:16]
    cipher_text = raw_cipher[16:]

    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_plain = decryptor.update(cipher_text) + decryptor.finalize()

    # PKCS7去填充后解析为JSON对象
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plain_bytes = unpadder.update(padded_plain) + unpadder.finalize()
    plain_obj = json.loads(plain_bytes.decode("utf-8"))
    if not isinstance(plain_obj, dict):
        raise ValueError("decrypted payload is not a json object")
    return plain_obj


@router.post("/feishu/event")
async def feishu_event_callback(request: Request) -> dict[str, Any]:
    """统一接收飞书事件回调。"""

    payload = await request.json()
    if isinstance(payload, dict) and "encrypt" in payload:
        settings = get_settings()
        encrypt_text = payload.get("encrypt")
        if not isinstance(encrypt_text, str) or not encrypt_text.strip():
            raise HTTPException(status_code=400, detail="invalid encrypt payload")
        try:
            payload = _decrypt_feishu_payload(encrypt_text, settings.feishu_encrypt_key)
        except Exception as exc:  # noqa: BLE001
            logger.exception("飞书加密消息解密失败")
            raise HTTPException(status_code=400, detail="decrypt payload failed") from exc

    if "challenge" in payload:
        return {"challenge": payload.get("challenge")}

    settings = get_settings()
    verify_token = _extract_verification_token(payload)
    if settings.feishu_verification_token and verify_token != settings.feishu_verification_token:
        raise HTTPException(status_code=403, detail="invalid verification token")

    header = payload.get("header", {})
    event_id = None
    event_type = None
    if isinstance(header, dict):
        event_id = header.get("event_id")
        event_type = header.get("event_type")

    if event_id and await _is_duplicate_event(str(event_id)):
        logger.info("忽略重复事件: %s", event_id)
        return {"ok": True, "deduplicated": True}

    event = payload.get("event", {})
    if not isinstance(event, dict):
        return {"ok": True}

    if event_type and event_type != "im.message.receive_v1":
        return {"ok": True, "ignored": event_type}

    sender_open_id = _extract_sender_open_id(event)
    sender_type = str(event.get("sender", {}).get("sender_type", ""))
    if sender_type and sender_type != "user":
        return {"ok": True, "ignored": "non-user"}

    bot_open_ids = await get_sender().get_bot_open_ids()
    if sender_open_id and sender_open_id in bot_open_ids:
        logger.info("忽略Bot自身消息: %s", sender_open_id)
        return {"ok": True, "ignored": "self-message"}

    message = event.get("message", {})
    if not isinstance(message, dict):
        return {"ok": True, "ignored": "invalid-message"}

    chat_id = str(message.get("chat_id", "")).strip()
    message_type = str(message.get("message_type", "")).strip()
    content = str(message.get("content", ""))
    user_text = _extract_message_text(message_type, content)

    if not chat_id or not user_text:
        return {"ok": True, "ignored": "empty"}

    asyncio.create_task(_dispatch_to_game_master(chat_id, user_text))
    return {"ok": True}


@router.post("/feishu/card_action")
async def feishu_card_action(request: Request) -> dict[str, Any]:
    """接收飞书卡片按钮回调。"""

    payload = await request.json()
    logger.info("卡片回调原始payload keys: %s", list(payload.keys()) if isinstance(payload, dict) else type(payload))

    # 处理加密payload — 依次尝试旁白和NPC的encrypt_key
    if isinstance(payload, dict) and "encrypt" in payload:
        settings = get_settings()
        encrypt_text = payload.get("encrypt")
        if isinstance(encrypt_text, str) and encrypt_text.strip():
            keys_to_try = [settings.feishu_encrypt_key]
            if settings.npc_encrypt_key:
                keys_to_try.append(settings.npc_encrypt_key)
            decrypted = False
            for key in keys_to_try:
                if not key:
                    continue
                try:
                    payload = _decrypt_feishu_payload(encrypt_text, key)
                    decrypted = True
                    break
                except Exception:  # noqa: BLE001
                    continue
            if not decrypted:
                logger.warning("卡片回调所有密钥解密失败")
                return {}

    if "challenge" in payload:
        return {"challenge": payload.get("challenge")}

    logger.info("卡片回调payload: %s", json.dumps(payload, ensure_ascii=False, default=str)[:500])

    # schema 2.0: action和context在event下；旧版在顶层
    event = payload.get("event", {})
    if isinstance(event, dict) and event.get("action"):
        action = event.get("action", {})
        context = event.get("context", {})
    else:
        action = payload.get("action", {})
        context = payload

    if not isinstance(action, dict):
        return {}

    value = action.get("value")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"action": value}

    if not isinstance(value, dict):
        return {}

    # 提取open_chat_id：schema 2.0在event.context中
    open_chat_id = ""
    if isinstance(context, dict):
        open_chat_id = str(context.get("open_chat_id", "")).strip()
    if not open_chat_id:
        open_chat_id = str(payload.get("open_chat_id", "")).strip()

    if not open_chat_id:
        logger.warning("卡片回调缺少chat_id payload=%s", payload)
        return {}

    handler = _card_action_handler
    if handler is None:
        from src.engine.game_master import handle_card_action
        handler = handle_card_action

    action_name = value.get("action", "") if isinstance(value, dict) else ""
    if action_name in {"add_to_backlog", "skip_backlog_item"}:
        try:
            result = await handler(open_chat_id, value)
            if isinstance(result, dict) and result.get("card"):
                return result
        except Exception:
            logger.exception("同步卡片回调处理失败 action=%s", action_name)
        return {}

    asyncio.create_task(handler(open_chat_id, value))
    return {"toast": {"type": "info", "content": "处理中..."}}
