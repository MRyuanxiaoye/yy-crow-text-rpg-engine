"""飞书消息发送模块，支持旁白与NPC双Bot。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

FEISHU_BASE_URL = "https://open.feishu.cn"
TOKEN_REFRESH_GUARD_SECONDS = 120
TOKEN_INVALID_CODES = {99991663, 99991664, 99991665, 99991668}


class FeishuAPIError(RuntimeError):
    """飞书接口调用异常。"""


@dataclass
class TokenState:
    """单个Bot的token状态。"""

    token: str = ""
    expires_at: float = 0.0
    open_id: str | None = None


class FeishuChannel:
    """飞书Bot通道，负责独立token管理与发消息。"""

    def __init__(self, app_id: str, app_secret: str, channel_name: str) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.channel_name = channel_name
        self._state = TokenState()
        self._lock = asyncio.Lock()

    async def _refresh_tenant_token(self) -> str:
        """刷新tenant_access_token并缓存。"""

        url = f"{FEISHU_BASE_URL}/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)

        response.raise_for_status()
        data = response.json()
        code = int(data.get("code", -1))
        if code != 0:
            raise FeishuAPIError(
                f"[{self.channel_name}] 获取tenant_access_token失败: "
                f"code={code}, msg={data.get('msg')}"
            )

        token = data.get("tenant_access_token", "")
        expire = int(data.get("expire", 7200))
        if not token:
            raise FeishuAPIError(f"[{self.channel_name}] tenant_access_token为空")

        self._state.token = token
        self._state.expires_at = time.time() + max(0, expire - TOKEN_REFRESH_GUARD_SECONDS)
        logger.info("[%s] tenant_access_token刷新成功，expire=%s", self.channel_name, expire)
        return token

    async def get_tenant_token(self, force_refresh: bool = False) -> str:
        """获取可用token，必要时自动刷新。"""

        now = time.time()
        if (not force_refresh) and self._state.token and now < self._state.expires_at:
            return self._state.token

        async with self._lock:
            now = time.time()
            if (not force_refresh) and self._state.token and now < self._state.expires_at:
                return self._state.token
            return await self._refresh_tenant_token()

    async def get_bot_open_id(self) -> str | None:
        """获取Bot自身open_id，用于过滤机器人自发消息。"""

        if self._state.open_id:
            return self._state.open_id

        token = await self.get_tenant_token()
        url = f"{FEISHU_BASE_URL}/open-apis/bot/v3/info"
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)

        if response.status_code >= 400:
            logger.warning(
                "[%s] 获取bot信息失败，status=%s body=%s",
                self.channel_name,
                response.status_code,
                response.text,
            )
            return None

        data = response.json()
        if int(data.get("code", -1)) != 0:
            logger.warning(
                "[%s] 获取bot信息失败: code=%s msg=%s",
                self.channel_name,
                data.get("code"),
                data.get("msg"),
            )
            return None

        bot_data = data.get("data", {})
        open_id = (
            bot_data.get("open_id")
            or bot_data.get("bot_open_id")
            or (bot_data.get("bot", {}) if isinstance(bot_data.get("bot"), dict) else {}).get(
                "open_id"
            )
        )
        self._state.open_id = open_id
        return open_id

    @staticmethod
    def _is_token_invalid(resp_json: dict[str, Any]) -> bool:
        """判断错误是否由token失效导致。"""

        code = int(resp_json.get("code", -1))
        msg = str(resp_json.get("msg", ""))
        return code in TOKEN_INVALID_CODES or "tenant_access_token" in msg

    async def send_message(self, chat_id: str, msg_type: str, content: dict[str, Any]) -> None:
        """发送飞书消息，遇到token失效时自动重试一次。"""

        url = f"{FEISHU_BASE_URL}/open-apis/im/v1/messages"
        params = {"receive_id_type": "chat_id"}
        payload = {
            "receive_id": chat_id,
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
        }

        for attempt in range(2):
            token = await self.get_tenant_token(force_refresh=(attempt == 1))
            headers = {"Authorization": f"Bearer {token}"}

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, params=params, json=payload, headers=headers)

            response.raise_for_status()
            data = response.json()
            code = int(data.get("code", -1))
            if code == 0:
                return

            if attempt == 0 and self._is_token_invalid(data):
                logger.warning("[%s] token疑似失效，尝试刷新后重发", self.channel_name)
                continue

            raise FeishuAPIError(
                f"[{self.channel_name}] 发送消息失败: code={code}, msg={data.get('msg')}"
            )

        raise FeishuAPIError(f"[{self.channel_name}] 发送消息失败，重试后仍失败")


class FeishuSender:
    """飞书发送器，封装旁白与NPC双通道。"""

    def __init__(self) -> None:
        settings = get_settings()
        self._narrator_channel = FeishuChannel(
            app_id=settings.narrator_app_id,
            app_secret=settings.narrator_app_secret,
            channel_name="narrator",
        )
        self._npc_channel = FeishuChannel(
            app_id=settings.npc_app_id,
            app_secret=settings.npc_app_secret,
            channel_name="npc",
        )

    async def send_narrator_card(self, chat_id: str, card_json: dict[str, Any]) -> None:
        """使用旁白Bot发送交互卡片。"""

        await self._narrator_channel.send_message(
            chat_id=chat_id,
            msg_type="interactive",
            content=card_json,
        )

    async def send_npc_text(self, chat_id: str, role_name: str, text: str) -> None:
        """使用NPC Bot发送文本消息。"""

        await self._npc_channel.send_message(
            chat_id=chat_id,
            msg_type="text",
            content={"text": f"【{role_name}】{text}"},
        )

    async def send_npc_card(self, chat_id: str, card_json: dict[str, Any]) -> None:
        """使用NPC Bot发送交互卡片。"""

        await self._npc_channel.send_message(
            chat_id=chat_id,
            msg_type="interactive",
            content=card_json,
        )

    async def get_bot_open_ids(self) -> set[str]:
        """获取两个Bot的open_id集合。"""

        ids: set[str] = set()
        narrator_open_id = await self._narrator_channel.get_bot_open_id()
        npc_open_id = await self._npc_channel.get_bot_open_id()
        if narrator_open_id:
            ids.add(narrator_open_id)
        if npc_open_id:
            ids.add(npc_open_id)
        return ids


_sender_instance: FeishuSender | None = None


def get_sender() -> FeishuSender:
    """获取全局FeishuSender单例。"""

    global _sender_instance
    if _sender_instance is None:
        _sender_instance = FeishuSender()
    return _sender_instance


async def send_narrator_card(chat_id: str, card_json: dict[str, Any]) -> None:
    """用旁白Bot发送卡片消息。"""

    await get_sender().send_narrator_card(chat_id, card_json)


async def send_npc_card(chat_id: str, card_json: dict[str, Any]) -> None:
    """用NPC Bot发送卡片消息。"""

    await get_sender().send_npc_card(chat_id, card_json)


async def send_npc_text(chat_id: str, role_name: str, text: str) -> None:
    """用NPC Bot发送文本消息。"""

    await get_sender().send_npc_text(chat_id, role_name, text)


async def send_npc_text(chat_id: str, role_name: str, text: str) -> None:
    """用NPC Bot发送文本消息。"""

    await get_sender().send_npc_text(chat_id, role_name, text)
