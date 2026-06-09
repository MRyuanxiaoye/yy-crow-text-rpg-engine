"""LLM 客户端封装：DeepSeek（对话生成）+ Claude（演化推进判定）。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)

_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_SET_SYNTAX_PATTERN = re.compile(r'\{(\s*"(?:[^"\\]|\\.)*"\s*)\}')


class LLMClient:
    """统一LLM调用封装，默认使用DeepSeek。"""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.deepseek_model
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    async def chat(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
    ) -> str:
        """基础对话接口，返回纯文本。"""

        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    async def chat_json(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """JSON对话接口，自动清洗markdown代码块并解析。"""

        raw_text: str
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            raw_text = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_json使用response_format失败，回退普通模式: %s", exc)
            raw_text = await self.chat(system_prompt, user_content, temperature=temperature)

        parsed = self._try_parse_json(raw_text)
        if parsed is not None:
            return parsed

        # 修复常见LLM JSON错误：Python set语法 {"text"} -> "text"
        repaired = _SET_SYNTAX_PATTERN.sub(r"\1", raw_text)
        if repaired != raw_text:
            parsed = self._try_parse_json(repaired)
            if parsed is not None:
                return parsed

        match = _JSON_BLOCK_PATTERN.search(raw_text)
        if match:
            parsed = self._try_parse_json(match.group(1).strip())
            if parsed is not None:
                return parsed

        raise ValueError(f"LLM返回内容不是有效JSON: {raw_text}")

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """尝试解析JSON对象。"""

        if not text:
            return None

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
            return None
        except json.JSONDecodeError:
            return None


class ClaudeClient:
    """Claude客户端，用于演化推进判定等需要强推理的任务。"""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.claude_model
        self._client = AsyncAnthropic(
            api_key=settings.claude_api_key,
            base_url=settings.claude_base_url,
        )

    async def chat(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.5,
    ) -> str:
        """基础对话接口，返回纯文本。"""

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        content = response.content[0].text if response.content else ""
        return content.strip()

    async def chat_json(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """JSON对话接口，要求Claude返回JSON并解析。"""

        full_system = f"{system_prompt}\n\n请严格以JSON格式回复，不要包含其他内容。"
        raw_text = await self.chat(full_system, user_content, temperature=temperature)

        parsed = LLMClient._try_parse_json(raw_text)
        if parsed is not None:
            return parsed

        match = _JSON_BLOCK_PATTERN.search(raw_text)
        if match:
            parsed = LLMClient._try_parse_json(match.group(1).strip())
            if parsed is not None:
                return parsed

        raise ValueError(f"Claude返回内容不是有效JSON: {raw_text}")


_client_instance: LLMClient | None = None
_claude_instance: ClaudeClient | None = None


def get_llm_client() -> LLMClient:
    """返回全局DeepSeek客户端单例。"""

    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance


def get_claude_client() -> ClaudeClient:
    """返回全局Claude客户端单例。"""

    global _claude_instance
    if _claude_instance is None:
        _claude_instance = ClaudeClient()
    return _claude_instance
