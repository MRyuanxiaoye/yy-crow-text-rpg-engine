"""游戏交互追踪系统——记录运行时完整因果链，支持事后回溯诊断。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.state import GameState

logger = logging.getLogger(__name__)

TRACE_DIR = Path("data/traces")

_template_registry: dict[str, str] = {}
_template_counter: int = 0


class GameTracer:

    def __init__(self, chat_id: str, enabled: bool = True) -> None:
        self.chat_id = chat_id
        self.enabled = enabled
        safe_id = chat_id.replace("/", "_")
        self._dir = TRACE_DIR / safe_id
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def _current_file(self) -> Path:
        return self._dir / f"{time.strftime('%Y%m%d')}.jsonl"

    def record(self, entry_type: str, function: str, data: dict[str, Any], state_summary: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "chat_id": self.chat_id,
            "type": entry_type,
            "function": function,
            "data": data,
        }
        if state_summary:
            entry["state"] = state_summary
        try:
            with self._current_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("trace写入失败 chat_id=%s function=%s", self.chat_id, function)

    def record_llm_call(
        self,
        function: str,
        template_name: str,
        template_text: str,
        variables: dict[str, Any],
        raw_response: str,
        parsed_response: Any,
        state_summary: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        global _template_counter
        template_id = _template_registry.get(template_name)
        if template_id is None:
            _template_counter += 1
            template_id = f"tpl_{_template_counter}"
            _template_registry[template_name] = template_id
            self.record("prompt_template", function, {
                "template_id": template_id,
                "template_name": template_name,
                "template_text": template_text,
            })
        self.record("llm_call", function, {
            "template_id": template_id,
            "variables": variables,
            "raw_response": raw_response[:3000],
            "parsed_response": parsed_response if isinstance(parsed_response, (dict, list, str, int, float, bool, type(None))) else str(parsed_response)[:1000],
        }, state_summary)

    def record_turn_snapshot(self, state: "GameState") -> None:
        if not self.enabled:
            return
        from src.engine.state import GameState
        from dataclasses import asdict
        snapshot = asdict(state)
        snapshot.pop("conversation_history", None)
        snapshot.pop("current_talk_history", None)
        self.record("turn_snapshot", "turn_boundary", {"full_state": snapshot})

    def clear(self) -> None:
        if self._dir.exists():
            for f in self._dir.glob("*.jsonl"):
                f.unlink()
            logger.info("trace已清空 chat_id=%s", self.chat_id)

    @staticmethod
    def state_summary(state: "GameState") -> dict[str, Any]:
        return {
            "turn": state.turn,
            "game_time": dict(state.game_time) if isinstance(state.game_time, dict) else {},
            "phase": state.phase,
            "talking_to": state.talking_to,
            "court_active": bool(state.court_session.get("active")) if isinstance(state.court_session, dict) else False,
            "settlement_phase": state.settlement_phase,
            "npc_location_keys": list(state.npc_locations.keys()) if isinstance(state.npc_locations, dict) else [],
        }


_tracers: dict[str, GameTracer] = {}


def get_tracer(chat_id: str, enabled: bool = True) -> GameTracer:
    tracer = _tracers.get(chat_id)
    if tracer is None:
        tracer = GameTracer(chat_id, enabled)
        _tracers[chat_id] = tracer
    else:
        tracer.enabled = enabled
    return tracer
