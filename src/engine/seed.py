"""后果种子机制。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

from src.engine.dice import roll_3d6
from src.engine.state import GameState

SEVERITY_CHANCES = {
    "low": 2,
    "medium": 3,
    "high": 4,
}

_SEVERITY_ALIASES = {
    "低": "low",
    "轻微": "low",
    "小": "low",
    "1": "low",
    "中": "medium",
    "中等": "medium",
    "普通": "medium",
    "2": "medium",
    "高": "high",
    "严重": "high",
    "重大": "high",
    "3": "high",
    "4": "high",
}


@dataclass
class ConsequenceSeed:
    """一颗延迟后果种子。"""

    seed_id: str
    source: str
    severity: str
    directions: list[str]
    remaining_chances: int
    created_at: dict[str, int]
    triggered: bool = False

    @classmethod
    def from_mapping(cls, payload: MutableMapping[str, Any]) -> "ConsequenceSeed":
        """从状态中的 dict 恢复种子对象，并保留回写引用。"""

        severity = _normalize_severity(payload.get("severity", "low"))
        remaining_chances = _safe_int(
            payload.get("remaining_chances", SEVERITY_CHANCES[severity]),
            SEVERITY_CHANCES[severity],
        )
        seed = cls(
            seed_id=str(payload.get("seed_id") or _generate_seed_id()),
            source=str(payload.get("source", "") or ""),
            severity=severity,
            directions=_normalize_directions(payload.get("directions")),
            remaining_chances=max(0, remaining_chances),
            created_at=_normalize_time(payload.get("created_at")),
            triggered=bool(payload.get("triggered", False)),
        )
        seed._raw = payload
        seed._sync_raw()
        return seed

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化的状态数据。"""

        return {
            "seed_id": self.seed_id,
            "source": self.source,
            "severity": self.severity,
            "directions": list(self.directions),
            "remaining_chances": self.remaining_chances,
            "created_at": dict(self.created_at),
            "triggered": self.triggered,
        }

    def _sync_raw(self) -> None:
        """把对象字段回写到原始状态 dict。"""

        raw = getattr(self, "_raw", None)
        if raw is not None:
            raw.update(self.to_dict())


def plant_seed(state: GameState, source: str, severity: str, directions: list[str]) -> ConsequenceSeed:
    """创建后果种子并加入游戏状态。"""

    normalized_severity = _normalize_severity(severity)
    seed = ConsequenceSeed(
        seed_id=_generate_seed_id(),
        source=str(source).strip(),
        severity=normalized_severity,
        directions=_normalize_directions(directions),
        remaining_chances=SEVERITY_CHANCES[normalized_severity],
        created_at=dict(state.game_time),
        triggered=False,
    )
    raw_seed = seed.to_dict()
    seed._raw = raw_seed
    state.consequence_seeds.append(raw_seed)
    return seed


def check_seeds(
    state: GameState,
    candidate_filter_fn: Callable[..., bool],
) -> list[ConsequenceSeed]:
    """筛选当前时机适合触发的存活种子，返回候选池。"""

    candidates: list[ConsequenceSeed] = []
    for seed in get_active_seeds(state):
        if _passes_filter(candidate_filter_fn, state, seed):
            candidates.append(seed)
    return candidates


def trigger_seed_selection(candidates: list[ConsequenceSeed]) -> ConsequenceSeed | None:
    """候选池非空时，通过 3d6 掷骰随机选择一颗种子触发。"""

    if not candidates:
        return None
    _, total = roll_3d6()
    chosen = candidates[(total - 3) % len(candidates)]
    chosen.triggered = True
    chosen._sync_raw()
    return chosen


def consume_unchosen(candidates: list[ConsequenceSeed], chosen: ConsequenceSeed | None) -> list[str]:
    """消耗未被选中候选种子的存活机会，返回消亡种子 ID。"""

    expired_ids: list[str] = []
    chosen_id = chosen.seed_id if chosen is not None else ""
    for seed in candidates:
        if seed.seed_id == chosen_id:
            continue
        seed.remaining_chances = max(0, seed.remaining_chances - 1)
        seed._sync_raw()
        if seed.remaining_chances <= 0:
            expired_ids.append(seed.seed_id)
    return expired_ids


def get_active_seeds(state: GameState) -> list[ConsequenceSeed]:
    """获取所有仍然存活、尚未触发的后果种子。"""

    active: list[ConsequenceSeed] = []
    for item in state.consequence_seeds:
        seed = _coerce_seed(item)
        if seed is None:
            continue
        if seed.triggered or seed.remaining_chances <= 0:
            continue
        active.append(seed)
    return active


def _coerce_seed(item: Any) -> ConsequenceSeed | None:
    """兼容 dict 和已构造的 ConsequenceSeed。"""

    if isinstance(item, ConsequenceSeed):
        return item
    if isinstance(item, MutableMapping):
        return ConsequenceSeed.from_mapping(item)
    return None


def _passes_filter(candidate_filter_fn: Callable[..., bool], state: GameState, seed: ConsequenceSeed) -> bool:
    """兼容 filter(seed) 与 filter(state, seed) 两种写法。"""

    try:
        return bool(candidate_filter_fn(seed))
    except TypeError:
        return bool(candidate_filter_fn(state, seed))


def _generate_seed_id() -> str:
    """生成短随机种子 ID。"""

    return f"seed_{random.randrange(16**8):08x}"


def _normalize_severity(severity: Any) -> str:
    """规范化严重度为 low/medium/high。"""

    text = str(severity or "low").strip().lower()
    if text in SEVERITY_CHANCES:
        return text
    return _SEVERITY_ALIASES.get(text, "low")


def _normalize_directions(directions: Any) -> list[str]:
    """清洗 AI 生成的后果方向列表。"""

    if not isinstance(directions, list):
        return []
    return [str(direction).strip() for direction in directions if str(direction).strip()]


def _normalize_time(value: Any) -> dict[str, int]:
    """清洗游戏时间字段。"""

    if not isinstance(value, dict):
        return {}
    return {str(key): _safe_int(item) for key, item in value.items()}


def _safe_int(value: Any, default: int = 0) -> int:
    """安全转换整数。"""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


__all__ = [
    "ConsequenceSeed",
    "SEVERITY_CHANCES",
    "check_seeds",
    "consume_unchosen",
    "get_active_seeds",
    "plant_seed",
    "trigger_seed_selection",
]
