"""角色成长系统。"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.engine.dice import RESULT_CRITICAL_SUCCESS, RESULT_SUCCESS
from src.engine.dimension import (
    CHARACTER_DIMENSIONS,
    DEFAULT_DIMENSION_VALUE,
    EXTENSION_DIMENSIONS,
    RELATION_NUMERIC_DIMENSIONS,
    WORLD_STATE_DIMENSIONS,
    DimensionReference,
    get_dimension_value,
    RelationDimensions,
    update_dimension_value,
)
from src.engine.state import GameState

ACTION_GROWTH_CHANCE = 0.2
LEGACY_METRIC_DIMENSION_MAP = {
    "国库": "财政",
    "军力": "兵力",
    "边防": "兵力",
    "民心": "民心",
    "朝廷稳定": "派系势力",
}


@dataclass
class GrowthRecord:
    """一次成长事件记录。"""

    time: dict[str, int]
    action_type: str
    dimension: str
    delta: int
    reason: str


@dataclass(frozen=True)
class _DimensionChange:
    """内部使用的维度变化结构。"""

    name: str
    delta: int
    category: str = "auto"
    npc_id: str = ""
    reason: str = "决策成长"


def apply_decision_growth(state: GameState, decision_effects: dict) -> list[GrowthRecord]:
    """应用决策直接带来的维度变化，并记录成长日志。

    决策成长信任 AI 给出的结构化效果：角色、世界、扩展和关系维度均可由此入口应用。
    """

    records: list[GrowthRecord] = []
    for change in _iter_dimension_changes(decision_effects):
        if change.delta == 0:
            continue
        try:
            actual_delta = _apply_dimension_delta(state, change)
        except (KeyError, TypeError, ValueError):
            continue
        if actual_delta == 0:
            continue
        record = GrowthRecord(
            time=dict(state.game_time),
            action_type="decision",
            dimension=change.name,
            delta=actual_delta,
            reason=change.reason,
        )
        _append_growth_record(state, record)
        records.append(record)
    return records


def apply_action_growth(state: GameState, result_tier: str, main_dimension: str) -> GrowthRecord | None:
    """根据行动判定结果尝试提升主角色维度。

    只有成功或大成功会触发 20% 成长检定；角色维度只涨不降。
    """

    dimension_name = str(main_dimension).strip()
    if not dimension_name or dimension_name not in CHARACTER_DIMENSIONS:
        return None
    if result_tier not in {RESULT_SUCCESS, RESULT_CRITICAL_SUCCESS}:
        return None
    if random.random() >= ACTION_GROWTH_CHANCE:
        return None

    reference = DimensionReference(name=dimension_name, category="character")
    state.dimensions.character.setdefault(dimension_name, DEFAULT_DIMENSION_VALUE)
    old_value = get_dimension_value(state.dimensions, reference)
    new_value = update_dimension_value(state.dimensions, reference, 1)
    actual_delta = max(0, new_value - old_value)
    if actual_delta <= 0:
        return None

    record = GrowthRecord(
        time=dict(state.game_time),
        action_type="action",
        dimension=dimension_name,
        delta=actual_delta,
        reason=f"{result_tier}后通过行动成长检定",
    )
    _append_growth_record(state, record)
    return record


def _apply_dimension_delta(state: GameState, change: _DimensionChange) -> int:
    """调用维度系统应用增量，返回实际变化值。"""

    reference = DimensionReference(name=change.name, category=change.category, npc_id=change.npc_id)
    _ensure_dimension_initialized(state, reference)
    if _resolve_growth_category(reference) == "character" and change.delta < 0:
        return 0
    old_value = get_dimension_value(state.dimensions, reference)
    new_value = update_dimension_value(state.dimensions, reference, change.delta)
    return new_value - old_value


def _ensure_dimension_initialized(state: GameState, reference: DimensionReference) -> None:
    """为合法但尚未出现的维度补默认值。"""

    category = _resolve_growth_category(reference)
    if category == "character" and reference.name in CHARACTER_DIMENSIONS:
        state.dimensions.character.setdefault(reference.name, DEFAULT_DIMENSION_VALUE)
    elif category == "world" and reference.name in WORLD_STATE_DIMENSIONS:
        state.dimensions.world.setdefault(reference.name, DEFAULT_DIMENSION_VALUE)
    elif category == "extension" and reference.name in EXTENSION_DIMENSIONS:
        state.dimensions.extensions.setdefault(reference.name, DEFAULT_DIMENSION_VALUE)
    elif category == "relation_numeric" and reference.name in RELATION_NUMERIC_DIMENSIONS and reference.npc_id:
        relation = state.dimensions.relations.setdefault(reference.npc_id, RelationDimensions())
        relation.values.setdefault(reference.name, DEFAULT_DIMENSION_VALUE)


def _resolve_growth_category(reference: DimensionReference) -> str:
    """解析成长系统关心的维度类型。"""

    if reference.category != "auto":
        return reference.category
    if reference.name in CHARACTER_DIMENSIONS:
        return "character"
    if reference.name in WORLD_STATE_DIMENSIONS:
        return "world"
    if reference.name in EXTENSION_DIMENSIONS:
        return "extension"
    if reference.name in RELATION_NUMERIC_DIMENSIONS:
        return "relation_numeric"
    return "auto"


def _iter_dimension_changes(effects: Mapping[str, Any] | None) -> list[_DimensionChange]:
    """兼容多种 effects 写法，抽取维度变化。"""

    if not isinstance(effects, Mapping):
        return []

    changes: list[_DimensionChange] = []
    for key in ("dimensions", "dimension_delta", "dimension_deltas"):
        payload = effects.get(key)
        if isinstance(payload, Mapping):
            changes.extend(_changes_from_payload(payload))
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    change = _change_from_item(item)
                    if change is not None:
                        changes.append(change)

    # 兼容模板中的 role_dimensions/world_dimensions 写法。
    changes.extend(_changes_from_group(effects.get("role_dimensions"), "character", "决策成长"))
    changes.extend(_changes_from_group(effects.get("character"), "character", "决策成长"))
    changes.extend(_changes_from_group(effects.get("角色维度"), "character", "决策成长"))
    changes.extend(_changes_from_group(effects.get("world_dimensions"), "world", "决策成长"))
    changes.extend(_changes_from_group(effects.get("world"), "world", "决策成长"))
    changes.extend(_changes_from_group(effects.get("世界维度"), "world", "决策成长"))

    # 兼容旧 metrics：历史实现将其视作世界维度变化。
    changes.extend(_changes_from_group(effects.get("metrics"), "auto", "决策成长"))
    return changes


def _changes_from_payload(payload: Mapping[str, Any]) -> list[_DimensionChange]:
    """从 dimensions 风格载荷中抽取维度变化。"""

    changes: list[_DimensionChange] = []
    category_keys = {
        "character": "character",
        "role_dimensions": "character",
        "角色维度": "character",
        "world": "world",
        "world_dimensions": "world",
        "世界维度": "world",
        "extensions": "extension",
        "extension": "extension",
        "扩展维度": "extension",
    }
    for key, category in category_keys.items():
        changes.extend(_changes_from_group(payload.get(key), category, "决策成长"))

    relations = payload.get("relations") or payload.get("relation_numeric") or payload.get("关系维度")
    if isinstance(relations, Mapping):
        for npc_id, relation_patch in relations.items():
            changes.extend(_relation_changes(str(npc_id), relation_patch))

    change = _change_from_item(payload)
    if change is not None:
        changes.append(change)
    return changes


def _changes_from_group(group: Any, category: str, default_reason: str) -> list[_DimensionChange]:
    """从 {维度名: 增量} 分组中抽取变化。"""

    if not isinstance(group, Mapping):
        return []
    changes: list[_DimensionChange] = []
    for name, raw_delta in group.items():
        delta = _safe_int(raw_delta)
        if delta == 0:
            continue
        dimension_name = str(name)
        dimension_category = category
        if category == "auto" and dimension_name in LEGACY_METRIC_DIMENSION_MAP:
            dimension_name = LEGACY_METRIC_DIMENSION_MAP[dimension_name]
            dimension_category = "world"
        changes.append(_DimensionChange(name=dimension_name, delta=delta, category=dimension_category, reason=default_reason))
    return changes


def _relation_changes(npc_id: str, relation_patch: Any) -> list[_DimensionChange]:
    """从 NPC 关系维度补丁中抽取数值变化。"""

    if not isinstance(relation_patch, Mapping):
        return []
    value_patch = relation_patch.get("values") if isinstance(relation_patch.get("values"), Mapping) else relation_patch
    changes: list[_DimensionChange] = []
    for name, raw_delta in value_patch.items():
        if name in {"values", "tags", "tags_add", "tags_remove"}:
            continue
        delta = _safe_int(raw_delta)
        if delta == 0:
            continue
        changes.append(
            _DimensionChange(
                name=str(name),
                delta=delta,
                category="relation_numeric",
                npc_id=npc_id,
                reason="决策成长",
            )
        )
    return changes


def _change_from_item(item: Mapping[str, Any]) -> _DimensionChange | None:
    """从单条 {name, delta} 变化中抽取结构。"""

    if not {"name", "delta"}.issubset(item.keys()):
        return None
    name = str(item.get("name", "")).strip()
    if not name:
        return None
    delta = _safe_int(item.get("delta", 0))
    if delta == 0:
        return None
    return _DimensionChange(
        name=name,
        delta=delta,
        category=str(item.get("category", "auto") or "auto"),
        npc_id=str(item.get("npc_id", "") or ""),
        reason=str(item.get("reason", "决策成长") or "决策成长"),
    )


def _append_growth_record(state: GameState, record: GrowthRecord) -> None:
    """把成长记录以可序列化 dict 形式写入状态。"""

    state.growth_log.append(asdict(record))


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
    "ACTION_GROWTH_CHANCE",
    "GrowthRecord",
    "apply_action_growth",
    "apply_decision_growth",
]
