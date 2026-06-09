"""游戏状态管理与存档读写。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .dimension import (
    CHARACTER_DIMENSIONS,
    DEFAULT_DIMENSION_VALUE,
    EXTENSION_DIMENSIONS,
    RELATION_NUMERIC_DIMENSIONS,
    RELATION_TAG_DIMENSIONS,
    WORLD_STATE_DIMENSIONS,
    DimensionReference,
    DimensionState,
    RelationDimensions,
    clamp_dimension_value,
    update_dimension_value,
)


logger = logging.getLogger(__name__)


LEGACY_METRIC_DIMENSION_MAP: dict[str, str] = {
    "国库": "财政",
    "军力": "兵力",
    "边防": "兵力",
    "民心": "民心",
    "朝廷稳定": "派系势力",
}


@dataclass
class GameState:
    """游戏运行态数据。"""

    script_id: str = "mingmo"
    phase: str = "free_dialogue"
    turn: int = 0
    game_date: str = "第一天"
    player_role: str = "emperor"
    game_length: str = "medium"
    dimensions: DimensionState = field(default_factory=lambda: DimensionState(
        character={"武力": 5, "智谋": 5, "口才": 5, "意志": 5, "感知": 5, "魅力": 5},
        world={"财政": 4, "兵力": 5, "民心": 5, "士气": 5, "派系势力": 4},
    ))
    storylines: dict[str, dict] = field(default_factory=dict)
    active_npcs: dict[str, dict] = field(default_factory=dict)
    accessible_npcs: list[str] = field(default_factory=list)
    cast: dict[str, str] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)
    decisions_completed: list[str] = field(default_factory=list)
    npc_statuses: dict[str, str] = field(default_factory=dict)
    current_scene: dict = field(default_factory=dict)
    conversation_history: list[dict] = field(default_factory=list)

    # 时间系统：连续时间流替代章节推进。
    game_time: dict[str, int] = field(default_factory=lambda: {"year": 1, "month": 8})
    turns_since_last_advance: int = 0

    # 决策确认制与玩家待执行行动。
    pending_actions: list[dict] = field(default_factory=list)
    turns_without_action: int = 0
    advance_queue: list[dict] = field(default_factory=list)

    # 新回合制待办与结算阶段。
    backlog: list[dict[str, Any]] = field(default_factory=list)
    delayed_queue: list[dict[str, Any]] = field(default_factory=list)
    turn_agenda: list[dict[str, Any]] = field(default_factory=list)
    settlement_phase: str = "free"

    # 目标驱动系统。
    player_goal: dict[str, Any] = field(default_factory=dict)
    pressure_sources: list[dict] = field(default_factory=list)
    consequence_seeds: list[dict] = field(default_factory=list)
    information_pool: list[dict] = field(default_factory=list)
    growth_log: list[dict] = field(default_factory=list)
    previous_world_dimensions: dict = field(default_factory=dict)
    side_quests: list = field(default_factory=list)

    # 事件系统。
    active_events: list[dict] = field(default_factory=list)
    event_history: list[str] = field(default_factory=list)
    event_cooldowns: dict[str, int] = field(default_factory=dict)

    # 对话模式。
    talking_to: str = ""
    conversation_topic: str = ""
    present_npcs: list[str] = field(default_factory=list)
    conversation_initiator: str = ""
    npc_join_round: dict[str, int] = field(default_factory=dict)
    visit_queue: list[dict] = field(default_factory=list)
    gm_hints_enabled: bool = True
    dialogue_summary: dict[str, Any] = field(default_factory=dict)
    current_talk_history: list[dict] = field(default_factory=list)

    # 长期记忆。
    npc_memory: dict[str, dict] = field(default_factory=dict)
    world_memory: list[dict] = field(default_factory=list)
    npc_memories: dict[str, list[dict]] = field(default_factory=dict)
    world_summary: str = ""
    npc_locations: dict[str, dict] = field(default_factory=dict)
    pending_gm_interpretation: dict[str, Any] = field(default_factory=dict)
    court_session: dict[str, Any] = field(default_factory=dict)
    trace_enabled: bool = True
    board_regions: dict[str, Any] = field(default_factory=dict)
    discovered_clues: list[dict[str, Any]] = field(default_factory=list)
    promises: list[dict] = field(default_factory=list)
    confirmed_decisions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """导出可序列化状态字典。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GameState:
        """从字典恢复状态，保持与存档读取一致。"""

        manager = get_state_manager()
        return manager._state_from_payload(payload)


class StateManager:
    """状态管理器，负责加载、保存、效果应用与状态描述。"""

    def __init__(self, save_dir: str = "data/saves") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_path(self, chat_id: str) -> Path:
        """根据 chat_id 生成存档路径。"""

        safe_chat_id = chat_id.replace("/", "_")
        return self.save_dir / f"{safe_chat_id}.json"

    def load(self, chat_id: str) -> GameState:
        """从 JSON 存档加载状态，不存在时返回默认状态。"""

        path = self._save_path(chat_id)
        if not path.exists():
            return GameState()

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        history_all = payload.get("conversation_history_all")
        if not isinstance(history_all, list):
            history_all = payload.get("conversation_history", [])
        if not isinstance(history_all, list):
            history_all = []

        if "metrics" in payload and "dimensions" not in payload and "dimension_state" not in payload:
            logger.warning("读取到旧格式存档 metrics，已按新维度池进行降级迁移: %s", path)

        return self._state_from_payload(payload, history_all=history_all)

    def _state_from_payload(self, payload: Mapping[str, Any], history_all: list[dict] | None = None) -> GameState:
        """把原始存档映射为 GameState。"""

        if history_all is None:
            history_all = payload.get("conversation_history_all")
            if not isinstance(history_all, list):
                history_all = payload.get("conversation_history", [])
            if not isinstance(history_all, list):
                history_all = []

        return GameState(
            script_id=str(payload.get("script_id", "mingmo")),
            phase=str(payload.get("phase", "free_dialogue")),
            turn=self._safe_int(payload.get("turn", 0)),
            game_date=str(payload.get("game_date", "第一天")),
            player_role=str(payload.get("player_role", "emperor")),
            game_length=str(payload.get("game_length", "medium")),
            dimensions=self._load_dimensions(payload),
            storylines=self._as_dict(payload.get("storylines")),
            active_npcs=self._as_dict(payload.get("active_npcs")),
            accessible_npcs=[str(item) for item in self._as_list(payload.get("accessible_npcs"))],
            cast={str(k): str(v) for k, v in self._as_dict(payload.get("cast")).items()},
            decisions=self._as_list(payload.get("decisions")),
            decisions_completed=[str(item) for item in self._as_list(payload.get("decisions_completed"))],
            npc_statuses={str(k): str(v) for k, v in self._as_dict(payload.get("npc_statuses")).items()},
            current_scene=self._as_dict(payload.get("current_scene")),
            conversation_history=history_all[-30:],
            game_time=self._as_int_dict(payload.get("game_time"), {"year": 1, "month": 8}),
            turns_since_last_advance=self._safe_int(payload.get("turns_since_last_advance", 0)),
            pending_actions=self._as_dict_list(payload.get("pending_actions")),
            turns_without_action=self._safe_int(payload.get("turns_without_action", 0)),
            advance_queue=self._normalize_advance_queue(payload.get("advance_queue")),
            backlog=self._normalize_backlog(payload.get("backlog", payload.get("pending_backlog"))),
            delayed_queue=self._normalize_delayed_queue(payload.get("delayed_queue")),
            turn_agenda=self._normalize_agenda(payload.get("turn_agenda")),
            settlement_phase=str(payload.get("settlement_phase", payload.get("phase_marker", "free"))),
            player_goal=self._as_dict(payload.get("player_goal")),
            pressure_sources=self._as_dict_list(payload.get("pressure_sources")),
            consequence_seeds=self._normalize_consequence_seeds(payload.get("consequence_seeds")),
            information_pool=self._normalize_information_pool(payload.get("information_pool")),
            growth_log=self._as_dict_list(payload.get("growth_log")),
            previous_world_dimensions=self._as_int_dict(payload.get("previous_world_dimensions"), {}),
            side_quests=self._normalize_side_quests(payload.get("side_quests")),
            active_events=self._as_dict_list(payload.get("active_events")),
            event_history=[str(item) for item in self._as_list(payload.get("event_history"))],
            event_cooldowns=self._as_int_dict(payload.get("event_cooldowns"), {}),
            talking_to=str(payload.get("talking_to", "")),
            conversation_topic=str(payload.get("conversation_topic", "")),
            present_npcs=[str(item) for item in self._as_list(payload.get("present_npcs")) if str(item).strip()],
            conversation_initiator=str(payload.get("conversation_initiator", "")),
            npc_join_round={str(k): self._safe_int(v) for k, v in self._as_dict(payload.get("npc_join_round")).items()},
            visit_queue=self._as_dict_list(payload.get("visit_queue")),
            gm_hints_enabled=bool(payload.get("gm_hints_enabled", True)),
            dialogue_summary=self._as_dict(payload.get("dialogue_summary")),
            current_talk_history=self._as_dict_list(payload.get("current_talk_history")),
            npc_memory=self._normalize_npc_memory(payload.get("npc_memory"), payload.get("npc_memories")),
            world_memory=self._as_dict_list(payload.get("world_memory")),
            npc_memories={
                str(npc_id): self._as_dict_list(memories)
                for npc_id, memories in self._as_dict(payload.get("npc_memories")).items()
            },
            world_summary=str(payload.get("world_summary", "")),
            npc_locations={str(k): dict(v) if isinstance(v, dict) else {} for k, v in self._as_dict(payload.get("npc_locations")).items()},
            pending_gm_interpretation=self._as_dict(payload.get("pending_gm_interpretation")),
            court_session=self._as_dict(payload.get("court_session")),
            trace_enabled=bool(payload.get("trace_enabled", True)),
            board_regions=self._as_dict(payload.get("board_regions")),
            discovered_clues=self._as_dict_list(payload.get("discovered_clues")),
            promises=self._as_dict_list(payload.get("promises")),
            confirmed_decisions=self._as_dict_list(payload.get("confirmed_decisions")),
        )

    def save(self, chat_id: str, state: GameState) -> None:
        """将状态保存到 JSON，额外保留完整对话历史。"""

        path = self._save_path(chat_id)

        existing_history_all: list[dict] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                existing_payload = json.load(f)
            existing_history = existing_payload.get("conversation_history_all")
            if isinstance(existing_history, list):
                existing_history_all = existing_history
            elif isinstance(existing_payload.get("conversation_history"), list):
                existing_history_all = list(existing_payload["conversation_history"])

        # 以最近窗口对齐，避免重复追加。
        merged_history = self._merge_history(existing_history_all, state.conversation_history)

        data = state.to_dict()
        data["dimensions"] = self._dump_dimensions(state.dimensions)
        data["conversation_history"] = merged_history[-30:]
        data["conversation_history_all"] = merged_history
        data["backlog"] = self._normalize_backlog(data.get("backlog"))
        data["delayed_queue"] = self._normalize_delayed_queue(data.get("delayed_queue"))
        data["turn_agenda"] = self._normalize_agenda(data.get("turn_agenda"))
        data["side_quests"] = self._normalize_side_quests(data.get("side_quests"))

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def apply_effects(self, state: GameState, effects: dict[str, Any]) -> None:
        """将决策后果应用到状态。"""

        self._apply_dimension_effects(state, effects)

        npc_changes = effects.get("npc", {})
        if not isinstance(npc_changes, dict):
            npc_changes = effects.get("active_npcs", {}) if isinstance(effects.get("active_npcs"), dict) else {}
        for npc_name, patch in npc_changes.items():
            if not isinstance(patch, dict):
                continue
            current = dict(state.active_npcs.get(str(npc_name), {}))
            for field_name, value in patch.items():
                if isinstance(value, int) and isinstance(current.get(field_name), int):
                    current[field_name] = current[field_name] + value
                else:
                    current[field_name] = value
            state.active_npcs[str(npc_name)] = current

        storyline_changes = effects.get("storyline", {})
        if not isinstance(storyline_changes, dict):
            storyline_changes = effects.get("storylines", {}) if isinstance(effects.get("storylines"), dict) else {}
        for line_id, patch in storyline_changes.items():
            if isinstance(patch, dict):
                current = dict(state.storylines.get(str(line_id), {}))
                current.update(patch)
                state.storylines[str(line_id)] = current

        if "phase" in effects and isinstance(effects["phase"], str):
            state.phase = effects["phase"]

        if "game_date" in effects and isinstance(effects["game_date"], str):
            state.game_date = effects["game_date"]

        if "game_time" in effects and isinstance(effects["game_time"], dict):
            state.game_time.update(self._as_int_dict(effects["game_time"], {}))

        if "current_scene" in effects and isinstance(effects["current_scene"], dict):
            state.current_scene = dict(effects["current_scene"])

        if "cast" in effects and isinstance(effects["cast"], dict):
            state.cast.update({str(k): str(v) for k, v in effects["cast"].items()})

        if "decision_record" in effects and isinstance(effects["decision_record"], dict):
            state.decisions.append(effects["decision_record"])

        if "decisions_completed" in effects and isinstance(effects["decisions_completed"], list):
            for decision_id in effects["decisions_completed"]:
                self._append_unique(state.decisions_completed, str(decision_id).strip())

        if "npc_statuses" in effects and isinstance(effects["npc_statuses"], dict):
            for npc_name, status in effects["npc_statuses"].items():
                npc_name_str = str(npc_name).strip()
                status_str = str(status).strip()
                if npc_name_str and status_str:
                    state.npc_statuses[npc_name_str] = status_str

        if "player_goal" in effects and isinstance(effects["player_goal"], dict):
            state.player_goal.update(effects["player_goal"])

        self._replace_or_extend_list(state, effects, "pressure_sources", "add_pressure_sources")
        self._replace_or_extend_list(state, effects, "consequence_seeds", "add_consequence_seeds")
        self._replace_or_extend_list(state, effects, "information_pool", "add_information")
        self._replace_or_extend_list(state, effects, "growth_log", "add_growth_log")
        self._replace_or_extend_list(state, effects, "advance_queue", "add_advance_actions")
        self._replace_or_extend_list(state, effects, "active_events", "add_active_events")
        self._replace_or_extend_list(state, effects, "pending_actions", "add_pending_actions")

        if "advance_action" in effects and isinstance(effects["advance_action"], dict):
            state.advance_queue.append(effects["advance_action"])

        if "event_history" in effects and isinstance(effects["event_history"], list):
            for event_id in effects["event_history"]:
                self._append_unique(state.event_history, str(event_id).strip())

        if "event_cooldowns" in effects and isinstance(effects["event_cooldowns"], dict):
            state.event_cooldowns.update(self._as_int_dict(effects["event_cooldowns"], {}))

        if "promises" in effects and isinstance(effects["promises"], list):
            state.promises.extend(item for item in effects["promises"] if isinstance(item, dict))

        if "turn_increment" in effects and isinstance(effects["turn_increment"], int):
            state.turn += effects["turn_increment"]

        # 兼容旧效果字段：章节字段已废弃，仅记录警告，不再写入状态。
        if "chapter_id" in effects or "current_chapter_events_triggered" in effects:
            logger.warning("忽略旧章节制 effects 字段: %s", [k for k in ("chapter_id", "current_chapter_events_triggered") if k in effects])

    def get_dimension_description(self, state: GameState) -> str:
        """将所有启用维度转换成文字描述。"""

        lines: list[str] = []
        self._append_dimension_lines(lines, "角色维度", state.dimensions.character, CHARACTER_DIMENSIONS)
        self._append_dimension_lines(lines, "世界维度", state.dimensions.world, WORLD_STATE_DIMENSIONS)
        self._append_dimension_lines(lines, "扩展维度", state.dimensions.extensions, EXTENSION_DIMENSIONS)

        if state.dimensions.relations:
            id_to_name = {str(npc_id): str(name) for name, npc_id in state.cast.items()}
            lines.append("【关系维度】")
            for npc_id, relation in state.dimensions.relations.items():
                npc_name = id_to_name.get(str(npc_id), str(npc_id))
                for name, value in relation.values.items():
                    definition = RELATION_NUMERIC_DIMENSIONS.get(name)
                    description = definition.description if definition else "自定义关系数值维度"
                    lines.append(f"{npc_name}.{name}：{value}（{self._dimension_score_to_text(value)}）- {description}")
                if relation.tags:
                    tag_desc = []
                    for tag in relation.tags:
                        definition = RELATION_TAG_DIMENSIONS.get(tag)
                        desc = definition.description if definition else "自定义关系标签"
                        tag_desc.append(f"{tag}（{desc}）")
                    lines.append(f"{npc_name}.关系标签：" + "、".join(tag_desc))

        return "\n".join(lines)

    def get_brief_world_summary(self, state: GameState) -> str:
        """生成行动面板使用的一行世界维度摘要。"""

        if not state.dimensions.world:
            return "世界维度未明"
        return " ".join(f"{name}:{value}" for name, value in state.dimensions.world.items())

    def get_dimension_trends(self, state: GameState) -> dict[str, str]:
        """比较上一回合快照，返回世界维度趋势箭头。"""

        previous = state.previous_world_dimensions
        if not previous:
            return {name: "━" for name in state.dimensions.world}

        trends: dict[str, str] = {}
        for name, value in state.dimensions.world.items():
            previous_value = previous.get(name)
            if previous_value is None:
                trends[name] = "━"
            elif value > previous_value:
                trends[name] = "▲"
            elif value < previous_value:
                trends[name] = "▼"
            else:
                trends[name] = "━"
        return trends

    def get_weak_dimensions(self, state: GameState, threshold: int = 3) -> list[dict]:
        """返回低于阈值的世界维度列表。"""

        weak_dimensions: list[dict] = []
        for name, value in state.dimensions.world.items():
            if value <= threshold:
                weak_dimensions.append({"name": name, "value": value, "label": "⚠ 薄弱"})
        return weak_dimensions

    def get_metrics_description(self, state: GameState) -> str:
        """兼容旧调用，返回新维度描述。"""

        logger.warning("get_metrics_description 已废弃，请改用 get_dimension_description")
        return self.get_dimension_description(state)

    def _apply_dimension_effects(self, state: GameState, effects: Mapping[str, Any]) -> None:
        """应用新维度效果，并兼容旧 metrics 效果。"""

        dimension_payloads = [effects.get("dimensions"), effects.get("dimension_delta"), effects.get("dimension_deltas")]
        for payload in dimension_payloads:
            if isinstance(payload, dict):
                self._apply_dimension_payload(state, payload)
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        self._apply_dimension_change_item(state, item)

        metrics_delta = effects.get("metrics", {})
        if isinstance(metrics_delta, dict):
            logger.warning("读取到旧格式 effects.metrics，已按新维度池进行降级迁移")
            legacy_delta = self._legacy_metric_deltas_to_dimensions(metrics_delta)
            self._apply_dimension_payload(state, {"world": legacy_delta})

    def _apply_dimension_payload(self, state: GameState, payload: Mapping[str, Any]) -> None:
        """应用分组形式的维度增量。"""

        category_map = {
            "character": "character",
            "角色维度": "character",
            "world": "world",
            "世界维度": "world",
            "extensions": "extension",
            "extension": "extension",
            "扩展维度": "extension",
        }
        for key, category in category_map.items():
            deltas = payload.get(key)
            if isinstance(deltas, dict):
                for name, delta in deltas.items():
                    self._update_dimension(state, str(name), self._safe_int(delta), category)

        relations = payload.get("relations") or payload.get("relation_numeric") or payload.get("关系维度")
        if isinstance(relations, dict):
            for npc_id, relation_patch in relations.items():
                self._apply_relation_patch(state, str(npc_id), relation_patch)

        if {"name", "delta"}.issubset(payload.keys()):
            self._apply_dimension_change_item(state, payload)

    def _apply_dimension_change_item(self, state: GameState, item: Mapping[str, Any]) -> None:
        """应用单条维度增量。"""

        name = str(item.get("name", "")).strip()
        if not name:
            return
        category = str(item.get("category", "auto") or "auto")
        npc_id = str(item.get("npc_id", "") or "")
        self._update_dimension(state, name, self._safe_int(item.get("delta", 0)), category, npc_id=npc_id)

    def _apply_relation_patch(self, state: GameState, npc_id: str, relation_patch: Any) -> None:
        """应用单个 NPC 的关系维度变化。"""

        if not isinstance(relation_patch, dict):
            return
        relation = state.dimensions.relations.setdefault(npc_id, RelationDimensions())
        value_patch = relation_patch.get("values") if isinstance(relation_patch.get("values"), dict) else relation_patch
        for name, delta in value_patch.items():
            if name in {"values", "tags", "tags_add", "tags_remove"}:
                continue
            self._update_dimension(state, str(name), self._safe_int(delta), "relation_numeric", npc_id=npc_id)

        for tag in self._as_list(relation_patch.get("tags_add")):
            tag_name = str(tag).strip()
            if tag_name and tag_name not in relation.tags:
                relation.tags.append(tag_name)
        for tag in self._as_list(relation_patch.get("tags_remove")):
            tag_name = str(tag).strip()
            if tag_name in relation.tags:
                relation.tags.remove(tag_name)
        if isinstance(relation_patch.get("tags"), list):
            relation.tags = [str(tag) for tag in relation_patch["tags"] if str(tag).strip()]

    def _update_dimension(
        self,
        state: GameState,
        name: str,
        delta: int,
        category: str = "auto",
        *,
        npc_id: str = "",
    ) -> None:
        """通过维度系统更新数值，并在缺省时补默认初值。"""

        if delta == 0:
            return
        reference = DimensionReference(name=name, category=category, npc_id=npc_id)
        try:
            self._ensure_dimension_initialized(state.dimensions, reference)
            update_dimension_value(state.dimensions, reference, delta)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("跳过无法应用的维度变化 name=%s category=%s npc_id=%s delta=%s err=%s", name, category, npc_id, delta, exc)

    def _ensure_dimension_initialized(self, dimensions: DimensionState, reference: DimensionReference) -> None:
        """为动态出现但合法的维度补默认初值。"""

        category = reference.category
        if category == "auto":
            if reference.name in CHARACTER_DIMENSIONS:
                category = "character"
            elif reference.name in WORLD_STATE_DIMENSIONS:
                category = "world"
            elif reference.name in EXTENSION_DIMENSIONS:
                category = "extension"
            elif reference.name in RELATION_NUMERIC_DIMENSIONS:
                category = "relation_numeric"

        if category == "character" and reference.name in CHARACTER_DIMENSIONS:
            dimensions.character.setdefault(reference.name, DEFAULT_DIMENSION_VALUE)
        elif category == "world" and reference.name in WORLD_STATE_DIMENSIONS:
            dimensions.world.setdefault(reference.name, DEFAULT_DIMENSION_VALUE)
        elif category == "extension" and reference.name in EXTENSION_DIMENSIONS:
            dimensions.extensions.setdefault(reference.name, DEFAULT_DIMENSION_VALUE)
        elif category == "relation_numeric" and reference.name in RELATION_NUMERIC_DIMENSIONS and reference.npc_id:
            relation = dimensions.relations.setdefault(reference.npc_id, RelationDimensions())
            relation.values.setdefault(reference.name, DEFAULT_DIMENSION_VALUE)

    def _load_dimensions(self, payload: Mapping[str, Any]) -> DimensionState:
        """从新旧存档中解析运行态维度。"""

        raw_dimensions = payload.get("dimensions") or payload.get("dimension_state")
        if isinstance(raw_dimensions, dict):
            return self._dimension_state_from_dict(raw_dimensions)

        raw_metrics = payload.get("metrics")
        if isinstance(raw_metrics, dict):
            state = GameState().dimensions
            state.world.update(self._legacy_metrics_to_world_dimensions(raw_metrics))
            return state

        return GameState().dimensions

    def _dimension_state_from_dict(self, payload: Mapping[str, Any]) -> DimensionState:
        """反序列化 DimensionState 与 RelationDimensions。"""

        return DimensionState(
            character=self._as_dimension_values(payload.get("character")),
            world=self._as_dimension_values(payload.get("world")),
            extensions=self._as_dimension_values(payload.get("extensions") or payload.get("extension")),
            relations=self._as_relations(payload.get("relations")),
        )

    def _dump_dimensions(self, dimensions: DimensionState) -> dict[str, Any]:
        """序列化 DimensionState 与 RelationDimensions。"""

        return {
            "character": dict(dimensions.character),
            "world": dict(dimensions.world),
            "extensions": dict(dimensions.extensions),
            "relations": {
                npc_id: {"values": dict(relation.values), "tags": list(relation.tags)}
                for npc_id, relation in dimensions.relations.items()
            },
        }

    def _as_relations(self, value: Any) -> dict[str, RelationDimensions]:
        """解析关系维度存档。"""

        if not isinstance(value, dict):
            return {}
        relations: dict[str, RelationDimensions] = {}
        for npc_id, relation_payload in value.items():
            if isinstance(relation_payload, RelationDimensions):
                relations[str(npc_id)] = RelationDimensions(
                    values=dict(relation_payload.values),
                    tags=list(relation_payload.tags),
                )
            elif isinstance(relation_payload, dict):
                values = relation_payload.get("values", relation_payload.get("relations", {}))
                tags = relation_payload.get("tags", [])
                relations[str(npc_id)] = RelationDimensions(
                    values=self._as_dimension_values(values),
                    tags=[str(tag) for tag in self._as_list(tags) if str(tag).strip()],
                )
        return relations

    def _legacy_metrics_to_world_dimensions(self, metrics: Mapping[str, Any]) -> dict[str, int]:
        """将旧 0-100 指标降级迁移到 0-10 世界维度。"""

        grouped: dict[str, list[int]] = {}
        for metric_name, value in metrics.items():
            dimension_name = LEGACY_METRIC_DIMENSION_MAP.get(str(metric_name))
            if not dimension_name:
                continue
            grouped.setdefault(dimension_name, []).append(clamp_dimension_value(round(self._safe_int(value) / 10)))
        return {name: round(sum(values) / len(values)) for name, values in grouped.items() if values}

    def _legacy_metric_deltas_to_dimensions(self, metrics: Mapping[str, Any]) -> dict[str, int]:
        """将旧指标增量降级迁移到世界维度增量。"""

        result: dict[str, int] = {}
        for metric_name, value in metrics.items():
            dimension_name = LEGACY_METRIC_DIMENSION_MAP.get(str(metric_name))
            if not dimension_name:
                continue
            result[dimension_name] = result.get(dimension_name, 0) + self._scale_legacy_delta(self._safe_int(value))
        return result

    def _append_dimension_lines(
        self,
        lines: list[str],
        title: str,
        values: Mapping[str, int],
        definitions: Mapping[str, Any],
    ) -> None:
        """追加某一类维度描述。"""

        if not values:
            return
        lines.append(f"【{title}】")
        for name, value in values.items():
            definition = definitions.get(name)
            description = definition.description if definition else "自定义维度"
            lines.append(f"{name}：{value}（{self._dimension_score_to_text(value)}）- {description}")

    @staticmethod
    def _dimension_score_to_text(score: int) -> str:
        """0-10 维度值映射到区间描述。"""

        if score <= 2:
            return "极低"
        if score <= 4:
            return "偏低"
        if score <= 6:
            return "中等"
        if score <= 8:
            return "偏高"
        return "极高"

    @staticmethod
    def _scale_legacy_delta(delta: int) -> int:
        """将旧 0-100 指标增量换算为新 0-10 维度增量。"""

        if delta == 0:
            return 0
        magnitude = max(1, round(abs(delta) / 10))
        return magnitude if delta > 0 else -magnitude

    @staticmethod
    def _as_dict(value: Any) -> dict:
        """安全转换 dict。"""

        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _as_list(value: Any) -> list:
        """安全转换 list。"""

        return list(value) if isinstance(value, list) else []

    @classmethod
    def _as_int_dict(cls, value: Any, default: dict[str, int]) -> dict[str, int]:
        """安全转换整数 dict。"""

        if not isinstance(value, dict):
            return dict(default)
        result: dict[str, int] = {}
        for key, item in value.items():
            result[str(key)] = cls._safe_int(item)
        return result

    @classmethod
    def _as_dimension_values(cls, value: Any) -> dict[str, int]:
        """安全转换维度值 dict。"""

        if not isinstance(value, dict):
            return {}
        return {str(key): clamp_dimension_value(cls._safe_int(item)) for key, item in value.items()}

    @staticmethod
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

    @staticmethod
    def _append_unique(target: list[str], value: str) -> None:
        """非空且不重复时追加字符串。"""

        if value and value not in target:
            target.append(value)

    @staticmethod
    def _replace_or_extend_list(state: GameState, effects: Mapping[str, Any], field_name: str, add_field_name: str) -> None:
        """支持列表字段整体替换或增量追加。"""

        if field_name in effects and isinstance(effects[field_name], list):
            setattr(state, field_name, StateManager._normalize_named_list(field_name, effects[field_name]))
        if add_field_name in effects and isinstance(effects[add_field_name], list):
            current = getattr(state, field_name)
            current.extend(StateManager._normalize_named_list(field_name, effects[add_field_name]))

    @staticmethod
    def _normalize_named_list(field_name: str, value: Any) -> list[dict]:
        """按字段名规范化目标驱动列表。"""

        if field_name == "consequence_seeds":
            return StateManager._normalize_consequence_seeds(value)
        if field_name == "information_pool":
            return StateManager._normalize_information_pool(value)
        if field_name == "advance_queue":
            return StateManager._normalize_advance_queue(value)
        return StateManager._as_dict_list(value)

    @staticmethod
    def _as_dict_list(value: Any) -> list[dict]:
        """安全转换 dict 列表。"""

        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    @classmethod
    def _normalize_npc_memory(cls, value: Any, legacy_value: Any = None) -> dict[str, dict]:
        """规范化NPC三层记忆，并兼容旧 npc_memories 字段。"""

        result: dict[str, dict] = {}
        if isinstance(value, dict):
            for npc_id, memory in value.items():
                if not isinstance(memory, dict):
                    continue
                result[str(npc_id)] = {
                    "key_facts": cls._as_dict_list(memory.get("key_facts")),
                    "talk_summaries": cls._as_dict_list(memory.get("talk_summaries")),
                    "recent_talks": cls._as_dict_list(memory.get("recent_talks"))[-2:],
                }
        if result or not isinstance(legacy_value, dict):
            return result
        for npc_id, memories in legacy_value.items():
            key_facts: list[dict] = []
            for index, memory in enumerate(cls._as_dict_list(memories), start=1):
                content = str(memory.get("content") or memory.get("fact") or "").strip()
                if not content:
                    continue
                key_facts.append(
                    {
                        "id": str(memory.get("id") or f"legacy_{index}"),
                        "type": str(memory.get("type") or "information"),
                        "content": content,
                        "round": cls._safe_int(memory.get("round"), 0),
                    }
                )
            if key_facts:
                result[str(npc_id)] = {"key_facts": key_facts, "talk_summaries": [], "recent_talks": []}
        return result

    @staticmethod
    def _normalize_consequence_seeds(value: Any) -> list[dict]:
        """规范化后果种子，确保核心字段存在。"""

        severity_aliases = {
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
        default_chances = {"low": 2, "medium": 3, "high": 4}
        seeds: list[dict] = []
        for item in StateManager._as_dict_list(value):
            seed = dict(item)
            seed["source"] = str(seed.get("source", "")).strip()
            severity_text = str(seed.get("severity", "low") or "low").strip().lower()
            seed["severity"] = severity_text if severity_text in default_chances else severity_aliases.get(severity_text, "low")
            directions = seed.get("directions", [])
            seed["directions"] = [str(direction) for direction in directions] if isinstance(directions, list) else []
            seed["remaining_chances"] = max(
                0,
                StateManager._safe_int(seed.get("remaining_chances", default_chances[seed["severity"]]), default_chances[seed["severity"]]),
            )
            seed["seed_id"] = str(seed.get("seed_id", "")).strip()
            seed["created_at"] = StateManager._as_int_dict(seed.get("created_at"), {})
            seed["triggered"] = bool(seed.get("triggered", False))
            seeds.append(seed)
        return seeds

    @staticmethod
    def _normalize_information_pool(value: Any) -> list[dict]:
        """规范化信息池，限制信息类型为三类。"""

        information_items: list[dict] = []
        allowed_types = {"fact", "persistent", "escalating"}
        for item in StateManager._as_dict_list(value):
            information = dict(item)
            info_type = str(information.get("type", "fact")).strip()
            information["type"] = info_type if info_type in allowed_types else "fact"
            information_items.append(information)
        return information_items

    @staticmethod
    def _normalize_advance_queue(value: Any) -> list[dict]:
        """规范化待执行行动队列，确保存在时间成本。"""

        actions: list[dict] = []
        for item in StateManager._as_dict_list(value):
            action = dict(item)
            time_cost = action.get("time_cost", action.get("time_cost_units", action.get("cost", 0)))
            action["time_cost"] = max(0, StateManager._safe_int(time_cost))
            actions.append(action)
        return actions

    @staticmethod
    def _normalize_backlog(value: Any) -> list[dict[str, Any]]:
        """规范化当前回合待办队列。"""

        items: list[dict[str, Any]] = []
        for item in StateManager._as_dict_list(value):
            backlog_item = dict(item)
            backlog_item["id"] = str(backlog_item.get("id", "")).strip()
            backlog_item["description"] = str(backlog_item.get("description", "")).strip()
            backlog_item["source"] = str(backlog_item.get("source", "player_text")).strip() or "player_text"
            duration = str(backlog_item.get("duration", "instant")).strip()
            backlog_item["duration"] = duration if duration in {"instant", "delayed"} else "instant"
            backlog_item["delay_count"] = max(0, StateManager._safe_int(backlog_item.get("delay_count", 0)))
            backlog_item["removable"] = bool(backlog_item.get("removable", True))
            severity_warning = backlog_item.get("severity_warning")
            backlog_item["severity_warning"] = str(severity_warning).strip() if severity_warning else ""
            backlog_item["deterministic"] = bool(backlog_item.get("deterministic", False))
            backlog_item["deterministic_effects"] = backlog_item.get("deterministic_effects", {}) if isinstance(backlog_item.get("deterministic_effects"), dict) else {}
            backlog_item["dc"] = StateManager._safe_int(backlog_item.get("dc", 0))
            backlog_item["modifier"] = StateManager._safe_int(backlog_item.get("modifier", 0))
            backlog_item["main_dimension"] = str(backlog_item.get("main_dimension", "")).strip()
            backlog_item["predicted_effects"] = backlog_item.get("predicted_effects", {}) if isinstance(backlog_item.get("predicted_effects"), dict) else {}
            items.append(backlog_item)
        return items

    @staticmethod
    def _normalize_side_quests(value: Any) -> list[dict[str, Any]]:
        items = StateManager._as_dict_list(value) if not isinstance(value, list) else [dict(v) for v in value if isinstance(v, dict)]
        result = []
        seen_names: set[str] = set()
        for item in items:
            name = str(item.get("name", "")).strip()
            desc = str(item.get("description", "")).strip()
            if not name or name == "未命名支线" or not desc:
                continue
            if name in seen_names:
                continue
            seen_names.add(name)
            result.append(item)
        return result

    @staticmethod
    def _normalize_delayed_queue(value: Any) -> list[dict[str, Any]]:
        """规范化延迟推进队列。"""

        items: list[dict[str, Any]] = []
        for item in StateManager._as_dict_list(value):
            delayed_item = dict(item)
            delayed_item["remaining_advances"] = max(0, StateManager._safe_int(delayed_item.get("remaining_advances", 0)))
            if isinstance(delayed_item.get("backlog_item"), dict):
                delayed_item["backlog_item"] = StateManager._normalize_backlog([delayed_item["backlog_item"]])[0]
            items.append(delayed_item)
        return items

    @staticmethod
    def _normalize_agenda(value: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if not isinstance(value, list):
            return items
        for item in value:
            if not isinstance(item, dict):
                continue
            agenda_item = dict(item)
            agenda_item["id"] = str(agenda_item.get("id", "")).strip()
            agenda_item["description"] = str(agenda_item.get("description", "")).strip()
            agenda_item["source"] = str(agenda_item.get("source", "system")).strip() or "system"
            agenda_item["urgency"] = str(agenda_item.get("urgency", "normal")).strip()
            agenda_item["consequence"] = str(agenda_item.get("consequence", "")).strip()
            agenda_item["resolved"] = bool(agenda_item.get("resolved", False))
            agenda_item["resolved_turn"] = int(agenda_item.get("resolved_turn", 0)) if agenda_item.get("resolved_turn") else 0
            agenda_item["relevant_npcs"] = [str(n) for n in agenda_item.get("relevant_npcs", [])] if isinstance(agenda_item.get("relevant_npcs"), list) else []
            items.append(agenda_item)
        return items

    @staticmethod
    def _merge_history(existing: list[dict], recent_window: list[dict]) -> list[dict]:
        """合并历史，尽量避免重复追加最近窗口。"""

        if not existing:
            return list(recent_window)
        if not recent_window:
            return list(existing)

        max_overlap = min(len(existing), len(recent_window))
        overlap = 0
        for n in range(max_overlap, 0, -1):
            if existing[-n:] == recent_window[:n]:
                overlap = n
                break

        return existing + recent_window[overlap:]


_state_manager_instance: StateManager | None = None


def get_state_manager() -> StateManager:
    """获取全局 StateManager 单例。"""

    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = StateManager()
    return _state_manager_instance
