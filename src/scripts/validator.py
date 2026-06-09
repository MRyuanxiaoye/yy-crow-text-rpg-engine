"""剧本验证器。"""

from __future__ import annotations

import asyncio
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from src.scripts.loader import BASE_SCRIPTS_DIR, load_role, load_world

Severity = Literal["error", "warning"]


@dataclass
class ValidationIssue:
    """单条验证问题。"""

    severity: Severity
    category: str
    message: str
    location: str


@dataclass
class ValidationReport:
    """剧本验证报告。"""

    script_id: str
    world_issues: list[ValidationIssue] = field(default_factory=list)
    role_issues: dict[str, list[ValidationIssue]] = field(default_factory=dict)
    passed: bool = True
    summary: str = ""


@dataclass(frozen=True)
class _DimensionCondition:
    """内部使用的维度条件。"""

    key: str
    operator: str
    value: int
    location: str


@dataclass
class _Bounds:
    """内部使用的理论取值范围。"""

    minimum: int
    maximum: int


_REQUIRED_WORLD_FIELDS = (
    "script_id",
    "name",
    "dimension_definitions",
    "time_system",
    "global_npc_pool",
    "optional_roles",
)
_CONDITION_OPERATORS = {">", ">=", "<", "<=", "==", "=", "!="}
def validate_world(world_data: dict) -> list[ValidationIssue]:
    """验证世界层数据。"""

    issues: list[ValidationIssue] = []

    for field_name in _REQUIRED_WORLD_FIELDS:
        value = world_data.get(field_name)
        if value is None or value == "" or value == [] or value == {}:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="structure",
                    message=f"世界层缺少必需字段：{field_name}",
                    location=field_name,
                )
            )

    definitions = _as_dict(world_data.get("dimension_definitions"))
    for field_name in ("world", "relationship_numeric", "relationship_tags"):
        if not _as_list(definitions.get(field_name)):
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="structure",
                    message=f"维度定义不完整：dimension_definitions.{field_name} 不能为空",
                    location=f"dimension_definitions.{field_name}",
                )
            )

    issues.extend(check_npc_conflict_graph(world_data))
    issues.extend(check_event_coverage(world_data, None))
    return issues


def validate_role(role_data: dict, world_data: dict) -> list[ValidationIssue]:
    """验证角色层数据。"""

    issues: list[ValidationIssue] = []
    role_info = _as_dict(role_data.get("role"))
    role_id = str(role_info.get("role_id") or "")
    optional_role_ids = _optional_role_ids(world_data)

    if not role_id:
        issues.append(
            ValidationIssue(
                severity="error",
                category="structure",
                message="角色层缺少 role.role_id",
                location="role.role_id",
            )
        )
    elif role_id not in optional_role_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                category="reference",
                message=f"角色ID未在世界层 optional_roles 中声明：{role_id}",
                location="role.role_id",
            )
        )

    issues.extend(check_dimension_reachability(role_data, world_data))
    issues.extend(check_ending_reachability(role_data, world_data))
    issues.extend(check_pressure_counterplay(role_data))
    issues.extend(_check_role_npc_references(role_data, world_data))
    issues.extend(_check_information_references(role_data, world_data))
    issues.extend(_check_pressure_event_references(role_data, world_data))
    issues.extend(_check_protagonist_npcs(role_data, world_data))
    issues.extend(check_event_coverage(world_data, role_data))
    return issues


def validate_script(script_id: str) -> ValidationReport:
    """加载并验证一个新格式剧本。"""

    try:
        world_data = _run_async(load_world(script_id))
    except Exception as exc:
        issue = ValidationIssue(
            severity="error",
            category="structure",
            message=f"世界层加载失败：{exc}",
            location="world.yaml",
        )
        return ValidationReport(
            script_id=script_id,
            world_issues=[issue],
            role_issues={},
            passed=False,
            summary="验证未通过：世界层加载失败。",
        )

    world_issues = validate_world(world_data)
    role_issues: dict[str, list[ValidationIssue]] = {}

    for role_id, role_path in _discover_existing_roles(script_id, world_data):
        try:
            role_data = _run_async(load_role(script_id, role_id)) if role_path is None else _load_role_file(role_path)
        except Exception as exc:  # 验证器需要把加载失败转成结构问题。
            role_issues[role_id] = [
                ValidationIssue(
                    severity="error",
                    category="structure",
                    message=f"角色层加载失败：{exc}",
                    location=f"roles.{role_id}",
                )
            ]
            continue
        role_issues[role_id] = validate_role(role_data, world_data)

    all_issues = [*world_issues, *[issue for issues in role_issues.values() for issue in issues]]
    error_count = sum(1 for issue in all_issues if issue.severity == "error")
    warning_count = sum(1 for issue in all_issues if issue.severity == "warning")
    passed = error_count == 0
    summary = (
        f"验证通过：世界层 {len(world_issues)} 个问题，"
        f"角色层 {sum(len(issues) for issues in role_issues.values())} 个问题，"
        f"error {error_count} 个，warning {warning_count} 个。"
        if passed
        else f"验证未通过：发现 error {error_count} 个，warning {warning_count} 个。"
    )

    return ValidationReport(
        script_id=script_id,
        world_issues=world_issues,
        role_issues=role_issues,
        passed=passed,
        summary=summary,
    )


def check_dimension_reachability(role_data: dict, world_data: dict) -> list[ValidationIssue]:
    """检查每个世界维度是否至少有两种理论影响来源。"""

    issues: list[ValidationIssue] = []
    world_dimension_ids = _world_dimension_ids(world_data)
    influence_sources = _collect_world_dimension_influences(world_data, role_data)

    for dimension_id in world_dimension_ids:
        sources = influence_sources.get(dimension_id, set())
        if len(sources) < 2:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="reachability",
                    message=f'世界维度"{dimension_id}"影响来源不足，当前仅发现 {len(sources)} 种',
                    location=f"dimension_definitions.world.{dimension_id}",
                )
            )
    return issues


def check_ending_reachability(role_data: dict, world_data: dict) -> list[ValidationIssue]:
    """检查结局条件在理论维度范围内是否可达。"""

    issues: list[ValidationIssue] = []
    endings = _parsed_or_raw_endings(role_data)
    catalog = _as_list(endings.get("catalog"))
    bottom_lines = {
        str(item.get("bottom_line_id")): item
        for item in _as_list(endings.get("bottom_lines"))
        if isinstance(item, dict) and item.get("bottom_line_id")
    }
    bounds = _build_theoretical_bounds(role_data, world_data)

    for index, ending in enumerate(catalog):
        if not isinstance(ending, dict):
            continue
        ending_id = str(ending.get("ending_id") or f"catalog[{index}]")
        conditions = ending.get("conditions")
        location = f"endings.catalog[{index}].conditions"
        if not conditions:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="reachability",
                    message=f'结局"{ending_id}"缺少结构化条件',
                    location=location,
                )
            )
            continue
        if not _condition_tree_satisfiable(conditions, bounds, bottom_lines, f"endings.catalog[{index}].conditions"):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="reachability",
                    message=f'结局"{ending_id}"按已声明效果理论不可达（玩家行动可动态影响维度）',
                    location=location,
                )
            )
    return issues


def check_npc_conflict_graph(world_data: dict) -> list[ValidationIssue]:
    """检查NPC之间是否存在目标或阵营对立。"""

    npcs = _as_list(world_data.get("global_npc_pool"))
    factions = _as_list(_as_dict(world_data.get("world_background")).get("factions"))
    faction_conflicts: set[tuple[str, str]] = set()
    for faction in factions:
        if not isinstance(faction, dict):
            continue
        faction_id = str(faction.get("faction_id") or "")
        for target in _as_list(faction.get("conflict_with")):
            if faction_id and target:
                faction_conflicts.add((faction_id, str(target)))
                faction_conflicts.add((str(target), faction_id))

    faction_to_npcs: dict[str, list[dict]] = defaultdict(list)
    for npc in npcs:
        if not isinstance(npc, dict):
            continue
        faction_id = npc.get("faction_id")
        if faction_id:
            faction_to_npcs[str(faction_id)].append(npc)

    for left_faction, right_faction in faction_conflicts:
        if faction_to_npcs.get(left_faction) and faction_to_npcs.get(right_faction):
            return []

    for left_index, left_npc in enumerate(npcs):
        if not isinstance(left_npc, dict):
            continue
        for right_index in range(left_index + 1, len(npcs)):
            right_npc = npcs[right_index]
            if isinstance(right_npc, dict) and _npc_goals_conflict(left_npc, right_npc):
                return []

    return [
        ValidationIssue(
            severity="error",
            category="balance",
            message="NPC冲突图不足，未发现至少一对目标对立的NPC",
            location="global_npc_pool",
        )
    ]


def check_event_coverage(world_data: dict, role_data: dict | None) -> list[ValidationIssue]:
    """检查固定时间线事件是否存在连续空窗或高密度。"""

    issues: list[ValidationIssue] = []
    time_system = _as_dict(world_data.get("time_system"))
    start = _as_dict(time_system.get("start"))
    max_time = _as_dict(time_system.get("max_time"))
    if not start or not max_time:
        return [
            ValidationIssue(
                severity="warning",
                category="structure",
                message="时间系统缺少 start 或 max_time，无法检查事件覆盖空窗",
                location="time_system",
            )
        ]

    unit = str(time_system.get("unit") or "")
    start_index = _time_to_index(start, unit)
    end_index = _time_to_index(max_time, unit)
    if start_index is None or end_index is None or end_index < start_index:
        return [
            ValidationIssue(
                severity="warning",
                category="structure",
                message="时间系统范围无法解析，无法检查事件覆盖空窗",
                location="time_system",
            )
        ]

    density: dict[int, int] = defaultdict(int)
    for event, _location in _timeline_events(world_data, role_data):
        at_index = _time_to_index(_as_dict(event.get("at")), unit)
        if at_index is not None:
            density[at_index] += 1

    empty_streak = 0
    high_streak = 0
    first_empty_start: int | None = None
    first_high_start: int | None = None
    for index in range(start_index, end_index + 1):
        count = density.get(index, 0)
        if count == 0:
            if empty_streak == 0:
                first_empty_start = index
            empty_streak += 1
        else:
            empty_streak = 0
            first_empty_start = None

        if count > 2:
            if high_streak == 0:
                first_high_start = index
            high_streak += 1
        else:
            high_streak = 0
            first_high_start = None

        if empty_streak == 4:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="balance",
                    message="时间线存在连续超过3个时间段没有固定事件（随机池和条件事件可填充）",
                    location=f"time_system[{first_empty_start}-{index}]",
                )
            )
        if high_streak == 4:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="balance",
                    message="时间线存在连续超过3个高密度时间段",
                    location=f"time_system[{first_high_start}-{index}]",
                )
            )
    return issues


def check_pressure_counterplay(role_data: dict) -> list[ValidationIssue]:
    """检查衰减型压力源是否配置缓解手段。"""

    issues: list[ValidationIssue] = []
    pressure_sources = _parsed_or_raw_pressure_sources(role_data)
    for index, pressure in enumerate(_as_list(pressure_sources.get("decay"))):
        if not isinstance(pressure, dict):
            continue
        counterplay = [item for item in _as_list(pressure.get("counterplay")) if item]
        if not counterplay:
            pressure_id = pressure.get("pressure_id") or f"decay[{index}]"
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="balance",
                    message=f'衰减型压力源"{pressure_id}"缺少 counterplay',
                    location=f"pressure_sources.decay[{index}].counterplay",
                )
            )
    return issues


def _check_role_npc_references(role_data: dict, world_data: dict) -> list[ValidationIssue]:
    """检查角色关系中的NPC引用。"""

    issues: list[ValidationIssue] = []
    known_npc_ids = _npc_ids(world_data)
    for index, relation in enumerate(_as_list(role_data.get("npc_relationships"))):
        if not isinstance(relation, dict):
            continue
        npc_id = relation.get("npc_id")
        if npc_id and str(npc_id) not in known_npc_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="reference",
                    message=f"NPC关系引用了不存在的NPC：{npc_id}",
                    location=f"npc_relationships[{index}].npc_id",
                )
            )
    return issues


def _check_information_references(role_data: dict, world_data: dict) -> list[ValidationIssue]:
    """检查信息池中的NPC引用。"""

    issues: list[ValidationIssue] = []
    known_npc_ids = _npc_ids(world_data)
    information_pool = role_data.get("parsed_information_pool") or role_data.get("information_pool")
    for index, info in enumerate(_as_list(information_pool)):
        if not isinstance(info, dict):
            continue
        for npc_id in _as_list(info.get("known_by")):
            if str(npc_id) not in known_npc_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="reference",
                        message=f"信息池引用了不存在的知情NPC：{npc_id}",
                        location=f"information_pool[{index}].known_by",
                    )
                )
    return issues


def _check_pressure_event_references(role_data: dict, world_data: dict) -> list[ValidationIssue]:
    """检查压力源引用的事件ID。"""

    issues: list[ValidationIssue] = []
    event_ids = _event_ids(world_data, role_data)
    pressure_sources = _parsed_or_raw_pressure_sources(role_data)
    for group_name in ("milestones", "reactions"):
        for index, pressure in enumerate(_as_list(pressure_sources.get(group_name))):
            if not isinstance(pressure, dict):
                continue
            event_id = pressure.get("event_id")
            if event_id and str(event_id) not in event_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="reference",
                        message=f"压力源引用了不存在的事件：{event_id}",
                        location=f"pressure_sources.{group_name}[{index}].event_id",
                    )
                )
    return issues


def _check_protagonist_npcs(role_data: dict, world_data: dict) -> list[ValidationIssue]:
    """检查前主角NPC引用和可用性。"""

    issues: list[ValidationIssue] = []
    npcs_by_id = {
        str(npc.get("npc_id")): npc
        for npc in _as_list(world_data.get("global_npc_pool"))
        if isinstance(npc, dict) and npc.get("npc_id")
    }
    for index, protagonist in enumerate(_as_list(role_data.get("protagonist_npcs"))):
        if not isinstance(protagonist, dict):
            continue
        npc_id = protagonist.get("npc_id")
        npc = npcs_by_id.get(str(npc_id)) if npc_id else None
        if npc is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="reference",
                    message=f"前主角NPC不存在于世界NPC池：{npc_id}",
                    location=f"protagonist_npcs[{index}].npc_id",
                )
            )
            continue
        if npc.get("usable_as_player_role") is not True:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="reference",
                    message=f"前主角NPC未标记为可作为玩家角色：{npc_id}",
                    location=f"protagonist_npcs[{index}].npc_id",
                )
            )
    return issues


def _as_dict(value: Any) -> dict:
    """安全转换为字典。"""

    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    """安全转换为列表。"""

    return value if isinstance(value, list) else []


def _definition_id(item: Any) -> str:
    """提取维度定义ID。"""

    if not isinstance(item, dict):
        return ""
    return str(item.get("id") or item.get("name") or "")


def _world_dimension_ids(world_data: dict) -> list[str]:
    """获取世界维度ID。"""

    definitions = _as_dict(world_data.get("dimension_definitions"))
    return [dimension_id for item in _as_list(definitions.get("world")) if (dimension_id := _definition_id(item))]


def _role_dimension_ids(role_data: dict) -> list[str]:
    """获取角色维度ID。"""

    role_dimensions = _as_dict(role_data.get("role_dimensions"))
    return [dimension_id for item in _as_list(role_dimensions.get("definitions")) if (dimension_id := _definition_id(item))]


def _optional_role_ids(world_data: dict) -> set[str]:
    """获取世界层声明的可选角色ID。"""

    return {
        str(role.get("role_id"))
        for role in _as_list(world_data.get("optional_roles"))
        if isinstance(role, dict) and role.get("role_id")
    }


def _npc_ids(world_data: dict) -> set[str]:
    """获取世界NPC ID集合。"""

    return {
        str(npc.get("npc_id"))
        for npc in _as_list(world_data.get("global_npc_pool"))
        if isinstance(npc, dict) and npc.get("npc_id")
    }


def _parsed_or_raw_pressure_sources(role_data: dict) -> dict:
    """兼容 loader 解析后与原始压力源结构。"""

    parsed = role_data.get("parsed_pressure_sources")
    if isinstance(parsed, dict):
        return parsed
    return _as_dict(role_data.get("pressure_sources"))


def _parsed_or_raw_endings(role_data: dict) -> dict:
    """兼容 loader 解析后与原始结局结构。"""

    parsed = role_data.get("parsed_endings")
    if isinstance(parsed, dict):
        return parsed
    return _as_dict(role_data.get("endings"))


def _event_ids(world_data: dict, role_data: dict | None) -> set[str]:
    """收集世界层与角色层事件ID。"""

    event_ids: set[str] = set()
    event_groups = (
        world_data.get("timeline_milestones"),
        world_data.get("condition_events"),
        _as_dict(world_data.get("random_event_pool")).get("entries"),
    )
    if role_data is not None:
        role_events = _as_dict(role_data.get("role_events"))
        event_groups += (role_events.get("timeline"), role_events.get("conditional"), role_events.get("random"))

    for group in event_groups:
        for event in _as_list(group):
            if isinstance(event, dict) and event.get("event_id"):
                event_ids.add(str(event.get("event_id")))
    return event_ids


def _timeline_events(world_data: dict, role_data: dict | None) -> list[tuple[dict, str]]:
    """收集带固定时间点的事件。"""

    events: list[tuple[dict, str]] = []
    for index, event in enumerate(_as_list(world_data.get("timeline_milestones"))):
        if isinstance(event, dict):
            events.append((event, f"timeline_milestones[{index}]"))

    if role_data is not None:
        role_events = _as_dict(role_data.get("role_events"))
        for index, event in enumerate(_as_list(role_events.get("timeline"))):
            if isinstance(event, dict):
                events.append((event, f"role_events.timeline[{index}]"))
        pressure_sources = _parsed_or_raw_pressure_sources(role_data)
        for index, pressure in enumerate(_as_list(pressure_sources.get("milestones"))):
            if isinstance(pressure, dict) and isinstance(pressure.get("at"), dict):
                events.append((pressure, f"pressure_sources.milestones[{index}]"))
    return events


def _time_to_index(value: dict, unit: str) -> int | None:
    """把时间字典转换为连续索引。"""

    if not value:
        return None
    if "turn" in value and ("回合" in unit or unit.lower() == "turn"):
        return _safe_int(value.get("turn"))
    year = _safe_int(value.get("year", 1)) or 1
    month = _safe_int(value.get("month", 1)) or 1
    week = _safe_int(value.get("week", 1)) or 1
    day = _safe_int(value.get("day", 1)) or 1
    turn = _safe_int(value.get("turn", 1)) or 1

    unit_lower = unit.lower()
    if "年" in unit or unit_lower == "year":
        return year
    if "月" in unit or unit_lower == "month":
        return (year - 1) * 12 + month
    if "周" in unit or unit_lower == "week":
        return (year - 1) * 52 + week
    if "天" in unit or "日" in unit or unit_lower == "day":
        return (year - 1) * 372 + (month - 1) * 31 + day
    return turn if "turn" in value else (year - 1) * 12 + month


def _safe_int(value: Any) -> int | None:
    """安全转换整数。"""

    return value if isinstance(value, int) else None


def _npc_goals_conflict(left_npc: dict, right_npc: dict) -> bool:
    """用目标文本和不可接受结果做保守冲突判断。"""

    left_goal = _goal_text(left_npc, include_unacceptable=False)
    right_goal = _goal_text(right_npc, include_unacceptable=False)
    left_unacceptable = _goal_text(left_npc, include_goals=False)
    right_unacceptable = _goal_text(right_npc, include_goals=False)

    if left_goal and right_unacceptable and _text_overlap(left_goal, right_unacceptable):
        return True
    if right_goal and left_unacceptable and _text_overlap(right_goal, left_unacceptable):
        return True
    combined = f"{left_goal} {right_goal}"
    opposite_pairs = (
        ("维护", "推翻"),
        ("维持", "颠覆"),
        ("扩张", "遏制"),
        ("统一", "割据"),
        ("独立", "控制"),
        ("镇压", "起义"),
        ("公开", "隐瞒"),
    )
    return any(left in combined and right in combined for left, right in opposite_pairs)


def _goal_text(npc: dict, *, include_goals: bool = True, include_unacceptable: bool = True) -> str:
    """拼接NPC目标层文本。"""

    goal_layer = _as_dict(npc.get("goal_layer"))
    parts: list[str] = []
    if include_goals:
        parts.append(str(goal_layer.get("long_term_goal") or ""))
        parts.extend(str(item) for item in _as_list(goal_layer.get("short_term_goals")))
    if include_unacceptable:
        parts.extend(str(item) for item in _as_list(goal_layer.get("unacceptable_outcomes")))
    return " ".join(part for part in parts if part)


def _text_overlap(left: str, right: str) -> bool:
    """检查两个中文短文本是否存在有意义重叠。"""

    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    return bool(left_tokens & right_tokens)


def _meaningful_tokens(text: str) -> set[str]:
    """提取用于保守匹配的中文片段。"""

    tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,}", text))
    stop_words = {"自身", "长期", "短期", "目标", "利益", "风险", "彻底", "示例", "阵营"}
    return {token for token in tokens if token not in stop_words}


def _collect_world_dimension_influences(world_data: dict, role_data: dict) -> dict[str, set[str]]:
    """收集影响世界维度的来源路径。"""

    influences: dict[str, set[str]] = defaultdict(set)

    def add_effects(effects: Any, source: str) -> None:
        if not isinstance(effects, dict):
            return
        world_dimensions = effects.get("world_dimensions")
        if isinstance(world_dimensions, dict):
            for dimension_id, delta in world_dimensions.items():
                if isinstance(delta, int) and delta != 0:
                    influences[str(dimension_id)].add(source)
        world_impact = effects.get("world_impact")
        if isinstance(world_impact, dict):
            for dimension_id, delta in world_impact.items():
                if isinstance(delta, int) and delta != 0:
                    influences[str(dimension_id)].add(source)

    for index, event in enumerate(_as_list(world_data.get("timeline_milestones"))):
        add_effects(_as_dict(event).get("default_effects"), f"timeline_milestones[{index}]")
    for index, event in enumerate(_as_list(world_data.get("condition_events"))):
        add_effects(_as_dict(event).get("default_effects"), f"condition_events[{index}]")
    for index, event in enumerate(_as_list(_as_dict(world_data.get("random_event_pool")).get("entries"))):
        add_effects(_as_dict(event).get("default_effects"), f"random_event_pool.entries[{index}]")

    pressure_sources = _parsed_or_raw_pressure_sources(role_data)
    for index, pressure in enumerate(_as_list(pressure_sources.get("decay"))):
        if not isinstance(pressure, dict):
            continue
        dimension_id = pressure.get("dimension")
        rate = pressure.get("rate")
        if dimension_id and isinstance(rate, int) and rate != 0:
            influences[str(dimension_id)].add(f"pressure_sources.decay[{index}]")

    role_events = _as_dict(role_data.get("role_events"))
    for group_name in ("timeline", "conditional", "random"):
        for index, event in enumerate(_as_list(role_events.get(group_name))):
            add_effects(_as_dict(event).get("default_effects"), f"role_events.{group_name}[{index}]")

    for index, protagonist in enumerate(_as_list(role_data.get("protagonist_npcs"))):
        for driver_index, driver in enumerate(_as_list(_as_dict(protagonist).get("behavior_driver"))):
            add_effects({"world_impact": _as_dict(driver).get("world_impact")}, f"protagonist_npcs[{index}].behavior_driver[{driver_index}]")

    # 兜底递归扫描，兼容后续脚本把效果放在 decisions 或自定义 effects 下。
    for path, mapping in _walk_dicts(role_data):
        if path.startswith("role_events") or path.startswith("pressure_sources") or path.startswith("protagonist_npcs"):
            continue
        add_effects(mapping, path)
    return influences


def _build_theoretical_bounds(role_data: dict, world_data: dict) -> dict[str, _Bounds]:
    """根据初值与所有已声明变化估算理论上下界。"""

    bounds: dict[str, _Bounds] = {}
    for dimension_id, initial, minimum, maximum in _world_dimension_ranges(world_data):
        bounds[dimension_id] = _Bounds(initial, initial)
        bounds[f"world:{dimension_id}"] = _Bounds(initial, initial)
        bounds[f"range:{dimension_id}"] = _Bounds(minimum, maximum)

    for dimension_id, initial, minimum, maximum in _role_dimension_ranges(role_data):
        bounds[dimension_id] = _Bounds(initial, initial)
        bounds[f"role:{dimension_id}"] = _Bounds(initial, initial)
        bounds[f"range:{dimension_id}"] = _Bounds(minimum, maximum)

    parsed_dimensions = _as_dict(role_data.get("parsed_dimensions"))
    for dimension_id, value in _as_dict(parsed_dimensions.get("world")).items():
        if isinstance(value, int):
            _set_initial(bounds, str(dimension_id), value)
            _set_initial(bounds, f"world:{dimension_id}", value)
    for dimension_id, value in _as_dict(parsed_dimensions.get("character")).items():
        if isinstance(value, int):
            _set_initial(bounds, str(dimension_id), value)
            _set_initial(bounds, f"role:{dimension_id}", value)

    for path, mapping in _walk_dicts({"world": world_data, "role": role_data}):
        _apply_declared_effects(bounds, mapping, path)

    pressure_sources = _parsed_or_raw_pressure_sources(role_data)
    for index, pressure in enumerate(_as_list(pressure_sources.get("decay"))):
        if not isinstance(pressure, dict):
            continue
        dimension_id = pressure.get("dimension")
        rate = pressure.get("rate")
        if not dimension_id or not isinstance(rate, int) or rate == 0:
            continue
        key = str(dimension_id)
        _ensure_bound(bounds, key)
        declared_range = bounds.get(f"range:{key}", _Bounds(0, 10))
        if rate < 0:
            floor = pressure.get("floor") if isinstance(pressure.get("floor"), int) else declared_range.minimum
            bounds[key].minimum = min(bounds[key].minimum, max(declared_range.minimum, floor))
        else:
            ceiling = pressure.get("ceiling") if isinstance(pressure.get("ceiling"), int) else declared_range.maximum
            bounds[key].maximum = max(bounds[key].maximum, min(declared_range.maximum, ceiling))

    for key, value in list(bounds.items()):
        if key.startswith("range:"):
            continue
        range_bound = bounds.get(f"range:{key}")
        if range_bound is None and ":" in key:
            range_bound = bounds.get(f"range:{key.split(':', 1)[1]}")
        minimum = range_bound.minimum if range_bound else 0
        maximum = range_bound.maximum if range_bound else 10
        value.minimum = max(minimum, min(maximum, value.minimum))
        value.maximum = max(minimum, min(maximum, value.maximum))
    return bounds


def _set_initial(bounds: dict[str, _Bounds], key: str, value: int) -> None:
    """设置初始理论范围。"""

    bounds[key] = _Bounds(value, value)


def _world_dimension_ranges(world_data: dict) -> list[tuple[str, int, int, int]]:
    """读取世界维度初始值与范围。"""

    result: list[tuple[str, int, int, int]] = []
    definitions = _as_dict(world_data.get("dimension_definitions"))
    for item in _as_list(definitions.get("world")):
        if not isinstance(item, dict):
            continue
        dimension_id = _definition_id(item)
        if not dimension_id:
            continue
        minimum, maximum = _range_tuple(item.get("range"))
        initial = item.get("initial") if isinstance(item.get("initial"), int) else item.get("default", 5)
        initial = initial if isinstance(initial, int) else 5
        result.append((dimension_id, initial, minimum, maximum))
    return result


def _role_dimension_ranges(role_data: dict) -> list[tuple[str, int, int, int]]:
    """读取角色维度初始值与范围。"""

    result: list[tuple[str, int, int, int]] = []
    role_dimensions = _as_dict(role_data.get("role_dimensions"))
    for item in _as_list(role_dimensions.get("definitions")):
        if not isinstance(item, dict):
            continue
        dimension_id = _definition_id(item)
        if not dimension_id:
            continue
        minimum, maximum = _range_tuple(item.get("range"))
        initial = item.get("initial") if isinstance(item.get("initial"), int) else item.get("default", 5)
        initial = initial if isinstance(initial, int) else 5
        result.append((dimension_id, initial, minimum, maximum))
    return result


def _range_tuple(value: Any) -> tuple[int, int]:
    """读取范围，默认0-10。"""

    if isinstance(value, list) and len(value) >= 2 and isinstance(value[0], int) and isinstance(value[1], int):
        return value[0], value[1]
    return 0, 10


def _apply_declared_effects(bounds: dict[str, _Bounds], mapping: dict, path: str) -> None:
    """把声明式效果合入理论范围。"""

    for effect_key, prefix in (("world_dimensions", "world"), ("role_dimensions", "role"), ("world_impact", "world")):
        effects = mapping.get(effect_key)
        if isinstance(effects, dict):
            for dimension_id, delta in effects.items():
                if isinstance(delta, int) and delta != 0:
                    _apply_delta(bounds, str(dimension_id), delta)
                    _apply_delta(bounds, f"{prefix}:{dimension_id}", delta)

    relation_deltas = mapping.get("npc_relation_deltas")
    if isinstance(relation_deltas, dict):
        for npc_id, deltas in relation_deltas.items():
            if isinstance(deltas, dict):
                for metric, delta in deltas.items():
                    if isinstance(delta, int) and delta != 0:
                        _apply_delta(bounds, f"relation:{npc_id}:{metric}", delta)


def _apply_delta(bounds: dict[str, _Bounds], key: str, delta: int) -> None:
    """应用单次理论变化。"""

    _ensure_bound(bounds, key)
    if delta > 0:
        bounds[key].maximum += delta
    else:
        bounds[key].minimum += delta


def _ensure_bound(bounds: dict[str, _Bounds], key: str) -> None:
    """保证理论范围存在。"""

    if key not in bounds:
        bounds[key] = _Bounds(5, 5)


def _condition_tree_satisfiable(condition: Any, bounds: dict[str, _Bounds], bottom_lines: dict[str, dict], location: str) -> bool:
    """判断条件树是否可能满足。"""

    if not isinstance(condition, dict):
        return True
    if "all" in condition:
        children_possible = all(
            _condition_tree_satisfiable(item, bounds, bottom_lines, f"{location}.all[{index}]")
            for index, item in enumerate(_as_list(condition.get("all")))
        )
        return children_possible and _combined_dimension_conditions_possible(
            _collect_all_dimension_conditions(condition, location), bounds
        )
    if "any" in condition:
        items = _as_list(condition.get("any"))
        return not items or any(
            _condition_tree_satisfiable(item, bounds, bottom_lines, f"{location}.any[{index}]")
            for index, item in enumerate(items)
        )
    if "not" in condition:
        return True
    if condition.get("bottom_line_triggered"):
        bottom_line = bottom_lines.get(str(condition.get("bottom_line_triggered")))
        return True if bottom_line is None else _condition_tree_satisfiable(bottom_line.get("condition"), bounds, bottom_lines, location)

    dimension_condition = _extract_dimension_condition(condition, location)
    if dimension_condition is None:
        return True
    return _dimension_condition_possible(dimension_condition, bounds)


def _extract_dimension_condition(condition: dict, location: str) -> _DimensionCondition | None:
    """从条件节点提取维度约束。"""

    if "dimension" in condition:
        dimension_id = str(condition.get("dimension"))
        operator = str(condition.get("operator") or "")
        value = condition.get("value")
        if operator in _CONDITION_OPERATORS and isinstance(value, int):
            return _DimensionCondition(dimension_id, operator, value, location)
    npc_relation = condition.get("npc_relation")
    if isinstance(npc_relation, dict):
        npc_id = npc_relation.get("npc_id")
        metric = npc_relation.get("metric")
        operator = str(npc_relation.get("operator") or "")
        value = npc_relation.get("value")
        if npc_id and metric and operator in _CONDITION_OPERATORS and isinstance(value, int):
            return _DimensionCondition(f"relation:{npc_id}:{metric}", operator, value, location)
    return None


def _dimension_condition_possible(condition: _DimensionCondition, bounds: dict[str, _Bounds]) -> bool:
    """判断单个维度条件是否可能满足。"""

    bound = _lookup_bound(bounds, condition.key)

    operator = "==" if condition.operator == "=" else condition.operator
    if operator == ">":
        return bound.maximum > condition.value
    if operator == ">=":
        return bound.maximum >= condition.value
    if operator == "<":
        return bound.minimum < condition.value
    if operator == "<=":
        return bound.minimum <= condition.value
    if operator == "==":
        return bound.minimum <= condition.value <= bound.maximum
    if operator == "!=":
        return bound.minimum != condition.value or bound.maximum != condition.value
    return True


def _lookup_bound(bounds: dict[str, _Bounds], key: str) -> _Bounds:
    """按维度键查找理论范围。"""

    bound = bounds.get(key)
    if bound is None and key.startswith("relation:"):
        bound = _Bounds(0, 10)
    if bound is None:
        bound = bounds.get(key.split(":", 1)[-1], _Bounds(0, 10))
    return bound


def _collect_all_dimension_conditions(condition: Any, location: str) -> list[_DimensionCondition]:
    """收集 all 分支下必须同时满足的维度条件。"""

    if not isinstance(condition, dict):
        return []
    extracted = _extract_dimension_condition(condition, location)
    if extracted is not None:
        return [extracted]
    if "all" not in condition:
        return []

    result: list[_DimensionCondition] = []
    for index, item in enumerate(_as_list(condition.get("all"))):
        result.extend(_collect_all_dimension_conditions(item, f"{location}.all[{index}]"))
    return result


def _combined_dimension_conditions_possible(conditions: list[_DimensionCondition], bounds: dict[str, _Bounds]) -> bool:
    """判断同一 all 分支的维度条件是否互相兼容。"""

    by_key: dict[str, list[_DimensionCondition]] = defaultdict(list)
    for condition in conditions:
        by_key[condition.key].append(condition)

    for key, key_conditions in by_key.items():
        bound = _lookup_bound(bounds, key)
        possible_min = bound.minimum
        possible_max = bound.maximum
        forbidden_values: set[int] = set()
        for condition in key_conditions:
            operator = "==" if condition.operator == "=" else condition.operator
            if operator == ">":
                possible_min = max(possible_min, condition.value + 1)
            elif operator == ">=":
                possible_min = max(possible_min, condition.value)
            elif operator == "<":
                possible_max = min(possible_max, condition.value - 1)
            elif operator == "<=":
                possible_max = min(possible_max, condition.value)
            elif operator == "==":
                possible_min = max(possible_min, condition.value)
                possible_max = min(possible_max, condition.value)
            elif operator == "!=":
                forbidden_values.add(condition.value)
        if possible_min > possible_max:
            return False
        if possible_min == possible_max and possible_min in forbidden_values:
            return False
    return True


def _walk_dicts(value: Any, path: str = "") -> list[tuple[str, dict]]:
    """递归遍历所有字典节点。"""

    found: list[tuple[str, dict]] = []
    if isinstance(value, dict):
        found.append((path, value))
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.extend(_walk_dicts(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_dicts(child, f"{path}[{index}]"))
    return found


def _discover_existing_roles(script_id: str, world_data: dict) -> list[tuple[str, Path | None]]:
    """发现已存在的角色层文件。"""

    root = BASE_SCRIPTS_DIR / script_id
    roles: list[tuple[str, Path | None]] = []
    seen: set[str] = set()

    for role_id in _optional_role_ids(world_data):
        if _role_file_exists(root, role_id):
            roles.append((role_id, None))
            seen.add(role_id)

    for path in _role_yaml_files(root):
        loaded_role_id = _read_role_id_from_yaml(path)
        if loaded_role_id and loaded_role_id not in seen:
            roles.append((loaded_role_id, path))
            seen.add(loaded_role_id)
    return roles


def _load_role_file(path: Path) -> dict:
    """直接加载发现到的角色层文件。"""

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return _as_dict(data)


def _role_file_exists(root: Path, role_id: str) -> bool:
    """检查 loader 支持的角色层候选路径是否存在。"""

    return any(
        candidate.exists()
        for candidate in (
            root / f"role_{role_id}.yaml",
            root / "roles" / f"role_{role_id}.yaml",
            root / "roles" / f"{role_id}.yaml",
            root / f"{role_id}.yaml",
        )
    )


def _role_yaml_files(root: Path) -> list[Path]:
    """列出可能的角色层YAML文件。"""

    files: list[Path] = []
    for base in (root, root / "roles"):
        if not base.exists():
            continue
        for path in base.glob("*.yaml"):
            if path.name == "world.yaml":
                continue
            files.append(path)
    return files


def _read_role_id_from_yaml(path: Path) -> str | None:
    """从角色层文件读取 role_id。"""

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except Exception:
        return None
    role_info = _as_dict(_as_dict(data).get("role"))
    role_id = role_info.get("role_id")
    return str(role_id) if role_id else None


def _run_async(awaitable: Any) -> Any:
    """在同步验证接口中运行异步加载器。"""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # 线程中需要显式带回异常。
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_world",
    "validate_role",
    "validate_script",
    "check_dimension_reachability",
    "check_ending_reachability",
    "check_npc_conflict_graph",
    "check_event_coverage",
    "check_pressure_counterplay",
]
