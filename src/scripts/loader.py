"""剧本加载器。"""

from __future__ import annotations

import asyncio
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import yaml

from src.engine.dimension import (
    CHARACTER_DIMENSIONS,
    EXTENSION_DIMENSIONS,
    RELATION_NUMERIC_DIMENSIONS,
    RELATION_TAG_DIMENSIONS,
    WORLD_STATE_DIMENSIONS,
    clamp_dimension_value,
    load_dimension_config_from_dict,
)

BASE_SCRIPTS_DIR = Path("scripts")
logger = logging.getLogger(__name__)

LEGACY_WORLD_DIMENSION_ALIASES: dict[str, str] = {
    "国库": "财政",
    "军力": "兵力",
    "边防": "兵力",
    "民心": "民心",
    "朝廷稳定": "派系势力",
}


def _script_root(script_id: str) -> Path:
    """获取剧本根目录。"""

    return BASE_SCRIPTS_DIR / script_id


def _world_path(script_id: str) -> Path:
    """获取新格式世界层文件路径。"""

    return _script_root(script_id) / "world.yaml"


def is_new_format_script(script_id: str) -> bool:
    """检测剧本是否使用世界层+角色层新格式。"""

    return _world_path(script_id).exists()


def _warn_deprecated(func_name: str) -> None:
    """提示旧格式加载函数已废弃但仍保留兼容。"""

    warnings.warn(
        f"{func_name} 是旧格式加载接口，建议新剧本改用 load_world/load_role。",
        DeprecationWarning,
        stacklevel=2,
    )


def _load_yaml_file_sync(path: Path) -> Any:
    """读取并解析YAML文件（同步实现，仅供to_thread调用）。"""

    if not path.exists():
        raise FileNotFoundError(f"剧本文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _time_to_month_index(value: Any) -> int:
    """将 {year, month} 转为可比较月份序号。"""

    if not isinstance(value, dict):
        return 0
    try:
        year = int(value.get("year", 0))
        month = int(value.get("month", 0))
    except (TypeError, ValueError):
        return 0
    return year * 12 + month


def _condition_to_int(value: Any, default: int = 0) -> int:
    """条件比较使用的安全整数转换。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compare_condition_value(left: Any, op: str, right: Any) -> bool:
    """通用条件比较器，兼容数值和文本。"""

    if op in {"<=", ">=", "<", ">"}:
        left_num = _condition_to_int(left)
        right_num = _condition_to_int(right)
        if op == "<=":
            return left_num <= right_num
        if op == ">=":
            return left_num >= right_num
        if op == "<":
            return left_num < right_num
        return left_num > right_num

    left_text = "" if left is None else str(left)
    right_text = "" if right is None else str(right)
    if op == "!=":
        return left_text != right_text
    return left_text == right_text


def _state_dimension_value(state: Any, key: str) -> Any:
    """从维度池或旧 metrics 中读取条件值。"""

    metrics = getattr(state, "metrics", None)
    if isinstance(metrics, dict) and key in metrics:
        return metrics.get(key)

    dimensions = getattr(state, "dimensions", None)
    for group_name in ("world", "character", "extensions"):
        group = getattr(dimensions, group_name, None) if dimensions is not None else None
        if isinstance(group, dict) and key in group:
            return group.get(key)

    alias = LEGACY_WORLD_DIMENSION_ALIASES.get(key)
    world = getattr(dimensions, "world", None) if dimensions is not None else None
    if alias and isinstance(world, dict):
        return world.get(alias)
    return None


def _get_timeline_condition_value(state: Any, npc_name: str, key: str) -> Any:
    """读取 NPC timeline 条件左值。"""

    if key in {"turn", "回合"}:
        return getattr(state, "turn", 0)
    if key in {"decision_completed", "decisions_completed"}:
        return getattr(state, "decisions_completed", [])
    if key in {"npc_status", "npc_statuses"}:
        return getattr(state, "npc_statuses", {}).get(npc_name)
    if key.startswith("npc_status."):
        target = key.split(".", 1)[1].strip()
        return getattr(state, "npc_statuses", {}).get(target)
    if key in {"chapter_id", "章节"}:
        return getattr(state, "chapter_id", "")
    return _state_dimension_value(state, key)


def _eval_timeline_string_condition(state: Any, npc_name: str, condition: str) -> bool:
    """解析 NPC timeline 字符串条件。"""

    cond = condition.strip()
    if not cond:
        return False

    complete_match = re.match(r"^完成决策[:：]\s*(.+)$", cond)
    if complete_match:
        decision_id = complete_match.group(1).strip()
        return bool(decision_id) and decision_id in getattr(state, "decisions_completed", [])

    contains_match = re.match(
        r"^(decision_completed|decisions_completed)\s+contains\s+(.+)$",
        cond,
        re.IGNORECASE,
    )
    if contains_match:
        _, decision_id = contains_match.groups()
        return decision_id.strip() in getattr(state, "decisions_completed", [])

    normalized = cond.replace(" ", "")
    match = re.match(r"^(.+?)(<=|>=|==|!=|<|>)(.+)$", normalized)
    if not match:
        logger.warning("无法解析NPC时间线条件 npc=%s condition=%s", npc_name, condition)
        return False

    left_key, op, right_raw = match.groups()
    left_val = _get_timeline_condition_value(state, npc_name, left_key)
    if left_key in {"decision_completed", "decisions_completed"} and op in {"==", "!="}:
        exists = right_raw.strip() in getattr(state, "decisions_completed", [])
        return exists if op == "==" else not exists
    return _compare_condition_value(left_val, op, right_raw)


def _eval_timeline_dict_condition(state: Any, npc_name: str, condition: dict[str, Any]) -> bool:
    """解析 NPC timeline 字典条件。"""

    ctype = str(condition.get("type", "metric")).strip().lower()
    if ctype in {"metric", "dimension"}:
        name = str(condition.get("name") or condition.get("dimension") or condition.get("metric") or "").strip()
        op = str(condition.get("op") or condition.get("operator") or "<=")
        value = condition.get("value", condition.get("threshold", 0))
        return _compare_condition_value(_state_dimension_value(state, name), op, value)
    if ctype == "turn":
        op = str(condition.get("op") or condition.get("operator") or ">=")
        return _compare_condition_value(getattr(state, "turn", 0), op, condition.get("value", 0))
    if ctype in {"decision", "decision_completed"}:
        decision_id = str(condition.get("decision_id", "")).strip()
        return bool(decision_id) and decision_id in getattr(state, "decisions_completed", [])
    if ctype == "npc_status":
        target_npc = str(condition.get("npc", npc_name)).strip() or npc_name
        status = str(condition.get("status", "")).strip()
        op = str(condition.get("op") or condition.get("operator") or "==")
        return _compare_condition_value(getattr(state, "npc_statuses", {}).get(target_npc), op, status)

    logger.warning("未知NPC时间线条件类型 npc=%s type=%s", npc_name, ctype)
    return False


def _eval_timeline_condition(state: Any, npc_name: str, condition: Any) -> bool:
    """统一 NPC timeline 条件判定入口。"""

    if isinstance(condition, str):
        return _eval_timeline_string_condition(state, npc_name, condition)
    if isinstance(condition, dict):
        return _eval_timeline_dict_condition(state, npc_name, condition)
    return False


def _latest_decision_choice(state: Any, decision_id: str) -> str | None:
    """读取指定决策最近一次结构化选择。"""

    for record in reversed(getattr(state, "decisions", [])):
        if not isinstance(record, dict):
            continue
        record_decision_id = str(record.get("decision_id", "")).strip()
        if record_decision_id and record_decision_id != decision_id:
            continue
        choice = record.get("choice") or record.get("choice_key") or record.get("decision")
        if choice:
            return str(choice)
    return None


def _is_npc_timeline_event_triggered(event: dict[str, Any], npc_name: str, game_time: dict, state: Any) -> bool:
    """判断 NPC 新格式时间线事件是否已触发。"""

    if _time_to_month_index(event.get("time")) > _time_to_month_index(game_time):
        return False

    trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
    trigger_type = str(trigger.get("type", "timeline")).strip().lower()
    if trigger_type == "timeline":
        return True

    if state is None:
        return False

    if trigger_type == "condition":
        conditions = trigger.get("conditions", [])
        if not isinstance(conditions, list) or not conditions:
            return False
        mode = str(trigger.get("mode", "all")).strip().lower()
        results = [_eval_timeline_condition(state, npc_name, condition) for condition in conditions]
        return any(results) if mode == "any" else all(results)

    if trigger_type == "decision":
        decision_id = str(trigger.get("decision_id", "")).strip()
        if not decision_id or decision_id not in getattr(state, "decisions_completed", []):
            return False
        expected_choice = str(trigger.get("choice", "")).strip().lower()
        if not expected_choice:
            return True
        actual_choice = (_latest_decision_choice(state, decision_id) or "").strip().lower()
        if actual_choice and actual_choice == expected_choice:
            return True
        expected_status = str(trigger.get("status") or event.get("status") or "").strip()
        current_status = str(getattr(state, "npc_statuses", {}).get(npc_name, "")).strip()
        return bool(expected_status) and current_status == expected_status

    logger.warning("未知NPC时间线触发类型 npc=%s event=%s type=%s", npc_name, event.get("id"), trigger_type)
    return False


async def _load_yaml_file(path: Path) -> Any:
    """异步读取并解析YAML文件。"""

    return await asyncio.to_thread(_load_yaml_file_sync, path)


def _ensure_dict(data: Any, path: Path, label: str) -> dict:
    """校验YAML根节点必须是对象。"""

    if not isinstance(data, dict):
        raise ValueError(f"{label} 格式错误，应为对象: {path}")
    return data


def _ensure_list(value: Any, label: str, path: Path) -> list:
    """校验字段必须是数组。"""

    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} 格式错误，应为数组: {path}")
    return value


def _dict_items(value: Any) -> list[dict]:
    """只保留数组中的对象项。"""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _extract_initial_dimension_values(definitions: Any) -> dict[str, int]:
    """从新格式维度定义中提取初始值。"""

    values: dict[str, int] = {}
    for item in _dict_items(definitions):
        dimension_id = item.get("id") or item.get("name")
        if not dimension_id:
            continue
        raw_value = item.get("initial", item.get("default", 5))
        if not isinstance(raw_value, int):
            logger.warning("维度初始值不是整数 dimension=%s value=%r，已使用默认值5", dimension_id, raw_value)
            raw_value = 5
        values[str(dimension_id)] = clamp_dimension_value(raw_value)
    return values


def _extract_dimension_names(definitions: Any) -> list[str]:
    """从新格式维度定义中提取维度ID列表。"""

    names: list[str] = []
    for item in _dict_items(definitions):
        dimension_id = item.get("id") or item.get("name")
        if dimension_id and str(dimension_id) not in names:
            names.append(str(dimension_id))
    return names


def _filter_known_values(values: dict[str, int], known_names: set[str], label: str) -> dict[str, int]:
    """过滤维度系统暂未内置的自定义维度。"""

    unknown_names = [name for name in values if name not in known_names]
    if unknown_names:
        logger.warning("%s包含自定义维度，维度池暂不接管: %s", label, unknown_names)
    return {name: value for name, value in values.items() if name in known_names}


def _filter_known_names(names: list[str], known_names: set[str], label: str) -> list[str]:
    """过滤维度系统暂未内置的关系标签或关系数值名。"""

    unknown_names = [name for name in names if name not in known_names]
    if unknown_names:
        logger.warning("%s包含自定义名称，维度池暂不接管: %s", label, unknown_names)
    return [name for name in names if name in known_names]


def _build_dimension_payload(world: dict, *, compatible_only: bool = False) -> dict[str, Any]:
    """将世界层维度定义转换为 dimension.py 可识别的配置格式。"""

    gameplay = world.get("gameplay") if isinstance(world.get("gameplay"), dict) else {}
    definitions = world.get("dimension_definitions") if isinstance(world.get("dimension_definitions"), dict) else {}

    world_values = _extract_initial_dimension_values(definitions.get("world"))
    character_values = _extract_initial_dimension_values(definitions.get("character"))
    extension_values = _extract_initial_dimension_values(definitions.get("extensions"))
    relation_numeric = _extract_dimension_names(definitions.get("relationship_numeric"))
    relation_tags = _extract_dimension_names(definitions.get("relationship_tags"))

    if compatible_only:
        world_values = _filter_known_values(world_values, set(WORLD_STATE_DIMENSIONS), "世界状态维度")
        character_values = _filter_known_values(character_values, set(CHARACTER_DIMENSIONS), "角色维度")
        extension_values = _filter_known_values(extension_values, set(EXTENSION_DIMENSIONS), "扩展维度")
        relation_numeric = _filter_known_names(relation_numeric, set(RELATION_NUMERIC_DIMENSIONS), "关系数值型维度")
        relation_tags = _filter_known_names(relation_tags, set(RELATION_TAG_DIMENSIONS), "关系标签型维度")

    return {
        "gameplay_types": gameplay.get("gameplay_types", []),
        "world_settings": gameplay.get("world_settings", []),
        "dimensions": {
            "world": world_values,
            "character": character_values,
            "extensions": extension_values,
            "relation_numeric": relation_numeric,
            "relation_tags": relation_tags,
        },
    }


def _parse_world_dimension_definitions(world: dict) -> dict[str, Any]:
    """解析并保留世界层声明的全部维度定义。"""

    definitions = world.get("dimension_definitions") if isinstance(world.get("dimension_definitions"), dict) else {}
    return {
        "world": _extract_initial_dimension_values(definitions.get("world")),
        "character": _extract_initial_dimension_values(definitions.get("character")),
        "extensions": _extract_initial_dimension_values(definitions.get("extensions")),
        "relation_numeric": _extract_dimension_names(definitions.get("relationship_numeric")),
        "relation_tags": _extract_dimension_names(definitions.get("relationship_tags")),
    }


def _attach_world_dimension_config(world: dict) -> None:
    """调用维度池解析世界层维度配置。"""

    world["parsed_dimensions"] = _parse_world_dimension_definitions(world)
    try:
        dimension_config = load_dimension_config_from_dict(_build_dimension_payload(world))
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("世界层维度配置含自定义项，改用内置维度子集解析: %s", exc)
        dimension_config = load_dimension_config_from_dict(_build_dimension_payload(world, compatible_only=True))

    world["dimension_config"] = dimension_config
    world["dimension_initial_state"] = dimension_config.create_state()


def _apply_world_dimension_overrides(world_values: dict[str, int], overrides: Any) -> dict[str, int]:
    """应用角色层对世界维度初始值的覆盖或修正。"""

    result = dict(world_values)
    if not isinstance(overrides, dict):
        return result

    for name, rule in overrides.items():
        current_value = result.get(str(name), 5)
        if isinstance(rule, int):
            result[str(name)] = clamp_dimension_value(rule)
        elif isinstance(rule, dict):
            if isinstance(rule.get("set"), int):
                result[str(name)] = clamp_dimension_value(rule["set"])
            elif isinstance(rule.get("add"), int):
                result[str(name)] = clamp_dimension_value(current_value + rule["add"])
            else:
                logger.warning("无法识别世界维度覆盖规则 dimension=%s rule=%r", name, rule)
        else:
            logger.warning("世界维度覆盖规则格式错误 dimension=%s rule=%r", name, rule)
    return result


def _parse_role_dimensions(role: dict, world: dict) -> dict[str, Any]:
    """解析角色维度与角色视角世界维度。"""

    role_dimensions = role.get("role_dimensions") if isinstance(role.get("role_dimensions"), dict) else {}
    world_definitions = world.get("dimension_definitions") if isinstance(world.get("dimension_definitions"), dict) else {}
    world_values = _extract_initial_dimension_values(world_definitions.get("world"))
    character_values = _extract_initial_dimension_values(role_dimensions.get("definitions"))
    relation_values: dict[str, dict[str, Any]] = {}

    for relation in _dict_items(role.get("npc_relationships")):
        npc_id = relation.get("npc_id")
        if not npc_id:
            continue
        numeric = relation.get("numeric") if isinstance(relation.get("numeric"), dict) else {}
        relation_values[str(npc_id)] = {
            "values": {str(name): clamp_dimension_value(value) for name, value in numeric.items() if isinstance(value, int)},
            "tags": [str(tag) for tag in relation.get("tags", []) if isinstance(tag, str)],
            "stance": relation.get("stance", "unknown"),
            "summary": relation.get("relationship_summary", ""),
            "reveal_policy": relation.get("reveal_policy", "partial"),
        }

    return {
        "character": character_values,
        "world": _apply_world_dimension_overrides(world_values, role_dimensions.get("world_dimension_overrides")),
        "relations": relation_values,
    }


def _parse_pressure_sources(role: dict, path: Path) -> dict[str, list[dict]]:
    """解析角色层压力源结构。"""

    pressure_sources = role.get("pressure_sources") if isinstance(role.get("pressure_sources"), dict) else {}
    parsed: dict[str, list[dict]] = {}
    for key in ("decay", "milestones", "reactions"):
        parsed[key] = _dict_items(_ensure_list(pressure_sources.get(key), f"pressure_sources.{key}", path))
    return parsed


def _parse_endings(role: dict, path: Path) -> dict[str, list[dict]]:
    """解析角色层结局条件结构。"""

    endings = role.get("endings") if isinstance(role.get("endings"), dict) else {}
    parsed = {
        "bottom_lines": _dict_items(_ensure_list(endings.get("bottom_lines"), "endings.bottom_lines", path)),
        "achievement_checks": _dict_items(_ensure_list(endings.get("achievement_checks"), "endings.achievement_checks", path)),
        "catalog": _dict_items(_ensure_list(endings.get("catalog"), "endings.catalog", path)),
    }
    parsed["catalog"].sort(key=lambda item: item.get("priority", 0) if isinstance(item.get("priority", 0), int) else 0, reverse=True)
    return parsed


def _parse_information_pool(role: dict, path: Path) -> list[dict]:
    """解析角色层信息池结构。"""

    information_pool = _dict_items(_ensure_list(role.get("information_pool"), "information_pool", path))
    seen_ids: set[str] = set()
    parsed: list[dict] = []
    for item in information_pool:
        info_id = item.get("info_id")
        if not info_id:
            logger.warning("信息池条目缺少info_id path=%s item=%r", path, item)
            continue
        if str(info_id) in seen_ids:
            logger.warning("信息池条目重复 info_id=%s path=%s", info_id, path)
            continue
        seen_ids.add(str(info_id))
        normalized = dict(item)
        normalized.setdefault("type", "fact")
        normalized.setdefault("known_by", [])
        normalized.setdefault("delivery_priority", "medium")
        normalized.setdefault("delivered", False)
        parsed.append(normalized)
    return parsed


def _load_script_yaml(script_id: str, name: str) -> dict:
    """同步加载剧本YAML文件。"""

    path = _script_root(script_id) / f"{name}.yaml"
    return _ensure_dict(_load_yaml_file_sync(path), path, f"{name}.yaml")


def load_board_regions(script_id: str) -> dict[str, Any]:
    """从剧本YAML加载棋盘区域定义，转换为运行时格式。"""

    world_data = _load_script_yaml(script_id, "world")
    raw_regions = world_data.get("board_regions", {})
    if not isinstance(raw_regions, dict):
        return {}

    regions: dict[str, Any] = {}
    for region_id, region_def in raw_regions.items():
        if not isinstance(region_def, dict):
            continue
        layers: list[dict[str, Any]] = []
        raw_layers = region_def.get("info_layers")
        if not isinstance(raw_layers, list):
            raw_layers = []
        for layer_def in raw_layers:
            if not isinstance(layer_def, dict):
                continue
            fog = None
            raw_fog = layer_def.get("initial_fog")
            if isinstance(raw_fog, dict):
                fog = {
                    "hint": str(raw_fog.get("hint", "")),
                    "unlock_condition": raw_fog.get("unlock_condition", {}),
                    "unlock_text": raw_fog.get("unlock_text"),
                    "unlocked": False,
                }
            layers.append(
                {
                    "id": str(layer_def.get("id", "")),
                    "label": str(layer_def.get("label", "")),
                    "known_text": layer_def.get("initial_known") or "",
                    "fog": fog,
                    "events": [],
                }
            )
        region_id_str = str(region_id)
        regions[region_id_str] = {
            "name": str(region_def.get("name", region_id_str)),
            "category": str(region_def.get("category", "geographic")),
            "layers": layers,
            "news": [],
        }
    return regions


def _role_candidates(script_id: str, role_id: str) -> list[Path]:
    """获取角色层文件候选路径。"""

    root = _script_root(script_id)
    return [
        root / f"role_{role_id}.yaml",
        root / "roles" / f"role_{role_id}.yaml",
        root / "roles" / f"{role_id}.yaml",
        root / f"{role_id}.yaml",
    ]


async def load_world(script_id: str) -> dict:
    """加载新格式世界层 world.yaml。"""

    path = _world_path(script_id)
    logger.info("加载世界层 script_id=%s path=%s", script_id, path)
    data = _ensure_dict(await _load_yaml_file(path), path, "world.yaml")

    yaml_script_id = data.get("script_id")
    if yaml_script_id and yaml_script_id != script_id:
        logger.warning("世界层script_id与目录名不一致 dir=%s yaml=%s", script_id, yaml_script_id)

    _ensure_list(data.get("global_npc_pool"), "global_npc_pool", path)
    _ensure_list(data.get("optional_roles"), "optional_roles", path)
    _attach_world_dimension_config(data)
    return data


async def load_role(script_id: str, role_id: str) -> dict:
    """加载新格式角色层 role_{role_id}.yaml。"""

    path = next((candidate for candidate in _role_candidates(script_id, role_id) if candidate.exists()), None)
    if path is None:
        candidates = ", ".join(str(candidate) for candidate in _role_candidates(script_id, role_id))
        raise FileNotFoundError(f"角色层文件不存在 role_id={role_id} candidates={candidates}")

    logger.info("加载角色层 script_id=%s role_id=%s path=%s", script_id, role_id, path)
    role = _ensure_dict(await _load_yaml_file(path), path, "角色层文件")
    world = await load_world(script_id)

    world_script_id = role.get("world_script_id")
    if world_script_id and world_script_id != script_id and world_script_id != world.get("script_id"):
        raise ValueError(f"角色层world_script_id不匹配: {world_script_id} != {script_id}")

    role_info = role.get("role") if isinstance(role.get("role"), dict) else {}
    loaded_role_id = role_info.get("role_id")
    if loaded_role_id and loaded_role_id != role_id:
        logger.warning("角色层role_id与请求不一致 request=%s yaml=%s", role_id, loaded_role_id)

    role["parsed_pressure_sources"] = _parse_pressure_sources(role, path)
    role["parsed_endings"] = _parse_endings(role, path)
    role["parsed_information_pool"] = _parse_information_pool(role, path)
    role["parsed_dimensions"] = _parse_role_dimensions(role, world)
    return role


async def load_npc_from_world(script_id: str, npc_id: str) -> dict:
    """从世界层global_npc_pool加载指定NPC完整定义。"""

    world = await load_world(script_id)
    for npc in _dict_items(world.get("global_npc_pool")):
        if npc.get("npc_id") == npc_id:
            return npc
    raise KeyError(f"世界层未找到NPC npc_id={npc_id} script_id={script_id}")


async def load_available_roles(script_id: str) -> list[dict]:
    """从世界层加载可选角色列表。"""

    world = await load_world(script_id)
    return _dict_items(world.get("optional_roles"))


async def load_manifest(script_id: str) -> dict:
    """加载剧本元信息。

    已废弃（deprecated）：旧章节制接口。新格式剧本请使用 load_world。
    """

    _warn_deprecated("load_manifest")
    if is_new_format_script(script_id):
        logger.info("检测到新格式剧本，load_manifest转为加载世界层 script_id=%s", script_id)
        return await load_world(script_id)

    path = _script_root(script_id) / "manifest.yaml"
    logger.info("加载manifest script_id=%s path=%s", script_id, path)
    return _ensure_dict(await _load_yaml_file(path), path, "manifest.yaml")


async def load_chapter(script_id: str, chapter_id: str) -> dict:
    """加载章节定义。

    已废弃（deprecated）：旧章节制接口。新格式剧本请使用 load_world/load_role。
    """

    _warn_deprecated("load_chapter")
    if is_new_format_script(script_id):
        raise ValueError(f"新格式剧本没有章节文件，请使用load_world/load_role: {script_id}")

    path = _script_root(script_id) / "chapters" / f"{chapter_id}.yaml"
    logger.info("加载章节 script_id=%s chapter_id=%s path=%s", script_id, chapter_id, path)
    return _ensure_dict(await _load_yaml_file(path), path, "章节文件")


async def load_npc_profile(script_id: str, npc_name: str) -> dict:
    """加载NPC人格卡。

    已废弃（deprecated）：旧NPC文件接口。新格式剧本请使用 load_npc_from_world。
    """

    _warn_deprecated("load_npc_profile")
    if is_new_format_script(script_id):
        logger.info("检测到新格式剧本，load_npc_profile转为加载世界层NPC script_id=%s npc=%s", script_id, npc_name)
        return await load_npc_from_world(script_id, npc_name)

    path = _script_root(script_id) / "fixed_npcs" / f"{npc_name}.yaml"
    logger.info("加载NPC人格卡 script_id=%s npc=%s path=%s", script_id, npc_name, path)
    return _ensure_dict(await _load_yaml_file(path), path, "NPC文件")


def load_npc_profile_at_time(script_id: str, npc_name: str, game_time: dict, state: Any) -> dict:
    """同步加载NPC在指定时间点的完整profile，兼容旧格式NPC文件。"""

    path = _script_root(script_id) / "fixed_npcs" / f"{npc_name}.yaml"
    logger.info("按时间加载NPC人格卡 script_id=%s npc=%s time=%s path=%s", script_id, npc_name, game_time, path)
    data = _ensure_dict(_load_yaml_file_sync(path), path, "NPC文件")

    if "character_seed" not in data:
        logger.warning("NPC文件为旧格式，降级使用原始profile script_id=%s npc=%s", script_id, npc_name)
        return data

    character_seed = data.get("character_seed") if isinstance(data.get("character_seed"), dict) else {}
    initial_state = data.get("initial_state") if isinstance(data.get("initial_state"), dict) else {}
    current_state = {
        "title": initial_state.get("title", ""),
        "faction": initial_state.get("faction", ""),
        "location": initial_state.get("location", ""),
        "status": initial_state.get("status", ""),
        "situation": initial_state.get("situation", ""),
        "public_profile": initial_state.get("public_profile", ""),
    }
    initial_relations = initial_state.get("initial_relations") if isinstance(initial_state.get("initial_relations"), dict) else {}

    experiences: list[dict[str, Any]] = []
    timeline = data.get("timeline") if isinstance(data.get("timeline"), list) else []
    for raw_event in timeline:
        if not isinstance(raw_event, dict):
            continue
        if not _is_npc_timeline_event_triggered(raw_event, npc_name, game_time, state):
            continue

        if raw_event.get("situation_update") is not None:
            current_state["situation"] = raw_event.get("situation_update")
        if raw_event.get("location") is not None:
            current_state["location"] = raw_event.get("location")
        if raw_event.get("status") is not None:
            current_state["status"] = raw_event.get("status")
        if raw_event.get("title_update") is not None:
            current_state["title"] = raw_event.get("title_update")
        if raw_event.get("public_profile_update") is not None:
            current_state["public_profile"] = raw_event.get("public_profile_update")

        experiences.append(
            {
                "id": raw_event.get("id", ""),
                "name": raw_event.get("name", ""),
                "situation_update": raw_event.get("situation_update", ""),
            }
        )

    return {
        "name": character_seed.get("name", npc_name),
        "character_seed": character_seed,
        "current_state": current_state,
        "experiences": experiences,
        "initial_relations": initial_relations,
        "is_new_format": True,
    }


async def load_decision_point(script_id: str, decision_id: str) -> dict:
    """加载决策点定义。

    已废弃（deprecated）：旧决策点接口。新格式剧本请使用角色/事件结构化数据。
    """

    _warn_deprecated("load_decision_point")
    if is_new_format_script(script_id):
        raise ValueError(f"新格式剧本没有decisions目录，请使用load_world/load_role: {script_id}")

    path = _script_root(script_id) / "decisions" / f"{decision_id}.yaml"
    logger.info("加载决策点 script_id=%s decision_id=%s path=%s", script_id, decision_id, path)
    return _ensure_dict(await _load_yaml_file(path), path, "决策点文件")


async def load_timeline(script_id: str) -> list[dict]:
    """加载时间线。

    已废弃（deprecated）：旧时间线接口。新格式剧本会映射到 timeline_milestones。
    """

    _warn_deprecated("load_timeline")
    if is_new_format_script(script_id):
        world = await load_world(script_id)
        return _dict_items(world.get("timeline_milestones"))

    path = _script_root(script_id) / "timeline.yaml"
    if not path.exists():
        return []
    logger.info("加载时间线 script_id=%s path=%s", script_id, path)
    data = await _load_yaml_file(path)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"timeline.yaml 格式错误，应为数组: {path}")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"timeline.yaml 项格式错误，应为对象: {path}")
    return data


async def load_npc_events(script_id: str, npc_name: str) -> list[dict]:
    """加载NPC事件列表，缺失时返回空数组。

    已废弃（deprecated）：旧NPC事件接口。新格式剧本请读取 world/role 事件池。
    """

    _warn_deprecated("load_npc_events")
    if is_new_format_script(script_id):
        logger.info("新格式剧本不再按NPC文件加载事件 script_id=%s npc=%s", script_id, npc_name)
        return []

    root = _script_root(script_id)
    candidates = [
        root / "fixed_npcs" / f"{npc_name}.yaml",
        root / "npc_events" / f"{npc_name}.yaml",
    ]

    for path in candidates:
        if not path.exists():
            continue
        logger.info("加载NPC事件 script_id=%s npc=%s path=%s", script_id, npc_name, path)
        data = await _load_yaml_file(path)
        if isinstance(data, dict):
            events = data.get("events", [])
            if isinstance(events, list):
                return [item for item in events if isinstance(item, dict)]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    logger.warning("未找到NPC事件文件 script_id=%s npc=%s", script_id, npc_name)
    return []


async def load_decision(script_id: str, decision_id: str) -> dict:
    """加载决策定义（与load_decision_point等价）。

    已废弃（deprecated）：旧决策点接口。新格式剧本请使用角色/事件结构化数据。
    """

    _warn_deprecated("load_decision")

    return await load_decision_point(script_id, decision_id)


async def load_endings(script_id: str, role: str) -> list[dict]:
    """按角色加载结局列表，缺失时返回空数组。

    已废弃（deprecated）：旧结局接口。新格式剧本请使用 load_role(...)["parsed_endings"]。
    """

    _warn_deprecated("load_endings")
    if is_new_format_script(script_id):
        role_data = await load_role(script_id, role)
        return role_data["parsed_endings"]["catalog"]

    path = _script_root(script_id) / "endings" / f"{role}_endings.yaml"
    if not path.exists():
        logger.warning("未找到结局文件 script_id=%s role=%s path=%s", script_id, role, path)
        return []
    logger.info("加载结局 script_id=%s role=%s path=%s", script_id, role, path)
    data = await _load_yaml_file(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        endings = data.get("endings", [])
        if isinstance(endings, list):
            return [item for item in endings if isinstance(item, dict)]
    return []


async def load_sideline(script_id: str, sideline_id: str) -> dict:
    """加载支线定义。

    已废弃（deprecated）：旧支线接口。新格式剧本请使用角色/事件结构化数据。
    """

    _warn_deprecated("load_sideline")
    if is_new_format_script(script_id):
        raise ValueError(f"新格式剧本没有sidelines目录，请使用load_world/load_role: {script_id}")

    path = _script_root(script_id) / "sidelines" / f"{sideline_id}.yaml"
    logger.info("加载支线 script_id=%s sideline_id=%s path=%s", script_id, sideline_id, path)
    return _ensure_dict(await _load_yaml_file(path), path, "支线文件")


async def load_event_file(script_id: str, filename: str) -> list[dict]:
    """加载events/目录下的事件文件，返回事件列表。

    已废弃（deprecated）：旧事件文件接口。新格式剧本会映射到世界层事件池。
    """

    _warn_deprecated("load_event_file")
    if is_new_format_script(script_id):
        world = await load_world(script_id)
        random_event_pool = world.get("random_event_pool")
        event_map = {
            "timeline.yaml": world.get("timeline_milestones"),
            "random_pool.yaml": random_event_pool.get("entries") if isinstance(random_event_pool, dict) else [],
            "disasters.yaml": world.get("condition_events"),
            "npc_events.yaml": [],
        }
        return _dict_items(event_map.get(filename, []))

    path = _script_root(script_id) / "events" / filename
    if not path.exists():
        return []
    data = await _load_yaml_file(path)
    if isinstance(data, dict):
        for key in ("timeline", "random_pool", "disasters", "npc_events"):
            if key in data and isinstance(data[key], list):
                return [item for item in data[key] if isinstance(item, dict)]
        first_list = next((v for v in data.values() if isinstance(v, list)), None)
        if first_list:
            return [item for item in first_list if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def load_all_fixed_npc_initial_locations(script_id: str) -> dict[str, dict[str, str]]:
    """加载剧本下所有 fixed_npcs 的初始位置信息。返回 {npc_name: {location, status, faction}}。"""
    base_path = Path(__file__).resolve().parent.parent.parent / "scripts" / script_id / "fixed_npcs"
    result: dict[str, dict[str, str]] = {}
    if not base_path.is_dir():
        return result
    for yaml_path in sorted(base_path.glob("*.yaml")):
        try:
            raw = _load_yaml_file_sync(yaml_path)
            if not isinstance(raw, dict) or "character_seed" not in raw:
                continue
            seed = raw.get("character_seed", {})
            initial = raw.get("initial_state", {})
            name = str(seed.get("name", yaml_path.stem))
            result[name] = {
                "location": str(initial.get("location", "未知")),
                "status": str(initial.get("status", "未知")),
                "faction": str(initial.get("faction", "未知")),
                "title": str(initial.get("title", "")),
            }
        except Exception:
            continue
    return result


def load_all_functional_npc_initial_locations(script_id: str) -> dict[str, dict[str, str]]:
    """加载剧本下所有 functional_npcs 的初始位置信息。返回 {npc_name: {location, status, faction}}。"""
    base_path = Path(__file__).resolve().parent.parent.parent / "scripts" / script_id / "functional_npcs"
    result: dict[str, dict[str, str]] = {}
    if not base_path.is_dir():
        return result
    for yaml_path in sorted(base_path.glob("*.yaml")):
        try:
            raw = _load_yaml_file_sync(yaml_path)
            if not isinstance(raw, dict) or "character_seed" not in raw:
                continue
            seed = raw.get("character_seed", {})
            initial = raw.get("initial_state", {})
            name = str(seed.get("name", yaml_path.stem))
            result[name] = {
                "location": str(initial.get("location", "未知")),
                "status": str(initial.get("status", "未知")),
                "faction": str(initial.get("faction", "未知")),
                "title": str(initial.get("title", "")),
            }
        except Exception:
            continue
    return result


def load_all_functional_npc_profiles(script_id: str) -> dict[str, dict]:
    """加载剧本下所有 functional_npcs 的完整profile，供 active_npcs 使用。"""
    base_path = Path(__file__).resolve().parent.parent.parent / "scripts" / script_id / "functional_npcs"
    result: dict[str, dict] = {}
    if not base_path.is_dir():
        return result
    for yaml_path in sorted(base_path.glob("*.yaml")):
        try:
            raw = _load_yaml_file_sync(yaml_path)
            if not isinstance(raw, dict) or "character_seed" not in raw:
                continue
            character_seed = raw.get("character_seed") if isinstance(raw.get("character_seed"), dict) else {}
            initial_state = raw.get("initial_state") if isinstance(raw.get("initial_state"), dict) else {}
            name = str(character_seed.get("name", yaml_path.stem))
            title = str(initial_state.get("title", ""))
            traits = character_seed.get("personality_traits") if isinstance(character_seed.get("personality_traits"), list) else []
            trait_text = "、".join(str(item) for item in traits if str(item).strip())
            speaking_style = str(character_seed.get("speaking_style", "")).strip()
            personality_parts = [part for part in (trait_text, speaking_style) if part]
            result[name] = {
                "npc_id": name,
                "name": name,
                "title": title,
                "personality": "；".join(personality_parts),
                "faction": str(initial_state.get("faction", "")),
                "character_seed": character_seed,
                "initial_state": initial_state,
                "is_functional": True,
            }
        except Exception:
            continue
    return result
