"""通用维度池系统。

本模块只维护维度定义、剧本维度配置、运行态维度值，以及判定修正值计算。
不同剧本可以只声明自己需要的维度子集，不需要启用完整维度池。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


logger = logging.getLogger(__name__)


DIMENSION_MIN_VALUE = 0
DIMENSION_MAX_VALUE = 10
DEFAULT_DIMENSION_VALUE = 5
ASSISTANT_DIMENSION_LIMIT = 2
ASSISTANT_BONUS_THRESHOLD = 7
MODIFIER_CAP = 5


@dataclass(frozen=True)
class DimensionDefinition:
    """维度定义，描述某个维度属于哪一类以及取值规则。"""

    name: str
    category: str
    description: str = ""
    min_value: int = DIMENSION_MIN_VALUE
    max_value: int = DIMENSION_MAX_VALUE
    per_npc: bool = False
    fixed_bonus: int = 0


@dataclass(frozen=True)
class DimensionReference:
    """一次判定中引用的维度。"""

    name: str
    category: str = "auto"
    npc_id: str = ""


@dataclass
class RelationDimensions:
    """单个 NPC 的关系维度。"""

    values: dict[str, int] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class DimensionState:
    """剧本运行态维度值。"""

    character: dict[str, int] = field(default_factory=dict)
    world: dict[str, int] = field(default_factory=dict)
    extensions: dict[str, int] = field(default_factory=dict)
    relations: dict[str, RelationDimensions] = field(default_factory=dict)


@dataclass
class DimensionConfig:
    """剧本维度配置，包含启用的维度子集和初始值。"""

    gameplay_types: list[str] = field(default_factory=list)
    world_settings: list[str] = field(default_factory=list)
    character: dict[str, int] = field(default_factory=dict)
    world: dict[str, int] = field(default_factory=dict)
    extensions: dict[str, int] = field(default_factory=dict)
    relation_numeric: list[str] = field(default_factory=list)
    relation_tags: list[str] = field(default_factory=list)
    relations: dict[str, RelationDimensions] = field(default_factory=dict)

    def create_state(self) -> DimensionState:
        """根据配置创建一份可写的运行态维度值。"""

        return DimensionState(
            character=dict(self.character),
            world=dict(self.world),
            extensions=dict(self.extensions),
            relations={
                npc_id: RelationDimensions(values=dict(relation.values), tags=list(relation.tags))
                for npc_id, relation in self.relations.items()
            },
        )


# 角色维度：描述玩家自身能力，统一范围 0-10。
CHARACTER_DIMENSIONS: dict[str, DimensionDefinition] = {
    "武力": DimensionDefinition("武力", "character", "个人战斗、军事指挥或武力威慑能力"),
    "智谋": DimensionDefinition("智谋", "character", "分析局势、谋划策略和识破计谋的能力"),
    "口才": DimensionDefinition("口才", "character", "说服、谈判、欺骗和公开表达能力"),
    "意志": DimensionDefinition("意志", "character", "承压、坚持、抗诱惑和精神韧性"),
    "感知": DimensionDefinition("感知", "character", "观察、直觉、侦查和察觉异常能力"),
    "魅力": DimensionDefinition("魅力", "character", "个人吸引力、亲和力和社交影响力"),
}

# 世界状态维度：描述剧本世界状态，统一范围 0-10。
WORLD_STATE_DIMENSIONS: dict[str, DimensionDefinition] = {
    "财政": DimensionDefinition("财政", "world", "可用钱粮、资金或经济余裕"),
    "兵力": DimensionDefinition("兵力", "world", "可调动的军事力量"),
    "补给": DimensionDefinition("补给", "world", "粮草、弹药、生存物资或后勤水平"),
    "情报": DimensionDefinition("情报", "world", "对局势、线索和敌情的掌握程度"),
    "民心": DimensionDefinition("民心", "world", "百姓、民众或基层群体的支持度"),
    "士气": DimensionDefinition("士气", "world", "军队、团队或组织的行动意愿"),
    "派系势力": DimensionDefinition("派系势力", "world", "所属派系或阵营的整体实力"),
}

# 数值型关系维度：每个 NPC 独立维护，统一范围 0-10。
RELATION_NUMERIC_DIMENSIONS: dict[str, DimensionDefinition] = {
    "好感度": DimensionDefinition("好感度", "relation_numeric", "NPC 对玩家个人的态度", per_npc=True),
    "忠诚度": DimensionDefinition("忠诚度", "relation_numeric", "NPC 对玩家的追随或服从意愿", per_npc=True),
    "利益绑定": DimensionDefinition("利益绑定", "relation_numeric", "玩家与 NPC 之间利害关系的深度", per_npc=True),
}

# 标签型关系维度：命中相关标签时提供固定 +1 加成。
RELATION_TAG_DIMENSIONS: dict[str, DimensionDefinition] = {
    "亲缘": DimensionDefinition("亲缘", "relation_tag", "血缘、姻亲或类亲属关系", per_npc=True, fixed_bonus=1),
    "旧交": DimensionDefinition("旧交", "relation_tag", "同窗、同袍、故友等旧关系", per_npc=True, fixed_bonus=1),
    "恩怨": DimensionDefinition("恩怨", "relation_tag", "救命之恩、杀亲之仇等强因果关系", per_npc=True, fixed_bonus=1),
    "师徒": DimensionDefinition("师徒", "relation_tag", "师徒、门生、旧部或上下级关系", per_npc=True, fixed_bonus=1),
}

# 世界观扩展维度：剧本可按需启用，仍遵循 0-10 数值范围。
EXTENSION_DIMENSIONS: dict[str, DimensionDefinition] = {
    "魔力": DimensionDefinition("魔力", "extension", "施法资源或魔法亲和力"),
    "法力": DimensionDefinition("法力", "extension", "施法资源或魔法储备"),
    "理智值": DimensionDefinition("理智值", "extension", "精神承受极限和稳定程度"),
    "魅惑": DimensionDefinition("魅惑", "extension", "诱导、操控和暗黑社交影响力"),
    "腐化度": DimensionDefinition("腐化度", "extension", "道德滑坡、污染或堕落程度"),
    "装备": DimensionDefinition("装备", "extension", "武器、护甲和冒险装备等级"),
    "身份隐匿度": DimensionDefinition("身份隐匿度", "extension", "真实身份或秘密暴露前的隐蔽程度"),
}

ALL_DIMENSION_DEFINITIONS: dict[str, DimensionDefinition] = {
    **CHARACTER_DIMENSIONS,
    **WORLD_STATE_DIMENSIONS,
    **RELATION_NUMERIC_DIMENSIONS,
    **RELATION_TAG_DIMENSIONS,
    **EXTENSION_DIMENSIONS,
}


# 玩法类型到推荐维度的映射，仅用于剧本未显式声明维度时自动裁剪。
GAMEPLAY_DIMENSION_MAP: dict[str, dict[str, list[str]]] = {
    "权谋政斗": {
        "character": ["口才", "智谋", "魅力", "意志"],
        "world": ["财政", "情报", "民心", "派系势力"],
        "relation_numeric": ["好感度", "忠诚度", "利益绑定"],
    },
    "战争策略": {
        "character": ["武力", "智谋", "意志"],
        "world": ["兵力", "补给", "民心", "士气"],
        "relation_numeric": ["忠诚度", "利益绑定"],
    },
    "悬疑推理": {
        "character": ["智谋", "口才", "感知", "意志"],
        "world": ["情报"],
        "relation_numeric": ["好感度"],
    },
    "冒险探索": {
        "character": ["武力", "智谋", "感知", "意志"],
        "world": ["补给", "士气"],
        "relation_numeric": ["好感度", "忠诚度"],
    },
    "社交推理": {
        "character": ["口才", "智谋", "感知", "魅力"],
        "world": ["情报"],
        "relation_numeric": ["好感度", "利益绑定"],
    },
    "经营模拟": {
        "character": ["口才", "智谋", "魅力"],
        "world": ["财政", "民心", "派系势力"],
        "relation_numeric": ["好感度", "利益绑定"],
    },
    "恐怖生存": {
        "character": ["武力", "智谋", "意志", "感知"],
        "world": ["补给", "情报", "士气"],
        "relation_numeric": ["好感度", "忠诚度"],
    },
}


WORLD_SETTING_EXTENSION_MAP: dict[str, list[str]] = {
    "奇幻": ["魔力"],
    "魔法": ["魔力", "法力"],
    "恐怖": ["理智值"],
    "克苏鲁": ["理智值"],
    "暗黑": ["魅惑", "腐化度"],
    "堕落": ["魅惑", "腐化度"],
    "冒险": ["装备"],
    "战斗": ["装备"],
    "谍战": ["身份隐匿度"],
    "隐匿": ["身份隐匿度"],
}


def validate_dimension_value(value: int, name: str = "维度值") -> int:
    """校验维度值是否在 0-10 范围内。"""

    if not isinstance(value, int):
        raise TypeError(f"{name}必须是整数: {value!r}")
    if value < DIMENSION_MIN_VALUE or value > DIMENSION_MAX_VALUE:
        raise ValueError(f"{name}必须在{DIMENSION_MIN_VALUE}-{DIMENSION_MAX_VALUE}之间: {value}")
    return value


def clamp_dimension_value(value: int) -> int:
    """将维度值限制在 0-10 范围内。"""

    return max(DIMENSION_MIN_VALUE, min(DIMENSION_MAX_VALUE, int(value)))


def calculate_modifier_from_values(
    main_value: int,
    auxiliary_values: Sequence[int] | None = None,
    tag_count: int = 0,
) -> int:
    """按主维度、辅助维度和标签数量计算判定修正值。

    规则：主维度值 // 3 + 最多两个辅助维度中每个大于等于 7 的 +1 + 每个标签 +1，
    最终限制在 [-5, +5]。
    """

    try:
        validate_dimension_value(main_value, "主维度值")
    except (TypeError, ValueError) as exc:
        logger.warning("主维度值无效，使用默认值%s: %s", DEFAULT_DIMENSION_VALUE, exc)
        main_value = DEFAULT_DIMENSION_VALUE
    if tag_count < 0:
        logger.warning("标签数量不能为负数，已按0处理: %s", tag_count)
        tag_count = 0

    modifier = main_value // 3
    for index, value in enumerate(list(auxiliary_values or [])[:ASSISTANT_DIMENSION_LIMIT], start=1):
        try:
            validate_dimension_value(value, f"辅助维度值{index}")
        except (TypeError, ValueError) as exc:
            logger.warning("辅助维度值无效，使用默认值%s: %s", DEFAULT_DIMENSION_VALUE, exc)
            value = DEFAULT_DIMENSION_VALUE
        if value >= ASSISTANT_BONUS_THRESHOLD:
            modifier += 1

    modifier += tag_count
    return max(-MODIFIER_CAP, min(MODIFIER_CAP, modifier))


def calculate_action_modifier(
    state: DimensionState,
    main_dimension: DimensionReference | str,
    auxiliary_dimensions: Sequence[DimensionReference | str] | None = None,
    tags: Sequence[DimensionReference | str] | None = None,
    npc_id: str = "",
) -> int:
    """根据运行态维度值计算一次行动的修正值。"""

    main_value = get_dimension_value(state, _ensure_reference(main_dimension, npc_id=npc_id))
    auxiliary_values = [
        get_dimension_value(state, _ensure_reference(item, npc_id=npc_id))
        for item in list(auxiliary_dimensions or [])[:ASSISTANT_DIMENSION_LIMIT]
    ]
    try:
        tag_count = count_available_tags(state, tags or [], npc_id=npc_id)
    except KeyError as exc:
        logger.warning("关系标签引用无效，已忽略: %s", exc)
        tag_count = 0
    return calculate_modifier_from_values(main_value, auxiliary_values, tag_count)


def get_dimension_value(state: DimensionState, reference: DimensionReference) -> int:
    """从运行态中读取某个维度值。"""

    try:
        category = _resolve_category(reference)
        _validate_reference_name(reference.name, category)
    except (KeyError, ValueError) as exc:
        logger.warning("维度引用无效，使用默认值%s: %s", DEFAULT_DIMENSION_VALUE, exc)
        return DEFAULT_DIMENSION_VALUE
    if category == "character":
        return _get_optional_value(state.character, reference.name, "角色维度")
    if category == "world":
        return _get_optional_value(state.world, reference.name, "世界状态维度")
    if category == "extension":
        return _get_optional_value(state.extensions, reference.name, "扩展维度")
    if category == "relation_numeric":
        try:
            npc_id = _require_npc_id(reference)
        except ValueError as exc:
            logger.warning("关系维度缺少NPC，使用默认值%s: %s", DEFAULT_DIMENSION_VALUE, exc)
            return DEFAULT_DIMENSION_VALUE
        relation = state.relations.get(npc_id)
        if relation is None:
            logger.warning("未找到NPC关系维度，使用默认值%s: %s", DEFAULT_DIMENSION_VALUE, npc_id)
            return DEFAULT_DIMENSION_VALUE
        return _get_optional_value(relation.values, reference.name, f"{npc_id}关系维度")
    logger.warning("不支持读取数值的维度类型，使用默认值%s: %s", DEFAULT_DIMENSION_VALUE, category)
    return DEFAULT_DIMENSION_VALUE


def set_dimension_value(state: DimensionState, reference: DimensionReference, value: int) -> int:
    """设置维度值，要求新值必须在 0-10 范围内。"""

    validated_value = validate_dimension_value(value, reference.name)
    bucket = _get_value_bucket(state, reference)
    bucket[reference.name] = validated_value
    return validated_value


def update_dimension_value(
    state: DimensionState,
    reference: DimensionReference,
    delta: int,
    *,
    clamp: bool = True,
) -> int:
    """按增量更新维度值。

    默认会把更新后的值裁剪到 0-10；如果 clamp=False，则越界时抛出异常。
    """

    if not isinstance(delta, int):
        raise TypeError(f"维度增量必须是整数: {delta!r}")
    current_value = get_dimension_value(state, reference)
    next_value = current_value + delta
    if clamp:
        next_value = clamp_dimension_value(next_value)
    return set_dimension_value(state, reference, next_value)


def load_dimension_config_from_yaml(path: str | Path) -> DimensionConfig:
    """从剧本 YAML 文件加载维度配置。"""

    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"维度配置YAML格式错误，应为对象: {yaml_path}")
    return load_dimension_config_from_dict(payload)


def load_dimension_config_from_dict(payload: Mapping[str, Any]) -> DimensionConfig:
    """从剧本配置对象加载维度配置。"""

    gameplay_types = _as_str_list(payload.get("gameplay_types", []))
    world_settings = _as_str_list(payload.get("world_settings", []))
    dimension_payload = _extract_dimension_payload(payload)

    character = _parse_dimension_values(dimension_payload, ["character", "角色维度", "characters"])
    world = _parse_dimension_values(dimension_payload, ["world", "世界状态维度", "world_state"])
    extensions = _parse_dimension_values(dimension_payload, ["extensions", "extension", "扩展维度"])

    relation_numeric = _parse_dimension_names(
        dimension_payload,
        ["relation_numeric", "numeric_relations", "关系数值型"],
    )
    relation_tags = _parse_dimension_names(
        dimension_payload,
        ["relation_tags", "tags", "关系标签型"],
    )

    if not character and not world and not extensions:
        auto_character, auto_world, auto_extensions, auto_relations = _derive_dimensions(
            gameplay_types,
            world_settings,
        )
        character = {name: DEFAULT_DIMENSION_VALUE for name in auto_character}
        world = {name: DEFAULT_DIMENSION_VALUE for name in auto_world}
        extensions = {name: DEFAULT_DIMENSION_VALUE for name in auto_extensions}
        if not relation_numeric:
            relation_numeric = auto_relations

    if not relation_numeric:
        relation_numeric = list(RELATION_NUMERIC_DIMENSIONS)
    if not relation_tags:
        relation_tags = list(RELATION_TAG_DIMENSIONS)

    _validate_dimension_bucket(character, CHARACTER_DIMENSIONS, "角色维度")
    _validate_dimension_bucket(world, WORLD_STATE_DIMENSIONS, "世界状态维度")
    _validate_dimension_bucket(extensions, EXTENSION_DIMENSIONS, "扩展维度")
    _validate_dimension_names(relation_numeric, RELATION_NUMERIC_DIMENSIONS, "关系数值型维度")
    _validate_dimension_names(relation_tags, RELATION_TAG_DIMENSIONS, "关系标签型维度")

    relations = _parse_relations(dimension_payload.get("relations") or payload.get("npc_relations", {}))
    for npc_id, relation in relations.items():
        _validate_dimension_bucket(relation.values, RELATION_NUMERIC_DIMENSIONS, f"{npc_id}关系数值型维度")
        _validate_dimension_names(relation.tags, RELATION_TAG_DIMENSIONS, f"{npc_id}关系标签型维度")

    return DimensionConfig(
        gameplay_types=gameplay_types,
        world_settings=world_settings,
        character=character,
        world=world,
        extensions=extensions,
        relation_numeric=relation_numeric,
        relation_tags=relation_tags,
        relations=relations,
    )


def count_available_tags(
    state: DimensionState,
    tags: Sequence[DimensionReference | str],
    npc_id: str = "",
) -> int:
    """统计本次行动实际命中的关系标签数量。"""

    count = 0
    for item in tags:
        reference = _ensure_reference(item, category="relation_tag", npc_id=npc_id)
        if reference.name not in RELATION_TAG_DIMENSIONS:
            raise KeyError(f"未知关系标签型维度: {reference.name}")
        if reference.npc_id:
            relation = state.relations.get(reference.npc_id)
            if relation is not None and reference.name in relation.tags:
                count += 1
        else:
            count += 1
    return count


def _derive_dimensions(
    gameplay_types: Sequence[str],
    world_settings: Sequence[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """根据玩法类型和世界观自动推导推荐维度子集。"""

    character: list[str] = []
    world: list[str] = []
    extensions: list[str] = []
    relation_numeric: list[str] = []

    for gameplay_type in gameplay_types:
        mapping = GAMEPLAY_DIMENSION_MAP.get(gameplay_type, {})
        character.extend(mapping.get("character", []))
        world.extend(mapping.get("world", []))
        relation_numeric.extend(mapping.get("relation_numeric", []))

    for world_setting in world_settings:
        extensions.extend(WORLD_SETTING_EXTENSION_MAP.get(world_setting, []))

    return (
        _unique(character),
        _unique(world),
        _unique(extensions),
        _unique(relation_numeric),
    )


def _extract_dimension_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """兼容 dimensions / dimension_pool / 顶层声明三种写法。"""

    for key in ("dimensions", "dimension_pool", "dimension_config"):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    return dict(payload)


def _parse_dimension_values(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, int]:
    """解析维度初始值，支持列表或 name->value 映射。"""

    raw_value = _first_existing(payload, keys)
    if raw_value is None:
        return {}
    if isinstance(raw_value, list):
        return {str(name): DEFAULT_DIMENSION_VALUE for name in raw_value if isinstance(name, str)}
    if isinstance(raw_value, dict):
        result: dict[str, int] = {}
        for name, value in raw_value.items():
            if isinstance(value, int):
                result[str(name)] = validate_dimension_value(value, str(name))
            elif value is None:
                result[str(name)] = DEFAULT_DIMENSION_VALUE
            else:
                raise TypeError(f"维度{name}初始值必须是整数或空值: {value!r}")
        return result
    raise TypeError(f"维度声明必须是列表或对象: {raw_value!r}")


def _parse_dimension_names(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> list[str]:
    """解析只声明名称、不声明数值的维度列表。"""

    raw_value = _first_existing(payload, keys)
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return _unique(str(item) for item in raw_value if isinstance(item, str))
    if isinstance(raw_value, dict):
        return _unique(str(name) for name in raw_value.keys())
    raise TypeError(f"维度名称声明必须是列表或对象: {raw_value!r}")


def _parse_relations(raw_value: Any) -> dict[str, RelationDimensions]:
    """解析 per-NPC 关系初始值。"""

    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise TypeError(f"NPC关系维度声明必须是对象: {raw_value!r}")

    relations: dict[str, RelationDimensions] = {}
    for npc_id, relation_payload in raw_value.items():
        if not isinstance(relation_payload, dict):
            raise TypeError(f"NPC关系声明必须是对象: {npc_id}")
        value_payload = relation_payload.get("values") or relation_payload.get("relations") or {}
        if not isinstance(value_payload, dict):
            raise TypeError(f"NPC数值型关系声明必须是对象: {npc_id}")
        values: dict[str, int] = {}
        for name, value in value_payload.items():
            if not isinstance(value, int):
                raise TypeError(f"NPC数值型关系初始值必须是整数: {npc_id}.{name}={value!r}")
            values[str(name)] = validate_dimension_value(value, f"{npc_id}.{name}")
        tags = _as_str_list(relation_payload.get("tags", []))
        relations[str(npc_id)] = RelationDimensions(values=values, tags=tags)
    return relations


def _validate_dimension_bucket(
    values: Mapping[str, int],
    definitions: Mapping[str, DimensionDefinition],
    label: str,
) -> None:
    """校验某类维度名称和值是否合法。"""

    for name, value in values.items():
        if name not in definitions:
            raise KeyError(f"未知{label}: {name}")
        validate_dimension_value(value, name)


def _validate_dimension_names(
    names: Sequence[str],
    definitions: Mapping[str, DimensionDefinition],
    label: str,
) -> None:
    """校验维度名称是否合法。"""

    for name in names:
        if name not in definitions:
            raise KeyError(f"未知{label}: {name}")


def _get_value_bucket(state: DimensionState, reference: DimensionReference) -> dict[str, int]:
    """获取可写的维度值容器。"""

    category = _resolve_category(reference)
    _validate_reference_name(reference.name, category)
    if category == "character":
        return state.character
    if category == "world":
        return state.world
    if category == "extension":
        return state.extensions
    if category == "relation_numeric":
        npc_id = _require_npc_id(reference)
        relation = state.relations.setdefault(npc_id, RelationDimensions())
        return relation.values
    raise ValueError(f"不支持设置数值的维度类型: {category}")


def _validate_reference_name(name: str, category: str) -> None:
    """校验引用的维度名称与类型是否匹配。"""

    definitions_by_category = {
        "character": CHARACTER_DIMENSIONS,
        "world": WORLD_STATE_DIMENSIONS,
        "extension": EXTENSION_DIMENSIONS,
        "relation_numeric": RELATION_NUMERIC_DIMENSIONS,
        "relation_tag": RELATION_TAG_DIMENSIONS,
    }
    definitions = definitions_by_category.get(category)
    if definitions is None:
        raise ValueError(f"未知维度类型: {category}")
    if name not in definitions:
        raise KeyError(f"维度{name}不属于类型{category}")


def _resolve_category(reference: DimensionReference) -> str:
    """在 category=auto 时根据维度名称推断类型。"""

    if reference.category != "auto":
        return reference.category
    if reference.name in CHARACTER_DIMENSIONS:
        return "character"
    if reference.name in WORLD_STATE_DIMENSIONS:
        return "world"
    if reference.name in RELATION_NUMERIC_DIMENSIONS:
        return "relation_numeric"
    if reference.name in EXTENSION_DIMENSIONS:
        return "extension"
    if reference.name in RELATION_TAG_DIMENSIONS:
        return "relation_tag"
    raise KeyError(f"未知维度: {reference.name}")


def _ensure_reference(
    value: DimensionReference | str,
    category: str = "auto",
    npc_id: str = "",
) -> DimensionReference:
    """把字符串维度名转为 DimensionReference。"""

    if isinstance(value, DimensionReference):
        if not value.npc_id and npc_id:
            return DimensionReference(name=value.name, category=value.category, npc_id=npc_id)
        return value
    return DimensionReference(name=str(value), category=category, npc_id=npc_id)


def _require_npc_id(reference: DimensionReference) -> str:
    """关系维度必须携带 NPC 标识。"""

    if not reference.npc_id:
        raise ValueError(f"关系维度必须指定npc_id: {reference.name}")
    return reference.npc_id


def _get_required_value(values: Mapping[str, int], name: str, label: str) -> int:
    """读取必需维度值，不存在时给出明确错误。"""

    if name not in values:
        raise KeyError(f"未启用或未初始化{label}: {name}")
    return validate_dimension_value(values[name], name)


def _get_optional_value(values: Mapping[str, int], name: str, label: str) -> int:
    """读取维度值，缺失时返回默认值避免LLM误引用导致中断。"""

    if name not in values:
        logger.warning("未启用或未初始化%s，使用默认值%s: %s", label, DEFAULT_DIMENSION_VALUE, name)
        return DEFAULT_DIMENSION_VALUE
    try:
        return validate_dimension_value(values[name], name)
    except (TypeError, ValueError) as exc:
        logger.warning("%s取值无效，使用默认值%s: %s", label, DEFAULT_DIMENSION_VALUE, exc)
        return DEFAULT_DIMENSION_VALUE


def _first_existing(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """按别名顺序读取第一个存在的字段。"""

    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _as_str_list(value: Any) -> list[str]:
    """把 YAML 中的字符串列表安全转换为 list[str]。"""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _unique(values: Iterable[str]) -> list[str]:
    """保持顺序去重。"""

    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


__all__ = [
    "ASSISTANT_BONUS_THRESHOLD",
    "ASSISTANT_DIMENSION_LIMIT",
    "CHARACTER_DIMENSIONS",
    "DEFAULT_DIMENSION_VALUE",
    "DIMENSION_MAX_VALUE",
    "DIMENSION_MIN_VALUE",
    "EXTENSION_DIMENSIONS",
    "MODIFIER_CAP",
    "RELATION_NUMERIC_DIMENSIONS",
    "RELATION_TAG_DIMENSIONS",
    "WORLD_STATE_DIMENSIONS",
    "DimensionConfig",
    "DimensionDefinition",
    "DimensionReference",
    "DimensionState",
    "RelationDimensions",
    "calculate_action_modifier",
    "calculate_modifier_from_values",
    "clamp_dimension_value",
    "count_available_tags",
    "get_dimension_value",
    "load_dimension_config_from_dict",
    "load_dimension_config_from_yaml",
    "set_dimension_value",
    "update_dimension_value",
    "validate_dimension_value",
]
