"""行动分类公式系统。维度驱动的确定性结算参数计算。"""

from __future__ import annotations
from typing import Any, Mapping


# 行动类别定义：每个类别有主维度、辅助维度、dc公式、effects公式
ACTION_CATEGORIES: dict[str, dict[str, Any]] = {
    'military': {
        'name': '军事',
        'keywords': ['调兵', '出征', '围剿', '守城', '伏击', '练兵', '军', '兵', '战', '攻', '防', '征'],
        'main_dimension': '兵力',
        'aux_dimensions': ['士气', '智谋'],
        'dc_formula': 'base_dc + (10 - target_兵力) - 智谋_bonus',
        'effects_template': {
            'success': {'兵力': 1, '士气': 1},
            'partial': {'兵力': 0, '士气': -1},
            'failure': {'兵力': -1, '士气': -2},
            'critical_failure': {'兵力': -2, '士气': -3, '民心': -1},
        },
    },
    'fiscal': {
        'name': '财政',
        'keywords': ['拨银', '拨款', '赏赐', '修建', '赈济', '税', '银', '钱', '粮', '饷', '库', '赈'],
        'main_dimension': '财政',
        'aux_dimensions': ['智谋'],
        'dc_formula': 'base_dc + cost_factor',
        'effects_template': {
            'success': {'财政': -1, '民心': 2},
            'partial': {'财政': -1, '民心': 1},
            'failure': {'财政': -1, '民心': 0},
            'critical_failure': {'财政': -2, '民心': -1},
        },
    },
    'civil': {
        'name': '民生',
        'keywords': ['减赋', '赈灾', '安抚', '治理', '改革', '教化', '民', '灾', '赋'],
        'main_dimension': '民心',
        'aux_dimensions': ['智谋', '口才'],
        'dc_formula': 'base_dc + unrest_factor',
        'effects_template': {
            'success': {'民心': 2, '财政': -1},
            'partial': {'民心': 1, '财政': -1},
            'failure': {'民心': 0, '财政': -1},
            'critical_failure': {'民心': -1, '财政': -1},
        },
    },
    'political': {
        'name': '政治',
        'keywords': ['弹劾', '任免', '拉拢', '清洗', '结盟', '联姻', '朝', '官', '臣', '党', '派'],
        'main_dimension': '派系势力',
        'aux_dimensions': ['口才', '魅力', '智谋'],
        'dc_formula': 'base_dc + faction_resistance',
        'effects_template': {
            'success': {'派系势力': -1},
            'partial': {'派系势力': 0},
            'failure': {'派系势力': 1},
            'critical_failure': {'派系势力': 2, '民心': -1},
        },
    },
    'intelligence': {
        'name': '情报',
        'keywords': ['密探', '刺探', '调查', '监视', '暗查', '情报', '探', '查', '密', '侦'],
        'main_dimension': '情报',
        'aux_dimensions': ['智谋', '感知'],
        'dc_formula': 'base_dc + secrecy_factor',
        'effects_template': {
            'success': {'情报': 1},
            'partial': {'情报': 0},
            'failure': {'情报': 0, '派系势力': 1},
            'critical_failure': {'情报': -1, '派系势力': 1},
        },
    },
    'diplomacy': {
        'name': '外交',
        'keywords': ['谈判', '议和', '使节', '朝贡', '封赏', '招降', '和', '盟', '降'],
        'main_dimension': '口才',
        'aux_dimensions': ['魅力', '智谋'],
        'dc_formula': 'base_dc + opponent_strength',
        'effects_template': {
            'success': {'民心': 1, '士气': 1},
            'partial': {'财政': -1},
            'failure': {'士气': -1},
            'critical_failure': {'士气': -2, '民心': -1},
        },
    },
}

# dc 基准值
BASE_DC = 8
# 维度 bonus 阈值
BONUS_THRESHOLD = 7
PENALTY_THRESHOLD = 3


# 行动性质：决定行动是否需要判定、延迟多久
ACTION_NATURE_KEYWORDS = {
    'command': {
        'keywords': ['准奏', '批准', '同意', '免职', '任命', '下旨', '拨银', '拨款', '赏赐', '下诏', '圣旨', '恩准', '册封'],
        'description': '皇帝直接命令/资源调配',
        'needs_roll': False,
        'delay': 0,
    },
    'local_exec': {
        'keywords': ['抄家', '审讯', '召见', '传唤', '查抄', '逮捕', '搜查', '问话', '面见'],
        'description': '本地执行的行政行动',
        'needs_roll': True,
        'delay': 1,
    },
    'remote_exec': {
        'keywords': ['调兵', '运粮', '催收', '巡视', '出使', '远征', '增援', '运送', '押送', '催办'],
        'description': '需要跨地域执行的行动',
        'needs_roll': True,
        'delay': 2,
    },
    'contested': {
        'keywords': ['说服', '谈判', '拉拢', '弹劾', '改革', '变法', '清洗', '招降', '议和'],
        'description': '有对抗/不确定性的行动',
        'needs_roll': True,
        'delay': 1,
    },
}


def classify_action_nature(description: str, category: str = '') -> dict[str, Any]:
    """判断行动性质：是否需要判定、延迟多久。"""
    desc = description.lower() if description else ''
    best_nature = 'local_exec'
    best_score = 0
    for nature_id, nature_def in ACTION_NATURE_KEYWORDS.items():
        score = sum(1 for kw in nature_def['keywords'] if kw in desc)
        if score > best_score:
            best_score = score
            best_nature = nature_id
    nature = ACTION_NATURE_KEYWORDS[best_nature]
    return {
        'nature': best_nature,
        'needs_roll': nature['needs_roll'],
        'default_delay': nature['delay'],
        'description': nature['description'],
    }


def classify_action(description: str, main_dimension_hint: str = '') -> str:
    """根据描述文本和维度提示分类行动类别。"""
    desc_lower = description.lower()
    # 先检查 main_dimension 提示
    dim_to_category = {
        '兵力': 'military', '士气': 'military',
        '财政': 'fiscal',
        '民心': 'civil',
        '派系势力': 'political',
        '情报': 'intelligence',
        '口才': 'diplomacy', '魅力': 'diplomacy',
    }
    if main_dimension_hint in dim_to_category:
        return dim_to_category[main_dimension_hint]
    # 关键词匹配
    best_category = 'political'  # 默认
    best_score = 0
    for cat_id, cat_def in ACTION_CATEGORIES.items():
        score = sum(1 for kw in cat_def['keywords'] if kw in desc_lower)
        if score > best_score:
            best_score = score
            best_category = cat_id
    return best_category


def calculate_dc(category: str, world_dims: Mapping[str, int], char_dims: Mapping[str, int]) -> int:
    """根据行动类别和当前维度计算难度等级。"""
    cat = ACTION_CATEGORIES.get(category, ACTION_CATEGORIES['political'])
    dc = BASE_DC
    # 主维度越低，事情越难
    main_dim = cat['main_dimension']
    main_val = world_dims.get(main_dim, char_dims.get(main_dim, 5))
    dc += max(0, 5 - main_val)  # 维度5=+0, 维度3=+2, 维度1=+4
    # 智谋 bonus
    intelligence = char_dims.get('智谋', 5)
    if intelligence >= BONUS_THRESHOLD:
        dc -= 1
    elif intelligence <= PENALTY_THRESHOLD:
        dc += 1
    return max(4, min(16, dc))


def calculate_modifier(category: str, world_dims: Mapping[str, int], char_dims: Mapping[str, int]) -> int:
    """根据行动类别和当前维度计算判定修正值。"""
    cat = ACTION_CATEGORIES.get(category, ACTION_CATEGORIES['political'])
    modifier = 0
    # 主维度 bonus
    main_dim = cat['main_dimension']
    main_val = world_dims.get(main_dim, char_dims.get(main_dim, 5))
    if main_val >= BONUS_THRESHOLD:
        modifier += 2
    elif main_val >= 5:
        modifier += 1
    elif main_val <= PENALTY_THRESHOLD:
        modifier -= 1
    # 辅助维度 bonus（每个符合条件的+1，最多+2）
    aux_bonus = 0
    for aux_name in cat.get('aux_dimensions', []):
        aux_val = char_dims.get(aux_name, world_dims.get(aux_name, 5))
        if aux_val >= BONUS_THRESHOLD:
            aux_bonus += 1
    modifier += min(2, aux_bonus)
    return max(-5, min(5, modifier))


def predict_effects(category: str, outcome_tier: str) -> dict[str, int]:
    """根据类别和结果等级返回预设效果模板。"""
    cat = ACTION_CATEGORIES.get(category, ACTION_CATEGORIES['political'])
    tier_map = {
        '大成功': 'success',
        '成功': 'success',
        '部分成功': 'partial',
        '失败': 'failure',
        '大失败': 'critical_failure',
    }
    template_key = tier_map.get(outcome_tier, 'partial')
    effects = dict(cat['effects_template'].get(template_key, {}))
    # 大成功额外加成
    if outcome_tier == '大成功':
        for dim_name in effects:
            if effects[dim_name] > 0:
                effects[dim_name] += 1
    return effects


def get_predicted_effects_all_tiers(category: str) -> dict[str, dict[str, int]]:
    """返回所有结果等级的预测效果，用于行动确认卡片。"""
    cat = ACTION_CATEGORIES.get(category, ACTION_CATEGORIES['political'])
    return {
        'success': dict(cat['effects_template'].get('success', {})),
        'partial': dict(cat['effects_template'].get('partial', {})),
        'failure': dict(cat['effects_template'].get('failure', {})),
    }


def format_predicted_effects(effects: Mapping[str, int]) -> str:
    """格式化预测效果为可读文本。"""
    parts = []
    for dim, val in effects.items():
        if val > 0:
            parts.append(f'{dim}+{val}')
        elif val < 0:
            parts.append(f'{dim}{val}')
    return ' '.join(parts) if parts else '无明确变化'
