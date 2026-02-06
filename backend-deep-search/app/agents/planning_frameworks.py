"""
Planning Frameworks for Different Query Types and Domains
Provides structured analysis templates to guide LLM planning

V2: Added person framework and framework fusion logic
"""

from typing import List, Dict

FRAMEWORKS = {
    # ========== Person Analysis (人物分析) ==========
    "person_history": {
        "name": "历史人物深度分析框架",
        "entity_type": "person",
        "mandatory_dimensions": [
            {
                "dimension": "Life Overview",
                "description": "生平概述：出生、成长、关键转折、结局",
                "search_hint": "生卒年、早年经历、人生轨迹"
            },
            {
                "dimension": "Character & Motivation",
                "description": "性格特点与内在动机",
                "search_hint": "性格、野心、驱动力、价值观"
            },
            {
                "dimension": "Key Decisions & Actions",
                "description": "关键决策与行动",
                "search_hint": "重要决定、转折点、标志性事件"
            },
            {
                "dimension": "Relationships & Networks",
                "description": "人际关系与政治网络",
                "search_hint": "盟友、敌人、家族、政治联盟"
            },
            {
                "dimension": "Historical Evaluation",
                "description": "历史评价与争议",
                "search_hint": "后世评价、功过争议、不同史料观点"
            }
        ]
    },
    
    "person_general": {
        "name": "人物分析框架",
        "entity_type": "person",
        "mandatory_dimensions": [
            {
                "dimension": "Background & Early Life",
                "description": "背景与早年经历",
                "search_hint": "出生、成长环境、教育"
            },
            {
                "dimension": "Achievements & Contributions",
                "description": "成就与贡献",
                "search_hint": "主要成就、代表作品、核心贡献"
            },
            {
                "dimension": "Key Events",
                "description": "关键事件",
                "search_hint": "人生转折点、重要决定"
            },
            {
                "dimension": "Legacy & Impact",
                "description": "影响与遗产",
                "search_hint": "对后世的影响、历史地位"
            }
        ]
    },

    # ========== Event Analysis (事件分析) ==========
    "event_history": {
        "name": "历史事件深度分析框架",
        "entity_type": "event",
        "mandatory_dimensions": [
            {
                "dimension": "Background & Causes",
                "description": "背景与起因",
                "search_hint": "历史背景、导火索、深层原因"
            },
            {
                "dimension": "Key Phases & Timeline",
                "description": "关键阶段与时间线",
                "search_hint": "时间节点、阶段划分、进程演变"
            },
            {
                "dimension": "Core Actors",
                "description": "核心人物与势力",
                "search_hint": "关键人物、主要势力、角色作用"
            },
            {
                "dimension": "Mechanisms & Dynamics",
                "description": "机制与动态",
                "search_hint": "运作机制、力量对比、关键转折"
            },
            {
                "dimension": "Consequences & Impact",
                "description": "后果与影响",
                "search_hint": "直接后果、长远影响、历史意义"
            }
        ]
    },

    # ========== Case Study (案例研究) - 保留兼容 ==========
    "case_study_history": {
        "name": "历史案例深度分析框架",
        "entity_type": "event",
        "mandatory_dimensions": [
            {
                "dimension": "Background Context",
                "description": "时代背景、起源、定义",
                "search_hint": "历史概况、地理位置、时间线"
            },
            {
                "dimension": "Environmental Factors",
                "description": "地理环境、气候、资源",
                "search_hint": "地理决定论、环境如何塑造"
            },
            {
                "dimension": "Core Mechanisms",
                "description": "关键机制、因果链条",
                "search_hint": "为什么、如何导致、内在逻辑"
            },
            {
                "dimension": "Technological/Organizational Edge",
                "description": "技术优势、组织形式",
                "search_hint": "独特的工具、战术、体制"
            },
            {
                "dimension": "Comparative Analysis",
                "description": "与其他案例的对比",
                "search_hint": "差异、独特性、相似案例"
            }
        ]
    },
    
    "case_study_tech": {
        "name": "技术案例分析框架",
        "mandatory_dimensions": [
            {
                "dimension": "Technical Principles",
                "description": "技术原理、科学基础",
                "search_hint": "底层原理、物理机制"
            },
            {
                "dimension": "Implementation",
                "description": "实现方式、架构设计",
                "search_hint": "如何实现、技术栈、架构"
            },
            {
                "dimension": "Advantages & Limitations",
                "description": "优势与局限",
                "search_hint": "为什么选择、缺点、权衡"
            },
            {
                "dimension": "Applications & Impact",
                "description": "应用场景、实际影响",
                "search_hint": "用在哪里、成功案例、影响"
            }
        ]
    },
    
    "case_study_general": {
        "name": "通用案例分析框架",
        "mandatory_dimensions": [
            {
                "dimension": "Background & Definition",
                "description": "背景与定义",
                "search_hint": "是什么、基本概念"
            },
            {
                "dimension": "Key Characteristics",
                "description": "核心特征",
                "search_hint": "主要特点、独特之处"
            },
            {
                "dimension": "Causal Mechanisms",
                "description": "因果机制",
                "search_hint": "为什么会这样、内在逻辑"
            },
            {
                "dimension": "Impact & Significance",
                "description": "影响与意义",
                "search_hint": "重要性、影响范围"
            }
        ]
    },
    
    # ========== Why Questions (为什么) ==========
    "why_general": {
        "name": "深度因果分析框架",
        "mandatory_dimensions": [
            {
                "dimension": "Direct Causes",
                "description": "直接原因",
                "search_hint": "立即原因、触发因素"
            },
            {
                "dimension": "Root Causes",
                "description": "根本原因",
                "search_hint": "底层逻辑、结构性原因"
            },
            {
                "dimension": "Mechanisms",
                "description": "作用机制",
                "search_hint": "如何导致、传导链条"
            },
            {
                "dimension": "Counter-examples & Exceptions",
                "description": "反例与例外",
                "search_hint": "什么情况下不成立、边界条件"
            }
        ]
    },
    
    # ========== How Questions (如何) ==========
    "how_general": {
        "name": "过程与方法分析框架",
        "mandatory_dimensions": [
            {
                "dimension": "Overview",
                "description": "整体流程概览",
                "search_hint": "总体步骤、框架"
            },
            {
                "dimension": "Detailed Steps",
                "description": "详细步骤",
                "search_hint": "逐步分解、具体操作"
            },
            {
                "dimension": "Key Challenges",
                "description": "关键难点",
                "search_hint": "难在哪、常见错误"
            },
            {
                "dimension": "Best Practices",
                "description": "最佳实践",
                "search_hint": "高手怎么做、优化技巧"
            }
        ]
    },
    
    # ========== Comparison (对比) ==========
    "compare_general": {
        "name": "对比分析框架",
        "mandatory_dimensions": [
            {
                "dimension": "Similarities",
                "description": "相似之处",
                "search_hint": "共同点、相同特征"
            },
            {
                "dimension": "Differences",
                "description": "差异之处",
                "search_hint": "区别、对立特征"
            },
            {
                "dimension": "Trade-offs",
                "description": "权衡与取舍",
                "search_hint": "各自优势、适用场景"
            }
        ]
    },
    
    # ========== Trend Analysis (趋势) ==========
    "trend_general": {
        "name": "趋势分析框架",
        "mandatory_dimensions": [
            {
                "dimension": "Current State",
                "description": "现状",
                "search_hint": "最新进展、现在情况"
            },
            {
                "dimension": "Historical Evolution",
                "description": "历史演变",
                "search_hint": "如何发展到今天、关键节点"
            },
            {
                "dimension": "Driving Forces",
                "description": "驱动力",
                "search_hint": "什么推动了变化"
            },
            {
                "dimension": "Future Outlook",
                "description": "未来展望",
                "search_hint": "趋势预测、可能方向"
            }
        ]
    },
    
    # ========== What Questions (是什么) ==========
    "what_general": {
        "name": "概念解析框架",
        "mandatory_dimensions": [
            {
                "dimension": "Definition & Origin",
                "description": "定义与起源",
                "search_hint": "准确定义、概念来源"
            },
            {
                "dimension": "Core Components",
                "description": "核心组成",
                "search_hint": "包含哪些部分、结构"
            },
            {
                "dimension": "Applications & Examples",
                "description": "应用与实例",
                "search_hint": "具体例子、实际应用"
            },
            {
                "dimension": "Related Concepts",
                "description": "相关概念",
                "search_hint": "类似概念、对比区别"
            }
        ]
    }
}

def select_framework(query_type: str, domain: str = "general") -> dict:
    """
    Select appropriate framework based on query classification
    
    Args:
        query_type: Type of query (person, event, case_study, why, how, compare, trend, what)
        domain: Domain of the query (history, tech, science, philosophy, business, general)
    
    Returns:
        Framework dictionary with mandatory dimensions
    """
    # Try domain-specific framework first
    key = f"{query_type}_{domain}"
    if key in FRAMEWORKS:
        return FRAMEWORKS[key]
    
    # Fallback to general framework for that query type
    fallback_key = f"{query_type}_general"
    if fallback_key in FRAMEWORKS:
        return FRAMEWORKS[fallback_key]
    
    # Ultimate fallback: case_study_general
    return FRAMEWORKS["case_study_general"]


def select_fused_framework(
    primary_type: str, 
    secondary_types: List[str], 
    domain: str = "general"
) -> Dict:
    """
    Fuse multiple frameworks: primary framework gets all dimensions (for deep-dive),
    secondary frameworks contribute 2 dimensions each (for shallow coverage).
    
    Args:
        primary_type: Main entity type (person, event, concept, etc.)
        secondary_types: Related entity types
        domain: Domain context
    
    Returns:
        Fused framework with primary and secondary dimensions marked
    """
    # Get primary framework
    primary_fw = select_framework(primary_type, domain)
    
    primary_dims = []
    for dim in primary_fw.get("mandatory_dimensions", []):
        dim_copy = dict(dim)
        dim_copy["priority"] = "primary"
        primary_dims.append(dim_copy)
    
    # Get secondary dimensions (only top 2 from each secondary framework)
    secondary_dims = []
    for sec_type in secondary_types[:2]:  # Max 2 secondary types
        sec_fw = select_framework(sec_type, domain)
        for dim in sec_fw.get("mandatory_dimensions", [])[:2]:  # Max 2 dims per secondary
            dim_copy = dict(dim)
            dim_copy["priority"] = "secondary"
            dim_copy["source_type"] = sec_type
            secondary_dims.append(dim_copy)
    
    return {
        "name": f"{primary_fw['name']}（融合）",
        "primary_type": primary_type,
        "secondary_types": secondary_types,
        "primary_dimensions": primary_dims,
        "secondary_dimensions": secondary_dims,
        "all_dimensions": primary_dims + secondary_dims
    }


def get_entity_type_hint(entity_name: str) -> str:
    """
    Provide hints for LLM to determine entity type.
    This is used in the classification prompt.
    """
    return """
    判断实体类型：
    - 'person': 具体的人物（如"安禄山"、"李白"、"乔布斯"）
    - 'event': 历史事件或事件系列（如"安史之乱"、"工业革命"、"911事件"）
    - 'concept': 抽象概念或理论（如"量子力学"、"民主制度"、"机器学习"）
    - 'thing': 具体事物（如"德文卷毛猫"、"特斯拉Model 3"、"长城"）
    - 'organization': 组织机构（如"苹果公司"、"唐朝政府"、"NASA"）
    """

