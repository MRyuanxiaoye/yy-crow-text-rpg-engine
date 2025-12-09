"""
Planning Frameworks for Different Query Types and Domains
Provides structured analysis templates to guide LLM planning
"""

FRAMEWORKS = {
    # ========== Case Study (案例研究) ==========
    "case_study_history": {
        "name": "历史案例深度分析框架",
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
        query_type: Type of query (case_study, why, how, compare, trend, what)
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

