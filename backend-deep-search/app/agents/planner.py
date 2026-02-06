from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.llm_factory import get_llm
from app.agents.planning_frameworks import select_framework, select_fused_framework

class SubTopic(BaseModel):
    sub_query: str = Field(description="Specific search query in Chinese")
    sub_query_en: str = Field(description="Specific search query in English")
    reason: str = Field(description="The reason why this sub-topic is important")
    priority: str = Field(default="primary", description="'primary' for deep-dive, 'secondary' for shallow coverage")
    parent_dimension: str = Field(default="", description="Parent dimension this sub-topic belongs to")

class ResearchPlan(BaseModel):
    sub_topics: List[SubTopic] = Field(description="List of sub-topics")

class EntityAnalysis(BaseModel):
    """Enhanced MetaPlan with entity type recognition"""
    focus_entity: str = Field(description="The main entity user wants to know about")
    focus_type: str = Field(description="Type of focus entity: 'person', 'event', 'concept', 'thing', 'organization'")
    related_entities: List[str] = Field(description="Related entities that may be mentioned (max 3)")
    related_types: List[str] = Field(description="Types of related entities")
    domain: str = Field(description="Domain: 'history', 'tech', 'science', 'philosophy', 'business', 'general'")
    search_depth: str = Field(default="deep", description="'shallow', 'medium', 'deep'")

class DeepDiveQuery(BaseModel):
    sub_query: str = Field(description="Deep-dive search query in Chinese")
    sub_query_en: str = Field(description="Deep-dive search query in English")
    reason: str = Field(description="Why this deep-dive is important")

class DeepDivePlan(BaseModel):
    deep_queries: List[DeepDiveQuery] = Field(description="List of deep-dive queries")


def plan_research(query: str) -> dict:
    """
    V5: Entity-Aware Framework Fusion with Layered Search
    
    Stage 1: Entity Analysis (identify focus entity and related entities)
    Stage 2: Framework Fusion (primary framework + secondary framework dimensions)
    Stage 3: Layered Query Generation
        - Primary dimensions: full deep-dive (macro + 2-3 sub-queries each)
        - Secondary dimensions: shallow coverage (1 query only, no deep-dive)
    """
    import time
    start_time = time.time()
    
    llm = get_llm(temperature=0.3)
    
    # ============ Step 1: Entity Analysis ============
    print(f"LOG_CHAIN [Planner V5] Step 1: Entity Analysis...", flush=True)
    
    entity_parser = JsonOutputParser(pydantic_object=EntityAnalysis)
    entity_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个实体识别专家。分析用户问题，识别出：
        
        1. **focus_entity**: 用户最想了解的核心实体（具体名称）
           - 例："讲讲安禄山" → "安禄山"
           - 例："安史之乱是怎么回事" → "安史之乱"
        
        2. **focus_type**: 核心实体的类型
           - 'person': 具体人物（安禄山、李白、乔布斯）
           - 'event': 历史事件（安史之乱、工业革命）
           - 'concept': 抽象概念（量子力学、民主制度）
           - 'thing': 具体事物（德文卷毛猫、长城）
           - 'organization': 组织机构（苹果公司、NASA）
        
        3. **related_entities**: 与核心实体紧密相关的其他实体（最多3个）
           - "安禄山" → ["安史之乱", "唐玄宗", "杨贵妃"]
           - "安史之乱" → ["安禄山", "郭子仪", "唐玄宗"]
        
        4. **related_types**: 相关实体的类型（与 related_entities 一一对应）
        
        5. **domain**: 领域 ('history', 'tech', 'science', 'philosophy', 'business', 'general')
        
        6. **search_depth**: 深度 ('deep' 为默认)
        
        请输出 JSON。
        """),
        ("user", "用户问题: {query}\n\n{format_instructions}")
    ])
    
    try:
        entity_chain = entity_prompt | llm | entity_parser
        entity_plan = entity_chain.invoke({
            "query": query, 
            "format_instructions": entity_parser.get_format_instructions()
        })
        
        focus_entity = entity_plan.get("focus_entity", query)
        focus_type = entity_plan.get("focus_type", "concept")
        related_entities = entity_plan.get("related_entities", [])[:3]
        related_types = entity_plan.get("related_types", [])[:3]
        domain = entity_plan.get("domain", "general")
        
        print(f"LOG_CHAIN [Planner V5] Focus: {focus_entity} ({focus_type})", flush=True)
        print(f"LOG_CHAIN [Planner V5] Related: {related_entities} ({related_types})", flush=True)
        
    except Exception as e:
        print(f"❌ [Planner] Entity analysis failed: {e}", flush=True)
        focus_entity = query
        focus_type = "concept"
        related_entities = []
        related_types = []
        domain = "general"
    
    # ============ Step 2: Framework Fusion ============
    print(f"LOG_CHAIN [Planner V5] Step 2: Framework Fusion...", flush=True)
    
    # Get unique related types for secondary frameworks
    secondary_types = list(set(related_types))[:2]  # Max 2 secondary types
    
    fused_fw = select_fused_framework(focus_type, secondary_types, domain)
    
    primary_dims = fused_fw.get("primary_dimensions", [])
    secondary_dims = fused_fw.get("secondary_dimensions", [])
    
    print(f"LOG_CHAIN [Planner V5] Primary dimensions: {len(primary_dims)}, Secondary dimensions: {len(secondary_dims)}", flush=True)
    
    # ============ Step 3: Generate Primary Queries (with deep-dive) ============
    print(f"LOG_CHAIN [Planner V5] Step 3: Generating primary queries...", flush=True)
    
    final_parser = JsonOutputParser(pydantic_object=ResearchPlan)
    
    primary_desc = "\n".join([
        f"{i+1}. **{dim['dimension']}**: {dim['description']}\n   搜索提示: {dim['search_hint']}"
        for i, dim in enumerate(primary_dims)
    ])
    
    primary_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个执行规划师。为每个维度生成搜索查询。
        
        **核心实体**: {focus_entity}
        **相关实体**: {related_entities}（可以在搜索词中提及，但必须与核心实体关联）
        
        **分析维度（主要）**：
        {framework_description}
        
        **要求**：
        1. 所有搜索词必须以 "{focus_entity}" 为视角和核心
        2. 第一个搜索词必须是 "{focus_entity} 全貌背景概述"
        3. 相关实体可以出现，但必须与核心实体关联
           - ✅ "{focus_entity}与{related_entity}的关系"
           - ❌ "{related_entity}的影响"（脱离了核心实体）
        4. 搜索词要揭示因果机制，不是简单罗列
        5. 每个维度生成 1 个搜索词
        
        请输出 JSON。
        """),
        ("user", "用户原始问题: {query}\n\n{format_instructions}")
    ])
    
    all_topics = []
    
    try:
        primary_chain = primary_prompt | llm | final_parser
        primary_plan = primary_chain.invoke({
            "query": query,
            "focus_entity": focus_entity,
            "related_entities": ", ".join(related_entities) if related_entities else "无",
            "related_entity": related_entities[0] if related_entities else "",
            "framework_description": primary_desc,
            "format_instructions": final_parser.get_format_instructions()
        })
        
        primary_topics = primary_plan.get("sub_topics", [])
        print(f"LOG_PLANNER [Macro Queries] Generated {len(primary_topics)} primary macro queries:", flush=True)
        
        # Add primary topics with priority="primary"
        for idx, topic in enumerate(primary_topics):
            topic_dict = dict(topic) if hasattr(topic, '__dict__') else topic
            topic_dict["priority"] = "primary"
            topic_dict["parent_dimension"] = ""
            all_topics.append(topic_dict)
            # Log each macro query
            print(f"LOG_PLANNER   [{idx+1}] {topic_dict.get('sub_query', '')[:60]}...", flush=True)
        
    except Exception as e:
        print(f"❌ [Planner] Primary query generation failed: {e}", flush=True)
        all_topics.append({
            "sub_query": f"{focus_entity} 深度分析",
            "sub_query_en": f"{focus_entity} deep analysis",
            "reason": "Fallback",
            "priority": "primary",
            "parent_dimension": ""
        })
        primary_topics = all_topics
    
    # ============ Step 4: Deep-Dive for Primary Dimensions ONLY ============
    print(f"LOG_CHAIN [Planner V5] Step 4: Deep-dive for primary dimensions...", flush=True)
    
    deep_parser = JsonOutputParser(pydantic_object=DeepDivePlan)
    deep_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个深度分析师。针对给定的宏观维度，生成 2 个**更深入、更具体**的子搜索词。

        **核心实体**: {focus_entity}
        **宏观维度**: {macro_dimension}
        **宏观搜索词**: {macro_query}
        
        **要求**：
        1. 子搜索词必须以 "{focus_entity}" 为核心视角
        2. 比宏观搜索词**更具体、更深入**
        3. 揭示该维度下的**细节、机制、关键节点**
        4. 只生成 2 个子搜索词
        
        请输出 JSON，包含 2 个 deep_queries。
        """),
        ("user", "{format_instructions}")
    ])
    
    # Only deep-dive for primary topics (skip the first one which is background overview)
    for i, topic in enumerate(primary_topics[1:], start=1):  # Skip first (background)
        macro_query = topic.get("sub_query", "") if isinstance(topic, dict) else topic.sub_query
        macro_reason = topic.get("reason", "") if isinstance(topic, dict) else topic.reason
        
        try:
            deep_chain = deep_prompt | llm | deep_parser
            deep_result = deep_chain.invoke({
                "focus_entity": focus_entity,
                "macro_dimension": macro_reason,
                "macro_query": macro_query,
                "format_instructions": deep_parser.get_format_instructions()
            })
            
            deep_queries = deep_result.get("deep_queries", [])[:2]  # Max 2
            print(f"LOG_PLANNER [Deep-Dive] Dim {i} '{macro_reason[:30]}...' -> {len(deep_queries)} sub-queries:", flush=True)
            
            for dq in deep_queries:
                print(f"LOG_PLANNER     -> {dq.get('sub_query', '')[:50]}...", flush=True)
                all_topics.append({
                    "sub_query": dq.get("sub_query", ""),
                    "sub_query_en": dq.get("sub_query_en", ""),
                    "reason": dq.get("reason", ""),
                    "priority": "primary",  # Deep-dive queries are also primary
                    "parent_dimension": macro_reason
                })
        except Exception as e:
            print(f"⚠️ [Planner] Deep-dive for dim {i} failed: {e}", flush=True)
            continue
    
    # ============ Step 5: Generate Secondary Queries (NO deep-dive) ============
    if secondary_dims:
        print(f"LOG_CHAIN [Planner V5] Step 5: Generating secondary queries (shallow)...", flush=True)
        
        secondary_desc = "\n".join([
            f"- **{dim['dimension']}**: {dim['description']} (来源: {dim.get('source_type', 'related')})"
            for dim in secondary_dims
        ])
        
        secondary_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个执行规划师。为相关维度生成**浅层搜索查询**（辅助覆盖）。

            **核心实体**: {focus_entity}
            **相关实体**: {related_entities}
            
            **辅助维度（浅层覆盖）**：
            {secondary_description}
            
            **要求**：
            1. 每个维度只生成 1 个搜索词
            2. 搜索词必须与 "{focus_entity}" 关联
            3. 这些是辅助信息，不需要太深入
            
            请输出 JSON。
            """),
            ("user", "用户原始问题: {query}\n\n{format_instructions}")
        ])
        
        try:
            secondary_chain = secondary_prompt | llm | final_parser
            secondary_plan = secondary_chain.invoke({
                "query": query,
                "focus_entity": focus_entity,
                "related_entities": ", ".join(related_entities) if related_entities else "无",
                "secondary_description": secondary_desc,
                "format_instructions": final_parser.get_format_instructions()
            })
            
            secondary_topics = secondary_plan.get("sub_topics", [])
            print(f"LOG_PLANNER [Secondary Queries] Generated {len(secondary_topics)} secondary queries:", flush=True)
            
            # Add secondary topics with priority="secondary"
            for idx, topic in enumerate(secondary_topics):
                topic_dict = dict(topic) if hasattr(topic, '__dict__') else topic
                topic_dict["priority"] = "secondary"
                topic_dict["parent_dimension"] = "secondary"
                all_topics.append(topic_dict)
                # Log each secondary query
                print(f"LOG_PLANNER   [S{idx+1}] {topic_dict.get('sub_query', '')[:60]}...", flush=True)
                
        except Exception as e:
            print(f"⚠️ [Planner] Secondary query generation failed: {e}", flush=True)
        
    end_time = time.time()
    
    # Count by priority
    primary_count = sum(1 for t in all_topics if t.get("priority") == "primary")
    secondary_count = sum(1 for t in all_topics if t.get("priority") == "secondary")
    
    print(f"", flush=True)
    print(f"LOG_PLANNER ========== SUMMARY ==========", flush=True)
    print(f"LOG_PLANNER Focus Entity: {focus_entity} ({focus_type})", flush=True)
    print(f"LOG_PLANNER Related: {related_entities}", flush=True)
    print(f"LOG_PLANNER Total Queries: {len(all_topics)} (primary: {primary_count}, secondary: {secondary_count})", flush=True)
    print(f"LOG_PLANNER Time: {end_time - start_time:.2f}s", flush=True)
    print(f"LOG_PLANNER ===============================", flush=True)
    print(f"⏱️ [Planner V5] Planning took {end_time - start_time:.2f}s", flush=True)

    return {
        "domain": fused_fw['name'],
        "role_name": "Deep Analyst",
        "persona_instruction": f"以 {focus_entity} 为核心视角，用瀑布式结构深度分析。相关实体（{', '.join(related_entities)}）可以穿插其中，但必须服务于对 {focus_entity} 的理解。",
        "search_strategy": "layered",
        "output_format_instruction": "Waterfall-style knowledge cascade",
        "sub_topics": all_topics,
        "focus_entity": focus_entity,
        "focus_type": focus_type,
        "related_entities": related_entities
    }


if __name__ == "__main__":
    import json
    
    test_queries = [
        "讲讲安禄山",
        "安史之乱是怎么回事",
        "德文卷毛猫是什么",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing: {query}")
        print('='*80)
        plan = plan_research(query)
        print(f"\nResult:")
        print(f"  Focus: {plan.get('focus_entity')} ({plan.get('focus_type')})")
        print(f"  Related: {plan.get('related_entities')}")
        print(f"  Total queries: {len(plan.get('sub_topics', []))}")
        for i, t in enumerate(plan.get('sub_topics', [])[:5]):
            print(f"    {i+1}. [{t.get('priority', '?')}] {t.get('sub_query', '')[:50]}...")
