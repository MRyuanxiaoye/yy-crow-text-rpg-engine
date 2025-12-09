from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.llm_factory import get_llm
from app.agents.planning_frameworks import select_framework

class SubTopic(BaseModel):
    sub_query: str = Field(description="Specific search query in Chinese (for domestic search)")
    sub_query_en: str = Field(description="Specific search query in English (for global depth search)")
    reason: str = Field(description="The reason why this sub-topic is important")

class ResearchPlan(BaseModel):
    sub_topics: List[SubTopic] = Field(description="List of sub-topics based on framework dimensions")

class MetaPlan(BaseModel):
    query_type: str = Field(description="Query type: 'case_study', 'why', 'how', 'compare', 'trend', 'what'")
    domain: str = Field(description="Domain: 'history', 'tech', 'science', 'philosophy', 'business', 'general'")
    search_depth: str = Field(description="'shallow', 'medium', 'deep'")
    primary_subject: str = Field(description="The specific subject to research. If user asks for 'a/one example', YOU must select the most typical one.")
    user_constraints: str = Field(description="Specific constraints (e.g., 'only one', 'compare A and B')")

def plan_research(query: str) -> dict:
    """
    V3: Framework-Guided Planning with Structured Templates
    Stage 1: Query Classification + Subject Selection
    Stage 2: Framework Selection
    Stage 3: LLM fills the framework with specific queries
    """
    import time
    start_time = time.time()
    
    llm = get_llm(temperature=0.3)  # Lower temp for more consistent classification
    
    # --- Step 1: Query Classification + Subject Selection ---
    print(f"LOG_CHAIN [Planner V3] Step 1: Classifying query...", flush=True)
    meta_parser = JsonOutputParser(pydantic_object=MetaPlan)
    meta_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个查询分类专家。分析用户问题，输出：
        
        1. **query_type** (问题类型):
           - 'case_study': 要求深入分析一个具体案例（如"介绍一个部族"、"分析某个技术"）
           - 'why': 询问原因（如"为什么X会Y"）
           - 'how': 询问方法/过程（如"如何做X"、"怎样实现Y"）
           - 'compare': 对比分析（如"A和B的区别"）
           - 'trend': 趋势分析（如"X的未来发展"、"Y的演变"）
           - 'what': 概念解释（如"什么是X"、"X是什么"）
        
        2. **domain** (领域):
           - 'history': 历史、历史人物、历史事件
           - 'tech': 技术、编程、工程
           - 'science': 自然科学、物理、生物
           - 'philosophy': 哲学、思想、理论
           - 'business': 商业、管理、经济
           - 'general': 其他或跨领域
        
        3. **search_depth** (深度需求):
           - 'shallow': 只需概述
           - 'medium': 需要一定深度
           - 'deep': 需要深入因果分析（默认选这个）
        
        4. **primary_subject** (核心主体): 
           - 如果用户问"介绍一个案例"，你必须从你的知识库中选定**最典型的具体例子**。
           - 不要返回泛泛的类别（如"严寒部族"），要返回具体名称（如"因纽特人"）。
        
        5. **user_constraints** (约束条件): 提取用户的特殊要求（如"只要一个"、"对比A和B"）
        
        请输出 JSON。
        """),
        ("user", "用户问题: {query}\n\n{format_instructions}")
    ])
    
    try:
        meta_chain = meta_prompt | llm | meta_parser
        meta_plan = meta_chain.invoke({
            "query": query, 
            "format_instructions": meta_parser.get_format_instructions()
        })
        print(f"LOG_CHAIN [Planner V3] Classification: type={meta_plan.get('query_type')}, domain={meta_plan.get('domain')}, depth={meta_plan.get('search_depth')}", flush=True)
        print(f"LOG_CHAIN [Planner V3] Selected Subject: {meta_plan.get('primary_subject')}", flush=True)
    except Exception as e:
        print(f"❌ [Planner] Classification failed: {e}", flush=True)
        # Fallback meta plan
        meta_plan = {
            "query_type": "case_study",
            "domain": "general",
            "search_depth": "deep",
            "primary_subject": query,
            "user_constraints": ""
        }

    # --- Step 2: Load Framework ---
    framework = select_framework(meta_plan.get('query_type'), meta_plan.get('domain'))
    print(f"LOG_CHAIN [Planner V3] Loaded Framework: '{framework['name']}' with {len(framework['mandatory_dimensions'])} dimensions", flush=True)
    
    # --- Step 3: LLM Instantiation (Framework-Guided) ---
    print(f"LOG_CHAIN [Planner V3] Step 3: Generating queries based on framework...", flush=True)
    final_parser = JsonOutputParser(pydantic_object=ResearchPlan)
    
    # Build framework description for prompt
    framework_desc = "\n".join([
        f"{i+1}. **{dim['dimension']}**: {dim['description']}\n   搜索提示: {dim['search_hint']}"
        for i, dim in enumerate(framework['mandatory_dimensions'])
    ])
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个执行规划师。你的任务是**严格按照给定的分析框架**，为每个维度生成具体的搜索查询。
        
        **强制框架（MANDATORY FRAMEWORK）**：
        {framework_description}
        
        **核心研究对象**: {subject}
        **用户约束**: {constraints}
        
        **执行要求**：
        1. **严格遵循框架**：你必须为上述**每个维度**生成至少一个搜索词。不要遗漏任何维度。
        2. **聚焦对象**：所有搜索词必须严格围绕 `{subject}` 展开。严禁跑题去搜其他对象。
        3. **深度优先**：搜索词应该揭示**因果机制、内在逻辑**，而不是简单罗列事实。
           - ❌ 浅层示例："{subject}的历史"
           - ✅ 深度示例："{subject}的地理环境如何影响了其军事战术的演变"
        4. **混合策略**：
           - 第一个搜索词应当是针对 `{subject}` 的**全貌背景调研**（确保 Writer 有上下文）
           - 后续搜索词针对各个框架维度进行**深度下钻**
        
        请输出 JSON，确保 `sub_topics` 数量 >= 框架维度数量。
        """),
        ("user", "用户原始问题: {query}\n\n{format_instructions}")
    ])
    
    try:
        final_chain = final_prompt | llm | final_parser
        final_plan = final_chain.invoke({
            "query": query,
            "subject": meta_plan.get('primary_subject', query),
            "constraints": meta_plan.get('user_constraints', "无特殊约束"),
            "framework_description": framework_desc,
            "format_instructions": final_parser.get_format_instructions()
        })
        
        print(f"✅ [Planner V3] Generated {len(final_plan.get('sub_topics', []))} sub-topics", flush=True)
        
    except Exception as e:
        print(f"❌ [Planner] Framework instantiation failed: {e}", flush=True)
        # Fallback: Generate basic queries
        return {
            "domain": meta_plan.get('domain', 'general'),
            "role_name": "Deep Analyst",
            "persona_instruction": f"Deeply analyze {meta_plan.get('primary_subject')}",
            "search_strategy": "deep_dive",
            "output_format_instruction": "Markdown",
            "sub_topics": [
                {"sub_query": f"{meta_plan.get('primary_subject')} 深度分析", "sub_query_en": f"{meta_plan.get('primary_subject')} deep analysis", "reason": "Fallback query"}
            ]
        }
        
    end_time = time.time()
    print(f"⏱️ [Planner V3] Total Planning took {end_time - start_time:.2f}s", flush=True)

    # Construct final return dict
    return {
        "domain": framework['name'],
        "role_name": "Deep Analyst",
        "persona_instruction": f"Analyze {meta_plan.get('primary_subject')} using the {framework['name']} framework. Focus on causal mechanisms and deep insights.",
        "search_strategy": meta_plan.get('search_depth', 'deep'),
        "output_format_instruction": "Markdown report with structured analysis",
        "sub_topics": final_plan.get("sub_topics", [])
    }

if __name__ == "__main__":
    # Test with different query types
    test_queries = [
        "介绍一个严寒地区的部族，要求是他们足够善战，曾经打下过大片疆土",
        "为什么蒙古帝国能够在短时间内征服如此广阔的领土",
        "如何实现一个高性能的分布式缓存系统",
        "React和Vue的区别是什么"
    ]
    
    import json
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing: {query}")
        print('='*80)
        plan = plan_research(query)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
