from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.llm_factory import get_llm

class SubTopic(BaseModel):
    sub_query: str = Field(description="Specific search query in Chinese (for domestic search)")
    sub_query_en: str = Field(description="Specific search query in English (for global depth search)")
    reason: str = Field(description="The reason why this sub-topic is important")

class ConceptFramework(BaseModel):
    core_concept: str = Field(description="The core concept or definition of the topic")
    macro_dimensions: List[str] = Field(description="Key macro dimensions (e.g., History, Policy, Geography) covering the topic")
    micro_examples: List[str] = Field(description="Specific representative examples (People, Events, Objects) to reflect the macro dimensions")

class ResearchPlan(BaseModel):
    domain: str = Field(description="The domain of the query (e.g., 'Technical', 'Historical', 'Lifestyle')")
    role_name: str = Field(description="The specific persona role name (e.g., 'Senior UX Designer', 'Military Historian')")
    persona_instruction: str = Field(description="Specific instruction for the persona to follow")
    search_strategy: str = Field(description="Strategy for sourcing information (e.g., 'Official Docs only', 'Primary sources')")
    output_format_instruction: str = Field(description="Detailed instruction on how to format the final report")
    concept_framework: ConceptFramework = Field(description="The conceptual framework defining macro dimensions and micro examples")
    sub_topics: List[SubTopic] = Field(description="List of 3-5 core sub-topics, derived from the concept framework")

def plan_research(query: str) -> dict:
    """
    Uses DeepSeek with Meta-Prompting to adaptively plan research with Concept Framework.
    """
    llm = get_llm(temperature=0.5)
    
    parser = JsonOutputParser(pydantic_object=ResearchPlan)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能体规划大师。你的任务是分析用户问题，制定深度的研究计划。

        **核心思考逻辑 (Thinking Process)**：
        为了确保资料查询的完整性，你需要构建一个“宏观骨架+微观血肉”的概念框架 (Concept Framework)。
        - **宏观骨架**: 确定问题涉及的核心维度。
        - **微观血肉**: 联想具体的案例来支撑宏观维度（这些案例用于生成搜索关键词，确保我们能查到细节）。

        **关键区别 (Critical Instruction)**：
        - “宏观+微观”框架 **仅用于生成搜索子问题 (sub_topics)**，目的是为了查全资料。
        - **最终报告的输出范式 (Output Format)**：不要机械地按照“宏观+微观”来写！
          - 要求生成一份**深度、饱满、逻辑连贯**的指令。
          - 告诉 Writer：你要像一位渊博的教授或资深研究员，将查到的宏观理论和微观案例**有机融合**。
          - 案例是用来证明观点的，不是用来凑数的。如果案例不合适，可以舍弃。
          - 目标是：**面面俱到，解释清晰，有所深度**。

        **任务步骤**：
        1. 定义领域与角色。
        2. 构建 Concept Framework (用于指导搜索)。
        3. 编写 Output Format Instruction (用于指导写作风格，强调“饱满”、“深度”、“非范式化”)。
        4. 基于 Concept Framework 拆解出 3-5 个具体的搜索子问题 (Sub-topics)。
           - **关键升级**: 对于每个子问题，同时生成 **中文查询词 (`sub_query`)** 和 **英文查询词 (`sub_query_en`)**。
           - 英文查询词应针对国际高质量内容进行优化（例如：将“Transformer架构”翻译为 "Transformer architecture deep dive" 或 "Transformer explained"）。

        请严格按照 JSON 格式返回。
        """),
        ("user", "用户问题: {query}\n\n{format_instructions}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "query": query,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        print(f"Planner Error: {e}")
        # Fallback plan
        return {
            "domain": "General",
            "role_name": "Research Assistant",
            "persona_instruction": "Answer accurately.",
            "search_strategy": "General web search",
            "output_format_instruction": "Markdown report",
            "concept_framework": {
                "core_concept": query,
                "macro_dimensions": ["General Info"],
                "micro_examples": []
            },
            "sub_topics": [{"sub_query": query, "reason": "Fallback"}]
        }

if __name__ == "__main__":
    # Test
    plan = plan_research("深圳的崛起是怎么样的")
    import json
    print(json.dumps(plan, ensure_ascii=False, indent=2))
