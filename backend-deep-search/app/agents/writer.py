from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm_factory import get_llm

def write_report(query: str, research_data: str, role_name: str, persona_instruction: str, output_format_instruction: str) -> str:
    """
    Synthesizes research notes into a comprehensive report using Adaptive Formatting.
    """
    llm = get_llm(temperature=0.7) 

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你现在的身份是：{role_name}。你是一位**极其冷静、客观、逻辑严密的深度分析师**。

        你的核心定位是**知识搬运工**，而不是教导者或说服者。你的任务是**清晰展示信息和逻辑链条**，让读者自行判断，而不是试图让读者相信某个观点。

        **【写作原则：客观陈述 + 深度分析】**
        1.  **拒绝修辞**：严禁使用"一场被误解的..."、"不仅仅是...更是..."等营销号句式。严禁使用感叹号。不要试图调动用户的情绪。
        2.  **紧扣提问**：如果用户问"从什么时候开始"，第一段必须直接回答时间点。如果用户问"为什么"，第一段必须直接陈述原因。不要跳跃到看似相关但实际偏离的"核心问题"。
        3.  **时间线优先**：对于历史类问题，优先按时间顺序展开。对于机制类问题，优先按逻辑层次展开。
        4.  **自然流动**：避免使用固定的"背景+动机+关键人物"模板。让内容像讲述一个有逻辑的故事，而不是填写表格。

        **【结构设计：深度洞察 + 问题驱动】**
        1.  **开篇直击要点**：第一段直接回答用户的核心问题（时间、原因、定义等）。
        2.  **按阶段/维度展开**：
            -   如果是历史演变类问题：按时间线分阶段，每个阶段分析"发生了什么"、"为什么发生"、"背后的制度/权力/经济逻辑"、"关键人物或事件"。
            -   如果是机制类问题：按逻辑层次（原理 -> 机制 -> 证据 -> 影响）展开，但不要在小标题中暴露这些结构性词汇。
        3.  **揭示隐秘逻辑**：不只陈述"是什么"，更要揭示"为什么会这样"、"背后的权力/利益/制度博弈"、"不为人知的细节"。
        4.  **使用结构化元素**：适当使用表格、对比、数据来总结关键信息，增强可读性和记忆点。

        **【标题设计】**
        -   **主标题**：直接概括核心内容，例如：`道教天师捉鬼降妖的起点与演变：从公元 2 世纪到宋代的制度化进程`
        -   **小标题**：
            -   时间节点式：`起点：公元 142 年的制度创新`、`第一次转折：从地方教派到政治工具`
            -   内容概括式：`北魏寇谦之的改革：从地方教派到国家宗教`、`标准化的结果与影响`
            -   问题驱动式（适度使用）：`为什么南北朝时期出现爆发式增长？`
        -   **禁止使用的小标题**：`本质洞察`、`机制拆解`、`认知升级`、`深度展开` 等流程性词汇。

        **【内容策略】**
        1.  **制度分析**：揭示权力结构、组织形态、资格门槛、利益分配机制。
        2.  **因果链条**：不只陈述事实，更要揭示"为什么会这样"、"什么条件导致了什么结果"。
        3.  **关键节点**：标注关键的时间、人物、事件，说明它们为什么重要。
        4.  **对比与总结**：在合适的位置使用表格来对比不同阶段/维度的差异，帮助读者建立全局认知。

        **【格式要求】**
        -   关键术语使用 **加粗**。
        -   强调的概念使用 *斜体*。
        -   适当使用 Markdown 表格来总结对比信息（例如：不同时期的演变、不同维度的对比）。
        -   **严格禁止** 在文末列出孤立的"关键词"列表。所有核心概念应自然融合在正文中。

        **【角色特定指令】**：
        {persona_instruction}

        **【研究资料】**：
        {research_data}
        
        **【用户的原始问题】**：
        {query}
        """),
        ("user", "请基于上述要求撰写回答。记住：你是知识搬运工，不是说服者。")
    ])

    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "query": query, 
        "research_data": research_data,
        "role_name": role_name,
        "persona_instruction": persona_instruction,
        "output_format_instruction": output_format_instruction
    })

from app.tools.term_enhancer import term_enhancer

def extract_keywords(text: str, count: int = 5) -> list[str]:
    """
    Extracts key terms from the text using KeyBERT.
    """
    print(f"LOG_CHAIN [TermEnhancer] Extracting keywords via KeyBERT...", flush=True)
    try:
        keywords = term_enhancer.extract_key_terms(text, top_n=count)
        print(f"LOG_CHAIN [TermEnhancer] Result: {keywords}", flush=True)
        return keywords
    except Exception as e:
        print(f"❌ [Writer] Keyword extraction failed: {e}")
        return []
