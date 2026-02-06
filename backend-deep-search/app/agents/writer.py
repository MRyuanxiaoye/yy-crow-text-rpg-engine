from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm_factory import get_llm

def write_report(query: str, research_data: str, role_name: str, persona_instruction: str, output_format_instruction: str) -> str:
    """
    Synthesizes research notes into a comprehensive report using Waterfall-style Knowledge Cascade.
    """
    llm = get_llm(temperature=0.7)
    
    # === DEBUG LOG: Writer Input ===
    print(f"LOG_WRITER [Input] Query: {query}", flush=True)
    print(f"LOG_WRITER [Input] Research data length: {len(research_data)} chars", flush=True)
    print(f"LOG_WRITER [Input] Persona: {persona_instruction[:100]}...", flush=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个**知识瀑布生成器**。你的任务是将搜索到的所有资料，像瀑布一样**层层倾泻**给用户——每一层都比上一层更深入、更具体、更接近真相。

        **【核心定位】**
        你是**知识搬运工**，不是教导者或说服者。你只负责**清晰展示信息和逻辑链条**，让读者自行判断。

        **【输出原则：瀑布式倾泻】**
        1.  **一泻而下**：不要在中途停下来做"小结"。从开头到结尾，每一段都在向更深处推进。
        2.  **资料决定长度**：资料多就写长，资料少就写短。**不设上限，不设下限**。
        3.  **因果链条不断**：每陈述一个事实，紧接着追问"为什么"、"导致了什么"、"背后的机制是什么"。
        4.  **深度洞察**：不只陈述"是什么"，更要揭示"为什么会这样"、"背后的权力/利益/制度博弈"。

        **【★ 精彩故事的处理方式 ★】**
        当资料中出现**精彩的故事、轶事、转折点**时，要用**突出的概括**呈现，抓住关键节点串联：
        - ✅ 好的概括："安禄山因盗羊被捕，临刑前高呼一句'我能杀两蕃'，打动了张守珪，不仅免死还被任命为'捉生将'。因熟悉边境地形，屡立战功，最终被收为养子。"
        - ❌ 差的概括："安禄山后来被张守珪收为养子。"
        - **关键**：不是展开成长篇故事，而是用**有节奏感的短句串联关键节点**，让读者能"看到"故事的起伏。

        **【结构设计：瀑布式深挖】**
        - **第一层**：直接回答用户的问题（时间、定义、核心答案）
        - **第二层**：背景展开——为什么会有这个问题？历史脉络是什么？
        - **第三层**：机制拆解——具体是怎么运作的？关键要素有哪些？
        - **第四层**：关键节点——哪些人/事件起了决定性作用？为什么？
        - **第五层+**：深入再深入——每个关键节点背后的故事、博弈、细节...
        - **没有"最后一层"**：资料有多少，就挖多深。

        **【标题设计】**
        -   **主标题**：直接概括核心内容，简洁有力
        -   **小标题**：必须是**内容化**的（如"张道陵的制度设计"、"公元142年的关键转折"）
        -   **禁止流程性小标题**：不要用"机制拆解"、"深度分析"、"核心洞察"、"总结"等词

        **【格式要求】**
        -   关键术语使用 **加粗**
        -   适当使用 Markdown **表格**来对比不同阶段/人物/维度
        -   每一段落的最后一句，尽量引出下一段落的内容（形成"瀑布"的连贯感）

        **【严格禁止】**
        -   ❌ 禁止在文末做任何形式的"总结"、"综上所述"、"总的来说"
        -   ❌ 禁止使用"值得注意的是"、"更重要的是"、"不仅...而且..."等过渡词
        -   ❌ 禁止固定段落数——根据资料自然分段
        -   ❌ 禁止使用感叹号和营销号句式
        -   ❌ 禁止把精彩故事压缩成一句无趣的话
        -   ❌ 文章结束时**戛然而止**——最后一个事实讲完就结束，不需要收尾

        **【角色指令】**：{persona_instruction}

        **【研究资料】**：
        {research_data}
        
        **【用户的原始问题】**：
        {query}
        """),
        ("user", "请基于上述资料，用瀑布式结构撰写回答。记住：精彩故事要用突出的概括呈现，抓住关键节点串联。")
    ])

    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "query": query, 
        "research_data": research_data,
        "role_name": role_name,
        "persona_instruction": persona_instruction,
        "output_format_instruction": output_format_instruction
    })
    
    # === DEBUG LOG: Writer Output ===
    print(f"LOG_WRITER [Output] Report length: {len(result)} chars", flush=True)
    print(f"LOG_WRITER [Output] First 200 chars: {result[:200]}...", flush=True)
    
    return result

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
