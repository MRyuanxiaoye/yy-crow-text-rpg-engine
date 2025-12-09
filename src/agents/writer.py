from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm_factory import get_llm

def write_report(query: str, research_data: str, role_name: str, persona_instruction: str, output_format_instruction: str) -> str:
    """
    Synthesizes research notes into a comprehensive report using Adaptive Formatting.
    """
    llm = get_llm(temperature=0.7) 

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你现在的身份是：{role_name}。
        
        你的任务是基于提供的研究资料，撰写一篇深度回答，回应用户问题："{query}"。

        **注意**：你的研究资料可能包含**英文**和**中文**的内容。
        - 请充分利用英文资料中的高密度信息。
        - 最终报告必须使用**中文**撰写。
        - 在引用英文资料时，请自然地将其内容融合到中文语境中，无需生硬翻译，而是侧重于原意的精准传达。

        **核心要求**：
        1. **饱满充实**：不要只写干巴巴的结论。对每一个核心论点，都要利用资料中的细节进行充分的展开和解释。
        2. **深度融合**：你拥有一些“微观案例”的资料，请**自然地**将它们穿插在你的论述中作为证据，而不要生硬地列出“案例1、案例2”。
        3. **逻辑连贯**：文章应该有起承转合，像一篇高质量的深度科普文章或学术综述，而不是简单的清单（Listicle）。
        4. **弥补空白**：如果资料中缺少某些关键环节（例如“创世神话”的具体内容），请明确指出资料缺失，并基于你的通用知识库（General Knowledge）进行**有理有据的补充推演**，但必须标注“基于通理推测”。

        **角色特定指令**：
        {persona_instruction}

        **格式要求**：
        {output_format_instruction}

        **研究资料**：
        {research_data}
        """),
        ("user", "请开始撰写。")
    ])

    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "query": query, 
        "research_data": research_data,
        "role_name": role_name,
        "persona_instruction": persona_instruction,
        "output_format_instruction": output_format_instruction
    })

def extract_keywords(text: str, count: int = 5) -> list[str]:
    """
    Extracts key terms from the text using LLM.
    """
    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a keyword extraction tool. Extract {count} complex or specialized terms from the text that might need explanation. Return ONLY the terms, separated by commas. No intro/outro."),
        ("user", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": text[:3000], "count": count}) # Limit context to save tokens
    return [term.strip() for term in result.split(",") if term.strip()]
