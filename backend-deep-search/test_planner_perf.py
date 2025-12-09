import os
import time
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM (Using the same config as in app)
llm = ChatOpenAI(
    model="deepseek-chat", 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base="https://api.deepseek.com",
    temperature=0.5
)

COMPLEX_PROMPT = """你是一个智能体规划大师。你的任务是分析用户问题，制定深度的研究计划。

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

请严格按照 JSON 格式返回，不要包含 markdown 格式化符号 ```json。
"""

SIMPLE_PROMPT = """你是一个搜索规划助手。
请分析用户问题 "{query}"，并拆解出 3-5 个具体的搜索子问题 (Sub-topics)。
对于每个子问题，生成中文查询词 (sub_query) 和 英文查询词 (sub_query_en)。
请直接返回 JSON 格式。
"""

def test_prompt(name, prompt_template, query):
    print(f"\n--- Testing {name} ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("user", "用户问题: {query}")
    ])
    
    chain = prompt | llm
    
    start = time.time()
    try:
        response = chain.invoke({"query": query})
        end = time.time()
        
        content = response.content
        token_count = len(content) / 4 # Approximate
        duration = end - start
        speed = token_count / duration
        
        print(f"⏱️ Duration: {duration:.2f}s")
        print(f"📝 Content Length: {len(content)} chars (~{int(token_count)} tokens)")
        print(f"🚀 Speed: {speed:.2f} tokens/s")
        print(f"👀 Preview: {content[:200]}...")
        
        # Check if it contains thinking process (DeepSeek specific)
        if "<think>" in content:
            print("⚠️  Response contains <think> tags!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    query = "深圳的崛起历程是怎样的"
    
    # 1. Test Complex Prompt (Current Production)
    test_prompt("Complex Prompt (Current)", COMPLEX_PROMPT, query)
    
    # 2. Test Simple Prompt
    test_prompt("Simple Prompt", SIMPLE_PROMPT, query)

