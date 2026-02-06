"""
LLM Factory - 双模型策略
- GPT-4o: 架构师 (逻辑校验、冲突检测)
- DeepSeek: 创作者 (生成创意、润色设定)
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../backend-deep-search/.env")
load_dotenv()


def get_architect_llm(temperature: float = 0.3):
    """
    获取架构师 LLM (GPT-4o)
    用于：逻辑校验、冲突检测、结构分析
    低温度 = 更严谨
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        api_key=api_key,
        max_tokens=4096
    )


def get_creator_llm(temperature: float = 0.7):
    """
    获取创作者 LLM (DeepSeek)
    用于：生成创意、扩展设定、润色描述
    高温度 = 更有创意
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found")
    
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        max_tokens=4096
    )


def get_llm(role: str = "creator", temperature: float = None):
    """
    统一入口：根据角色获取对应 LLM
    
    Args:
        role: "architect" (GPT-4o) 或 "creator" (DeepSeek)
        temperature: 可选，覆盖默认温度
    """
    if role == "architect":
        return get_architect_llm(temperature if temperature is not None else 0.3)
    else:
        return get_creator_llm(temperature if temperature is not None else 0.7)


# 测试连接
if __name__ == "__main__":
    print("Testing LLM connections...")
    
    try:
        architect = get_architect_llm()
        result = architect.invoke("Say 'Architect ready' in 3 words")
        print(f"✅ Architect (GPT-4o): {result.content}")
    except Exception as e:
        print(f"❌ Architect failed: {e}")
    
    try:
        creator = get_creator_llm()
        result = creator.invoke("Say 'Creator ready' in 3 words")
        print(f"✅ Creator (DeepSeek): {result.content}")
    except Exception as e:
        print(f"❌ Creator failed: {e}")

