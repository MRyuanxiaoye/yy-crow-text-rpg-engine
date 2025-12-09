import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm(temperature: float = 0.7):
    """
    Returns the configured LLM client.
    Defaults to DeepSeek via OpenAI-compatible interface.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")

    return ChatOpenAI(
        model="deepseek-chat",  # DeepSeek-V3 model name
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
        max_tokens=4096
    )

