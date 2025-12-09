import os
import time
import requests
from typing import List, Dict
from tavily import TavilyClient
from exa_py import Exa
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.llm_factory import get_llm

class Researcher:
    def __init__(self):
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.exa = Exa(api_key=os.getenv("EXA_API_KEY"))
        self.ddg_search = DuckDuckGoSearchRun() # 备用免费搜索引擎
        self.bocha_api_key = os.getenv("BOCHA_API_KEY")
        self.llm = get_llm(temperature=0.3)
        
    def _bocha_search(self, query: str, count: int = 3) -> List[Dict]:
        """博查 API - 专注中文高质量生态"""
        if not self.bocha_api_key:
            return []
            
        try:
            url = "https://api.bochaai.com/v1/web-search"
            headers = {
                "Authorization": f"Bearer {self.bocha_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "query": query,
                "freshness": "noLimit", # 不限时间
                "summary": True,
                "count": count
            }
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Bocha response structure adaptation
                results = []
                for item in data.get("data", {}).get("webPages", {}).get("value", []):
                    results.append({
                        "title": item.get("name"),
                        "url": item.get("url"),
                        "content": item.get("summary") or item.get("snippet")
                    })
                return results
        except Exception as e:
            print(f"⚠️ [Researcher] Bocha Search failed: {e}")
        return []

    def _exa_search(self, query: str, category: str = "research paper") -> List[Dict]:
        """Exa Neural Search - 专注英文深度内容"""
        try:
            # Auto-detect if we want papers or blogs based on query context?
            # For now, let's trust the Neural Search to find high quality content.
            response = self.exa.search_and_contents(
                query,
                type="neural",
                use_autoprompt=True,
                num_results=3,
                text=True
            )
            return [
                {"title": r.title, "url": r.url, "content": r.text[:1500]} 
                for r in response.results
            ]
        except Exception as e:
            print(f"⚠️ [Researcher] Exa Search failed: {e}")
            return []

    # --- 增加重试机制 ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _tavily_search_safe(self, query, **kwargs):
        return self.tavily.search(query=query, **kwargs)

    def research_topic(self, query_zh: str, query_en: str = "", domain: str = "General") -> dict:
        """
        执行 4-Engine 混合搜索策略
        """
        logs = []
        all_results = []
        
        # 1. Global Depth (English) - Exa
        if query_en:
            msg = f"🧠 [Researcher] Neural Search (Exa): {query_en}"
            print(msg)
            logs.append(msg)
            exa_results = self._exa_search(query_en)
            if exa_results:
                all_results.extend(exa_results)
                logs.append(f"✅ Exa found {len(exa_results)} deep articles")

        # 2. Domestic Precision (Chinese) - Bocha
        msg = f"🐼 [Researcher] Ecosystem Search (Bocha): {query_zh}"
        print(msg)
        logs.append(msg)
        bocha_results = self._bocha_search(query_zh)
        if bocha_results:
            all_results.extend(bocha_results)
            logs.append(f"✅ Bocha found {len(bocha_results)} domestic sources")
        
        # 3. Global Facts (English/Chinese) - Tavily
        # Use English query for Tavily if available for better fact checking, else Chinese
        tavily_query = query_en if query_en else query_zh
        msg = f"🔎 [Researcher] Fact Search (Tavily): {tavily_query}"
        print(msg)
        logs.append(msg)
        try:
            tavily_resp = self._tavily_search_safe(
                query=tavily_query,
                search_depth="advanced",
                max_results=3,
                include_raw_content=True
            )
            t_results = tavily_resp.get('results', [])
            all_results.extend(t_results)
            logs.append(f"✅ Tavily found {len(t_results)} fact pages")
        except Exception as e:
            logs.append(f"❌ Tavily failed: {e}")

        # 4. Fallback - DDG (Only if total results are low)
        if len(all_results) < 2:
            msg = f"🦆 [Researcher] Low data, triggering Fallback (DDG): {query_zh}"
            print(msg)
            logs.append(msg)
            try:
                ddg_text = self.ddg_search.invoke(query_zh)
                if ddg_text:
                    all_results.append({"title": "DDG Backup", "url": "ddg", "content": ddg_text})
            except Exception as e:
                logs.append(f"❌ DDG failed: {e}")

        # Analyze
        if not all_results:
            return {"summary": "NO_DATA", "logs": logs}
            
        summary = self.analyze_and_select(query_zh, all_results) # Always summarize in context of the original user query
        return {"summary": summary, "logs": logs}

    def analyze_and_select(self, query: str, results: List[Dict]) -> str:
        """
        Reads search results and summarizes.
        """
        if not results:
            return "NO_DATA"

        # Context formatting
        context = "\n\n".join([
            f"Source: {r.get('title', 'Unknown')} ({r.get('url', 'Unknown')})\nContent: {r.get('content', '')[:1200]}..." 
            for r in results
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个严谨的研究助手。你的任务是根据提供的搜索结果摘要，提取出能回答查询 "{query}" 的核心信息。

            注意：资料可能包含英文和中文。请综合理解，**并用中文**进行总结。

            要求：
            1. **提取事实**：只提取与查询紧密相关的事实。不要遗漏任何具体的实体、数据、定义。
            2. **判断长文价值**：如果你发现某个链接的内容似乎是一篇很有价值的长文（如书籍章节、长篇教程），请在总结中特别注明 "FOUND_LONG_TEXT: <URL>"，以便后续模块进行深度阅读。
            3. **去噪**：忽略广告和无关导航。
            """),
            ("user", "搜索结果:\n{context}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"query": query, "context": context})
        return summary
