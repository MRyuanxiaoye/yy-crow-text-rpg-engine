import os
import asyncio
import aiohttp
from typing import List, Dict
from tavily import TavilyClient, AsyncTavilyClient
from exa_py import Exa
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
from app.llm_factory import get_llm
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

class Researcher:
    def __init__(self):
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.exa_key = os.getenv("EXA_API_KEY")
        self.bocha_key = os.getenv("BOCHA_API_KEY")
        
        # Explicit logging for debugging env vars (masked partly for security)
        print(f"DEBUG: TAVILY_API_KEY loaded: {bool(self.tavily_key)}", flush=True)
        print(f"DEBUG: EXA_API_KEY loaded: {bool(self.exa_key)}", flush=True)
        print(f"DEBUG: BOCHA_API_KEY loaded: {bool(self.bocha_key)}", flush=True)
        
        if not self.exa_key:
            print("⚠️ [Researcher] EXA_API_KEY is missing! Please check .env file.", flush=True)

        try:
            self.tavily = TavilyClient(api_key=self.tavily_key) 
            self.async_tavily = AsyncTavilyClient(api_key=self.tavily_key) 
        except Exception as e:
            print(f"❌ [Researcher] Tavily init failed: {e}")

        try:
            self.exa = Exa(api_key=self.exa_key) if self.exa_key else None
        except Exception as e:
            print(f"❌ [Researcher] Exa init failed: {e}")
            self.exa = None

        self.ddg_search = DuckDuckGoSearchRun() 
        self.bocha_api_key = self.bocha_key
        self.llm = get_llm(temperature=0.3)
        
    async def _bocha_search_async(self, query: str, count: int = 3) -> List[Dict]:
        """Async Bocha API"""
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
                "freshness": "noLimit",
                "summary": True,
                "count": count
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=10, ssl=False) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get("data", {}).get("webPages", {}).get("value", []):
                            results.append({
                                "title": item.get("name"),
                                "url": item.get("url"),
                                "content": item.get("summary") or item.get("snippet")
                            })
                        return results
        except Exception as e:
            print(f"⚠️ [Researcher] Async Bocha Search failed: {e}")
        return []

    async def _exa_search_async(self, query: str) -> List[Dict]:
        """Async Exa Wrapper (Exa SDK is sync, so we wrap it)"""
        if not self.exa:
            print("⚠️ [Researcher] Exa client not initialized, skipping search.", flush=True)
            return []
            
        # Note: Exa currently doesn't have a native async method in public SDK usually, 
        # so we run it in a thread to not block the event loop.
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._exa_search_sync_impl, query)
        except Exception as e:
            print(f"⚠️ [Researcher] Async Exa Search failed: {e}", flush=True)
            return []

    def _exa_search_sync_impl(self, query: str) -> List[Dict]:
        if not self.exa:
            return []
        try:
            print(f"DEBUG: Executing Exa search for '{query}'...", flush=True)
            # Exa SDK 2.0+ Migration:
            # 1. Use search() instead of search_and_contents()
            # 2. 'use_autoprompt' is valid for search() in some versions, but error says "Invalid option".
            #    It implies for 'neural' search it might be implied or parameter name changed.
            #    Checking docs: use_autoprompt is indeed a parameter for search(). 
            #    However, the error `Invalid option: 'use_autoprompt'` usually comes from the API response validation 
            #    if the SDK passes it incorrectly or if the account tier/model doesn't support it.
            #    Safest fix: Remove it for now to unblock.
            # 3. 'text=True' becomes 'contents={"text": True}'
            
            response = self.exa.search(
                query,
                type="neural",
                num_results=3,
                contents={"text": True}
            )
            
            results = [
                {"title": r.title, "url": r.url, "content": r.text[:1500] if r.text else ""} 
                for r in response.results
            ]
            print(f"DEBUG: Exa returned {len(results)} results.", flush=True)
            return results
        except Exception as e:
            print(f"❌ [Researcher] Exa API Error: {e}", flush=True)
            return []

    async def _tavily_search_async(self, query: str) -> List[Dict]:
        """Async Tavily Search"""
        try:
            # The async client search method is awaitable
            tavily_resp = await self.async_tavily.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_raw_content=True
            )
            return tavily_resp.get('results', [])
        except Exception as e:
            print(f"⚠️ [Researcher] Async Tavily Search failed: {e}")
            return []

    async def research_topic_async(self, query_zh: str, query_en: str = "", domain: str = "General") -> dict:
        """
        Execute 4-Engine Search in Parallel
        """
        logs = []
        tasks = []
        
        # 1. Exa (Neural) - Global Depth
        if query_en:
            logs.append(f"🧠 [Researcher] Neural Search (Exa): {query_en}")
            tasks.append(self._exa_search_async(query_en))
        else:
            tasks.append(asyncio.sleep(0)) # No-op

        # 2. Bocha (Ecosystem) - Domestic Precision
        logs.append(f"🐼 [Researcher] Ecosystem Search (Bocha): {query_zh}")
        tasks.append(self._bocha_search_async(query_zh))
        
        # 3. Tavily (Fact) - Global Facts
        tavily_query = query_en if query_en else query_zh
        logs.append(f"🔎 [Researcher] Fact Search (Tavily): {tavily_query}")
        tasks.append(self._tavily_search_async(tavily_query))

        # Run all concurrently
        results_list = await asyncio.gather(*tasks)
        
        # Unpack
        exa_results = results_list[0] if query_en else []
        bocha_results = results_list[1]
        tavily_results = results_list[2]
        
        print(f"LOG_CHAIN [Researcher Internal] Search results: Exa={len(exa_results)}, Bocha={len(bocha_results)}, Tavily={len(tavily_results)}", flush=True)
        
        all_results = []
        if exa_results: 
            all_results.extend(exa_results)
            logs.append(f"✅ Exa found {len(exa_results)} articles")
        if bocha_results: 
            all_results.extend(bocha_results)
            logs.append(f"✅ Bocha found {len(bocha_results)} sources")
        if tavily_results: 
            all_results.extend(tavily_results)
            logs.append(f"✅ Tavily found {len(tavily_results)} pages")
            
        # 4. Fallback DDG (Sync fallback if needed, but we skip for speed optimization if we have data)
        if len(all_results) < 2:
             # Only run fallback if others failed badly
             # Wrap sync DDG in executor
             try:
                 loop = asyncio.get_event_loop()
                 ddg_text = await loop.run_in_executor(None, self.ddg_search.invoke, query_zh)
                 if ddg_text:
                     all_results.append({"title": "DDG Backup", "url": "ddg", "content": ddg_text})
                     logs.append("🦆 Fallback DDG triggered")
             except:
                 pass

        if not all_results:
            return {"summary": "NO_DATA", "logs": logs}
            
        # Summarize (This is an LLM call, we can await it if we make analyze_and_select async, 
        # or just run it in executor. For now, LangChain invoke is sync, so executor.)
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, self.analyze_and_select, query_zh, all_results)
        
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
