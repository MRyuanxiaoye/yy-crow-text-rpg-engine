import trafilatura
import jieba
import chromadb
import uuid
import asyncio
import aiohttp
from typing import List, Dict
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings 
from app.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import requests
from tavily import TavilyClient

class HybridReader:
    def __init__(self):
        # ... (init stays same)
        # Best Practice: Use OpenAI's text-embedding-3-small for superior multilingual support
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        self.chroma_client = chromadb.Client() # Ephemeral in-memory client
        self.llm = get_llm(temperature=0.3)
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    async def fetch_text_async(self, url: str) -> str:
        """Async Level 1: Crawler with aiohttp"""
        text = ""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        downloaded = await response.text()
                        # Trafilatura extract is CPU bound, run in executor
                        loop = asyncio.get_event_loop()
                        text = await loop.run_in_executor(None, trafilatura.extract, downloaded)
        except Exception as e:
            # print(f"⚠️ [Reader] Async fetch failed: {e}")
            pass

        # Fallback logic (Sync for now or async wrapper needed for Tavily if we want fully async fallback)
        # For simplicity, if simple fetch fails, we skip fallback in async mode to keep it fast, 
        # OR we could wrap the tavily fallback in executor. Let's do executor.
        if not text or len(text) < 100:
            try:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, self._tavily_fallback_sync, url)
            except:
                pass
                
        return text if text else ""

    def _tavily_fallback_sync(self, url):
        # ... (Existing fallback logic)
        try:
            tavily_response = self.tavily.search(
                query=url,
                search_depth="advanced",
                include_raw_content=True,
                max_results=1
            )
            if tavily_response.get('results'):
                return tavily_response['results'][0].get('raw_content', "")
        except:
            return ""
        return ""

    # ... (Keep existing sync methods chunk_text, bm25_search, vector_search, summarize for now)
    # But we need an async entry point

    async def read_and_analyze_async(self, url: str, query: str) -> dict:
        """Async Entry Point"""
        logs = []
        msg = f"📖 [Reader] 正在异步精读: {url}"
        logs.append(msg)
        
        full_text = await self.fetch_text_async(url)
        
        if not full_text or len(full_text) < 200:
            return {"summary": f"抓取失败或过短: {url}", "logs": logs}
            
        # CPU bound tasks run in executor
        loop = asyncio.get_event_loop()
        
        # 1. Chunking
        all_chunks = await loop.run_in_executor(None, self.chunk_text, full_text)
        
        # 2. BM25
        bm25_chunks = await loop.run_in_executor(None, self.bm25_search, all_chunks, query, 20)
        
        # 3. Vector Search (Network bound due to OpenAI embedding, but chromadb is local)
        # Vector search calls embed_documents which calls OpenAI. It's sync in LangChain. 
        # So executor is good.
        final_chunks = await loop.run_in_executor(None, self.vector_search, bm25_chunks, query, 5)
        
        # 4. Summarize (LLM)
        summary = await loop.run_in_executor(None, self.summarize, final_chunks, query)
        
        final_summary = f"### 深度阅读报告 (来源: {url})\n{summary}"
        return {"summary": final_summary, "logs": logs}

    # ... (Keep original sync methods below for reference or mixed use if needed)
    def fetch_text(self, url: str) -> str:
    # ...
        """Level 1: Crawler (Enhanced with Headers + Tavily Fallback)"""
        text = ""
        
        # 1. Try Trafilatura with Headers
        try:
            # Custom headers to mimic a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            # Download with requests first to control headers
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                downloaded = response.text
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        except Exception as e:
            print(f"⚠️ [Reader] Trafilatura fetch failed: {e}")

        # 2. Fallback to Tavily Extract if failed or too short
        if not text or len(text) < 100:
            print(f"⚠️ [Reader] Crawler failed or anti-bot detected (len={len(text) if text else 0}). Switching to Tavily Extract...")
            try:
                # Use Tavily's extract endpoint (via search context actually, or specific extract API if available in SDK)
                # The Python SDK 'search' with 'include_raw_content' is what we have access to in this plan.
                # We can try to search for the specific URL to get its content.
                tavily_response = self.tavily.search(
                    query=url, # Searching the URL directly often triggers extraction
                    search_depth="advanced",
                    include_raw_content=True,
                    max_results=1
                )
                if tavily_response.get('results'):
                    text = tavily_response['results'][0].get('raw_content', "")
            except Exception as e:
                 print(f"❌ [Reader] Tavily fallback failed: {e}")
                 
        return text if text else ""

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Simple chunking strategy"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
            
        return chunks

    def bm25_search(self, chunks: List[str], query: str, top_k: int = 20) -> List[str]:
        """Level 2: BM25 Keyword Filtering (Free)"""
        tokenized_chunks = [list(jieba.cut(doc)) for doc in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
        tokenized_query = list(jieba.cut(query))
        # Safety check: if k > len(chunks), return all
        if top_k >= len(chunks):
            return chunks
            
        top_chunks = bm25.get_top_n(tokenized_query, chunks, n=top_k)
        return top_chunks

    def vector_search(self, chunks: List[str], query: str, top_k: int = 5) -> List[str]:
        """Level 3: Vector Search (High Quality with OpenAI)"""
        if not chunks:
            return []
            
        collection_name = f"temp_{uuid.uuid4().hex[:8]}"
        
        try:
            # 1. Embed Documents
            embeddings = self.embedding_function.embed_documents(chunks)
            
            collection = self.chroma_client.create_collection(name=collection_name)
            
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=[str(i) for i in range(len(chunks))]
            )
            
            # 2. Embed Query
            query_embedding = self.embedding_function.embed_query(query)
            
            # Safety: if k > len, cap it
            actual_k = min(top_k, len(chunks))
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k
            )
            
            return results['documents'][0]
        except Exception as e:
            print(f"❌ [Reader] Vector search error: {e}")
            return chunks[:top_k] # Fallback to first chunks
        finally:
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass

    def summarize(self, chunks: List[str], query: str) -> str:
        """Level 4: LLM Synthesis"""
        if not chunks:
            return "抓取内容为空，无法总结。"
            
        context = "\n\n".join(chunks)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个深度阅读助手。你的任务是阅读长文片段，回答用户的问题。
            
            用户问题: {query}
            
            长文精选片段:
            {context}
            
            请基于以上片段，总结与问题相关的所有关键信息。如果片段中没有答案，请直接说“文中未提及”。
            """),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context})

    def read_and_analyze(self, url: str, query: str) -> dict:
        """Main Entry Point. Returns dict with 'summary' and 'logs'."""
        logs = []
        
        msg = f"📖 [Reader] 正在抓取全书/长文: {url}"
        print(msg)
        logs.append(msg)
        
        full_text = self.fetch_text(url)
        
        if not full_text or len(full_text) < 200:
            msg = f"抓取失败或内容过短 (Length: {len(full_text)}). 跳过深度阅读。"
            logs.append(msg)
            return {"summary": msg, "logs": logs}
            
        msg = f"📖 [Reader] 全文长度: {len(full_text)} 字。正在切片..."
        print(msg)
        logs.append(msg)
        
        all_chunks = self.chunk_text(full_text)
        
        msg = f"🔍 [Reader] Level 2: BM25 初筛 (从 {len(all_chunks)} 个片段中选 20 个)..."
        print(msg)
        logs.append(msg)
        
        bm25_chunks = self.bm25_search(all_chunks, query, top_k=20)
        
        msg = f"🧬 [Reader] Level 3: OpenAI 向量精筛 (从 {len(bm25_chunks)} 个片段中选 5 个)..."
        print(msg)
        logs.append(msg)
        
        final_chunks = self.vector_search(bm25_chunks, query, top_k=5)
        
        msg = f"🧠 [Reader] Level 4: 深度总结..."
        print(msg)
        logs.append(msg)
        
        summary = self.summarize(final_chunks, query)
        
        final_summary = f"### 深度阅读报告 (来源: {url})\n{summary}"
        return {"summary": final_summary, "logs": logs}

if __name__ == "__main__":
    pass
