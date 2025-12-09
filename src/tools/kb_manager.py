import os
from typing import Optional, Dict, List
import asyncio
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings

class KnowledgeBaseManager:
    def __init__(self):
        url: str = os.getenv("SUPABASE_URL")
        key: str = os.getenv("SUPABASE_KEY")
        
        self.enabled = False
        if url and key:
            try:
                self.supabase: Client = create_client(url, key)
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                )
                self.enabled = True
                print("✅ [KB Manager] Supabase connected.")
            except Exception as e:
                print(f"⚠️ [KB Manager] Connection failed: {e}")
        else:
            print("⚠️ [KB Manager] SUPABASE_URL or SUPABASE_KEY missing. Caching disabled.")

    def check_cache(self, query: str, threshold: float = 0.95) -> Optional[str]:
        """
        Checks for semantic cache hit.
        """
        if not self.enabled:
            return None
            
        try:
            # 1. Embed query
            query_vec = self.embeddings.embed_query(query)
            
            # 2. RPC call to match_documents (we assume a 'match_search_cache' function exists in DB)
            # Or simple select if using pgvector
            # Here we assume a Supabase Edge Function or RPC 'match_search_cache'
            # signature: (query_embedding vector(1536), match_threshold float, match_count int)
            
            response = self.supabase.rpc(
                "match_search_cache",
                {
                    "query_embedding": query_vec,
                    "match_threshold": threshold,
                    "match_count": 1
                }
            ).execute()
            
            data = response.data
            if data and len(data) > 0:
                print(f"⚡ [KB Manager] Cache HIT! (Similarity: {data[0]['similarity']:.4f})")
                return data[0]['final_report']
            
        except Exception as e:
            print(f"⚠️ [KB Manager] Cache check error: {e}")
            
        return None

    def save_to_cache(self, query: str, report: str):
        """
        Saves the final report to cache.
        """
        if not self.enabled:
            return
            
        try:
            query_vec = self.embeddings.embed_query(query)
            
            data = {
                "query_text": query,
                "query_embedding": query_vec,
                "final_report": report
            }
            
            self.supabase.table("search_cache").insert(data).execute()
            print("💾 [KB Manager] Report saved to cache.")
        except Exception as e:
            print(f"⚠️ [KB Manager] Save to cache failed: {e}")

    def save_content_lake(self, items: List[Dict]):
        """
        Saves raw research content to the lake for RAG.
        """
        if not self.enabled or not items:
            return
            
        # This can be heavy, so we might want to run it in background or limit batch size
        # For prototype, we just print intent
        print(f"💾 [KB Manager] Saving {len(items)} items to Content Lake (Async placeholder)...")
        # Actual implementation would embed 'content' and insert into 'content_lake' table

