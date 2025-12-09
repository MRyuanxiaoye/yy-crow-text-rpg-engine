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
            
            # 2. RPC call to match_documents 
            # Fixed parameter name to avoid ambiguity: use 'query_embedding_vec' instead of 'query_embedding'
            # You MUST update the function in Supabase to accept 'query_embedding_vec' or use named notation carefully.
            # Let's try to use named notation with a distinct name if we updated the SQL function.
            # But since we can't easily update SQL from here without user action, let's assume the user WILL run the SQL I provided.
            # I will use the standard 'query_embedding' but rely on the SQL fix (using table alias).
            # Actually, to be safe, let's pass it as a list if the RPC supports it, or just hope the SQL fix resolves it.
            # Wait, the error was "column reference... is ambiguous". This is purely SQL side.
            # The Python client just passes a dict. The keys in the dict must match the FUNCTION ARGUMENTS.
            # If the function arg is named 'query_embedding' and the table col is 'query_embedding', that's the issue.
            # I will assume the user runs my SQL fix which keeps the arg name but fixes the query body.
            
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
                score = data[0].get('similarity', 0)
                print(f"⚡ [KB Manager] Potential Hit... Score: {score:.4f} (Threshold: {threshold})")
                
                if score >= threshold:
                    print(f"✅ [KB Manager] Cache HIT confirmed!")
                    return data[0]['final_report']
                else:
                    print(f"⚠️ [KB Manager] Cache MISS (Score too low)")
                    return None
            
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

