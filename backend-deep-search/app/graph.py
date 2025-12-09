from typing import TypedDict, List, Annotated, Optional
import operator
import re
from langgraph.graph import StateGraph, END
from langchain_community.callbacks import get_openai_callback

from app.agents.planner import plan_research, SubTopic
from app.agents.researcher import Researcher
from app.agents.writer import write_report, extract_keywords
from app.tools.reader_engine import HybridReader
from app.tools.kb_manager import KnowledgeBaseManager
import asyncio

# Define the State
class AgentState(TypedDict):
    query: str
    
    # Planner State
    domain: Optional[str]
    role_name: Optional[str]
    persona_instruction: Optional[str]
    search_strategy: Optional[str]
    output_format_instruction: Optional[str]
    # NOTE: LangGraph validates this against Pydantic models if defined as such.
    # If backend/app/agents/planner.py has a different SubTopic definition, this might fail validation
    # if the graph was initialized with old types.
    # To be safe, let's use List[dict] for a moment if we suspect type mismatch, 
    # but we should try to keep SubTopic.
    # sub_topics: List[SubTopic] 
    sub_topics: List[dict] # Relaxed type for debugging
    
    # Data State
    research_data: Annotated[List[str], operator.add]
    long_text_urls: Annotated[List[str], operator.add] # URLs to be deep-read
    
    # Output State
    final_report: str
    keywords: List[str] # New field for keywords
    logs: Annotated[List[str], operator.add]
    
    # Cost Tracking
    total_tokens: Optional[int]
    total_cost: Optional[float]

# Initialize Modules
researcher = Researcher()
reader = HybridReader()
kb_manager = KnowledgeBaseManager()

# --- Node Functions ---

def check_cache_node(state: AgentState):
    query = state["query"]
    logs = []
    
    logs.append(f"🧠 [Cache] 正在检查云端知识库: {query}...")
    cached_report = kb_manager.check_cache(query)
    
    if cached_report:
        logs.append(f"⚡ [Cache] 命中缓存！跳过搜索流程。")
        return {
            "final_report": cached_report,
            "logs": logs,
            "research_data": [],
            # Trick to skip other nodes: we'll handle this in conditional edges
            # But LangGraph standard way is conditional edge.
            # For simplicity in this linear graph, we can populate final_report and let others check?
            # Or better: Add a conditional edge after start.
        }
    else:
        logs.append(f"💨 [Cache] 未命中，开始新一轮深度搜索...")
        return {"logs": logs}

def plan_node(state: AgentState):
    import sys
    # print("DEBUG: Entering plan_node function...", flush=True)
    query = state["query"]
    print(f"LOG_CHAIN [Planner] Input State: query={query}", flush=True)
    log_msg = f"🧠 [Planner] 正在深度拆解问题: {query}..."
    print(log_msg, flush=True)
    
    try:
        # print("DEBUG: Starting get_openai_callback block...", flush=True)
        with get_openai_callback() as cb:
            # print("DEBUG: Calling plan_research...", flush=True)
            plan = plan_research(query)
            # print("DEBUG: plan_research returned.", flush=True)
            # cost_info = f"💰 [Planner] Cost: ${cb.total_cost:.4f} (Tokens: {cb.total_tokens})"
            # print(f"DEBUG: Cost info generated: {cost_info}", flush=True)
        
        # DEBUG: Print the raw plan object types
        # print(f"DEBUG: Raw plan type: {type(plan)}", flush=True)
        
        # print("DEBUG: Constructing state_update dict...", flush=True)
        # Manual check for keys to avoid KeyError if schema mismatch
        sub_topics = plan.get("sub_topics", [])
        # print(f"DEBUG: Extracted sub_topics (count: {len(sub_topics)})", flush=True)
        print(f"LOG_CHAIN [Planner] Output Plan: {len(sub_topics)} sub-topics", flush=True)
        
        return_value = {
            "domain": plan.get("domain", "General"),
            "role_name": plan.get("role_name", "Researcher"),
            "persona_instruction": plan.get("persona_instruction", "Answer accurately."),
            "search_strategy": plan.get("search_strategy", "General"),
            "output_format_instruction": plan.get("output_format_instruction", "Markdown"),
            "sub_topics": sub_topics,
            "logs": [
                log_msg, 
                # f"🗺️ [Planner] 领域: {plan.get('domain', 'General')} | 角色: {plan.get('role_name', 'Researcher')}",
                # cost_info
            ]
        }
        
        # print("DEBUG: plan_node finished execution, returning value.", flush=True)
        return return_value
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in plan_node: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            "logs": [f"❌ Planner crashed: {e}"],
            "sub_topics": [] 
        }

async def research_node(state: AgentState):
    # print("DEBUG: Entering research_node...", flush=True)
    query = state["query"]
    sub_topics = state["sub_topics"]
    # print(f"DEBUG: sub_topics count: {len(sub_topics)}", flush=True)
    print(f"LOG_CHAIN [Researcher] Input: {len(sub_topics)} sub-topics", flush=True)
    domain = state.get("domain", "General")
    
    logs = []
    gathered_info = []
    found_urls = []
    
    total = len(sub_topics)
    log_msg = f"🚀 [Researcher] 正在为“{query}”执行 {total} 个维度的深度搜索..."
    logs.append(log_msg)
    print(log_msg, flush=True)

    # Create async tasks
    tasks = []
    for i, topic in enumerate(sub_topics):
        q_zh = topic.get("sub_query", "")
        q_en = topic.get("sub_query_en", "")
        # We wrap the call with metadata so we know which topic it belongs to
        tasks.append(process_research_task(i, q_zh, q_en, domain, topic.get("reason", "")))

    # Run concurrently
    # return_exceptions=True allows one failure not to crash the whole batch
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Track logs to avoid duplicates if necessary, but simple list is fine
    
    for res in results:
        if isinstance(res, Exception):
            # logs.append(f"❌ Task failed: {res}")
            continue
            
        summary = res["summary"]
        # logs.extend(res["logs"]) # Skip technical logs
        
        long_text_matches = re.findall(r"FOUND_LONG_TEXT: (https?://[^\s]+)", summary)
        found_urls.extend(long_text_matches)
        
        if summary != "NO_DATA":
            data_entry = f"### 子主题: {res['q_zh']}\n**目的**: {res['reason']}\n**发现**: {summary}\n---"
            gathered_info.append(data_entry)

    logs.append(f"🔍 [Researcher] 已完成“{query}”的信息搜集，筛选出 {len(found_urls)} 篇核心文献。")

    print(f"LOG_CHAIN [Researcher] Output: {len(gathered_info)} items, {len(logs)} logs", flush=True)

    return {
        "research_data": gathered_info,
        "long_text_urls": found_urls,
        "logs": logs
    }

async def process_research_task(index, q_zh, q_en, domain, reason):
    """Helper for async research task"""
    import time
    start = time.time()
    # print(f"  Start Task {index}: {q_zh}")
    result = await researcher.research_topic_async(query_zh=q_zh, query_en=q_en, domain=domain)
    # print(f"  End Task {index}")
    end = time.time()
    # print(f"⏱️ [Task {index}] Search took {end - start:.2f}s", flush=True)
    
    return {
        "summary": result["summary"],
        "logs": [], # result["logs"] + [f"⏱️ [Task {index}] Time: {end - start:.2f}s"],
        "q_zh": q_zh,
        "reason": reason
    }

async def reader_node(state: AgentState):
    query = state["query"]
    urls = state.get("long_text_urls", [])
    urls = list(set(urls))
    
    print(f"LOG_CHAIN [Reader] Input: {len(urls)} URLs", flush=True)

    if not urls:
        return {"logs": [f"📚 [Reader] 关于“{query}”的搜索结果中未发现长文，跳过深度阅读。"]}
        
    logs = []
    deep_summaries = []
    
    log_msg = f"📚 [Reader] 正在深度阅读关于“{query}”的 {len(urls)} 篇核心资料..."
    logs.append(log_msg)
    print(log_msg, flush=True)
    
    import time
    start_time = time.time()

    tasks = []
    for url in urls:
        tasks.append(reader.read_and_analyze_async(url, state["query"]))
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    # print(f"⏱️ [Reader] Total reading took {end_time - start_time:.2f}s", flush=True)
    # logs.append(f"⏱️ [Reader] Total Time: {end_time - start_time:.2f}s")
    
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            # logs.append(f"❌ Reader task failed: {res}")
            continue
            
        deep_summaries.append(res["summary"])
        # logs.extend(res["logs"])
        # logs.append(f"✅ [Reader] 完成: {urls[i]}")

    print(f"LOG_CHAIN [Reader] Output: {len(deep_summaries)} summaries", flush=True)

    return {
        "research_data": deep_summaries,
        "logs": logs
    }


def write_node(state: AgentState):
    query = state["query"]
    data = state["research_data"]
    
    print(f"LOG_CHAIN [Writer] Input Data Length: {len(str(data))}", flush=True)

    role = state.get("role_name", "Researcher")
    p_instr = state.get("persona_instruction", "")
    fmt_instr = state.get("output_format_instruction", "")
    
    log_msg = f"✍️ [Writer] 正在为“{query}”撰写最终报告..."
    print(log_msg, flush=True)
    
    import time
    start_time = time.time()
    
    with get_openai_callback() as cb:
        full_context = "\n\n".join(data)
        if not full_context:
            full_context = "无外部资料。"
        
        report = write_report(query, full_context, role, p_instr, fmt_instr)
        # cost_msg = f"💰 [Writer] Cost: ${cb.total_cost:.4f}"
        
        # Async save to cache
        if kb_manager.enabled:
            kb_manager.save_to_cache(query, report)
    
    end_time = time.time()
    # print(f"⏱️ [Writer] Writing took {end_time - start_time:.2f}s", flush=True)
    print(f"LOG_CHAIN [Writer] Output Report Length: {len(report)}", flush=True)
            
    # Extract keywords
    try:
        keywords = extract_keywords(report)
    except:
        keywords = []
    
    return {
        "final_report": report,
        "keywords": keywords,
        "logs": [log_msg] # , cost_msg, f"⏱️ [Writer] Time: {end_time - start_time:.2f}s", "✅ [Writer] 报告完成，已存入云端知识库"]
    }

# --- Build Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", plan_node)
workflow.add_node("researcher", research_node)
workflow.add_node("reader", reader_node)
workflow.add_node("writer", write_node)

workflow.set_entry_point("check_cache")

# Conditional Edge
def route_after_cache(state: AgentState):
    if state.get("final_report"):
        return END
    return "planner"

workflow.add_conditional_edges(
    "check_cache",
    route_after_cache,
    {
        END: END,
        "planner": "planner"
    }
)

workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "reader")
workflow.add_edge("reader", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()
