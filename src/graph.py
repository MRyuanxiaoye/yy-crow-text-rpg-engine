from typing import TypedDict, List, Annotated, Optional
import operator
import re
from langgraph.graph import StateGraph, END
from langchain_community.callbacks import get_openai_callback

from src.agents.planner import plan_research, SubTopic
from src.agents.researcher import Researcher
from src.agents.writer import write_report, extract_keywords
from src.tools.reader_engine import HybridReader
from src.tools.kb_manager import KnowledgeBaseManager

# Define the State
class AgentState(TypedDict):
    query: str
    
    # Planner State
    domain: Optional[str]
    role_name: Optional[str]
    persona_instruction: Optional[str]
    search_strategy: Optional[str]
    output_format_instruction: Optional[str]
    sub_topics: List[SubTopic]
    
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
    query = state["query"]
    log_msg = f"🧠 [Planner] 正在深度拆解问题: {query}..."
    print(log_msg)
    
    with get_openai_callback() as cb:
        plan = plan_research(query)
        cost_info = f"💰 [Planner] Cost: ${cb.total_cost:.4f} (Tokens: {cb.total_tokens})"
    
    return {
        "domain": plan.get("domain", "General"),
        "role_name": plan.get("role_name", "Researcher"),
        "persona_instruction": plan.get("persona_instruction", "Answer accurately."),
        "search_strategy": plan.get("search_strategy", "General"),
        "output_format_instruction": plan.get("output_format_instruction", "Markdown"),
        "sub_topics": plan.get("sub_topics", []),
        "logs": [
            log_msg, 
            f"🗺️ [Planner] 领域: {plan.get('domain')} | 角色: {plan.get('role_name')}",
            cost_info
        ]
    }

def research_node(state: AgentState):
    sub_topics = state["sub_topics"]
    domain = state.get("domain", "General")
    
    gathered_info = []
    found_urls = []
    logs = []
    total_cost = 0.0
    total_tokens = 0
    
    total = len(sub_topics)
    
    # We only track LLM cost here (summarizer in researcher). Tavily cost is external.
    with get_openai_callback() as cb:
        for i, topic in enumerate(sub_topics):
            q_zh = topic.get("sub_query", "")
            q_en = topic.get("sub_query_en", "")
            reason = topic.get("reason", "")
            
            log_msg = f"🕵️‍♂️ [Researcher] ({i+1}/{total}) 正在挖掘: {q_zh} | {q_en}"
            print(log_msg)
            logs.append(log_msg)
            
            result = researcher.research_topic(query_zh=q_zh, query_en=q_en, domain=domain)
            summary = result["summary"]
            logs.extend(result["logs"])
            
            long_text_matches = re.findall(r"FOUND_LONG_TEXT: (https?://[^\s]+)", summary)
            found_urls.extend(long_text_matches)
            
            if summary != "NO_DATA":
                data_entry = f"### 子主题: {q_zh} (EN: {q_en})\n**目的**: {reason}\n**发现**: {summary}\n---"
                gathered_info.append(data_entry)
                logs.append(f"✅ [Researcher] 获取资料: {len(summary)} 字")
            else:
                logs.append(f"⚠️ [Researcher] 未找到有效资料: {q_zh}")
                
        total_cost = cb.total_cost
        total_tokens = cb.total_tokens
        logs.append(f"💰 [Researcher] LLM Cost: ${total_cost:.4f}")
            
    return {
        "research_data": gathered_info,
        "long_text_urls": found_urls,
        "logs": logs
    }

def reader_node(state: AgentState):
    urls = state.get("long_text_urls", [])
    urls = list(set(urls))
    
    if not urls:
        return {"logs": ["📚 [Reader] 没有发现需要深度阅读的长文，跳过。"]}
        
    logs = []
    deep_summaries = []
    total_cost = 0.0
    
    logs.append(f"📚 [Reader] 发现 {len(urls)} 篇长文，启动混合分级阅读器...")
    
    with get_openai_callback() as cb:
        for url in urls:
            log_msg = f"📖 [Reader] 正在精读: {url}"
            print(log_msg)
            logs.append(log_msg)
            
            try:
                result = reader.read_and_analyze(url, state["query"])
                deep_summaries.append(result["summary"])
                logs.extend(result["logs"])
                logs.append(f"✅ [Reader] 完成深度阅读: {len(result['summary'])} 字")
            except Exception as e:
                logs.append(f"❌ [Reader] 阅读失败 {url}: {e}")
        
        total_cost = cb.total_cost
        logs.append(f"💰 [Reader] LLM/Embed Cost: ${total_cost:.4f}")

    return {
        "research_data": deep_summaries,
        "logs": logs
    }

def write_node(state: AgentState):
    query = state["query"]
    data = state["research_data"]
    role = state.get("role_name", "Researcher")
    p_instr = state.get("persona_instruction", "")
    fmt_instr = state.get("output_format_instruction", "")
    
    log_msg = f"✍️ [Writer] 正在以 [{role}] 身份撰写报告..."
    print(log_msg)
    
    with get_openai_callback() as cb:
        full_context = "\n\n".join(data)
        if not full_context:
            full_context = "无外部资料。"
        
        report = write_report(query, full_context, role, p_instr, fmt_instr)
        cost_msg = f"💰 [Writer] Cost: ${cb.total_cost:.4f}"
        
        # Async save to cache
        if kb_manager.enabled:
            kb_manager.save_to_cache(query, report)
            
    # Extract keywords
    try:
        keywords = extract_keywords(report)
    except:
        keywords = []
    
    return {
        "final_report": report,
        "keywords": keywords,
        "logs": [log_msg, cost_msg, "✅ [Writer] 报告完成，已存入云端知识库"]
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
