"""
World Weaver - LangGraph 流程定义
多模型协作：
1. Creator (DeepSeek) - 世界观审计（查漏补缺）
2. Architect (GPT-4o) - 逻辑校验
3. Character (DeepSeek) - 角色顾问
"""

from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from app.agents.creator import CreatorAgent
from app.agents.architect import ArchitectAgent
from app.agents.character import CharacterAgent
from app.models import WorldGraph, WorldNode, ChatResponse


class WeaverState(TypedDict):
    """工作流状态"""
    # 输入
    user_message: str
    current_node_id: Optional[str]
    mode: str  # "audit" | "validate" | "chat" | "character"
    world_graph: Optional[WorldGraph]
    history: List[dict]  # 历史对话 [{"role": "user/assistant", "content": "..."}]
    
    # 额外上下文（用于角色模式）
    events_context: str
    other_characters: str
    
    # 中间状态
    context: str
    generated_content: str
    
    # 输出
    response: str
    conflicts: List[str]
    suggestions: List[str]
    new_nodes: List[dict]


# 初始化 Agents
creator = CreatorAgent()
architect = ArchitectAgent()
character = CharacterAgent()


def build_context_node(state: WeaverState) -> WeaverState:
    """
    Step 1: 构建上下文
    从世界图谱中提取当前节点相关的所有上下文
    """
    print("🔧 [Graph] Building context...", flush=True)
    
    world_graph = state.get("world_graph")
    node_id = state.get("current_node_id")
    
    if world_graph and node_id:
        context = world_graph.build_context(node_id)
    elif world_graph:
        context = f"【世界】{world_graph.name}\n{world_graph.description}"
    else:
        context = "（这是一个全新的世界，尚无任何设定）"
    
    print(f"🔧 [Graph] Context built: {len(context)} chars", flush=True)
    
    return {**state, "context": context}


def audit_node(state: WeaverState) -> WeaverState:
    """
    审计模式：使用 Creator (DeepSeek) 进行查漏补缺
    """
    print("🔍 [Creator] Auditing world setting...", flush=True)
    
    result = creator.audit_setting(
        user_request=state["user_message"],
        context=state["context"]
    )
    
    response_parts = []
    
    # 缺失部分
    if result.missing_parts:
        response_parts.append("📋 **发现的缺漏**：")
        for part in result.missing_parts:
            response_parts.append(f"  • {part}")
        response_parts.append("")
    
    # 矛盾
    if result.inconsistencies:
        response_parts.append("⚠️ **发现的矛盾**：")
        for inc in result.inconsistencies:
            response_parts.append(f"  • {inc}")
        response_parts.append("")
    
    # 建议
    if result.suggestions:
        response_parts.append("💡 **补充建议**（仅供参考）：")
        for sug in result.suggestions:
            response_parts.append(f"  • {sug}")
        response_parts.append("")
    
    # 问题
    if result.questions:
        response_parts.append("❓ **需要思考的问题**：")
        for q in result.questions:
            response_parts.append(f"  • {q}")
    
    if not response_parts:
        response_parts.append("✅ 当前设定看起来比较完整，暂未发现明显缺漏。")
    
    print(f"🔍 [Creator] Audit complete: {len(result.missing_parts)} gaps found", flush=True)
    
    return {
        **state,
        "generated_content": "\n".join(response_parts),
        "suggestions": result.suggestions,
        "new_nodes": []
    }


def chat_node(state: WeaverState) -> WeaverState:
    """
    对话模式：自由对话（带历史记忆）
    """
    history = state.get("history", [])
    print(f"💬 [Creator] Chatting with {len(history)} history messages...", flush=True)
    
    response = creator.chat(
        user_message=state["user_message"],
        context=state["context"],
        history=history
    )
    
    return {
        **state,
        "generated_content": response,
        "new_nodes": []
    }


def validate_node(state: WeaverState) -> WeaverState:
    """
    校验模式：使用 Architect (GPT-4o) 校验设定合理性
    """
    print("🔬 [Architect] Validating setting...", flush=True)
    
    result = architect.validate_setting(
        setting=state["user_message"],
        validate_type="science"
    )
    
    response_parts = []
    
    if result.is_valid:
        response_parts.append("✅ **设定合理**")
    else:
        response_parts.append("⚠️ **设定需要调整**")
    
    response_parts.append(f"\n**分析**：{result.reasoning}")
    
    if result.scientific_basis:
        response_parts.append(f"\n**科学依据**：{result.scientific_basis}")
    
    if result.mythological_basis:
        response_parts.append(f"\n**玄学依据**：{result.mythological_basis}")
    
    print(f"🔬 [Architect] Validation complete", flush=True)
    
    return {
        **state,
        "generated_content": "\n".join(response_parts),
        "suggestions": result.suggestions,
        "new_nodes": []
    }


def character_node(state: WeaverState) -> WeaverState:
    """
    角色模式：使用 Character Agent 进行角色审计
    """
    print("👤 [Character] Auditing character...", flush=True)
    
    # 获取额外上下文
    events_context = state.get("events_context", "")
    other_characters = state.get("other_characters", "")
    
    result = character.audit_character(
        character_info=state["user_message"],
        world_context=state["context"],
        events_context=events_context
    )
    
    response_parts = []
    
    # 评分
    response_parts.append(f"📊 **角色评估**")
    response_parts.append(f"  • 设定完整度: {result.completeness_score}/100")
    response_parts.append(f"  • 事件关联度: {result.event_relevance_score}/100")
    response_parts.append("")
    
    # 空白
    if result.gaps:
        response_parts.append("📋 **设定空白**：")
        for gap in result.gaps:
            response_parts.append(f"  • {gap}")
        response_parts.append("")
    
    # 事件推动分析
    event_analysis = result.event_analysis
    
    if event_analysis.direct_impacts:
        response_parts.append("🎯 **直接推动的事件**：")
        for impact in event_analysis.direct_impacts:
            response_parts.append(f"  • {impact}")
        response_parts.append("")
    
    if event_analysis.indirect_impacts:
        response_parts.append("🔗 **间接影响**（通过影响其他角色）：")
        for impact in event_analysis.indirect_impacts:
            response_parts.append(f"  • {impact}")
        response_parts.append("")
    
    if event_analysis.missing_connections:
        response_parts.append("⚠️ **缺失的事件关联**：")
        for missing in event_analysis.missing_connections:
            response_parts.append(f"  • {missing}")
        response_parts.append("")
    
    if event_analysis.redundancy_warnings:
        response_parts.append("🔴 **功能冗余警告**：")
        for warning in event_analysis.redundancy_warnings:
            response_parts.append(f"  • {warning}")
        response_parts.append("")
    
    # 建议
    if result.suggestions:
        response_parts.append("💡 **改进建议**：")
        for sug in result.suggestions:
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sug.priority, "⚪")
            response_parts.append(f"  {priority_icon} [{sug.category}] {sug.content}")
            response_parts.append(f"     _理由: {sug.reason}_")
        response_parts.append("")
    
    if event_analysis.enhancement_suggestions:
        response_parts.append("🚀 **增强角色作用的建议**：")
        for sug in event_analysis.enhancement_suggestions:
            response_parts.append(f"  • {sug}")
    
    print(f"👤 [Character] Audit complete: completeness={result.completeness_score}, relevance={result.event_relevance_score}", flush=True)
    
    return {
        **state,
        "generated_content": "\n".join(response_parts),
        "suggestions": [s.content for s in result.suggestions],
        "new_nodes": []
    }


def conflict_check_node(state: WeaverState) -> WeaverState:
    """
    冲突检测：使用 Architect (GPT-4o) 检测逻辑冲突
    """
    print("⚔️ [Architect] Checking conflicts...", flush=True)
    
    if state["mode"] != "audit":
        return {**state, "conflicts": []}
    
    if not state["context"] or "全新的世界" in state["context"]:
        return {**state, "conflicts": []}
    
    result = architect.detect_conflicts(
        new_setting=state["generated_content"],
        existing_context=state["context"]
    )
    
    conflicts = result.conflicts if result.has_conflict else []
    suggestions = state.get("suggestions", []) + result.suggestions
    
    print(f"⚔️ [Architect] Found {len(conflicts)} conflicts", flush=True)
    
    return {
        **state,
        "conflicts": conflicts,
        "suggestions": suggestions
    }


def format_response_node(state: WeaverState) -> WeaverState:
    """
    格式化最终响应
    """
    print("📝 [Graph] Formatting response...", flush=True)
    
    response_parts = [state["generated_content"]]
    
    if state.get("conflicts"):
        response_parts.append("\n\n---\n⚠️ **检测到以下冲突**：")
        for conflict in state["conflicts"]:
            response_parts.append(f"- {conflict}")
    
    return {
        **state,
        "response": "\n".join(response_parts)
    }


def route_by_mode(state: WeaverState) -> str:
    """根据模式路由到不同的处理节点"""
    mode = state.get("mode", "chat")
    print(f"🔀 [Graph] Routing to mode: {mode}", flush=True)
    
    mode_map = {
        "audit": "audit",
        "validate": "validate",
        "character": "character",
        "chat": "chat"
    }
    return mode_map.get(mode, "chat")


def build_weaver_graph():
    """构建 LangGraph 工作流"""
    
    workflow = StateGraph(WeaverState)
    
    # 添加节点
    workflow.add_node("build_context", build_context_node)
    workflow.add_node("audit", audit_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("character", character_node)
    workflow.add_node("conflict_check", conflict_check_node)
    workflow.add_node("format_response", format_response_node)
    
    # 设置入口
    workflow.set_entry_point("build_context")
    
    # 上下文构建后，根据模式路由
    workflow.add_conditional_edges(
        "build_context",
        route_by_mode,
        {
            "audit": "audit",
            "chat": "chat",
            "validate": "validate",
            "character": "character"
        }
    )
    
    # 审计后进行冲突检测
    workflow.add_edge("audit", "conflict_check")
    
    # 对话/校验/角色直接格式化响应
    workflow.add_edge("chat", "format_response")
    workflow.add_edge("validate", "format_response")
    workflow.add_edge("character", "format_response")
    
    # 冲突检测后格式化响应
    workflow.add_edge("conflict_check", "format_response")
    
    # 格式化后结束
    workflow.add_edge("format_response", END)
    
    return workflow.compile()


# 编译图
weaver_graph = build_weaver_graph()


def run_weaver(
    user_message: str,
    mode: str = "chat",
    world_graph: Optional[WorldGraph] = None,
    current_node_id: Optional[str] = None,
    events_context: str = "",
    other_characters: str = "",
    history: List[dict] = None
) -> ChatResponse:
    """
    运行 World Weaver 工作流
    
    Args:
        user_message: 用户消息
        mode: 模式 - "audit"/"expand"/"validate"/"chat"/"character"
        world_graph: 世界图谱（可选）
        current_node_id: 当前选中的节点 ID（可选）
        events_context: 事件上下文（角色模式用）
        other_characters: 其他角色信息（角色模式用）
        history: 历史对话（最近几条）
    
    Returns:
        ChatResponse
    """
    if history is None:
        history = []
    
    print(f"\n{'='*50}", flush=True)
    print(f"🌍 [World Weaver] Starting workflow", flush=True)
    print(f"   Mode: {mode}", flush=True)
    print(f"   Message: {user_message[:50]}...", flush=True)
    print(f"   History: {len(history)} messages", flush=True)
    print(f"{'='*50}\n", flush=True)
    
    initial_state: WeaverState = {
        "user_message": user_message,
        "current_node_id": current_node_id,
        "mode": mode,
        "world_graph": world_graph,
        "history": history,
        "events_context": events_context,
        "other_characters": other_characters,
        "context": "",
        "generated_content": "",
        "response": "",
        "conflicts": [],
        "suggestions": [],
        "new_nodes": []
    }
    
    result = weaver_graph.invoke(initial_state)
    
    print(f"\n{'='*50}", flush=True)
    print(f"✅ [World Weaver] Workflow complete", flush=True)
    print(f"{'='*50}\n", flush=True)
    
    return ChatResponse(
        response=result["response"],
        suggestions=result.get("suggestions", []),
        conflicts=result.get("conflicts", []),
        new_nodes=[WorldNode(**n) for n in result.get("new_nodes", []) if n.get("name")]
    )
