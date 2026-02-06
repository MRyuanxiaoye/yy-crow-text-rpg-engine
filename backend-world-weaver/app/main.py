"""
World Weaver API
小说世界观构建工具的后端服务
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel

# Load env from parent directory or local
load_dotenv("../backend-deep-search/.env")
load_dotenv()

from app.models import (
    WorldGraph, WorldNode, NodeRelation, NodeType, RelationType,
    ChatRequest, ChatResponse, CreateNodeRequest, ValidateRequest
)
from app.graph import run_weaver

app = FastAPI(
    title="World Weaver API",
    description="小说世界观构建工具 - 双模型协作 (GPT-4o + DeepSeek)",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 内存存储（后续可替换为数据库） ==========
# 存储当前世界图谱
current_world: Optional[WorldGraph] = None


# ========== 基础端点 ==========

@app.get("/")
def read_root():
    return {
        "message": "World Weaver API is running",
        "version": "1.0.0",
        "models": {
            "architect": "GPT-4o (逻辑校验)",
            "creator": "DeepSeek (创意生成)"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY"))
        },
        "world_loaded": current_world is not None
    }


# ========== 世界管理 ==========

class CreateWorldRequest(BaseModel):
    name: str
    description: str = ""


@app.post("/world/create")
def create_world(request: CreateWorldRequest):
    """创建新世界"""
    global current_world
    current_world = WorldGraph(
        name=request.name,
        description=request.description
    )
    return {
        "message": f"世界 '{request.name}' 创建成功",
        "world_id": current_world.id
    }


@app.get("/world")
def get_world():
    """获取当前世界信息"""
    if not current_world:
        raise HTTPException(status_code=404, detail="尚未创建世界")
    return {
        "id": current_world.id,
        "name": current_world.name,
        "description": current_world.description,
        "node_count": len(current_world.nodes),
        "relation_count": len(current_world.relations)
    }


@app.get("/world/export")
def export_world():
    """导出完整世界数据（用于保存）"""
    if not current_world:
        raise HTTPException(status_code=404, detail="尚未创建世界")
    return current_world.model_dump()


class ImportWorldRequest(BaseModel):
    data: dict


@app.post("/world/import")
def import_world(request: ImportWorldRequest):
    """导入世界数据"""
    global current_world
    try:
        current_world = WorldGraph(**request.data)
        return {"message": f"世界 '{current_world.name}' 导入成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"导入失败: {e}")


# ========== 节点管理 ==========

@app.get("/nodes")
def get_nodes():
    """获取所有节点"""
    if not current_world:
        return {"nodes": []}
    return {
        "nodes": [node.model_dump() for node in current_world.nodes.values()]
    }


@app.get("/nodes/{node_id}")
def get_node(node_id: str):
    """获取单个节点详情"""
    if not current_world:
        raise HTTPException(status_code=404, detail="尚未创建世界")
    node = current_world.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="节点不存在")
    return node.model_dump()


@app.post("/nodes")
def create_node(request: CreateNodeRequest):
    """创建节点"""
    global current_world
    if not current_world:
        # 自动创建默认世界
        current_world = WorldGraph(name="未命名世界", description="")
    
    node = WorldNode(
        name=request.name,
        node_type=request.node_type,
        content=request.content,
        rules=request.rules,
        tags=request.tags,
        parent_id=request.parent_id,
        position_x=request.position_x,
        position_y=request.position_y
    )
    
    current_world.add_node(node)
    
    return {
        "message": f"节点 '{node.name}' 创建成功",
        "node": node.model_dump()
    }


@app.put("/nodes/{node_id}")
def update_node(node_id: str, request: CreateNodeRequest):
    """更新节点"""
    if not current_world:
        raise HTTPException(status_code=404, detail="尚未创建世界")
    
    node = current_world.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="节点不存在")
    
    # 更新字段
    node.name = request.name
    node.node_type = request.node_type
    node.content = request.content
    node.rules = request.rules
    node.tags = request.tags
    node.parent_id = request.parent_id
    node.position_x = request.position_x
    node.position_y = request.position_y
    
    return {
        "message": f"节点 '{node.name}' 更新成功",
        "node": node.model_dump()
    }


@app.delete("/nodes/{node_id}")
def delete_node(node_id: str):
    """删除节点"""
    if not current_world:
        raise HTTPException(status_code=404, detail="尚未创建世界")
    
    if node_id not in current_world.nodes:
        raise HTTPException(status_code=404, detail="节点不存在")
    
    node_name = current_world.nodes[node_id].name
    del current_world.nodes[node_id]
    
    # 同时删除相关关系
    current_world.relations = [
        r for r in current_world.relations 
        if r.source_id != node_id and r.target_id != node_id
    ]
    
    return {"message": f"节点 '{node_name}' 已删除"}


# ========== 关系管理 ==========

class CreateRelationRequest(BaseModel):
    source_id: str
    target_id: str
    relation_type: RelationType = RelationType.RELATED_TO
    description: str = ""


@app.get("/relations")
def get_relations():
    """获取所有关系"""
    if not current_world:
        return {"relations": []}
    return {
        "relations": [rel.model_dump() for rel in current_world.relations]
    }


@app.post("/relations")
def create_relation(request: CreateRelationRequest):
    """创建关系"""
    if not current_world:
        raise HTTPException(status_code=404, detail="尚未创建世界")
    
    # 验证节点存在
    if request.source_id not in current_world.nodes:
        raise HTTPException(status_code=404, detail="源节点不存在")
    if request.target_id not in current_world.nodes:
        raise HTTPException(status_code=404, detail="目标节点不存在")
    
    relation = NodeRelation(
        source_id=request.source_id,
        target_id=request.target_id,
        relation_type=request.relation_type,
        description=request.description
    )
    
    current_world.relations.append(relation)
    
    return {
        "message": "关系创建成功",
        "relation": relation.model_dump()
    }


# ========== AI 对话 ==========

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    与 AI 对话
    
    mode 选项：
    - "chat": 自由对话
    - "audit": 查漏补缺
    - "expand": 扩展分析
    - "validate": 校验设定合理性
    - "character": 角色分析
    """
    # 转换历史消息格式
    history = [{"role": m.role, "content": m.content} for m in request.history]
    
    try:
        response = run_weaver(
            user_message=request.message,
            mode=request.mode,
            world_graph=current_world,
            current_node_id=request.node_id,
            history=history
        )
        return response
    except Exception as e:
        print(f"❌ [API] Chat error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"对话出错: {e}")


@app.post("/validate")
def validate_setting(request: ValidateRequest):
    """
    校验设定合理性
    
    validate_type 选项：
    - "science": 科学角度
    - "mythology": 神话/玄学角度
    - "logic": 纯逻辑角度
    """
    try:
        response = run_weaver(
            user_message=request.setting,
            mode="validate",
            world_graph=current_world,
            current_node_id=request.node_id
        )
        return {
            "setting": request.setting,
            "validation": response.response,
            "suggestions": response.suggestions
        }
    except Exception as e:
        print(f"❌ [API] Validation error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"校验出错: {e}")


# ========== 上下文查询 ==========

@app.get("/context/{node_id}")
def get_context(node_id: str):
    """获取指定节点的完整上下文"""
    if not current_world:
        raise HTTPException(status_code=404, detail="尚未创建世界")
    
    context = current_world.build_context(node_id)
    return {"context": context}
