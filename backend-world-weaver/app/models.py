"""
World Weaver 数据模型
定义世界观节点、关系等核心数据结构
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class NodeType(str, Enum):
    """节点类型"""
    WORLD = "world"           # 世界（顶级）
    REGION = "region"         # 地区/国家
    RACE = "race"             # 种族
    CHARACTER = "character"   # 角色
    MAGIC_SYSTEM = "magic"    # 魔法体系
    TECHNOLOGY = "tech"       # 科技体系
    RELIGION = "religion"     # 宗教/信仰
    HISTORY = "history"       # 历史事件
    ORGANIZATION = "org"      # 组织/势力
    RULE = "rule"             # 规则/法则
    ITEM = "item"             # 物品/道具
    CREATURE = "creature"     # 生物
    CUSTOM = "custom"         # 自定义


class RelationType(str, Enum):
    """关系类型"""
    CONTAINS = "contains"         # 包含
    BELONGS_TO = "belongs_to"     # 属于
    DEPENDS_ON = "depends_on"     # 依赖
    CONFLICTS_WITH = "conflicts"  # 冲突
    RESTRICTS = "restricts"       # 限制
    ENABLES = "enables"           # 使能
    RELATED_TO = "related"        # 相关


class WorldNode(BaseModel):
    """世界观节点"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="节点名称")
    node_type: NodeType = Field(default=NodeType.CUSTOM, description="节点类型")
    content: str = Field(default="", description="详细描述")
    rules: List[str] = Field(default_factory=list, description="该节点的规则/约束")
    tags: List[str] = Field(default_factory=list, description="标签")
    parent_id: Optional[str] = Field(default=None, description="父节点 ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # 位置信息（用于前端 XMind 布局）
    position_x: float = Field(default=0.0)
    position_y: float = Field(default=0.0)


class NodeRelation(BaseModel):
    """节点之间的关系"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = Field(..., description="源节点 ID")
    target_id: str = Field(..., description="目标节点 ID")
    relation_type: RelationType = Field(default=RelationType.RELATED_TO)
    description: str = Field(default="", description="关系描述")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="关系强度")


class WorldGraph(BaseModel):
    """完整的世界观图谱"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="世界名称")
    description: str = Field(default="", description="世界简介")
    nodes: Dict[str, WorldNode] = Field(default_factory=dict)
    relations: List[NodeRelation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_node(self, node: WorldNode) -> WorldNode:
        """添加节点"""
        self.nodes[node.id] = node
        self.updated_at = datetime.now()
        return node
    
    def get_node(self, node_id: str) -> Optional[WorldNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[WorldNode]:
        """获取子节点"""
        return [n for n in self.nodes.values() if n.parent_id == node_id]
    
    def get_ancestors(self, node_id: str) -> List[WorldNode]:
        """获取所有祖先节点（用于上下文构建）"""
        ancestors = []
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            parent = self.nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors
    
    def get_related_nodes(self, node_id: str) -> List[WorldNode]:
        """获取所有关联节点"""
        related_ids = set()
        for rel in self.relations:
            if rel.source_id == node_id:
                related_ids.add(rel.target_id)
            elif rel.target_id == node_id:
                related_ids.add(rel.source_id)
        return [self.nodes[nid] for nid in related_ids if nid in self.nodes]
    
    def build_context(self, node_id: str) -> str:
        """为某个节点构建完整上下文（用于 AI 对话）"""
        context_parts = []
        
        # 1. 世界基础信息
        context_parts.append(f"【世界】{self.name}\n{self.description}")
        
        # 2. 祖先链（从顶层到当前节点）
        ancestors = self.get_ancestors(node_id)
        if ancestors:
            context_parts.append("\n【上下文层级】")
            for i, ancestor in enumerate(reversed(ancestors)):
                indent = "  " * i
                context_parts.append(f"{indent}└─ {ancestor.name} ({ancestor.node_type.value})")
                if ancestor.rules:
                    context_parts.append(f"{indent}   规则: {', '.join(ancestor.rules)}")
        
        # 3. 当前节点详情
        current = self.nodes.get(node_id)
        if current:
            context_parts.append(f"\n【当前节点】{current.name} ({current.node_type.value})")
            context_parts.append(f"描述: {current.content}")
            if current.rules:
                context_parts.append(f"规则: {', '.join(current.rules)}")
        
        # 4. 关联节点
        related = self.get_related_nodes(node_id)
        if related:
            context_parts.append("\n【关联设定】")
            for rel_node in related[:5]:  # 最多 5 个关联
                context_parts.append(f"- {rel_node.name}: {rel_node.content[:100]}...")
        
        return "\n".join(context_parts)


# ========== API 请求/响应模型 ==========

class HistoryMessage(BaseModel):
    """历史对话消息"""
    role: str = Field(..., description="角色: user/assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., description="用户消息")
    node_id: Optional[str] = Field(default=None, description="当前选中的节点 ID")
    mode: str = Field(default="chat", description="模式: chat/audit/expand/validate/character")
    history: List[HistoryMessage] = Field(default_factory=list, description="最近的对话历史")


class ChatResponse(BaseModel):
    """对话响应"""
    response: str = Field(..., description="AI 回复")
    suggestions: List[str] = Field(default_factory=list, description="建议的后续操作")
    conflicts: List[str] = Field(default_factory=list, description="检测到的冲突")
    new_nodes: List[WorldNode] = Field(default_factory=list, description="建议创建的新节点")


class CreateNodeRequest(BaseModel):
    """创建节点请求"""
    name: str
    node_type: NodeType = NodeType.CUSTOM
    content: str = ""
    rules: List[str] = []
    tags: List[str] = []
    parent_id: Optional[str] = None
    position_x: float = 0.0
    position_y: float = 0.0


class ValidateRequest(BaseModel):
    """校验请求"""
    setting: str = Field(..., description="要校验的设定")
    node_id: Optional[str] = Field(default=None, description="相关节点 ID")
    validate_type: str = Field(default="science", description="校验类型: science/logic/internal")

