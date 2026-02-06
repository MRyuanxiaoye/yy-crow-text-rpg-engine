"""
架构师 Agent (GPT-4o)
职责：
1. 逻辑校验 - 检查设定是否自洽
2. 冲突检测 - 发现与已有设定的矛盾
3. 结构分析 - 理解世界观的层级关系
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from app.llm_factory import get_architect_llm


class ConflictReport(BaseModel):
    """冲突检测报告"""
    has_conflict: bool = Field(description="是否存在冲突")
    conflicts: List[str] = Field(default_factory=list, description="冲突列表")
    severity: str = Field(default="none", description="严重程度: none/low/medium/high")
    suggestions: List[str] = Field(default_factory=list, description="修复建议")


class ValidationReport(BaseModel):
    """合理性校验报告"""
    is_valid: bool = Field(description="设定是否合理")
    reasoning: str = Field(description="推理过程")
    scientific_basis: Optional[str] = Field(default=None, description="科学依据（如有）")
    mythological_basis: Optional[str] = Field(default=None, description="神话/玄学依据（如有）")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")


class ArchitectAgent:
    """架构师 Agent - 使用 GPT-4o 进行逻辑分析"""
    
    def __init__(self):
        self.llm = get_architect_llm(temperature=0.2)
    
    def detect_conflicts(self, new_setting: str, existing_context: str) -> ConflictReport:
        """
        检测新设定与已有世界观的冲突
        
        Args:
            new_setting: 新的设定内容
            existing_context: 已有的世界观上下文
        """
        parser = JsonOutputParser(pydantic_object=ConflictReport)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个严谨的世界观架构师。你的任务是检测新设定与已有世界观之间的逻辑冲突。

**分析原则**：
1. 检查直接矛盾：新设定是否与已有规则直接冲突
2. 检查间接矛盾：新设定的推论是否会导致逻辑问题
3. 检查连锁反应：新设定会影响哪些已有设定

**严重程度判断**：
- none: 无冲突
- low: 小瑕疵，可以通过补充设定解决
- medium: 需要修改部分设定才能调和
- high: 根本性冲突，二者不可能同时成立

请用 JSON 格式输出分析结果。
{format_instructions}"""),
            ("user", """【已有世界观】
{existing_context}

【新设定】
{new_setting}

请分析新设定与已有世界观是否存在冲突。""")
        ])
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "existing_context": existing_context,
                "new_setting": new_setting,
                "format_instructions": parser.get_format_instructions()
            })
            return ConflictReport(**result)
        except Exception as e:
            print(f"❌ [Architect] Conflict detection failed: {e}")
            return ConflictReport(
                has_conflict=False,
                conflicts=[],
                severity="none",
                suggestions=["检测过程出错，请手动检查"]
            )
    
    def validate_setting(self, setting: str, validate_type: str = "science") -> ValidationReport:
        """
        校验设定的合理性
        
        Args:
            setting: 要校验的设定
            validate_type: 校验类型 - science(科学)/mythology(玄学)/logic(纯逻辑)
        """
        parser = JsonOutputParser(pydantic_object=ValidationReport)
        
        type_instructions = {
            "science": "请从科学角度（物理、化学、生物学等）分析这个设定是否合理。如果不完全合理，指出需要补充哪些前提条件。",
            "mythology": "请从神话、玄学、宗教传统的角度分析这个设定是否有根据。可以引用东西方神话体系。",
            "logic": "请纯粹从逻辑角度分析这个设定是否自洽，不考虑现实世界的科学规律。"
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个博学的世界观顾问，精通科学、神话、哲学。
            
{type_instruction}

**注意**：
- 小说世界观不需要完全符合现实，但需要内在逻辑自洽
- 如果设定需要某些前提条件才能成立，请明确指出
- 给出具体的改进建议，而不是笼统的评价

请用 JSON 格式输出。
{format_instructions}"""),
            ("user", """【待校验设定】
{setting}

请进行合理性分析。""")
        ])
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "setting": setting,
                "type_instruction": type_instructions.get(validate_type, type_instructions["logic"]),
                "format_instructions": parser.get_format_instructions()
            })
            return ValidationReport(**result)
        except Exception as e:
            print(f"❌ [Architect] Validation failed: {e}")
            return ValidationReport(
                is_valid=True,
                reasoning="校验过程出错",
                suggestions=["请手动检查设定合理性"]
            )
    
    def analyze_structure(self, world_context: str, user_query: str) -> str:
        """
        分析世界观结构，回答用户关于架构的问题
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个世界观架构分析师。基于给定的世界观信息，回答用户关于结构、关系、逻辑的问题。

**回答风格**：
- 简洁明了，直击要点
- 用结构化的方式呈现（列表、层级）
- 指出潜在的问题或可以优化的地方"""),
            ("user", """【世界观信息】
{world_context}

【用户问题】
{user_query}""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "world_context": world_context,
                "user_query": user_query
            })
            return result.content
        except Exception as e:
            print(f"❌ [Architect] Analysis failed: {e}")
            return f"分析出错: {e}"

