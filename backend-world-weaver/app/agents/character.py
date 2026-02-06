"""
角色顾问 Agent (DeepSeek)
职责：
1. 人物弧光 - 帮助完善角色的成长轨迹
2. 性格设定 - 分析和建议角色的性格特质
3. 事件推动 - 检查角色对故事事件的推动作用
4. 关系网络 - 分析角色与其他角色的关系
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from app.llm_factory import get_creator_llm


class CharacterArc(BaseModel):
    """角色弧光分析"""
    current_state: str = Field(description="角色当前状态/设定概述")
    missing_elements: List[str] = Field(default_factory=list, description="缺失的设定元素")
    arc_suggestions: List[str] = Field(default_factory=list, description="弧光发展建议")
    personality_gaps: List[str] = Field(default_factory=list, description="性格设定的空白点")
    questions: List[str] = Field(default_factory=list, description="需要作者思考的问题")


class EventImpactAnalysis(BaseModel):
    """角色对事件的推动作用分析"""
    direct_impacts: List[str] = Field(default_factory=list, description="直接推动的事件")
    indirect_impacts: List[str] = Field(default_factory=list, description="间接影响（通过影响其他角色）")
    missing_connections: List[str] = Field(default_factory=list, description="缺失的事件关联")
    redundancy_warnings: List[str] = Field(default_factory=list, description="角色功能冗余警告")
    enhancement_suggestions: List[str] = Field(default_factory=list, description="增强角色作用的建议")


class CharacterSuggestion(BaseModel):
    """角色设定建议"""
    category: str = Field(description="建议类别: 生平/性格/动机/关系/弧光/事件推动")
    content: str = Field(description="建议内容")
    reason: str = Field(description="建议理由")
    priority: str = Field(default="medium", description="优先级: high/medium/low")


class CharacterAuditResult(BaseModel):
    """角色审计结果"""
    completeness_score: int = Field(default=0, description="完整度评分 0-100")
    event_relevance_score: int = Field(default=0, description="事件关联度评分 0-100")
    gaps: List[str] = Field(default_factory=list, description="设定空白")
    suggestions: List[CharacterSuggestion] = Field(default_factory=list, description="改进建议")
    event_analysis: EventImpactAnalysis = Field(default_factory=EventImpactAnalysis, description="事件推动分析")


class CharacterAgent:
    """角色顾问 Agent - 使用 DeepSeek 进行角色设定分析"""
    
    def __init__(self):
        self.llm = get_creator_llm(temperature=0.5)
    
    def audit_character(self, character_info: str, world_context: str = "", events_context: str = "") -> CharacterAuditResult:
        """
        全面审计角色设定
        
        Args:
            character_info: 角色当前设定
            world_context: 世界观上下文
            events_context: 相关事件/剧情上下文
        """
        parser = JsonOutputParser(pydantic_object=CharacterAuditResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个**角色顾问**。你帮助作者完善角色设定，确保角色在故事中发挥应有的作用。

**审计维度**：

1. **完整性检查**（设定是否齐全）：
   - 生平：出生、成长背景、关键经历、当前状态
   - 性格：核心特质、优点、缺点、怪癖、禁忌
   - 动机：驱动力、恐惧、欲望、目标
   - 关系：与其他角色的关系、立场、羁绊
   - 弧光：角色如何成长/变化、转折点

2. **事件推动作用检查**（角色存在的意义）：
   - **直接推动**：这个角色直接导致了哪些事件发生？
   - **间接推动**：这个角色通过影响其他角色（如主角、配角），间接导致了哪些事件？
   - **功能检查**：这个角色的功能是否可以被其他角色替代？如果可以，是否存在冗余？
   - **缺失关联**：这个角色是否应该与某些事件有关联，但目前没有？

3. **评分标准**：
   - completeness_score: 设定完整度 (0-100)
   - event_relevance_score: 事件关联度 (0-100)，越高说明角色越重要

**输出原则**：
- 指出问题，提出建议，但不替作者做决定
- 所有建议都要说明理由
- 特别关注"这个角色为什么存在"这个核心问题

请用 JSON 格式输出。
{format_instructions}"""),
            ("user", """【世界观背景】
{world_context}

【相关事件/剧情】
{events_context}

【角色设定】
{character_info}

请对这个角色进行全面审计。""")
        ])
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "world_context": world_context if world_context else "（无世界观背景）",
                "events_context": events_context if events_context else "（无事件/剧情信息）",
                "character_info": character_info,
                "format_instructions": parser.get_format_instructions()
            })
            return CharacterAuditResult(**result)
        except Exception as e:
            print(f"❌ [Character] Audit failed: {e}")
            return CharacterAuditResult(
                completeness_score=0,
                event_relevance_score=0,
                gaps=["审计过程出错，请重试"],
                suggestions=[],
                event_analysis=EventImpactAnalysis()
            )
    
    def analyze_arc(self, character_info: str, story_timeline: str = "") -> CharacterArc:
        """
        分析角色弧光（成长轨迹）
        
        Args:
            character_info: 角色设定
            story_timeline: 故事时间线（可选）
        """
        parser = JsonOutputParser(pydantic_object=CharacterArc)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个**角色弧光分析师**。你帮助作者规划角色的成长轨迹。

**角色弧光要素**：
1. **起点**：角色开始时是什么样的人？
2. **触发**：什么事件/经历触发了角色的变化？
3. **挣扎**：角色在变化过程中经历了什么内心冲突？
4. **转折**：关键的转折点是什么？
5. **终点**：角色最终变成了什么样的人？

**分析角度**：
- 性格弧光：性格如何变化？
- 能力弧光：能力如何成长？
- 关系弧光：与他人的关系如何演变？
- 价值观弧光：信念/价值观如何转变？

**输出原则**：
- 指出当前设定中弧光的空白
- 提出可能的弧光方向（作为建议，不做决定）
- 提出帮助作者思考的问题

请用 JSON 格式输出。
{format_instructions}"""),
            ("user", """【角色设定】
{character_info}

【故事时间线】
{story_timeline}

请分析这个角色的弧光设定。""")
        ])
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "character_info": character_info,
                "story_timeline": story_timeline if story_timeline else "（无时间线信息）",
                "format_instructions": parser.get_format_instructions()
            })
            return CharacterArc(**result)
        except Exception as e:
            print(f"❌ [Character] Arc analysis failed: {e}")
            return CharacterArc(
                current_state="分析出错",
                missing_elements=[],
                arc_suggestions=[],
                personality_gaps=[],
                questions=[]
            )
    
    def check_event_impact(self, character_info: str, events: str, other_characters: str = "") -> EventImpactAnalysis:
        """
        检查角色对事件的推动作用
        
        Args:
            character_info: 角色设定
            events: 故事中的事件列表
            other_characters: 其他相关角色（用于分析间接影响）
        """
        parser = JsonOutputParser(pydantic_object=EventImpactAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个**故事结构分析师**。你负责检查角色在故事中的推动作用。

**核心问题**：这个角色为什么存在？他/她推动了什么？

**分析维度**：

1. **直接推动**：
   - 这个角色的行动直接导致了哪些事件？
   - 没有这个角色，哪些事件就不会发生？

2. **间接推动**：
   - 这个角色影响了哪些其他角色？
   - 通过影响其他角色，间接导致了什么事件？
   - 例如：A 角色激励了主角，主角才决定去做 X → A 间接推动了 X

3. **功能冗余检查**：
   - 这个角色的功能是否可以被其他角色替代？
   - 如果删除这个角色，故事是否还能进行？
   - 如果存在冗余，要么删除角色，要么增加独特功能

4. **缺失关联**：
   - 根据角色的设定，他/她是否应该参与某些事件，但目前没有？
   - 是否有未被利用的潜力？

**输出原则**：
- 客观分析，不做主观好坏判断
- 如果角色作用不明显，直接指出
- 提供增强角色作用的具体建议

请用 JSON 格式输出。
{format_instructions}"""),
            ("user", """【角色设定】
{character_info}

【故事事件】
{events}

【其他相关角色】
{other_characters}

请分析这个角色对事件的推动作用。""")
        ])
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "character_info": character_info,
                "events": events,
                "other_characters": other_characters if other_characters else "（无其他角色信息）",
                "format_instructions": parser.get_format_instructions()
            })
            return EventImpactAnalysis(**result)
        except Exception as e:
            print(f"❌ [Character] Event impact analysis failed: {e}")
            return EventImpactAnalysis(
                direct_impacts=[],
                indirect_impacts=[],
                missing_connections=["分析出错"],
                redundancy_warnings=[],
                enhancement_suggestions=[]
            )
    
    def chat(self, user_message: str, character_context: str = "", world_context: str = "") -> str:
        """
        角色设定自由对话
        
        Args:
            user_message: 用户消息
            character_context: 当前角色上下文
            world_context: 世界观上下文
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个**角色顾问**。你帮助作者思考和完善角色设定。

**你的职责**：
1. 帮助作者发现角色设定的空白
2. 提出帮助思考的问题
3. 建议可能的设定方向（但不替作者决定）
4. 检查角色在故事中的必要性和作用

**关键问题意识**：
- 这个角色为什么存在？
- 没有这个角色，故事会怎样？
- 这个角色推动了什么事件（直接或间接）？

**对话风格**：
- 像一个有经验的编辑
- 提出尖锐但有建设性的问题
- 不做主观好坏判断
- 所有建议都说明理由

【当前角色】
{character_context}

【世界观背景】
{world_context}"""),
            ("user", "{user_message}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "character_context": character_context if character_context else "（尚无角色信息）",
                "world_context": world_context if world_context else "（无世界观背景）",
                "user_message": user_message
            })
            return result
        except Exception as e:
            print(f"❌ [Character] Chat failed: {e}")
            return f"对话出错: {e}"

