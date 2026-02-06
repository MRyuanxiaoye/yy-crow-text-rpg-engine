"""
世界观审计员 Agent (DeepSeek)
职责：
1. 查漏补缺 - 发现世界观设定中的空白和不完整之处
2. 自洽检查 - 发现设定之间的矛盾
3. 建议补充 - 以建议形式提出可能的新设定（不替用户决定）
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from app.llm_factory import get_creator_llm


class AuditResult(BaseModel):
    """审计结果"""
    missing_parts: List[str] = Field(default_factory=list, description="缺失的设定部分")
    inconsistencies: List[str] = Field(default_factory=list, description="发现的矛盾或不一致")
    suggestions: List[str] = Field(default_factory=list, description="建议补充的内容（仅供参考）")
    questions: List[str] = Field(default_factory=list, description="需要作者进一步思考的问题")


class SettingSuggestion(BaseModel):
    """设定建议"""
    name: str = Field(description="建议的设定名称")
    content: str = Field(description="建议的设定内容（供参考）")
    reason: str = Field(description="为什么建议添加这个设定")
    priority: str = Field(default="medium", description="优先级: high/medium/low")


class ExpansionResult(BaseModel):
    """扩展分析结果"""
    gaps: List[str] = Field(default_factory=list, description="当前设定的空白点")
    suggestions: List[SettingSuggestion] = Field(default_factory=list, description="建议补充的子设定")
    questions: List[str] = Field(default_factory=list, description="需要进一步明确的问题")


class CreatorAgent:
    """世界观审计员 Agent - 使用 DeepSeek 进行查漏补缺"""
    
    def __init__(self):
        self.llm = get_creator_llm(temperature=0.5)  # 降低温度，更理性
    
    def audit_setting(self, user_request: str, context: str = "") -> AuditResult:
        """
        审计世界观设定，发现缺漏和矛盾
        
        Args:
            user_request: 用户的请求/要审计的内容
            context: 已有的世界观上下文
        """
        parser = JsonOutputParser(pydantic_object=AuditResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个**世界观审计员**。你的任务不是替用户创作，而是帮助用户完善他们的世界观。

**你的职责**：
1. **查漏**：根据已有设定，指出缺失的部分
   - 例如：地图有了，但气候、资源分布没设定
   - 例如：魔法体系有了，但消耗、限制、代价没说明
   
2. **补缺建议**：对于发现的空白，以**建议形式**提出补充方案
   - 不要直接替用户做决定
   - 提供多个可能的方向供选择
   - 说明每个建议的理由
   
3. **自洽检查**：发现设定之间的矛盾或不合理之处
   - 逻辑矛盾
   - 时间线冲突
   - 因果关系不通

**输出风格**：
- 像一个编辑审稿一样，指出问题，提出建议
- 不要美化或润色用户的文字
- 所有建议都是"仅供参考"，最终决定权在用户

请用 JSON 格式输出。
{format_instructions}"""),
            ("user", """【已有世界观】
{context}

【用户请求】
{user_request}

请对以上内容进行审计，指出缺漏和矛盾，并提出建议。""")
        ])
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "context": context if context else "（暂无已有设定）",
                "user_request": user_request,
                "format_instructions": parser.get_format_instructions()
            })
            return AuditResult(**result)
        except Exception as e:
            print(f"❌ [Creator] Audit failed: {e}")
            return AuditResult(
                missing_parts=[],
                inconsistencies=[],
                suggestions=["审计过程出错，请重试"],
                questions=[]
            )
    
    def analyze_gaps(self, setting_name: str, setting_content: str, context: str = "") -> ExpansionResult:
        """
        分析设定的空白点，提出补充建议
        
        Args:
            setting_name: 设定名称
            setting_content: 设定当前内容
            context: 世界观上下文
        """
        parser = JsonOutputParser(pydantic_object=ExpansionResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个**世界观顾问**。针对给定的设定，分析其空白点并提出补充建议。

**分析维度**：
1. **完整性**：这个设定还缺少哪些必要的组成部分？
2. **深度**：哪些地方可以进一步细化？
3. **关联性**：这个设定与世界的其他部分如何互动？是否有遗漏的联系？
4. **可行性**：这个设定在逻辑上是否站得住脚？

**建议原则**：
- 每个建议都要说明**为什么**需要补充
- 提供**具体的方向**，而不是笼统的概念
- 标注优先级（high: 必须有 / medium: 建议有 / low: 可选）
- 不要替用户做决定，提供选项

请用 JSON 格式输出。
{format_instructions}"""),
            ("user", """【世界观背景】
{context}

【待分析设定】
名称：{setting_name}
当前内容：{setting_content}

请分析这个设定的空白点，并提出补充建议。""")
        ])
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "context": context if context else "（无额外背景）",
                "setting_name": setting_name,
                "setting_content": setting_content,
                "format_instructions": parser.get_format_instructions()
            })
            return ExpansionResult(**result)
        except Exception as e:
            print(f"❌ [Creator] Gap analysis failed: {e}")
            return ExpansionResult(
                gaps=["分析过程出错"],
                suggestions=[],
                questions=[]
            )
    
    def chat(self, user_message: str, context: str = "", history: List[dict] = None) -> str:
        """
        自由对话模式 - 讨论世界观相关的任何话题（支持多轮对话）
        
        Args:
            user_message: 用户消息
            context: 世界观上下文
            history: 历史对话 [{"role": "user/assistant", "content": "..."}]
        """
        if history is None:
            history = []
        
        # 系统消息 - 纯粹的自由对话，不做审计
        system_message = f"""你是一个**世界观创作伙伴**。你和作者一起讨论、构思世界观设定。

**对话风格**：
- 像朋友一样轻松交流，一起头脑风暴
- 顺着作者的思路聊，帮助他们展开想法
- 可以提供灵感和创意，但尊重作者的决定
- 回答问题时简洁有帮助

**你可以做的**：
- 回答作者关于世界观构建的问题
- 一起讨论设定的可能性
- 提供创意灵感和参考
- 帮助作者理清思路

**你不应该做的**：
- 不要主动审计、指出缺漏（那是"查漏"功能的事）
- 不要主动校验合理性（那是"校验"功能的事）
- 不要替作者做决定

**【重要】可沉淀内容格式**：
当对话中产生了**具体的、可以直接作为设定的内容**时，请用 📌 开头标记，方便用户一键添加到节点。

例如：
- 📌 魔法消耗：施法者每次施法消耗固定魔力值
- 📌 魔力恢复：通过休息自然恢复，8小时完全恢复

只有**用户确认的具体设定**才用 📌 标记。讨论性、探索性的内容不要标记。

【当前世界观背景】
{context if context else "（这是一个全新的世界，尚无任何设定）"}"""

        # 构建多轮对话消息列表
        messages = [("system", system_message)]
        
        # 添加历史对话
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(("user", content))
            elif role == "assistant":
                messages.append(("assistant", content))
        
        # 添加当前用户消息
        messages.append(("user", user_message))
        
        # 使用 ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({})
            return result
        except Exception as e:
            print(f"❌ [Creator] Chat failed: {e}")
            return f"对话出错: {e}"
