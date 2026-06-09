"""游戏主控路由器。"""

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from src.engine.action import (
    ActionProposal,
    calculate_success_rate,
    create_action_proposal,
    create_passive_response,
    execute_action,
    resolve_partial_success,
)
from src.engine.dice import RESULT_PARTIAL_SUCCESS, JudgmentResult, judge as judge_roll, roll_3d6
from src.engine.dimension import (
    CHARACTER_DIMENSIONS,
    EXTENSION_DIMENSIONS,
    RELATION_NUMERIC_DIMENSIONS,
    DimensionState,
    RelationDimensions,
    WORLD_STATE_DIMENSIONS,
)
from src.engine.ending import check_bottom_lines, generate_ending_narration
from src.engine.formula import (
    calculate_dc,
    calculate_modifier,
    classify_action,
    classify_action_nature,
    format_predicted_effects,
    get_predicted_effects_all_tiers,
    predict_effects,
)
from src.engine.growth import apply_action_growth, apply_decision_growth
from src.engine.judge import judge_decision
from src.engine.narrator import narrate_action, narrate_decision_result, narrate_opening_briefing, narrate_query, narrate_scene, narrate_time_advance
from src.engine.npc_engine import generate_npc_reply
from src.engine.pressure import apply_decay as apply_pressure_decay, check_milestones, check_reactions
from src.engine.state import GameState, get_state_manager
from src.engine.time_system import AdvanceResult, format_game_time, process_advance_queue
from src.engine.tracer import get_tracer, GameTracer
from src.feishu.card_builder import (
    build_action_confirmation_card,
    build_action_panel,
    build_cost_choice_card,
    build_dialogue_summary_card,
    build_dice_result_card,
    build_empty_advance_confirmation_card,
    build_narration_card,
    build_npc_reply_card,
    build_passive_response_card,
    build_prologue_card,
    build_report_card,
    build_role_selection_card,
    build_settlement_confirmation_card,
    build_settlement_result_card,
    build_gm_interpretation_card,
    build_court_session_card,
    build_situation_card,
    build_board_card,
    build_decision_card,
    build_opening_card,
)
from src.feishu.sender import send_narrator_card, send_npc_card, send_npc_text
from src.llm.client import get_llm_client
from src.scripts.loader import (
    is_new_format_script,
    load_available_roles,
    load_manifest,
    load_npc_from_world,
    load_npc_profile,
    load_role,
    load_world,
    load_all_fixed_npc_initial_locations,
    load_all_functional_npc_initial_locations,
    load_all_functional_npc_profiles,
    load_board_regions,
)

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """
你是一个文字RPG游戏的意图识别器。根据玩家输入判断意图。

当前场景：{scene_description}
当前阶段：{phase}
在场NPC：{present_npcs}

玩家消息：{user_text}

判断意图类型，输出JSON：
{{"intent": "dialogue/action/decision/query/meta", "target_npc": "NPC名或null", "detail": "一句话说明"}}
""".strip()

ACTION_ANALYSIS_PROMPT = """
你是文字RPG行动解析器。请把玩家输入转为一次可掷骰的行动提案。

规则：
1. 只输出JSON，不要解释。
2. main_dimension 从角色维度中选择最贴切的一项；auxiliary_dimensions 可为空或1-2项。
3. dc 取 4/6/8/10/12/14；普通行动8，困难10，极难12以上。
4. time_cost 表示行动需要推进的时间单位：即时行动为0，普通计划为1，长期计划为2-3。
5. tags 可填写与关系或场景有关的标签；不确定则空数组。

JSON结构：
{"description":"行动描述","dc":8,"main_dimension":"智谋","auxiliary_dimensions":[],"tags":[],"time_cost":1,"npc_id":""}
""".strip()

ACTION_EFFECT_PROMPT = """
你是文字RPG行动后果结算器。根据掷骰结果生成结构化后果。

只输出JSON：
{
  "narrative_hint": "给旁白的一句话后果提示",
  "effects": {
    "dimensions": {"world": {}, "character": {}, "relations": {}},
    "add_consequence_seeds": [],
    "add_information": [],
    "advance_action": null,
    "add_active_events": []
  },
  "public_info": [{"type": "decree/event/announcement", "content": "公开结果"}],
  "cost_options": [
    {"id":"cost_1","description":"接受一项明确代价","effects":{"dimensions":{"world":{}}},"narrative_hint":"代价叙事提示"}
  ]
}

要求：
- 成功应落实目标；失败应改变条件或留下隐患；大失败可生成压力、事件或后果种子。
- 部分成功必须给2-3个代价选项。
- 不要使用旧指标字段。
- 只有政令、公开行动、众所周知的失败或事件才写入 public_info；密谋、暗查、私下安排返回空数组。
- narrative_hint 和 cost_options[].narrative_hint 必须简短具体，动作或结果先行，每句话承载具体信息或决策压力，不用却/竟/不禁/仿佛/似乎，不用排比句，不用四字成语堆砌。
""".strip()

TALK_SUMMARY_PROMPT = """
你是文字RPG的记忆提取器。请分析以下NPC对话。

## 第一步：逐轮扫描
按对话轮次（每次玩家发言算一轮）逐轮列出玩家的发言要点。每轮回答：
- 轮次编号
- 玩家说了什么（一句话概括）
- 是否包含决策、指令、批准、人事任命、制度安排、建议、试探（是/否）

## 第二步：提取结构化输出
基于第一步的逐轮扫描结果，提取：

1. summary: 一两句话概括本次对话要点
2. key_facts: 关键事实列表，每条包含 type 和 content。type 可选: promise/attitude/information/request/secret/conflict
3. candidate_items: 将第一步中标记为「是」的每一轮，转为候选行动/决策
4. public_info: 对话中产生的公开信息（政令、公开决定）。私密对话不算公开。

只输出JSON：
{
  "round_scan": [
    {"round": 1, "player_summary": "...", "has_decision": true}
  ],
  "summary": "...",
  "key_facts": [{"type": "...", "content": "..."}],
  "candidate_items": [
    {
      "description": "...",
      "duration": "instant/delayed",
      "delay_count": 0,
      "deterministic": false,
      "deterministic_effects": {},
      "dc": 8,
      "modifier": 0,
      "main_dimension": "智谋"
    }
  ],
  "public_info": [{"type": "...", "content": "..."}]
}

要求：
- 第一步必须覆盖每一轮玩家发言，不得跳过任何一轮。
- 第一步中标记为有决策的轮次，第二步必须有对应的 candidate_item，不得遗漏。
- 提取玩家表达的任何潜在行动意图，包括但不限于：明确决策、批准（准奏/照办/就这么定/可以）、建议（那不如...）、试探（要不先...）、疑问形式的意图（能不能让xxx...）、人事任命、制度安排、监察架构。宁多勿漏，最终由玩家在汇总卡片上选择是否执行。
- 没有候选事项时 candidate_items 返回空数组。
- duration 只能是 instant 或 delayed；delay_count 是延迟推进次数。
- 不要输出 predicted_effects 字段，该字段由系统自动计算。
- 没有关键事实或公开信息时，对应字段返回空数组。
- 字段名严格使用上述JSON格式中的名称，不要替换为其他名称。
""".strip()

NPC_INTERFERENCE_PROMPT = """
你是文字RPG的NPC干扰裁决器。请根据掷骰结果、NPC档案和维度变化，判断是否有NPC干扰。

只输出JSON：
{
  "interferences": [
    {
      "item_id": "待办id",
      "npc": "NPC名",
      "type": "flip/weaken/condition",
      "reason": "动机、触发条件、能力范围为何同时满足",
      "added_backlog": null
    }
  ]
}

规则：只有动机 × 触发条件 × 能力范围三条全满足时才允许干扰。没有干扰返回空数组。
""".strip()

SUMMON_TARGET_RESOLVE_PROMPT = """
你是文字RPG召见目标解析器。根据玩家原文和NPC名单，判断玩家想召见哪些NPC。

规则：
1. 只从给定NPC名单中选择，返回NPC全名。
2. 可根据姓氏、简称、官职或上下文称呼匹配全名。
3. 没有召见意图或无法确定时返回空数组。
4. 只输出JSON：{"targets":["NPC全名"]}
""".strip()

SETTLEMENT_MEMORY_PROMPT = """
你是文字RPG结算记忆裁决器。请根据本次结算结果更新记忆。

只输出JSON：
{
  "fulfilled_promise_ids": ["承诺事实id"],
  "public_info": [{"type": "decree/event/announcement", "content": "公开结果"}]
}

规则：
- 只有成功或确定执行的待办，且明确兑现了某项承诺，才把该承诺id放入 fulfilled_promise_ids。
- 对没有随结算叙事给出 public_info、但明显公开的结果，补充写入 public_info。
- 政令、公开行动、众所周知的失败或事件算公开；密谋、暗查和私下安排不要写入 public_info。
- 没有命中时返回空数组。
""".strip()

COMPACT_NARRATION_STYLE = """
风格要求：
- 动作先行：先写人物动作和画面，如"大步踏入""甲胄尘土未拂"，不要用形容词堆砌气氛
- 信息即戏剧：每句话都要承载具体信息或决策压力，如"粮道断三日""铁骑压境"，不要空洞的文学描写
- 简短有力：句子短，信息密度高，不要长篇大论
- 留白给玩家：结尾制造张力让玩家想开口，不要替玩家描述感受
- 禁止文学腔：不用却/竟/不禁/仿佛/似乎，不用排比句，不用四字成语堆砌
""".strip()

PROLOGUE_NARRATION_STYLE = """
风格要求：
- 动作先行：先写人物动作和画面，如"大步踏入""甲胄尘土未拂"，不要用形容词堆砌气氛
- 信息即戏剧：每句话都要承载具体信息或决策压力，如"粮道断三日""铁骑压境"，不要空洞的文学描写
- 3-5句话，简短有力，不要长篇大论
- 结尾制造张力让玩家想开口，不要替玩家描述感受
- 禁止文学腔：不用却/竟/不禁/仿佛/似乎，不用排比句，不用四字成语堆砌
""".strip()

GM_OPENING_PROMPT = """
你是文字RPG的GM。现在玩家刚选好角色，你需要用 2-3 句话简述局势，作为开局破冰。

角色身份：{identity}
称呼：{player_address}
所在地点：{start_location}
背景局势：{background}
初始局面：{initial_context}
当前压力：{pressure_summary}

要求：
- 第一句用"{player_address}"称呼玩家，直接说当前最紧迫的一件事
- 第二句补充一个正在恶化的隐患或刚收到的消息
- 如有第三句，点明此刻身边的人或等待处理的东西
- 总共不超过80字
- 语气：你是旁白/GM，不是NPC；像一个冷静的局势简报员
- 禁止：文学修辞、四字成语堆砌、形容词堆叠、排比句
- 只输出纯文本，不要JSON、不要标题
""".strip()

OPENING_CLOSING_PROMPT = """
你是文字RPG的GM。玩家刚完成开局信息获取，你需要用 1-2 句话做叙事收尾，把场景推向"该行动了"。

角色身份：{identity}
称呼：{player_address}
所在地点：{start_location}
刚刚做了什么：{action_summary}

要求：
- 一句话描述信息获取后角色面前的局面
- 一句话暗示下一步的紧迫感
- 总共不超过50字
- 语气：旁白，不是NPC
- 禁止：文学修辞、四字成语、排比句
- 只输出纯文本
""".strip()

VISIT_NARRATION_PROMPT = """
你是文字RPG的NPC入场生成器。NPC进来就直接说事。

NPC：{npc_name}（{npc_title}）
玩家身份：{player_role}
来访原因：{reason}
来访形式：{visit_form}（in_person=亲自到访, messenger=传令兵, letter=书信）

格式（严格遵守，不加任何修饰）：
- in_person: "{npc_name}行礼：'{player_title}，具体事情内容'"
- messenger: "传令兵呈上{npc_name}急报：'{player_title}，具体事情内容'"
- letter: "{npc_name}来信：'{player_title}，具体事情内容'"

规则：
- NPC必须用"{player_title}"称呼玩家
- NPC第一句话必须包含具体信息（数字、地名、人名、事件）
- 总共1句话，不超过50字
- 严禁：外貌描写、穿着描写、环境描写、动作细节、文学修辞、四字成语、任何形容词修饰
- 错误示例（禁止）："朝服下摆沾着墨渍"、"掀帘而入"、"面色凝重"、"匆匆赶来"
- 只输出文本，不要JSON
""".strip()

GM_DECISION_SCAN_PROMPT = '''你是文字RPG的GM决策扫描器。检查本轮对话中玩家是否做出了任何明确决策。

当前对话：
玩家：{player_text}
{npc_name}：{npc_reply}

最近对话上下文：{recent_context}
已确认决策列表：{confirmed_decisions}

规则：
- 只捕获玩家【明确表态】的决策，如「准了」「就这么办」「拨银二十万」「下旨」等
- 模糊讨论、询问、犹豫不算决策
- 如果玩家没有做任何决策，返回空列表
- 不要重复已确认决策列表中已有的决策
- 每个决策用一句话概括，不超过30字

输出JSON：
{{"decisions": [{{"id": "dec_xxx", "summary": "一句话决策摘要", "category": "军事/财政/人事/政策/外交"}}]}}
如果没有检测到决策，返回：{{"decisions": []}}
'''.strip()

GM_HINT_PROMPT = """
你是文字RPG的GM（游戏主持人）。基于当前对话生成玩家提示。
当前对话上下文：{context}
NPC最新发言：{npc_message}
玩家维度数据：{dimensions}

规则：
- frame提示：给出2-3个探索方向，简短（10字以内每个）。必须是方向性的引导（如"追问详情""有什么方案""先搁置"），绝不能是具体策略或行动方案（如"加征辽饷""裁撤冗费"）。具体策略应在玩家追问后由NPC展开。
- annotate提示：只在玩家相关维度 > 阈值时生成，揭示NPC未明说的信息
- alert提示：只在玩家即将做高风险决策时生成
- side_quest 字段（可选，大多数对话不应触发）：仅当对话中明确出现了一个【与当前主要议题不同的】独立可执行方案时才触发。判断标准：(1)必须是NPC明确提出的具体可操作建议，不是泛泛的讨论；(2)必须是独立于当前主话题的次要事项，如果和当前对话主题直接相关则不算支线；(3)之前几轮已经讨论过的同一件事不要重复触发。大部分对话不应触发支线任务。如果不确定是否应触发，就不要触发。
- 如果当前对话不需要提示，返回空列表

输出JSON：{{"frame": ["..."], "annotate": [{{"text": "...", "dimension": "军事", "threshold": 60}}], "alert": ["..."], "side_quest": {{"name": "...", "description": "...", "rewards": "...", "penalties": "..."}}}}
""".strip()

MULTI_TALK_ARBITER_PROMPT = """
你是文字RPG多人谈话的发言裁定器。根据玩家发言、在场NPC和当前话题，选择谁回应。

只输出JSON：
{"primary": "NPC名", "supplements": ["NPC名"]}

规则：
- primary 必须从在场NPC中选一个。
- supplements 可为空，最多2人。
- 若玩家明确点名，只让被点名NPC作为primary。
""".strip()

JOIN_BRIEF_PROMPT = f"""
你是文字RPG的GM。一个新NPC被召入正在进行的对话。请生成入场叙事和给新NPC的简短上下文。

入场叙事要求：
{COMPACT_NARRATION_STYLE}
- 入场叙事必须是1句话，只写动作（如"某某入内行礼"），不超过20字
- 严禁在narration中包含NPC说的话、对白、台词——NPC说话由别的模块生成
- 错误示例："某某行礼：'陛下，臣来了'"——禁止出现引号内的对话

只输出JSON：
{{"narration": "纯动作入场叙事，1句话，无对白", "brief": "给新NPC的上下文简报，50字以内"}}
""".strip()

GM_INTERPRET_PROMPT = """
你是这个文字RPG的游戏主持人（GM）。玩家发来了一段话，请以GM视角解读。

【当前世界状态】
{world_brief}

【玩家角色】
{player_role_brief}

【世界维度状态】
{dimension_status}

以下维度限制了玩家行动的可行性：
- 财政 <= 2 时，大规模拨银、赏赐、修建等行动应标为"有条件"或"不可行"
- 兵力 <= 2 时，大规模调兵、出征等军事行动应标为"有条件"或"不可行"
- 民心 <= 2 时，加税、征役等损害民生的行动应标为"有条件"（会有严重后果）
- 士气 <= 2 时，强令出战等需要军心的行动应标为"有条件"
- 派系势力 >= 8 时，直接对抗强势派系的行动应标为"有条件"（会遭到强烈反弹）
维度值在 3-7 之间时行动一般可行但可能有代价。

【代价预估精度】
玩家智谋为 {player_intelligence}。
- 智谋 >= 7：在 risk_hint 和 prerequisites 中给出具体数字和精确预估（如"预计消耗财政2-3点"、"大约需要2回合"）
- 智谋 4-6：给出定性描述（如"代价不小"、"需要一定时间"）
- 智谋 <= 3：只给模糊感觉（如"此事不易"、"恐有风险"），不给具体数字

【在场/可联系的NPC】
{npc_list_with_locations}
重要：target_npcs 中的名字必须严格使用上方NPC列表中出现过的精确名字，不要使用官职、别称或自创角色。如果玩家提到的角色无法匹配到列表中的任何NPC，请在 gm_note 中说明。

【玩家输入】
{player_text}

请判断玩家想做什么，评估可行性和代价，输出JSON：
{{
  "interpretation": "你对玩家意图的理解（一句话）",
  "feasibility": "可行/有条件/不可行",
  "options": [
    {{
      "type": "talk/action/query/summon/multi_talk",
      "description": "选项描述",
      "target_npcs": ["NPC名"],
      "time_cost": 0,
      "prerequisites": "前置条件说明",
      "risk_hint": "风险提示"
    }}
  ],
  "gm_note": "给玩家的GM补充说明"
}}

规则：
- type 可选：talk（发起对话）、action（执行行动）、query（查询信息）、summon（召回远处NPC）、multi_talk（多人同场对话）
- 如果NPC不在身边但可以召回，type应为summon并在prerequisites中说明距离和所需回合
- 如果NPC无法联系，在gm_note中说明原因
- 可以给出1-3个选项供玩家选择
- time_cost为0表示即时，>0表示需要消耗回合
""".strip()

DYNAMIC_NPC_GENERATE_PROMPT = '''你是一个文字RPG游戏的角色设计师。游戏需要一个新角色，请根据以下信息生成完整的角色设定。

【剧本背景】
{world_brief}

【当前时间】
{game_time}

【玩家角色】
{player_role}

【已有角色列表】（避免重名或矛盾）
{existing_npcs}

【需要的角色】
{npc_description}

请生成一个有个性、有内在张力的角色。不要泛泛写'忠厚老实'，要有明确的性格特点、价值取向和行为模式。

输出严格JSON格式：
{{
  "character_seed": {{
    "name": "角色真实姓名",
    "personality_traits": ["特质1", "特质2", "特质3"],
    "values": ["价值1", "价值2"],
    "speaking_style": "说话风格描述",
    "behavioral_tendencies": "行为倾向描述",
    "knowledge_domain": "擅长领域",
    "knowledge_blind_spots": "不了解的领域"
  }},
  "initial_state": {{
    "title": "官职或身份",
    "faction": "阵营",
    "location": "当前位置",
    "status": "当前状态",
    "situation": "当前处境（第一人称）",
    "public_profile": "一句话公开身份描述"
  }}
}}'''

FOG_UNLOCK_GENERATE_PROMPT = """你是文字RPG游戏的世界信息生成器。一个之前被迷雾笼罩的区域信息现已解锁。请根据游戏当前状态，为这条解锁信息生成具体内容。

区域：{region_name}
信息类别：{layer_label}
迷雾提示（之前玩家看到的模糊信息）：{fog_hint}
当前游戏时间：{game_time}
世界维度状态：{dimension_summary}

要求：
1. 基于迷雾提示扩写为1-2句具体情报内容
2. 内容要和当前游戏状态吻合
3. 不要添加任何前缀标记或格式装饰
4. 保持简洁（30-60字内）
5. 不要输出JSON，直接输出纯文本
6. 动作先行或情报先行，每句话承载具体信息或决策压力，不用却/竟/不禁/仿佛/似乎，不用排比句，不用四字成语堆砌

直接输出文本内容。""".strip()

VALID_INTENTS = {"dialogue", "action", "decision", "query", "meta"}
_REGION_GROUPS: list[set[str]] = [
    {"京畿", "京师", "紫禁城", "乾清宫东暖阁", "内阁值房", "司礼监直房"},
    {"辽东", "宁锦前线军机幕府", "宁远", "锦州"},
    {"关外", "沈阳", "盛京"},
    {"西北", "陕北", "陕西", "陕豫流民大营"},
    {"江南", "南京", "苏州"},
]

UNREACHABLE_STATUSES = {"死亡", "被囚", "被掳", "流亡"}
HOSTILE_FACTIONS = {"敌对外部", "后金方", "流寇方"}


COURT_ARBITER_PROMPT = """
你是朝会裁判，决定这一轮哪些NPC发言、按什么顺序。

【玩家角色】
{player_role}

【在场NPC】
{npc_list_with_brief}

【议题】
{topic}

【场景类型】
{scene_type}

【本轮上下文】
玩家说：{player_text}
上一轮发言记录：{last_round_summary}

【是否点名】
{addressed_npc}

请输出JSON数组：
[
  {{"npc": "NPC名", "mode": "active/react/silent", "expression": "沉默时的非语言反应（仅silent需要）", "reason": "简短理由"}}
]

mode说明：
- active：完整发言（50-160字）
- react：简短回应（20-60字）
- silent：不发言，只有非语言反应

排序规则（严格执行，不得按NPC列表输入顺序排列）：
- 被玩家点名的NPC必须排第一且mode=active
- 泛问时，根据以下因素综合排序：
  1. 议题相关度：该NPC的知识领域/职责是否与议题直接相关
  2. 性格主动性：性格主动强势的排前面，阴柔被动的排后面
  3. 利害关系：议题结果对该NPC利益影响越大越优先发言
  4. 场景正式度：formal场合官阶高者优先；private场合性格主导
  5. 玩家身份：考虑玩家的角色地位，NPC的发言应体现与玩家的关系
- 不是每个NPC都必须active，与议题无关或性格内敛的NPC应mode=silent或react
- 只输出JSON数组，不要解释
""".strip()


def _same_region(loc1: str, loc2: str) -> bool:
    for group in _REGION_GROUPS:
        if loc1 in group and loc2 in group:
            return True
    return loc1 == loc2


def _calculate_reachability(
    player_location: str,
    npc_location: str,
    npc_status: str,
    npc_faction: str,
    npc_title: str = "",
) -> dict[str, Any]:
    if npc_status in UNREACHABLE_STATUSES:
        return {
            "location": npc_location,
            "reachability": "unreachable",
            "recall_cost": 0,
            "reason": f"状态：{npc_status}",
            "title": npc_title,
        }
    if _same_region(player_location, npc_location):
        return {"location": npc_location, "reachability": "present", "recall_cost": 0, "title": npc_title}
    if npc_faction in HOSTILE_FACTIONS:
        return {
            "location": npc_location,
            "reachability": "unreachable",
            "recall_cost": 0,
            "reason": f"敌方阵营：{npc_faction}",
            "title": npc_title,
        }
    return {"location": npc_location, "reachability": "distant", "recall_cost": 2, "title": npc_title}


_PLAYER_ROLE_TITLES = {
    "emperor": ("大明天子", "陛下"),
    "grand_secretary": ("内阁首辅", "阁老"),
    "frontier_general": ("辽东督师", "大帅"),
    "chief_eunuch": ("司礼监秉笔", "公公"),
    "rebel_leader": ("闯军首领", "大王"),
}


def _player_role_and_title(state: GameState) -> tuple[str, str]:
    """返回 (玩家身份描述, NPC对玩家的称呼)。"""
    role_id = str(state.player_role or "emperor").strip()
    if role_id in _PLAYER_ROLE_TITLES:
        return _PLAYER_ROLE_TITLES[role_id]
    role_info = state.storylines.get("role", {}).get("role", {}) if isinstance(state.storylines.get("role"), dict) else {}
    identity = str(role_info.get("identity", "")).strip()
    return (identity or role_id, "大人")


def _npc_title(state: GameState, npc_name: str) -> str:
    """获取NPC当前头衔。"""
    loc = state.npc_locations.get(npc_name, {})
    if isinstance(loc, dict):
        return str(loc.get("title", ""))
    return ""


def _build_npc_location_text(state: GameState) -> str:
    """构建NPC位置列表文本供GM解读prompt使用。"""
    parts: list[str] = []
    reachability_desc = {
        "present": "在身边，可即时对话",
        "nearby": "附近，1回合可召见",
    }
    for npc_name, loc_data in state.npc_locations.items():
        if not isinstance(loc_data, dict):
            continue
        reachability = loc_data.get("reachability", "unknown")
        location = loc_data.get("location", "未知")
        if reachability == "distant":
            desc = f"远处（{location}），需{loc_data.get('recall_cost', 2)}回合召回"
        elif reachability == "unreachable":
            desc = f"无法联系（{loc_data.get('reason', location)}）"
        else:
            desc = reachability_desc.get(reachability, reachability)
        title = loc_data.get("title", "")
        name_part = f"{npc_name}（{title}）" if title else npc_name
        parts.append(f"- {name_part}（{location}，{desc}）")
    return "\n".join(parts) if parts else "暂无可联系的NPC"


def _player_location(state: GameState) -> str:
    scene = state.current_scene if isinstance(state.current_scene, Mapping) else {}
    location = str(scene.get("location") or "").strip()
    if location and location != "未明":
        return location
    role_info = state.storylines.get("role", {}).get("role", {}) if isinstance(state.storylines.get("role"), Mapping) else {}
    location = str(role_info.get("start_location") or "").strip()
    if location:
        return location
    for loc_data in state.npc_locations.values():
        if isinstance(loc_data, Mapping) and str(loc_data.get("reachability", "")) in {"present", "nearby"}:
            location = str(loc_data.get("location") or "").strip()
            if location:
                return location
    return "京师"


def _npc_location_data(state: GameState, npc_name: str) -> Mapping[str, Any]:
    loc = state.npc_locations.get(npc_name, {})
    if isinstance(loc, Mapping):
        return loc
    npc_id = state.cast.get(npc_name, npc_name)
    loc = state.npc_locations.get(str(npc_id), {})
    return loc if isinstance(loc, Mapping) else {}


def _is_npc_near_player(state: GameState, npc_name: str) -> bool:
    loc = _npc_location_data(state, npc_name)
    npc_location = str(loc.get("location") or "").strip()
    if not npc_location:
        return True
    return _same_region(_player_location(state), npc_location)


def _build_prologue_npc_presence_text(state: GameState) -> str:
    nearby_npcs: list[str] = []
    remote_npcs: list[str] = []
    for npc_name in _all_known_npc_names(state):
        if _is_npc_near_player(state, npc_name):
            _append_unique_str(nearby_npcs, npc_name)
        else:
            _append_unique_str(remote_npcs, npc_name)
    nearby_text = "、".join(nearby_npcs) if nearby_npcs else "无"
    remote_text = "、".join(remote_npcs) if remote_npcs else "无"
    return (
        f"在场NPC（可出现在场景物理描写中）：{nearby_text}。"
        f"不在场NPC（人在远方，不能出现在场景中）：{remote_text}"
    )


def _intent_to_gm_result(intent_result: dict[str, Any], player_text: str) -> dict[str, Any]:
    """将旧意图识别结果转换为GM解读格式作为降级兜底。"""
    intent = intent_result.get("intent", "action")
    target_npc = intent_result.get("target_npc")
    type_map = {"dialogue": "talk", "action": "action", "query": "query", "decision": "action", "meta": "query"}
    return {
        "interpretation": intent_result.get("detail", player_text),
        "feasibility": "可行",
        "options": [{
            "type": type_map.get(intent, "action"),
            "description": player_text,
            "target_npcs": [target_npc] if target_npc else [],
            "time_cost": 0,
            "prerequisites": "",
            "risk_hint": "",
        }],
        "gm_note": "",
    }


RESTART_KEYWORDS = {"重新开始", "重来", "再来", "重新开局"}
CONFIRM_WORDS = {"确认", "执行", "同意", "好", "是", "行"}
REJECT_WORDS = {"取消", "放弃", "撤回", "不", "否"}
DEFAULT_SCRIPT_ID = "template"
SUMMON_KEYWORDS = ("叫", "召", "宣", "请", "喊", "找")
DISMISS_KEYWORDS = ("退下", "先走", "不用了", "下去", "告退", "离开")
URGENCY_SCORE = {"low": 20, "minor": 25, "normal": 45, "medium": 50, "moderate": 55, "high": 70, "major": 75, "urgent": 90, "critical": 100}
DIMENSION_RESPONSIBLE_NPCS = {
    "财政": ("户部尚书", "毕自严"),
    "兵力": ("兵部尚书", "孙传庭", "袁崇焕", "孙承宗"),
    "民心": ("内阁首辅", "周延儒", "左都御史"),
    "士气": ("兵部尚书", "孙传庭", "袁崇焕"),
    "派系势力": ("内阁首辅", "周延儒", "温体仁", "左都御史"),
    "情报": ("锦衣卫指挥使",),
    "补给": ("户部尚书", "工部尚书", "兵部尚书"),
}


def _append_history(state: GameState, role: str, content: str, speaker: str | None = None) -> None:
    """写入会话历史，保留最近窗口。"""

    record: dict[str, Any] = {"role": role, "content": content}
    if speaker:
        record["speaker"] = speaker
    state.conversation_history.append(record)
    state.conversation_history = state.conversation_history[-30:]


def _parse_trace_command(text: str) -> str | None:
    t = text.strip().lower()
    if t in ("/trace on", "/trace off", "/trace clear"):
        return t.split()[-1]
    return None


async def _handle_trace_command(chat_id: str, state: GameState, cmd: str) -> None:
    tracer = get_tracer(chat_id)
    if cmd == "on":
        state.trace_enabled = True
        tracer.enabled = True
        msg = "Trace 记录已开启。"
    elif cmd == "off":
        state.trace_enabled = False
        tracer.enabled = False
        msg = "Trace 记录已关闭。"
    elif cmd == "clear":
        tracer.clear()
        msg = "Trace 记录已清空。"
    else:
        return
    await send_narrator_card(chat_id, build_narration_card("Trace", msg))


def _is_restart_command(text: str) -> bool:
    """判断是否为重开指令。"""

    return any(keyword in text.strip() for keyword in RESTART_KEYWORDS)


def _is_new_game(state: GameState) -> bool:
    """判断是否首次进入游戏。"""

    return state.turn == 0 and not state.conversation_history and state.phase not in {"choosing_role", "opening", "playing", "free"}


def _reset_new_game_runtime_state(state: GameState) -> None:
    """清理新开局必须重置的运行态。"""

    state.npc_memory = {}
    state.world_memory = []
    state.conversation_history = []
    state.current_talk_history = []
    state.backlog = []
    state.delayed_queue = []
    state.present_npcs = []
    state.conversation_initiator = ""
    state.npc_join_round = {}
    state.visit_queue = []
    state.settlement_phase = "free"


async def _classify_intent(state: GameState, user_text: str) -> dict[str, Any]:
    """调用LLM判定玩家意图，失败时做兜底推断。"""

    scene = state.current_scene if isinstance(state.current_scene, dict) else {}
    system_prompt = INTENT_SYSTEM_PROMPT.format(
        scene_description=str(scene.get("context", "未设定")),
        phase=state.phase,
        present_npcs=scene.get("present_npcs", []),
        user_text=user_text,
    )
    user_content = json.dumps(
        {
            "user_text": user_text,
            "phase": state.phase,
            "scene": scene,
            "recent_history": state.conversation_history[-8:],
        },
        ensure_ascii=False,
    )

    try:
        result = await get_llm_client().chat_json(system_prompt, user_content, temperature=0.1)
        intent = str(result.get("intent", "")).strip().lower()
        if intent not in VALID_INTENTS:
            raise ValueError(f"invalid intent: {intent}")
        target_npc = result.get("target_npc")
        return {
            "intent": intent,
            "target_npc": str(target_npc).strip() if target_npc not in (None, "", "null") else None,
            "detail": str(result.get("detail", "")),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("意图识别失败，使用规则兜底: %s", exc)
        return _fallback_intent(user_text)


def _fallback_intent(user_text: str) -> dict[str, Any]:
    """简单规则兜底，避免分类失败中断流程。"""

    text = user_text.strip()
    lowered = text.lower()
    if any(keyword in text for keyword in ["存档", "帮助", "退出", "重来", "重新开始"]):
        return {"intent": "meta", "target_npc": None, "detail": "规则命中meta关键词"}
    if any(keyword in text for keyword in ["多少", "几", "如何", "情况", "查看", "查询", "维度", "数值"]):
        return {"intent": "query", "target_npc": None, "detail": "规则命中query关键词"}
    if any(keyword in text for keyword in ["决定", "选择", "采纳", "传旨", "准奏", "下令", "任命"]):
        return {"intent": "decision", "target_npc": None, "detail": "规则命中decision关键词"}
    if any(keyword in text for keyword in ["派", "调查", "打探", "出巡", "训练", "修建", "潜入", "攻击", "救援", "搜寻"]):
        return {"intent": "action", "target_npc": None, "detail": "规则命中action关键词"}
    if lowered.endswith("?") or text.endswith("？"):
        return {"intent": "query", "target_npc": None, "detail": "规则命中问句"}
    return {"intent": "dialogue", "target_npc": None, "detail": "默认按对话处理"}


async def _gm_interpret(state: GameState, player_text: str) -> dict[str, Any]:
    """调用LLM以GM视角解读玩家自由文本，失败时降级到旧意图识别。"""
    world_brief = state.world_summary or str(state.current_scene.get("context", "局势未明"))
    role_info = state.storylines.get("role", {}).get("role", {}) if isinstance(state.storylines.get("role"), dict) else {}
    player_role_brief = str(role_info.get("identity", "") or state.player_role)
    npc_list_text = _build_npc_location_text(state)
    dimension_status = ' '.join(f'{name}:{value}/10' for name, value in state.dimensions.world.items()) if state.dimensions.world else '维度未初始化'
    player_intelligence = state.dimensions.character.get('智谋', 5)
    _gm_cid = getattr(state, '_runtime_chat_id', '')
    if _gm_cid:
        get_tracer(_gm_cid, enabled=state.trace_enabled).record(
            'gm_dimension_gate', '_gm_interpret',
            {'dimension_status': dimension_status, 'player_intelligence': player_intelligence},
            GameTracer.state_summary(state))
    system_prompt = GM_INTERPRET_PROMPT.format(
        world_brief=world_brief,
        player_role_brief=player_role_brief,
        dimension_status=dimension_status,
        player_intelligence=player_intelligence,
        npc_list_with_locations=npc_list_text,
        player_text=player_text,
    )
    user_content = json.dumps({
        "player_text": player_text,
        "recent_history": state.conversation_history[-8:],
        "game_time": state.game_time,
    }, ensure_ascii=False)
    try:
        result = await get_llm_client().chat_json(system_prompt, user_content, temperature=0.15)
        if not isinstance(result, dict) or "options" not in result:
            raise ValueError("GM解读返回格式不正确")
        _cid = getattr(state, "_runtime_chat_id", "")
        if _cid:
            _t = get_tracer(_cid, enabled=state.trace_enabled)
            _t.record_llm_call("_gm_interpret", "GM_INTERPRET_PROMPT", GM_INTERPRET_PROMPT, {"world_brief": world_brief[:200], "player_text": player_text, "npc_list": npc_list_text[:500]}, json.dumps(result, ensure_ascii=False)[:2000], result, _t.state_summary(state))
        return result
    except Exception as exc:
        logger.warning("GM解读失败，降级到意图识别: %s", exc)
        fallback = await _classify_intent(state, player_text)
        return _intent_to_gm_result(fallback, player_text)


async def _generate_dynamic_npc(state: GameState, npc_description: str, chat_id: str) -> dict[str, Any] | None:
    """根据自然语言描述动态生成NPC，并写入运行态。"""

    try:
        existing_lines: list[str] = []
        for npc_name, profile in state.active_npcs.items():
            if not isinstance(profile, dict):
                continue
            name = str(profile.get("name") or npc_name)
            title = str(profile.get("title") or state.npc_locations.get(name, {}).get("title", ""))
            existing_lines.append(f"- {name}" + (f"（{title}）" if title else ""))
        existing_npcs = "\n".join(existing_lines) if existing_lines else "暂无"
        prompt = DYNAMIC_NPC_GENERATE_PROMPT.format(
            world_brief=state.world_summary,
            game_time=format_game_time(state.game_time),
            player_role=state.player_role,
            existing_npcs=existing_npcs,
            npc_description=npc_description,
        )
        llm = get_llm_client()
        raw = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model="default",
            temperature=0.7,
        )
        raw_text = raw if isinstance(raw, str) else str(raw)
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("动态NPC生成返回中未找到JSON")
        result = json.loads(json_match.group())
        if not isinstance(result, dict):
            raise ValueError("动态NPC生成JSON不是对象")
        character_seed = result.get("character_seed")
        initial_state = result.get("initial_state")
        if not isinstance(character_seed, dict) or not isinstance(initial_state, dict):
            raise ValueError("动态NPC生成缺少character_seed或initial_state")
        npc_name = str(character_seed["name"]).strip()
        if not npc_name:
            raise ValueError("动态NPC名称为空")
        profile = {
            "npc_id": npc_name,
            "name": npc_name,
            "title": initial_state.get("title", ""),
            "personality": ", ".join(str(t) for t in character_seed.get("personality_traits", [])),
            "faction": initial_state.get("faction", "未知"),
            "character_seed": character_seed,
            "initial_state": initial_state,
            "is_dynamic": True,
        }
        state.active_npcs[npc_name] = profile
        state.cast[npc_name] = npc_name

        player_location = "京师"
        for loc_data in state.npc_locations.values():
            if isinstance(loc_data, dict) and loc_data.get("reachability") == "present":
                player_location = str(loc_data.get("location") or player_location)
                break
        npc_location = str(initial_state.get("location", "京师"))
        npc_status = str(initial_state.get("status", "在朝"))
        npc_faction = str(initial_state.get("faction", "朝廷方"))
        npc_title = str(initial_state.get("title", ""))
        state.npc_locations[npc_name] = _calculate_reachability(
            player_location,
            npc_location,
            npc_status,
            npc_faction,
            npc_title,
        )
        state.npc_locations[npc_name]["faction"] = npc_faction
        tracer = get_tracer(chat_id, enabled=state.trace_enabled)
        tracer.record(
            "dynamic_npc_generated",
            "_generate_dynamic_npc",
            {"description": npc_description, "generated_name": npc_name},
            tracer.state_summary(state),
        )
        logger.info("动态生成NPC: %s (请求: %s)", npc_name, npc_description)
        return profile
    except Exception as exc:  # noqa: BLE001
        logger.warning("动态生成NPC失败: %s (请求: %s)", exc, npc_description)
        return None


async def _ensure_npc_exists(state: GameState, npc_name: str, chat_id: str) -> str | None:
    """确保NPC存在。返回实际NPC名字（可能是通过title匹配到的），不存在且生成失败返回None。"""

    npc_name = str(npc_name).strip()
    if not npc_name:
        return None
    if npc_name in state.npc_locations:
        return npc_name
    for npc_id, profile in state.active_npcs.items():
        if not isinstance(profile, dict):
            continue
        if str(profile.get("title", "")).strip() == npc_name:
            return str(profile.get("name") or npc_id)
    for npc, loc_data in state.npc_locations.items():
        if isinstance(loc_data, dict) and str(loc_data.get("title", "")).strip() == npc_name:
            return npc
    profile = await _generate_dynamic_npc(state, npc_name, chat_id)
    if profile:
        return str(profile.get("name", "")).strip() or None
    return None


async def _initialize_new_game(chat_id: str, state: GameState) -> None:
    """检测剧本格式并进入新游戏初始化流程。"""

    _reset_new_game_runtime_state(state)
    if is_new_format_script(state.script_id):
        await _initialize_new_format_game(chat_id, state)
        return
    await _initialize_legacy_game(chat_id, state)


async def _initialize_new_format_game(chat_id: str, state: GameState) -> None:
    """新格式剧本先展示世界层角色列表。"""

    world = await load_world(state.script_id)
    roles = await load_available_roles(state.script_id)
    _reset_new_game_runtime_state(state)
    state.phase = "choosing_role"
    state.turn = 0
    state.player_role = ""
    state.storylines["world"] = _compact_world(world)
    state.storylines["available_roles"] = roles
    state.world_summary = str(world.get("summary") or world.get("description") or "世界已展开，等待你选择视角。")

    card = build_role_selection_card(
        title=str(world.get("name") or world.get("title") or state.script_id),
        description=state.world_summary,
        roles=roles,
    )
    await send_narrator_card(chat_id, card)
    _append_history(state, "assistant", "请选择一个玩家角色。", "narrator")


async def _initialize_legacy_game(chat_id: str, state: GameState) -> None:
    """旧格式剧本降级启动，不再进入章节推进。"""

    manifest = await load_manifest(state.script_id)
    _reset_new_game_runtime_state(state)
    state.phase = "free"
    state.settlement_phase = "free"
    state.player_role = str(manifest.get("player_role") or state.player_role or "player")
    state.world_summary = str(manifest.get("summary") or manifest.get("description") or "旧格式剧本已载入。")
    state.current_scene = {"context": state.world_summary, "present_npcs": []}
    _sync_time_from_config(state, manifest.get("time") if isinstance(manifest.get("time"), dict) else {})

    await send_narrator_card(chat_id, build_narration_card("开局", state.world_summary))
    _append_history(state, "assistant", state.world_summary, "narrator")
    await _start_turn_visits_or_panel(chat_id, state)


async def _choose_role(chat_id: str, state: GameState, role_id: str) -> None:
    """选定角色后加载角色层并初始化运行态。"""

    role = await load_role(state.script_id, role_id)
    world = await load_world(state.script_id)
    role_info = role.get("role") if isinstance(role.get("role"), dict) else {}
    parsed_dimensions = role.get("parsed_dimensions") if isinstance(role.get("parsed_dimensions"), dict) else {}

    state.player_role = role_id
    state.phase = "free"
    state.turn = 1
    state.dimensions = _dimension_state_from_role(parsed_dimensions)
    state.previous_world_dimensions = dict(state.dimensions.world)
    state.pressure_sources = _merge_pressure_sources(world, role)
    state.information_pool = list(role.get("parsed_information_pool", []))
    state.player_goal = dict(role.get("player_goal") if isinstance(role.get("player_goal"), dict) else {})
    parsed_endings = role.get("parsed_endings") if isinstance(role.get("parsed_endings"), dict) else {}
    if parsed_endings:
        state.player_goal["endings"] = parsed_endings
        state.player_goal.setdefault("bottom_lines", parsed_endings.get("bottom_lines", []))
    state.storylines["world"] = _compact_world(world)
    state.storylines["role"] = _compact_role(role)
    state.storylines["endings"] = parsed_endings
    state.active_npcs = _active_npcs_from_world(world, role)
    state.accessible_npcs = _accessible_npcs_from_role(role, role_info)
    state.cast = _build_cast(state.active_npcs)
    state.current_scene = _initial_scene(role_info)
    state.game_time = _safe_int_dict(_time_config_from_world(world).get("start"), {"year": 1, "month": 1})
    state.game_date = _format_state_time(state)
    state.advance_queue = []
    state.pending_actions = []
    state.backlog = []
    state.delayed_queue = []
    # 加载初始回合议程
    initial_agenda = role.get("initial_agenda") if isinstance(role.get("initial_agenda"), list) else []
    state.turn_agenda = [
        {
            "id": str(item.get("id", f"agenda_{i}")),
            "description": str(item.get("description", "")),
            "source": "script",
            "urgency": str(item.get("urgency", "normal")),
            "consequence": str(item.get("consequence", "")),
            "relevant_npcs": [str(n) for n in item.get("relevant_npcs", [])] if isinstance(item.get("relevant_npcs"), list) else [],
        }
        for i, item in enumerate(initial_agenda)
        if isinstance(item, dict) and str(item.get("description", "")).strip()
    ]
    _init_cid = getattr(state, '_runtime_chat_id', '')
    if _init_cid:
        get_tracer(_init_cid, enabled=state.trace_enabled).record(
            'initial_agenda_loaded', '_choose_role',
            {'agenda_count': len(state.turn_agenda), 'items': [{'id': a.get('id', ''), 'description': a.get('description', '')} for a in state.turn_agenda]},
            GameTracer.state_summary(state))
    state.settlement_phase = "free"
    state.npc_memory = {}
    state.world_memory = []
    state.conversation_history = []
    state.current_talk_history = []
    state.consequence_seeds = []
    state.active_events = []
    state.event_history = []
    state.event_cooldowns = {}
    state.growth_log = []
    state.talking_to = ""
    state.present_npcs = []
    state.conversation_initiator = ""
    state.npc_join_round = {}
    state.visit_queue = []
    # NPC 位置初始化
    player_start_location = str(role_info.get("start_location", "京师"))
    fixed_npc_locs = load_all_fixed_npc_initial_locations(state.script_id)
    for npc_name, loc_info in fixed_npc_locs.items():
        state.npc_locations[npc_name] = _calculate_reachability(
            player_start_location, loc_info["location"], loc_info["status"], loc_info["faction"], loc_info.get("title", ""),
        )
    func_npc_locs = load_all_functional_npc_initial_locations(state.script_id)
    for npc_name, loc_info in func_npc_locs.items():
        state.npc_locations[npc_name] = _calculate_reachability(
            player_start_location, loc_info["location"], loc_info["status"], loc_info["faction"], loc_info.get("title", ""),
        )

    func_npc_profiles = load_all_functional_npc_profiles(state.script_id)
    for npc_name, profile in func_npc_profiles.items():
        if npc_name not in state.active_npcs:
            state.active_npcs[npc_name] = profile
            state.cast[npc_name] = npc_name

    # 棋盘区域初始化
    state.board_regions = load_board_regions(state.script_id)
    state.discovered_clues = []
    state.current_talk_history = []
    state.world_summary = str(state.player_goal.get("core_goal") or role_info.get("identity") or "新局已开。")

    # 开局流程分发：有 opening 配置用新流程，否则降级旧 prologue
    opening_config = role.get("opening")
    if isinstance(opening_config, dict) and opening_config.get("action_modes"):
        await _run_opening(chat_id, state, role, role_info)
    else:
        await _run_legacy_prologue(chat_id, state, role_info)


async def _run_opening(chat_id: str, state: GameState, role: dict, role_info: dict) -> None:
    """GM主持开局流程：破冰 → 行为模式选择面板。"""

    opening_config = role.get("opening", {})
    world = await load_world(state.script_id)

    identity = str(role_info.get("identity", ""))
    narration_style = role.get("narration_style", {})
    if not isinstance(narration_style, dict):
        narration_style = {}
    player_address = str(narration_style.get("player_address", "你"))
    start_location = str(role_info.get("start_location", ""))
    background = ""
    world_bg = world.get("world_background")
    if isinstance(world_bg, dict):
        background = str(world_bg.get("public_situation", ""))
    elif isinstance(world_bg, str):
        background = world_bg
    initial_context = str(opening_config.get("icebreaker_hint", ""))
    if not initial_context:
        scene = role_info.get("initial_scene")
        initial_context = str(scene.get("context", "")) if isinstance(scene, dict) else ""

    pressure_lines = []
    for src in state.pressure_sources:
        if isinstance(src, dict) and str(src.get("source_type", src.get("type", ""))) == "decay":
            narrative = str(src.get("narrative", "")).strip()
            if narrative:
                pressure_lines.append(narrative)
    pressure_summary = "；".join(pressure_lines[:3]) if pressure_lines else "局势紧迫"

    prompt = GM_OPENING_PROMPT.format(
        identity=identity,
        player_address=player_address,
        start_location=start_location,
        background=background,
        initial_context=initial_context,
        pressure_summary=pressure_summary,
    )

    llm = get_llm_client()
    icebreaker_text = await llm.chat(
        "你是文字RPG的GM，负责开局破冰。",
        prompt,
        temperature=0.7,
    )
    icebreaker_text = icebreaker_text.strip()

    action_modes = opening_config.get("action_modes", [])
    card = build_opening_card(icebreaker_text, action_modes, player_address)
    await send_narrator_card(chat_id, card)
    _append_history(state, "assistant", icebreaker_text, "narrator")

    state.phase = "opening"
    state.pending_gm_interpretation = {
        "type": "opening",
        "action_modes": action_modes,
    }
    logger.info("GM开局已发送 chat_id=%s modes=%d", chat_id, len(action_modes))


async def _run_legacy_prologue(chat_id: str, state: GameState, role_info: dict) -> None:
    """旧版 prologue 流程（向后兼容无 opening 字段的角色）。"""

    scene_context = (
        f"{state.current_scene.get('context') or state.world_summary}\n\n"
        f"{_build_prologue_npc_presence_text(state)}。\n\n"
        "只有location在当前地点的NPC可以出现在场景的物理描写中。location在远方的NPC不能出现在场景中。\n\n"
        "这是游戏序幕/开场叙事，请按以下要求输出：\n"
        f"{PROLOGUE_NARRATION_STYLE}"
    )
    narration = await narrate_scene(state, scene_context)
    await send_narrator_card(
        chat_id,
        build_prologue_card(
            narration,
            _prologue_dimensions(state),
            _prologue_pressure_warnings(state),
            _prologue_main_quest(state),
            turn_agenda=state.turn_agenda,
        ),
    )
    _append_history(state, "assistant", narration, "narrator")

    await _send_action_panel(chat_id, state)
    await _build_visit_queue(state)
    if state.visit_queue:
        await send_narrator_card(chat_id, build_narration_card("朝务", "序幕结束，开始处理朝务。"))
        visit_text = await _present_next_visit(state)
        if visit_text:
            message, gm_note, npc_name = visit_text
            await send_npc_card(chat_id, build_npc_reply_card(npc_name, message, npc_title=_npc_title(state, npc_name), gm_note=gm_note))
            state.current_talk_history.append({"role": "assistant", "speaker": npc_name, "content": message, "type": "visit_entry"})
            logger.info("[DEBUG-VISIT] 序幕来访注入 current_talk_history len=%d, npc=%s, type=visit_entry", len(state.current_talk_history), npc_name)
            _append_history(state, "assistant", message, state.talking_to or "narrator")


async def handle_message(chat_id: str, user_text: str) -> None:
    """处理用户消息并完成路由、回复与自动存档。"""

    state_manager = get_state_manager()
    state = state_manager.load(chat_id)
    clean_text = user_text.strip()
    logger.info("收到玩家消息 chat_id=%s text=%s phase=%s", chat_id, clean_text, state.phase)

    persist_state = True
    try:
        tracer = get_tracer(chat_id, enabled=state.trace_enabled)
        state._runtime_chat_id = chat_id
        trace_cmd = _parse_trace_command(clean_text)
        if trace_cmd:
            await _handle_trace_command(chat_id, state, trace_cmd)
            return

        if _is_restart_command(clean_text):
            state = GameState(script_id=state.script_id or DEFAULT_SCRIPT_ID)
            await _initialize_new_game(chat_id, state)
            return

        if state.phase == "ended":
            msg = "此局已结束。若要重玩，请发送“重新开始”或“重来”。"
            await send_narrator_card(chat_id, build_narration_card("终局已定", msg))
            _append_history(state, "assistant", msg, "narrator")
            return

        if _is_new_game(state):
            await _initialize_new_game(chat_id, state)
            return

        if state.phase == "choosing_role":
            _append_history(state, "user", clean_text, "player")
            await _handle_role_text_choice(chat_id, state, clean_text)
            return

        if state.phase == "opening":
            msg = "请从上方卡片中选择一个行为模式。"
            await send_narrator_card(chat_id, build_narration_card("提示", msg))
            return

        _append_history(state, "user", clean_text, "player")
        tracer.record("route", "handle_message", {"player_text": clean_text}, tracer.state_summary(state))

        if state.phase == "talking" and state.talking_to:
            await _route_talking_message(chat_id, state, clean_text)
            return

        if state.phase == "court" and state.court_session.get("active"):
            await _handle_court_message(chat_id, state, clean_text)
            return

        if state.settlement_phase != "free":
            msg = "当前正在结算流程中，请先使用卡片按钮确认或返回修改。"
            await send_narrator_card(chat_id, build_narration_card("结算进行中", msg))
            _append_history(state, "assistant", msg, "narrator")
            return

        interpretation = await _gm_interpret(state, clean_text)
        options = interpretation.get("options", [])
        logger.info("GM解读结果 interpretation=%s options=%d", interpretation.get("interpretation"), len(options))

        # 单选项快速路由：明确对话且NPC在身边，直接进入
        if len(options) == 1:
            opt = options[0]
            opt_type = str(opt.get("type", "")).strip()
            targets = opt.get("target_npcs", []) if isinstance(opt.get("target_npcs"), list) else []
            if opt_type == "talk" and targets:
                target = str(targets[0]).strip()
                loc = state.npc_locations.get(target, {})
                if loc.get("reachability") == "present":
                    await _start_talk(chat_id, state, target)
                    return
            elif opt_type == "query":
                await _route_query(chat_id, state, clean_text)
                return
            elif opt_type == "multi_talk" and targets:
                resolved = []
                for t in targets:
                    actual = await _ensure_npc_exists(state, str(t).strip(), chat_id)
                    if actual:
                        resolved.append(actual)
                if resolved:
                    targets = resolved
                    all_present = all(
                        state.npc_locations.get(str(t).strip(), {}).get("reachability") == "present"
                        for t in targets
                    )
                    if all_present:
                        topic = str(opt.get("description", "")).strip() or "议事"
                        scene_type = "formal" if any(k in topic for k in ("朝", "议", "军", "政")) else "private"
                        await _init_court_session(chat_id, state, [str(t).strip() for t in targets], topic, scene_type)
                        return

        # 其他情况展示GM解读卡片供玩家选择
        state.pending_gm_interpretation = interpretation
        await send_narrator_card(chat_id, build_gm_interpretation_card(interpretation))
        _append_history(state, "assistant", str(interpretation.get("interpretation", "")), "narrator")

    finally:
        state_manager.save(chat_id, state)
        logger.info("状态已自动存档 chat_id=%s turn=%s phase=%s", chat_id, state.turn, state.phase)


async def handle_card_action(chat_id: str, action_value: dict[str, Any]) -> dict[str, Any] | None:
    """处理飞书卡片按钮回调。"""

    state_manager = get_state_manager()
    state = state_manager.load(chat_id)
    tracer = get_tracer(chat_id, enabled=state.trace_enabled)
    state._runtime_chat_id = chat_id
    action = str(action_value.get("action", "")).strip()
    logger.info("卡片回调 chat_id=%s action=%s value=%s", chat_id, action, action_value)
    tracer.record("card_action", "handle_card_action", {"action": action, "value": {k: str(v)[:200] for k, v in action_value.items()}}, tracer.state_summary(state))
    persist_state = True

    try:
        if action == "select_role":
            role_id = str(action_value.get("role_id", "")).strip()
            if role_id:
                await _choose_role(chat_id, state, role_id)

        elif action == "talk":
            await _start_talk(chat_id, state, str(action_value.get("npc", "")).strip())

        elif action == "end_talk":
            await _end_talk(chat_id, state)

        elif action == "query_status":
            await _send_status_report(chat_id, state)
            await _send_action_panel(chat_id, state)

        elif action == "add_to_backlog":
            return await _add_card_item_to_backlog(chat_id, state, action_value)

        elif action == "skip_backlog_item":
            return await _skip_dialogue_candidate(chat_id, state, action_value)

        elif action == "confirm_dialogue_summary":
            await _confirm_dialogue_summary(chat_id, state)

        elif action == "remove_backlog_item":
            await _remove_backlog_item(chat_id, state, action_value)

        elif action == "confirm_settlement":
            await _confirm_settlement(chat_id, state)

        elif action == "return_to_modify":
            await _return_to_modify(chat_id, state)

        elif action == "confirm_empty_advance":
            await _advance_world_phase(chat_id, state)

        elif action == "confirm_action":
            proposal = _proposal_from_payload(action_value.get("proposal") or action_value)
            await _confirm_action(chat_id, state, proposal)

        elif action == "abandon_action":
            await _abandon_action(chat_id, state, str(action_value.get("proposal_id", "")))

        elif action == "passive_response":
            proposal = _proposal_from_payload(action_value.get("proposal") or action_value)
            await _confirm_action(chat_id, state, proposal, passive=True)

        elif action == "partial_success_cost":
            await _apply_partial_success_cost(chat_id, state, action_value)

        elif action == "advance_time":
            await _preview_advance(chat_id, state)

        elif action == "gm_select":
            await _handle_gm_selection(chat_id, state, action_value)

        elif action == "opening_select":
            await _handle_opening_selection(chat_id, state, action_value)

        elif action == "gm_frame_input":
            text = str(action_value.get("text", "")).strip()
            if text:
                persist_state = False
                await handle_message(chat_id, text)

        elif action == "court_address":
            npc = str(action_value.get("npc", "")).strip()
            if state.court_session.get("active") and npc:
                state.court_session["addressed_npc"] = npc
                await send_narrator_card(chat_id, build_narration_card("点名", f"请对{npc}说话。"))

        elif action == "court_speak_all":
            if state.court_session.get("active"):
                state.court_session["addressed_npc"] = ""

        elif action == "confirm_decision":
            decision = action_value.get("decision")
            if isinstance(decision, dict) and str(decision.get("summary", "")).strip():
                decision["confirmed_turn"] = state.turn
                state.confirmed_decisions.append(decision)
                _mark_agenda_resolved(state, decision.get("summary", ""))
                await send_narrator_card(chat_id, build_narration_card("决策已确认", f"📜 {decision.get('summary', '')}——已记录。"))

        elif action == "revoke_decision":
            dec_id = str(action_value.get("decision_id", ""))
            await send_narrator_card(chat_id, build_narration_card("决策已撤回", f"已撤回决策 {dec_id}。"))

        elif action == "court_end":
            await _end_court_session(chat_id, state)

        else:
            logger.info("忽略未知卡片动作: %s", action)

        return None

    finally:
        if persist_state:
            state_manager.save(chat_id, state)
        logger.info("卡片回调处理完毕 chat_id=%s action=%s", chat_id, action)


def _mark_agenda_resolved(state: GameState, decision_summary: str) -> None:
    """将与决策匹配的议程项标记为已处理。"""
    if not decision_summary:
        return
    for agenda_item in state.turn_agenda:
        if not isinstance(agenda_item, dict):
            continue
        if agenda_item.get("resolved"):
            continue
        title = str(agenda_item.get("title", "")).strip()
        desc = str(agenda_item.get("description", "")).strip()
        if title and any(kw in decision_summary for kw in title.split("_") if len(kw) >= 2):
            agenda_item["resolved"] = True
            agenda_item["resolved_turn"] = state.turn
            continue
        if desc and len(desc) >= 4:
            overlap = sum(1 for c in desc if c in decision_summary)
            if overlap >= len(desc) * 0.3:
                agenda_item["resolved"] = True
                agenda_item["resolved_turn"] = state.turn


async def _handle_role_text_choice(chat_id: str, state: GameState, text: str) -> None:
    """支持玩家用文字选择角色。"""

    roles = await load_available_roles(state.script_id)
    normalized = text.strip().lower()
    for index, role in enumerate(roles, start=1):
        role_id = str(role.get("role_id", "")).strip()
        name = str(role.get("name", role_id)).strip()
        if normalized in {str(index), role_id.lower(), name.lower()} or name in text or role_id in text:
            await _choose_role(chat_id, state, role_id)
            return
    msg = "没有匹配到该角色，请点击角色卡片按钮，或输入角色序号/名称。"
    await send_narrator_card(chat_id, build_role_selection_card("请选择角色", msg, roles))
    _append_history(state, "assistant", msg, "narrator")


async def _route_dialogue(chat_id: str, state: GameState, text: str, target_npc: str | None) -> None:
    """把普通对话路由给目标NPC。"""

    npc_name = _pick_target_npc(state, text, target_npc)
    npc_profile = await _resolve_npc_profile(state, npc_name)
    reply = await generate_npc_reply(state, npc_name, npc_profile, text)
    await send_npc_text(chat_id, npc_name, reply)
    _append_history(state, "assistant", reply, npc_name)
    state.turn += 1
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record_turn_snapshot(state)
    state.turns_without_action += 1


async def _route_talking_message(chat_id: str, state: GameState, text: str) -> None:
    """单独谈话模式下持续路由给同一NPC。"""

    if await _handle_talk_summon(chat_id, state, text):
        return
    if await _handle_talk_dismiss(chat_id, state, text):
        return
    if len(_normalize_present_npcs(state)) > 1:
        await _route_multi_talking_message(chat_id, state, text)
        return

    npc_name = state.talking_to
    logger.info("[DEBUG-TALK] _route_talking_message npc=%s phase=%s talk_history_len=%d history_types=%s", npc_name, state.phase, len(state.current_talk_history), [h.get("type") for h in state.current_talk_history])
    npc_profile = await _resolve_npc_profile(state, npc_name)
    state.current_talk_history.append({"role": "user", "speaker": "玩家", "content": text})
    reply = await generate_npc_reply(state, npc_name, npc_profile, text)
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record("llm_call", "_route_talking_message", {"npc_name": npc_name, "player_text": text, "reply": reply[:500]}, GameTracer.state_summary(state))
    state.current_talk_history.append({"role": "assistant", "speaker": npc_name, "content": reply})
    hints = await _generate_gm_hints(state, reply, {"source": "talk", "npc": npc_name, "player_text": text})
    hint_text = _apply_side_quest_hint(state, hints)
    frame_options = _hint_frame_options(hints)
    await send_npc_card(chat_id, build_npc_reply_card(npc_name, reply, frame_options=frame_options, npc_title=_npc_title(state, npc_name), gm_note=hint_text))
    detected = await _gm_decision_scan(state, text, reply, npc_name)
    if detected:
        await send_narrator_card(chat_id, build_decision_card(detected, npc_name))
    _append_history(state, "assistant", reply, npc_name)
    state.turn += 1
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record_turn_snapshot(state)
    state.turns_without_action += 1


async def _route_action(chat_id: str, state: GameState, text: str, intent_result: Mapping[str, Any]) -> None:
    """主动行动转为本回合待办，不立即产生后果。"""

    proposal = await _create_proposal_from_text(state, text, intent_result)
    item = _backlog_item_from_proposal(proposal, source="player_text")
    state.backlog.append(item)
    state.phase = "free"
    state.settlement_phase = "free"
    await send_narrator_card(chat_id, build_narration_card("待办已记录", f"**已加入待办**\n{item['description']}"))
    _append_history(state, "assistant", f"已加入待办：{proposal.description}", "narrator")
    await _start_turn_visits_or_panel(chat_id, state)


async def _route_decision(chat_id: str, state: GameState, text: str) -> None:
    """决策意图同样加入本回合待办。"""

    proposal = await _create_proposal_from_text(state, text, {"intent": "decision", "detail": text})
    item = _backlog_item_from_proposal(proposal, source="player_text")
    state.backlog.append(item)
    state.phase = "free"
    state.settlement_phase = "free"
    await send_narrator_card(chat_id, build_narration_card("待办已记录", f"**已加入待办**\n{item['description']}"))
    _append_history(state, "assistant", f"已加入待办：{proposal.description}", "narrator")
    await _start_turn_visits_or_panel(chat_id, state)


async def _route_query(chat_id: str, state: GameState, text: str) -> None:
    """处理玩家信息查询。"""

    answer = await narrate_query(state, text)
    forces_summary = f"{state.world_summary or str(state.current_scene.get('context', '暂无势力动向。'))}\n\n{answer}"
    await send_narrator_card(chat_id, build_situation_card(_world_state_payload(state), forces_summary, state.backlog, state.delayed_queue))
    _append_history(state, "assistant", forces_summary, "narrator")
    state.turn += 1
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record_turn_snapshot(state)
    state.turns_without_action += 1


async def _route_meta(chat_id: str, state: GameState, text: str) -> None:
    """处理帮助、状态等元指令。"""

    if "帮助" in text:
        msg = "你可以：与NPC对话、描述行动并确认掷骰、点击推进时间、查看状态；发送“重新开始”可重开。"
    elif any(word in text for word in ["存档", "保存"]):
        msg = "当前进度会在每次消息或按钮操作后自动存档。"
    else:
        msg = "指令已收到。若要重开，请发送“重新开始”；若要查看局势，请点击“查看状态”。"
    await send_narrator_card(chat_id, build_narration_card("系统", msg))
    _append_history(state, "assistant", msg, "narrator")


async def _create_proposal_from_text(state: GameState, text: str, intent_result: Mapping[str, Any]) -> ActionProposal:
    """调用行动解析器并转为 action.py 的提案对象。"""

    user_content = json.dumps(
        {
            "player_input": text,
            "intent_detail": dict(intent_result),
            "dimension_description": get_state_manager().get_dimension_description(state),
            "scene": state.current_scene,
            "active_npcs": state.active_npcs,
            "time_policy": _time_config_from_state(state).get("action_time_cost", {}),
        },
        ensure_ascii=False,
    )
    try:
        data = await get_llm_client().chat_json(ACTION_ANALYSIS_PROMPT, user_content, temperature=0.15)
    except Exception as exc:  # noqa: BLE001
        logger.warning("行动解析失败，使用兜底提案: %s", exc)
        data = {}

    main_dimension = str(data.get("main_dimension") or data.get("main_dim") or _default_main_dimension(state)).strip()
    description = str(data.get("description") or text).strip()
    return create_action_proposal(
        description=description,
        dc=_safe_int(data.get("dc"), 8),
        state=state,
        main_dim=main_dimension,
        aux_dims=_as_str_list(data.get("auxiliary_dimensions") or data.get("aux_dims")),
        tags=_as_str_list(data.get("tags")),
        time_cost=max(0, _safe_int(data.get("time_cost"), 1)),
        npc_id=str(data.get("npc_id") or "").strip(),
    )


async def _confirm_action(chat_id: str, state: GameState, proposal: ActionProposal, passive: bool = False) -> None:
    """确认执行行动，掷骰、发结果卡并处理成长。"""

    _remove_pending_proposal(state, proposal)
    action_result = execute_action(proposal, state)
    structured = await _generate_action_effects(state, action_result)
    action_result.effects.update(structured.get("effects", {}))
    if structured.get("narrative_hint"):
        action_result.narrative_hint = str(structured["narrative_hint"])

    cost_options = structured.get("cost_options") if isinstance(structured.get("cost_options"), list) else []
    growth_record = apply_action_growth(state, action_result.judgment.result_tier, proposal.main_dimension)
    narration = await narrate_action(
        state,
        f"{proposal.description}\n判定：{action_result.judgment.result_tier}；提示：{action_result.narrative_hint}",
    )

    card = build_dice_result_card(action_result, narration=narration, growth=_serialize_record(growth_record) if growth_record else None)
    await send_narrator_card(chat_id, card)
    _append_history(state, "assistant", narration, "narrator")
    state.turn += 1
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record_turn_snapshot(state)
    state.turns_without_action = 0

    if action_result.judgment.result_tier == RESULT_PARTIAL_SUCCESS and cost_options:
        state.pending_actions.append(
            {
                "kind": "partial_success",
                "proposal": _proposal_payload(proposal),
                "base_effects": action_result.effects,
                "cost_options": cost_options,
            }
        )
        await send_narrator_card(chat_id, build_cost_choice_card(proposal, cost_options))
        return

    get_state_manager().apply_effects(state, action_result.effects)
    _queue_action_if_needed(state, proposal, action_result.effects, passive=passive)
    await _check_bottom_line_and_maybe_end(chat_id, state)


async def _generate_action_effects(state: GameState, action_result: Any) -> dict[str, Any]:
    """根据行动判定结果生成结构化后果。"""

    payload = {
        "proposal": _proposal_payload(action_result.proposal),
        "judgment": _serialize_record(action_result.judgment),
        "default_effects": action_result.effects,
        "state": {
            "game_time": state.game_time,
            "dimensions": get_state_manager().get_dimension_description(state),
            "scene": state.current_scene,
            "player_goal": state.player_goal,
        },
    }
    try:
        category = classify_action(action_result.proposal.description, action_result.proposal.main_dimension)
        predicted = get_predicted_effects_all_tiers(category)
        from src.engine.formula import ACTION_CATEGORIES

        cat_name = ACTION_CATEGORIES.get(category, {}).get("name", "通用")
        formula_constraint = (
            f"\n\n【公式约束】本行动类别：{cat_name}。"
            f"预计效果——成功：{format_predicted_effects(predicted.get('success', {}))}，"
            f"失败：{format_predicted_effects(predicted.get('failure', {}))}。"
            f"effects中的维度变化必须在上述预计范围±1以内。大成功可额外+1，大失败可额外-1。"
        )
        enhanced_prompt = ACTION_EFFECT_PROMPT + formula_constraint
        data = await get_llm_client().chat_json(enhanced_prompt, json.dumps(payload, ensure_ascii=False), temperature=0.35)
    except Exception as exc:  # noqa: BLE001
        logger.warning("行动后果生成失败，使用默认后果: %s", exc)
        data = {}
    if not isinstance(data, dict):
        return {}
    effects = data.get("effects") if isinstance(data.get("effects"), dict) else {}
    cost_options = data.get("cost_options") if isinstance(data.get("cost_options"), list) else []
    return {
        "narrative_hint": str(data.get("narrative_hint", "")).strip(),
        "effects": effects,
        "public_info": [item for item in _as_list(data.get("public_info")) if isinstance(item, dict)],
        "cost_options": cost_options,
    }


async def _apply_partial_success_cost(chat_id: str, state: GameState, action_value: Mapping[str, Any]) -> None:
    """应用玩家选择的部分成功代价。"""

    proposal = _proposal_from_payload(action_value.get("proposal") or action_value)
    cost_id = str(action_value.get("cost_id", "")).strip()
    pending = _find_partial_success(state, proposal)
    if not pending:
        msg = "这项代价选择已经失效。"
        await send_narrator_card(chat_id, build_narration_card("代价失效", msg))
        return

    cost_options = pending.get("cost_options", []) if isinstance(pending.get("cost_options"), list) else []
    selected_options = [option for option in cost_options if isinstance(option, dict) and str(option.get("id")) == cost_id]
    if not selected_options and cost_options:
        selected_options = [cost_options[0]]
    resolution = resolve_partial_success(proposal, selected_options)
    effects = dict(pending.get("base_effects", {})) if isinstance(pending.get("base_effects"), dict) else {}
    get_state_manager().apply_effects(state, effects)
    get_state_manager().apply_effects(state, resolution.get("effects", {}))
    _queue_action_if_needed(state, proposal, effects)
    state.pending_actions.remove(pending)

    narration = await narrate_decision_result(state, f"{proposal.description}的代价", resolution)
    await send_narrator_card(chat_id, build_narration_card("代价落定", narration))
    _append_history(state, "assistant", narration, "narrator")
    await _check_bottom_line_and_maybe_end(chat_id, state)


async def _add_card_item_to_backlog(chat_id: str, state: GameState, action_value: dict[str, Any]) -> dict[str, Any] | None:
    """把汇总卡候选项加入当前待办，返回更新后的卡片。"""

    summary = state.dialogue_summary
    if not summary or not isinstance(summary.get("candidates"), list):
        return None
    candidates = summary["candidates"]
    item_index = action_value.get("item_index")
    if not isinstance(item_index, int) or item_index < 0 or item_index >= len(candidates):
        return None
    candidate = candidates[item_index]
    if candidate.get("status") != "pending":
        return None
    item = candidate["item"]
    if not isinstance(item, dict):
        return None
    if not any(existing.get("id") == item.get("id") for existing in state.backlog):
        state.backlog.append(item)
    candidate["status"] = "added"
    _append_history(state, "assistant", f"已加入待办：{item.get('description', '')}", "narrator")
    updated_card = build_dialogue_summary_card(summary["npc_name"], summary["summary_text"], candidates)
    return {"card": {"type": "raw", "data": updated_card}}


async def _skip_dialogue_candidate(chat_id: str, state: GameState, action_value: dict[str, Any]) -> dict[str, Any] | None:
    """跳过汇总卡候选项，返回更新后的卡片。"""

    summary = state.dialogue_summary
    if not summary or not isinstance(summary.get("candidates"), list):
        return None
    candidates = summary["candidates"]
    item_index = action_value.get("item_index")
    if not isinstance(item_index, int) or item_index < 0 or item_index >= len(candidates):
        return None
    candidate = candidates[item_index]
    if candidate.get("status") != "pending":
        return None
    candidate["status"] = "skipped"
    updated_card = build_dialogue_summary_card(summary["npc_name"], summary["summary_text"], candidates)
    return {"card": {"type": "raw", "data": updated_card}}


async def _confirm_dialogue_summary(chat_id: str, state: GameState) -> None:
    """确认完毕，清除汇总状态并进入来访或行动面板。"""

    state.dialogue_summary = {}
    state.settlement_phase = "free"
    await _start_turn_visits_or_panel(chat_id, state)


async def _remove_backlog_item(chat_id: str, state: GameState, action_value: Mapping[str, Any]) -> None:
    """从确认卡移除一条待办并重绘确认卡。"""

    if state.settlement_phase != "confirming":
        await send_narrator_card(chat_id, build_narration_card("操作已失效", "当前不在结算确认阶段。"))
        return
    raw_item = action_value.get("item") if isinstance(action_value.get("item"), Mapping) else action_value
    item_id = str(raw_item.get("id", "")).strip() if isinstance(raw_item, Mapping) else ""
    description = str(raw_item.get("description", "")).strip() if isinstance(raw_item, Mapping) else ""
    state.backlog = [
        item for item in state.backlog
        if (item_id and str(item.get("id")) != item_id) or (not item_id and str(item.get("description")) != description)
    ]
    state.settlement_phase = "confirming"
    await send_narrator_card(chat_id, build_settlement_confirmation_card(state.backlog, _world_state_payload(state), state.delayed_queue))


async def _return_to_modify(chat_id: str, state: GameState) -> None:
    """返回自由阶段，保留当前待办。"""

    state.settlement_phase = "free"
    state.phase = "free"
    await _start_turn_visits_or_panel(chat_id, state)


async def _advance_time(chat_id: str, state: GameState) -> None:
    """玩家主动推进时间，处理到期行动和触发事件。"""

    result = process_advance_queue(state)
    narration = await narrate_time_advance(state, result)
    await send_narrator_card(chat_id, build_narration_card("时间推进", narration))
    _append_history(state, "assistant", narration, "narrator")

    for due_action in result.due_actions:
        await _resolve_due_action(chat_id, state, due_action)

    for event in result.triggered_events:
        await _handle_triggered_event(chat_id, state, event)

    if not await _check_bottom_line_and_maybe_end(chat_id, state):
        await _start_turn_visits_or_panel(chat_id, state)


async def _preview_advance(chat_id: str, state: GameState) -> None:
    """推进按钮先进入结算确认阶段。"""

    state.settlement_phase = "confirming"
    if not state.backlog:
        await send_narrator_card(chat_id, build_empty_advance_confirmation_card())
        return
    await send_narrator_card(
        chat_id,
        build_settlement_confirmation_card(
            state.backlog,
            _world_state_payload(state),
            state.delayed_queue,
        ),
    )


async def _confirm_settlement(chat_id: str, state: GameState) -> None:
    """确认后统一结算当前待办，并进入世界推进。"""

    if state.settlement_phase not in {"confirming", "settling"}:
        await send_narrator_card(chat_id, build_narration_card("操作已失效", "请先点击行动面板的推进按钮生成结算确认。"))
        return
    state.settlement_phase = "settling"
    backlog_items = list(state.backlog)
    state.backlog = []
    instant_results: list[dict[str, Any]] = []

    for item in backlog_items:
        if item.get("duration") == "delayed":
            delayed_item = dict(item)
            delayed_item["remaining_advances"] = max(1, _safe_int(item.get("delay_count"), 1))
            state.delayed_queue.append(delayed_item)
            await send_narrator_card(chat_id, build_settlement_result_card(item, {}, "进入延迟队列", f"此事将于 {delayed_item['remaining_advances']} 次推进后结算。", None))
            continue
        result = await _settle_backlog_item(state, item)
        instant_results.append(result)

    interferences = await _judge_npc_interference(state, instant_results)
    _apply_interferences_to_results(state, instant_results, interferences)
    _apply_settlement_effects(state, instant_results)
    # 收集即时结算的维度效果
    settlement_dim_changes: dict[str, int] = {}
    for result in instant_results:
        effects = result.get("effects") if isinstance(result.get("effects"), dict) else {}
        world_dims = effects.get("dimensions", {}).get("world", {}) if isinstance(effects.get("dimensions"), dict) else {}
        if isinstance(world_dims, dict):
            for dim_name, delta in world_dims.items():
                if isinstance(delta, int):
                    settlement_dim_changes[str(dim_name)] = settlement_dim_changes.get(str(dim_name), 0) + delta
    if settlement_dim_changes:
        _evolve_board_from_dimension_changes(state, settlement_dim_changes)
    newly_unlocked = await _check_board_fog_gates(state)
    if newly_unlocked:
        unlock_text = "\n".join(f"🔓 {u['region']}·{u['layer']}：{u['content']}" for u in newly_unlocked)
        await send_narrator_card(chat_id, build_narration_card("迷雾揭开", unlock_text))
    await _update_memory_after_settlement(state, instant_results, interferences)

    for result in instant_results:
        item = result["item"]
        await send_narrator_card(
            chat_id,
            build_settlement_result_card(
                item,
                result.get("dice_result", {}),
                str(result.get("outcome", "已结算")),
                str(result.get("narrative", "")),
                result.get("interference"),
            ),
        )
        _append_history(state, "assistant", str(result.get("narrative", "")), "narrator")

    if await _check_bottom_line_and_maybe_end(chat_id, state):
        return
    await _advance_world_phase(chat_id, state)


async def _advance_world_phase(chat_id: str, state: GameState) -> None:
    """执行世界推进、到期延迟项和被动事件入队。"""

    state.settlement_phase = "advancing"
    old_time = dict(state.game_time)
    from src.engine.time_system import advance_time as advance_game_time

    state.game_time = advance_game_time(state.game_time, units=1)
    state.turn += 1
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record_turn_snapshot(state)
    state.game_date = _format_state_time(state)

    state.previous_world_dimensions = dict(state.dimensions.world)
    decay_changes = apply_pressure_decay(state, state.pressure_sources)
    if decay_changes:
        _evolve_board_from_dimension_changes(state, decay_changes)
    triggered_events = [*check_milestones(state, state.pressure_sources), *check_reactions(state, state.pressure_sources)]
    due_delayed = _tick_delayed_queue(state)
    for item in due_delayed:
        if str(item.get("type", "")).strip() == "summon":
            npc = str(item.get("npc", "")).strip()
            if npc and npc in state.npc_locations:
                state.npc_locations[npc]["reachability"] = "present"
                state.npc_locations[npc]["recall_cost"] = 0
            msg = f"{npc}已到达，现可召见。"
            await send_narrator_card(chat_id, build_narration_card("人员到达", msg))
            _append_history(state, "assistant", msg, "narrator")
            continue
        result = await _settle_backlog_item(state, item)
        _apply_settlement_effects(state, [result])
        delayed_effects = result.get("effects") if isinstance(result.get("effects"), dict) else {}
        delayed_world = delayed_effects.get("dimensions", {}).get("world", {}) if isinstance(delayed_effects.get("dimensions"), dict) else {}
        if isinstance(delayed_world, dict) and delayed_world:
            _evolve_board_from_dimension_changes(state, delayed_world)
        newly_unlocked = await _check_board_fog_gates(state)
        if newly_unlocked:
            unlock_text = "\n".join(f"🔓 {u['region']}·{u['layer']}：{u['content']}" for u in newly_unlocked)
            await send_narrator_card(chat_id, build_narration_card("迷雾揭开", unlock_text))
        await _update_memory_after_settlement(state, [result], [])
        await send_narrator_card(chat_id, build_settlement_result_card(item, result.get("dice_result", {}), str(result.get("outcome", "已结算")), str(result.get("narrative", "")), None))

    for event in triggered_events:
        passive_item = _backlog_item_from_event(state, event)
        state.backlog.append(passive_item)
        _append_world_memory(state, {"type": "event", "content": str(event.get("description") or event.get("name") or passive_item.get("description", ""))}, state.turn)
    _inject_events_to_board(state, triggered_events)

    advance_text = (
        f"{format_game_time(old_time, _time_config_from_state(state).get('display_format', '崇祯{year}年{month}月'))} → {state.game_date}\n"
        f"自然变化：{json.dumps(decay_changes, ensure_ascii=False) if decay_changes else '无'}\n"
        f"新增被动事件：{len(triggered_events)} 项"
    )
    await send_narrator_card(chat_id, build_narration_card("世界推进", advance_text))
    _append_history(state, "assistant", advance_text, "narrator")
    if decay_changes:
        _append_world_memory(state, {"type": "dimension_change", "content": f"自然变化：{json.dumps(decay_changes, ensure_ascii=False)}"}, state.turn)
    if await _check_bottom_line_and_maybe_end(chat_id, state):
        return

    # 生成下回合议程
    next_agenda: list[dict[str, Any]] = []

    # 1. 从衰减变化生成议程
    if decay_changes:
        for dim_name, change_val in decay_changes.items():
            current_val = state.dimensions.world.get(str(dim_name), 5)
            if current_val <= 3:
                next_agenda.append(
                    {
                        "id": f"agenda_decay_{dim_name}_{state.turn}",
                        "description": f"{dim_name}持续恶化（当前{current_val}/10），亟需应对",
                        "source": "decay",
                        "urgency": "urgent" if current_val <= 2 else "normal",
                        "consequence": f"若不处理，{dim_name}将继续下滑",
                    }
                )

    # 2. 从即将到来的里程碑生成预警议程
    for ps in state.pressure_sources:
        if not isinstance(ps, dict):
            continue
        milestones = ps.get("milestones") if isinstance(ps.get("milestones"), list) else []
        for ms in milestones:
            if not isinstance(ms, dict):
                continue
            ms_time = ms.get("at") if isinstance(ms.get("at"), dict) else {}
            ms_year = ms_time.get("year", 999)
            ms_month = ms_time.get("month", 999)
            cur_year = state.game_time.get("year", 1)
            cur_month = state.game_time.get("month", 1)
            months_until = (ms_year - cur_year) * 12 + (ms_month - cur_month)
            if 0 < months_until <= 3 and str(ms.get("pressure_id", "")) not in state.event_history:
                next_agenda.append(
                    {
                        "id": f"agenda_ms_{ms.get('pressure_id', 'unknown')}",
                        "description": f"预警：{ms.get('description', '重大事件逼近')}（约{months_until}个月后）",
                        "source": "milestone",
                        "urgency": "urgent" if months_until <= 1 else "normal",
                        "consequence": str(ms.get("description", "")),
                    }
                )

    # 3. 从被动事件生成议程
    for event in triggered_events:
        if not isinstance(event, dict):
            continue
        next_agenda.append(
            {
                "id": f"agenda_event_{event.get('event_id', 'unknown')}_{state.turn}",
                "description": f"新发事件：{event.get('description', event.get('name', '突发状况'))}",
                "source": "event",
                "urgency": "urgent",
                "consequence": "需要立刻应对",
            }
        )

    # 4. 保留上回合未处理的紧急议程
    for old_item in state.turn_agenda:
        if not isinstance(old_item, dict):
            continue
        if str(old_item.get("urgency")) == "urgent" and str(old_item.get("source")) != "event":
            carried = dict(old_item)
            carried["description"] = "（延续）" + carried.get("description", "")
            carried["source"] = "carried"
            next_agenda.append(carried)

    state.turn_agenda = next_agenda[:8]
    _agenda_cid = getattr(state, '_runtime_chat_id', '')
    if _agenda_cid:
        get_tracer(_agenda_cid, enabled=state.trace_enabled).record(
            'agenda_generated', '_advance_world_phase',
            {'agenda_count': len(next_agenda[:8]), 'items': [{'id': a.get('id', ''), 'description': a.get('description', ''), 'source': a.get('source', ''), 'urgency': a.get('urgency', '')} for a in next_agenda[:8]]},
            GameTracer.state_summary(state))
    newly_unlocked = await _check_board_fog_gates(state)
    if newly_unlocked:
        unlock_text = "\n".join(f"🔓 {u['region']}·{u['layer']}：{u['content']}" for u in newly_unlocked)
        await send_narrator_card(chat_id, build_narration_card("迷雾揭开", unlock_text))
    state.settlement_phase = "free"
    state.phase = "free"
    await _start_turn_visits_or_panel(chat_id, state)


async def _settle_backlog_item(state: GameState, item: Mapping[str, Any]) -> dict[str, Any]:
    """结算单条待办，返回未应用的效果。"""

    normalized_item = dict(item)
    if normalized_item.get("deterministic"):
        effects = normalized_item.get("deterministic_effects", {}) if isinstance(normalized_item.get("deterministic_effects"), dict) else {}
        return {
            "item": normalized_item,
            "dice_result": {},
            "outcome": "确定执行",
            "effects": effects,
            "narrative": f"{normalized_item.get('description', '此事')}按既定安排执行。",
            "public_info": [],
        }

    dice_values, total = roll_3d6()
    modifier = _safe_int(normalized_item.get("modifier"), 0)
    dc = _safe_int(normalized_item.get("dc"), 8)
    outcome = judge_roll(total, modifier, dc)
    category = str(normalized_item.get("category", "")) or classify_action(str(normalized_item.get("description", "")), str(normalized_item.get("main_dimension", "")))
    formula_effects = predict_effects(category, outcome)
    judgment = JudgmentResult(dice_values=dice_values, total=total, modifier=modifier, dc=dc, final_value=total + modifier, result_tier=outcome, is_critical=outcome in {"大成功", "大失败"})
    proposal = _proposal_from_backlog_item(normalized_item)
    action_result = execute_action(proposal, state)
    action_result.judgment = judgment
    structured = await _generate_action_effects(state, action_result)
    effects = structured.get("effects", {}) if isinstance(structured.get("effects"), dict) else {}
    if not effects:
        effects = action_result.effects
    if not effects:
        effects = {"dimensions": {"world": formula_effects}}
        _settle_cid = getattr(state, '_runtime_chat_id', '')
        if _settle_cid:
            get_tracer(_settle_cid, enabled=state.trace_enabled).record(
                'formula_fallback', '_settle_backlog_item',
                {'item_id': str(normalized_item.get('id', '')), 'category': category, 'outcome': outcome, 'formula_effects': formula_effects},
                GameTracer.state_summary(state))
    narrative = str(structured.get("narrative_hint") or await narrate_action(state, f"{proposal.description}\n判定：{outcome}"))
    return {
        "item": normalized_item,
        "dice_result": _serialize_record(judgment) or {},
        "outcome": outcome,
        "effects": effects,
        "narrative": narrative,
        "public_info": structured.get("public_info", []),
    }


async def _judge_npc_interference(state: GameState, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """调用LLM判断NPC是否干扰结算结果。"""

    if not results or not state.active_npcs:
        return []
    payload = {
        "results": [{"item": result.get("item"), "outcome": result.get("outcome"), "dice": result.get("dice_result"), "effects": result.get("effects")} for result in results],
        "npcs": state.active_npcs,
        "dimensions": get_state_manager().get_dimension_description(state),
    }
    try:
        data = await get_llm_client().chat_json(NPC_INTERFERENCE_PROMPT, json.dumps(payload, ensure_ascii=False), temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NPC干扰判定失败，跳过: %s", exc)
        return []
    if not isinstance(data, dict):
        return []
    return [item for item in _as_list(data.get("interferences")) if isinstance(item, dict)]


def _apply_interferences_to_results(state: GameState, results: list[dict[str, Any]], interferences: list[dict[str, Any]]) -> None:
    """把NPC干扰写入结算结果。"""

    by_id = {str(result.get("item", {}).get("id")): result for result in results if isinstance(result.get("item"), dict)}
    for interference in interferences:
        result = by_id.get(str(interference.get("item_id", "")))
        if not result:
            continue
        result["interference"] = interference
        interference_type = str(interference.get("type", ""))
        if interference_type == "flip" and result.get("outcome") in {"成功", "大成功"}:
            result["outcome"] = "失败"
            result["effects"] = {}
        elif interference_type == "weaken":
            result["effects"] = _weaken_dimension_effects(result.get("effects", {}))
        elif interference_type == "condition" and isinstance(interference.get("added_backlog"), dict):
            state.backlog.append(_normalize_candidate_item(state, interference["added_backlog"], "passive_event"))


def _apply_settlement_effects(state: GameState, results: list[dict[str, Any]]) -> None:
    """统一应用所有结算效果。"""

    merged_effects = _merge_effects_for_settlement([result.get("effects") for result in results])
    if merged_effects:
        get_state_manager().apply_effects(state, merged_effects)


def _merge_effects_for_settlement(raw_effects: list[Any]) -> dict[str, Any]:
    """合并结算效果，维度先累加净值再应用。"""

    merged: dict[str, Any] = {"dimensions": {"world": {}, "character": {}, "extensions": {}, "relations": {}}}
    passthrough_lists = {
        "add_consequence_seeds": [],
        "add_information": [],
        "add_active_events": [],
        "add_growth_log": [],
    }
    for effects in raw_effects:
        if not isinstance(effects, dict):
            continue
        dimensions = effects.get("dimensions") if isinstance(effects.get("dimensions"), dict) else {}
        for category in ("world", "character", "extensions", "extension"):
            source_category = dimensions.get(category) if isinstance(dimensions.get(category), dict) else {}
            target_key = "extensions" if category == "extension" else category
            for name, delta in source_category.items():
                if isinstance(delta, int):
                    target = merged["dimensions"].setdefault(target_key, {})
                    target[str(name)] = target.get(str(name), 0) + delta
        relations = dimensions.get("relations") if isinstance(dimensions.get("relations"), dict) else {}
        for npc_id, patch in relations.items():
            if not isinstance(patch, dict):
                continue
            target_relation = merged["dimensions"]["relations"].setdefault(str(npc_id), {})
            values = patch.get("values") if isinstance(patch.get("values"), dict) else patch
            for name, delta in values.items():
                if isinstance(delta, int):
                    target_relation[str(name)] = target_relation.get(str(name), 0) + delta
        for key, target_list in passthrough_lists.items():
            value = effects.get(key)
            if isinstance(value, list):
                target_list.extend(item for item in value if isinstance(item, dict))
        for key in ("decision_record", "current_scene", "player_goal", "npc", "active_npcs"):
            if key in effects and key not in merged:
                merged[key] = effects[key]

    for key, value in passthrough_lists.items():
        if value:
            merged[key] = value
    dimensions = merged.get("dimensions", {})
    if isinstance(dimensions, dict):
        merged["dimensions"] = {key: value for key, value in dimensions.items() if value}
        if not merged["dimensions"]:
            merged.pop("dimensions", None)
    return merged


def _tick_delayed_queue(state: GameState) -> list[dict[str, Any]]:
    """推进延迟队列并取出到期项。"""

    due: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for item in state.delayed_queue:
        delayed_item = dict(item)
        delayed_item["remaining_advances"] = _safe_int(delayed_item.get("remaining_advances"), 0) - 1
        if delayed_item["remaining_advances"] <= 0:
            due.append(delayed_item)
        else:
            remaining.append(delayed_item)
    state.delayed_queue = remaining
    return due


async def _resolve_due_action(chat_id: str, state: GameState, due_action: Mapping[str, Any]) -> None:
    """处理推进后到期的行动。"""

    proposal_payload = due_action.get("proposal") if isinstance(due_action.get("proposal"), dict) else due_action
    proposal = _proposal_from_payload(proposal_payload)
    result = execute_action(proposal, state)
    structured = await _generate_action_effects(state, result)
    result.effects.update(structured.get("effects", {}))
    get_state_manager().apply_effects(state, result.effects)
    growth_record = apply_action_growth(state, result.judgment.result_tier, proposal.main_dimension)
    narration = await narrate_action(state, f"到期行动：{proposal.description}\n判定：{result.judgment.result_tier}")
    await send_narrator_card(chat_id, build_dice_result_card(result, narration=narration, growth=_serialize_record(growth_record) if growth_record else None))
    _append_history(state, "assistant", narration, "narrator")


async def _handle_triggered_event(chat_id: str, state: GameState, event: Mapping[str, Any]) -> None:
    """把触发事件转成被动响应卡片。"""

    event_id = str(event.get("event_id") or event.get("id") or "").strip()
    if event_id:
        state.event_history.append(event_id)
    state.active_events.append(dict(event))
    description = str(event.get("description") or event.get("name") or "突发事件逼近。")
    responses = _suggested_responses_from_event(event)
    passive_payload = create_passive_response(description, responses, state)
    await send_narrator_card(chat_id, build_passive_response_card(passive_payload))


def _suggested_responses_from_event(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """从事件定义生成被动响应候选。"""

    raw = event.get("suggested_responses") or event.get("recommended_responses") or event.get("directions")
    responses: list[dict[str, Any]] = []
    for item in _as_list(raw):
        if isinstance(item, dict):
            responses.append(item)
        else:
            responses.append({"description": str(item), "dc": 8, "main_dimension": "意志", "time_cost": 0})
    if not responses:
        responses = [
            {"description": "稳住局势，先控制损害", "dc": 8, "main_dimension": "意志", "time_cost": 0},
            {"description": "查清真相，再决定下一步", "dc": 8, "main_dimension": "感知", "time_cost": 1},
        ]
    return responses


async def _check_bottom_line_and_maybe_end(chat_id: str, state: GameState) -> bool:
    """每次推进或结算后检查底线，击穿则进入结算。"""

    broken = await check_bottom_lines(state)
    if not broken:
        return False
    ending_id = str(broken.get("ending_id") or broken.get("bottom_line_id") or "bottom_line_failure")
    state.phase = "ended"
    narration = await generate_ending_narration(state, ending_id)
    await send_narrator_card(chat_id, build_narration_card("结局", narration))
    _append_history(state, "assistant", narration, "narrator")
    return True


async def _start_talk(chat_id: str, state: GameState, npc_name: str) -> None:
    """进入与指定NPC的连续谈话。"""

    if not npc_name:
        return
    loc_data = state.npc_locations.get(npc_name, {})
    reachability = loc_data.get("reachability", "present")
    if reachability == "unreachable":
        reason = loc_data.get("reason", "无法联系")
        msg = f"无法与{npc_name}对话：{reason}"
        await send_narrator_card(chat_id, build_narration_card("无法召见", msg))
        _append_history(state, "assistant", msg, "narrator")
        return
    if reachability == "distant":
        recall_cost = loc_data.get("recall_cost", 2)
        location = loc_data.get("location", "远方")
        msg = f"{npc_name}目前在{location}，需{recall_cost}回合才能到达。请先通过自由输入发出召回令。"
        await send_narrator_card(chat_id, build_narration_card("距离过远", msg))
        _append_history(state, "assistant", msg, "narrator")
        return
    state.talking_to = npc_name
    state.present_npcs = [npc_name]
    state.conversation_initiator = "player"
    state.npc_join_round = {npc_name: 0}
    state.current_talk_history = []
    state.phase = "talking"
    # 从turn_agenda匹配当前NPC相关的议题
    state.conversation_topic = ""
    for agenda_item in state.turn_agenda:
        if not isinstance(agenda_item, dict):
            continue
        relevant = agenda_item.get("relevant_npcs", [])
        if isinstance(relevant, list) and npc_name in relevant:
            state.conversation_topic = str(agenda_item.get("description", "")).strip()
            break
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record("route", "_start_talk", {"npc_name": npc_name}, GameTracer.state_summary(state))
    if state.conversation_topic:
        msg = f"召见{npc_name}，议题：{state.conversation_topic}"
    else:
        msg = f"召见{npc_name}，请告知所议何事。"
    await send_narrator_card(chat_id, build_narration_card("召见", msg))
    _append_history(state, "assistant", msg, "narrator")


async def _handle_gm_selection(chat_id: str, state: GameState, action_value: dict[str, Any]) -> None:
    """处理玩家在GM解读卡片上的选择。"""
    interpretation = state.pending_gm_interpretation
    if not interpretation:
        return
    options = interpretation.get("options", [])
    option_index = int(action_value.get("option_index", 0))
    if option_index < 0 or option_index >= len(options):
        return
    option = options[option_index]
    opt_type = str(option.get("type", "action")).strip()
    targets = option.get("target_npcs", []) if isinstance(option.get("target_npcs"), list) else []
    description = str(option.get("description", "")).strip()
    state.pending_gm_interpretation = {}
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record("route", "_handle_gm_selection", {"option_index": option_index, "opt_type": opt_type, "targets": targets, "description": description}, GameTracer.state_summary(state))

    if opt_type == "talk" and targets:
        actual = await _ensure_npc_exists(state, str(targets[0]).strip(), chat_id)
        if actual is None:
            msg = f"无法找到或创建角色：{targets[0]}"
            await send_narrator_card(chat_id, build_narration_card("角色不存在", msg))
            _append_history(state, "assistant", msg, "narrator")
            return
        targets[0] = actual
        await _start_talk(chat_id, state, str(targets[0]).strip())
    elif opt_type == "summon" and targets:
        target_npc = str(targets[0]).strip()
        recall_cost = max(1, int(option.get("time_cost", 2) or 2))
        state.delayed_queue.append({
            "id": str(uuid.uuid4()),
            "type": "summon",
            "description": f"召回{target_npc}",
            "npc": target_npc,
            "remaining_advances": recall_cost,
            "source": "gm_interpret",
            "duration": "delayed",
            "delay_count": recall_cost,
            "removable": True,
            "severity_warning": "",
            "deterministic": True,
            "deterministic_effects": {},
            "dc": 0,
            "modifier": 0,
            "main_dimension": "",
            "auxiliary_dimensions": [],
            "tags": [],
            "npc_id": target_npc,
            "success_rate": 1.0,
            "predicted_effects": {},
        })
        msg = f"已发出召令，{target_npc}预计{recall_cost}回合后抵达。"
        await send_narrator_card(chat_id, build_narration_card("召回令", msg))
        _append_history(state, "assistant", msg, "narrator")
        await _start_turn_visits_or_panel(chat_id, state)
    elif opt_type == "query":
        await _route_query(chat_id, state, description or "查询")
    elif opt_type == "multi_talk" and targets:
        topic = description or "议事"
        scene_type = "formal" if any(k in topic for k in ("朝", "议", "军", "政")) else "private"
        await _init_court_session(chat_id, state, [str(t).strip() for t in targets], topic, scene_type)
    else:
        intent_result = {"intent": "action", "detail": description}
        await _route_action(chat_id, state, description, intent_result)


async def _handle_opening_selection(chat_id: str, state: GameState, action_value: dict[str, Any]) -> None:
    """处理开局行为模式选择，路由到对应功能。"""

    route = str(action_value.get("route", "")).strip()
    mode_id = str(action_value.get("mode_id", "")).strip()
    route_params = action_value.get("route_params", {})
    if not isinstance(route_params, dict):
        route_params = {}

    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record(
            "route", "_handle_opening_selection",
            {"mode_id": mode_id, "route": route},
            GameTracer.state_summary(state),
        )

    state.pending_gm_interpretation = {}

    if route == "status_briefing":
        query_label = str(action_value.get("label", "查看当前奏章与各地急报")).strip()
        answer = await narrate_opening_briefing(state, query_label)
        forces_summary = f"{state.world_summary or str(state.current_scene.get('context', ''))}\n\n{answer}"
        await send_narrator_card(chat_id, build_situation_card(_world_state_payload(state), forces_summary, state.backlog, state.delayed_queue))
        _append_history(state, "assistant", forces_summary, "narrator")
        await _finish_opening(chat_id, state)

    elif route == "court_session":
        await _finish_opening(chat_id, state)
        topic = str(route_params.get("topic", "开局议事")).strip()
        target_npcs = route_params.get("target_npcs", [])
        if not isinstance(target_npcs, list):
            target_npcs = []
        if not target_npcs:
            target_npcs = [
                name for name, loc in state.npc_locations.items()
                if isinstance(loc, dict) and loc.get("reachability") == "present"
            ][:3]
        scene_type = "formal" if any(k in topic for k in ("朝", "议", "军", "政", "阁")) else "private"
        await _init_court_session(chat_id, state, target_npcs, topic, scene_type)

    elif route == "npc_select":
        await _finish_opening(chat_id, state)
        npc_list = [
            name for name, loc in state.npc_locations.items()
            if isinstance(loc, dict) and loc.get("reachability") in ("present", "nearby")
        ]
        elements: list[dict[str, Any]] = [
            {"tag": "markdown", "content": "**选择要召见的人：**"},
        ]
        buttons = []
        for npc_name in npc_list:
            buttons.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": f"召见{npc_name}"},
                "type": "default",
                "value": {"action": "talk", "npc": npc_name},
            })
        if buttons:
            elements.append({"tag": "action", "actions": buttons})
        else:
            elements.append({"tag": "markdown", "content": "当前无可召见之人。"})
        card = {
            "config": {"wide_screen_mode": True, "enable_forward": True},
            "header": {"template": "blue", "title": {"tag": "plain_text", "content": "选人"}},
            "elements": elements,
        }
        await send_narrator_card(chat_id, card)

    else:
        await _send_status_report(chat_id, state)
        await _finish_opening(chat_id, state)


async def _finish_opening(chat_id: str, state: GameState) -> None:
    """开局收尾：首次展示棋盘迷雾 + 进入正式游戏。"""

    await _send_status_report(chat_id, state, first_time=True)
    state.phase = "free"


async def _init_court_session(chat_id: str, state: GameState, target_npcs: list[str], topic: str, scene_type: str = "formal") -> None:
    """初始化朝会/多人对话。"""
    present = []
    pending = []
    skipped = []
    resolved_npcs = []
    for npc in target_npcs:
        actual_name = await _ensure_npc_exists(state, npc, chat_id)
        if actual_name is None:
            skipped.append(npc)
            logger.warning("朝会目标NPC无法识别或生成失败，已跳过: %s", npc)
            continue
        resolved_npcs.append(actual_name)

    for npc in resolved_npcs:
        loc = state.npc_locations.get(npc, {})
        if not loc:
            skipped.append(npc)
            logger.warning("朝会目标NPC不存在于系统中，已跳过: %s", npc)
            continue
        reach = loc.get("reachability", "distant")
        if reach == "unreachable":
            continue
        if reach == "present":
            present.append(npc)
        else:
            eta = loc.get("recall_cost", 2)
            pending.append({"name": npc, "eta": eta})
            state.delayed_queue.append({
                "id": str(uuid.uuid4()),
                "type": "summon",
                "description": f"召回{npc}参加议事",
                "npc": npc,
                "remaining_advances": eta,
                "source": "court_session",
                "duration": "delayed",
                "delay_count": eta,
                "removable": False,
                "severity_warning": "",
                "deterministic": True,
                "deterministic_effects": {},
                "dc": 0,
                "modifier": 0,
                "main_dimension": "",
                "auxiliary_dimensions": [],
                "tags": [],
                "npc_id": npc,
                "success_rate": 1.0,
                "predicted_effects": {},
            })
    if not present:
        msg = "所有指定NPC当前不在身边，无法立即开始议事。" + (f"已派人召回{'、'.join(p['name'] for p in pending)}。" if pending else "")
        await send_narrator_card(chat_id, build_narration_card("无法召开", msg))
        _append_history(state, "assistant", msg, "narrator")
        await _start_turn_visits_or_panel(chat_id, state)
        return
    state.talking_to = ""
    state.court_session = {
        "active": True,
        "npcs": present,
        "topic": topic,
        "scene_type": scene_type,
        "round": 0,
        "history": [],
        "addressed_npc": "",
    }
    state.phase = "court"
    msg = f"朝会议事已开启，议题：{topic}。在场：{'、'.join(present)}。"
    if pending:
        msg += f" 已派人召回：{'、'.join(p['name'] for p in pending)}。"
    if skipped:
        msg += f" （{', '.join(skipped)}不在朝中，无法参与。）"
    await send_narrator_card(chat_id, build_narration_card("议事开启", msg))
    _append_history(state, "assistant", msg, "narrator")
    card = build_court_session_card(topic, 0, [], present, pending)
    await send_narrator_card(chat_id, card)


async def _handle_court_message(chat_id: str, state: GameState, player_text: str) -> None:
    """处理朝会中玩家的发言，裁定NPC发言序列并依次生成回复。"""
    court = state.court_session
    if not court.get("active"):
        return
    npcs = court.get("npcs", [])
    topic = court.get("topic", "")
    scene_type = court.get("scene_type", "formal")
    addressed = court.get("addressed_npc", "")
    court["round"] = court.get("round", 0) + 1
    round_num = court["round"]
    court.setdefault("history", []).append({"round": round_num, "speaker": "玩家", "mode": "speak", "content": player_text})
    last_round = [h for h in court.get("history", []) if h.get("round") == round_num - 1]
    last_summary = "; ".join(f"{h['speaker']}：{h['content'][:30]}" for h in last_round if h.get("content")) if last_round else "无"
    npc_briefs = []
    for n in npcs:
        title = _npc_title(state, n)
        loc_info = state.npc_locations.get(n, {})
        faction = loc_info.get("faction", "")
        npc_info = state.active_npcs.get(state.cast.get(n, n), {})
        if not npc_info:
            for cid, cinfo in state.active_npcs.items():
                if str(cinfo.get("name", "")) == n:
                    npc_info = cinfo
                    break
        traits = "、".join(npc_info.get("personality_traits", [])) if npc_info.get("personality_traits") else ""
        tendency = str(npc_info.get("behavioral_tendencies", ""))[:80]
        brief_parts = [f"{n}（{title}，{faction}）"]
        if traits:
            brief_parts.append(f"  性格：{traits}")
        if tendency:
            brief_parts.append(f"  行为倾向：{tendency}")
        npc_briefs.append("\n".join(brief_parts))
    role_info = state.active_npcs.get(state.cast.get(state.player_role, state.player_role), {})
    player_role_desc = str(role_info.get("identity", "") or state.player_role)
    arbiter_prompt = COURT_ARBITER_PROMPT.format(
        player_role=player_role_desc,
        npc_list_with_brief="\n".join(npc_briefs),
        topic=topic,
        scene_type=scene_type,
        player_text=player_text,
        last_round_summary=last_summary,
        addressed_npc=addressed or "未点名",
    )
    llm = get_llm_client()
    try:
        raw = await llm.chat(arbiter_prompt, player_text, temperature=0.4)
        raw_text = raw if isinstance(raw, str) else str(raw)
        import re as _re
        json_match = _re.search(r"\[.*\]", raw_text, _re.DOTALL)
        if json_match:
            speaking_order = json.loads(json_match.group())
        else:
            speaking_order = [{"npc": n, "mode": "active", "reason": "默认发言"} for n in npcs]
    except Exception:
        logger.exception("朝会裁定LLM调用失败，使用默认顺序")
        speaking_order = [{"npc": n, "mode": "active", "reason": "默认发言"} for n in npcs]
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record_llm_call("_handle_court_message.arbiter", "COURT_ARBITER_PROMPT", COURT_ARBITER_PROMPT, {"topic": topic, "npcs": npcs, "player_text": player_text}, str(speaking_order)[:2000], speaking_order, GameTracer.state_summary(state))
    responses = []
    prior_speakers = []
    for entry in speaking_order:
        npc_name = str(entry.get("npc", "")).strip()
        mode = str(entry.get("mode", "silent")).strip()
        if npc_name not in npcs:
            continue
        if mode == "silent":
            expression = str(entry.get("expression", "沉默不语"))
            responses.append({"npc": npc_name, "mode": "silent", "content": "", "expression": expression})
            court["history"].append({"round": round_num, "speaker": npc_name, "mode": "silent", "content": expression})
            continue
        npc_profile = await _resolve_npc_profile(state, npc_name)
        court_ctx = {
            "scene_type": scene_type,
            "topic": topic,
            "present_npcs": npcs,
            "mode": mode,
            "reason": str(entry.get("reason", "")),
            "prior_speakers": list(prior_speakers),
        }
        reply = await generate_npc_reply(state, npc_name, npc_profile, player_text, court_context=court_ctx)
        responses.append({"npc": npc_name, "mode": mode, "content": reply, "expression": ""})
        prior_speakers.append({"npc": npc_name, "content": reply})
        court["history"].append({"round": round_num, "speaker": npc_name, "mode": mode, "content": reply})
        _append_history(state, "assistant", reply, npc_name)
    court["addressed_npc"] = ""
    pending = [{"name": item["npc"], "eta": item.get("remaining_advances", 1)} for item in state.delayed_queue if item.get("type") == "summon" and item.get("source") == "court_session"]
    card = build_court_session_card(topic, round_num, responses, npcs, pending)
    await send_narrator_card(chat_id, card)
    all_npc_replies = ' '.join(r.get('content', '')[:100] for r in responses if r.get('mode') != 'silent')
    if all_npc_replies:
        court_detected = await _gm_decision_scan(state, player_text, all_npc_replies, '朝会')
        if court_detected:
            await send_narrator_card(chat_id, build_decision_card(court_detected, '朝会'))
    state.turn += 1
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record_turn_snapshot(state)


async def _end_court_session(chat_id: str, state: GameState) -> None:
    """结束朝会，LLM提炼对话内容，展示候选行动，归档NPC记忆。"""
    court = state.court_session
    if not court.get("active"):
        return
    history = court.get("history", [])
    topic = court.get("topic", "")
    npcs = court.get("npcs", [])

    # 转换朝会历史为 talk_history 格式
    talk_history: list[dict] = []
    for h in history:
        speaker = h.get("speaker", "")
        mode = h.get("mode", "")
        content = h.get("content", "")
        if speaker == "玩家" or speaker == "player":
            talk_history.append({"role": "user", "content": content})
        elif mode == "silent":
            talk_history.append({"role": "assistant", "content": f"{speaker}（沉默，{content}）"})
        else:
            talk_history.append({"role": "assistant", "content": f"{speaker}：{content}"})

    # LLM提炼
    summary = await _summarize_talk(state, talk_history, npc_names=npcs)
    if summary:
        _update_board_from_key_facts(state, summary.get("key_facts", []))

    # 归档每个NPC的三层记忆
    for npc_name in npcs:
        _archive_talk_memory(state, npc_name, talk_history, summary)

    # 重置朝会状态
    state.court_session = {}
    state.phase = "free"

    # 展示摘要卡 + 候选行动
    summary_text = str(summary.get("summary", "")).strip() if summary else f"议题「{topic}」议事结束，未整理出明确候选待办。"
    if not summary_text:
        summary_text = f"议题「{topic}」议事结束。"
    candidate_items = [_normalize_candidate_item(state, item, "dialogue_summary") for item in _as_list(summary.get("candidate_items")) if isinstance(item, dict)] if summary else []
    candidates = [{"item": item, "status": "pending"} for item in candidate_items]
    npc_label = "、".join(npcs)
    state.dialogue_summary = {"npc_name": npc_label, "summary_text": summary_text, "candidates": candidates}
    await send_narrator_card(chat_id, build_dialogue_summary_card(npc_label, summary_text, candidates))
    _append_history(state, "assistant", f"朝会纪要（{npc_label}）：{summary_text}", "narrator")
    if not candidate_items:
        state.dialogue_summary = {}
        await _start_turn_visits_or_panel(chat_id, state)


async def _end_talk(chat_id: str, state: GameState) -> None:
    """结束连续谈话，发送候选待办汇总卡。"""

    npc_name = state.talking_to
    present_npcs = _normalize_present_npcs(state)
    talk_history = list(state.current_talk_history)
    summary = await _summarize_talk(state, talk_history, npc_names=present_npcs if len(present_npcs) > 1 else None)
    if summary:
        _update_board_from_key_facts(state, summary.get("key_facts", []))
    archive_targets = present_npcs or ([npc_name] if npc_name else [])
    for archive_npc in archive_targets:
        _archive_talk_memory(state, archive_npc, talk_history, summary)
    state.talking_to = ""
    state.conversation_topic = ""
    state.present_npcs = []
    state.conversation_initiator = ""
    state.npc_join_round = {}
    state.current_talk_history = []
    state.phase = "free"
    state.settlement_phase = "free"
    summary_text = str(summary.get("summary", "")).strip() if summary else "对话结束，未整理出明确候选待办。"
    candidate_items = [_normalize_candidate_item(state, item, "dialogue_summary") for item in _as_list(summary.get("candidate_items")) if isinstance(item, dict)] if summary else []
    candidates = [{"item": item, "status": "pending"} for item in candidate_items]
    npc_label = "、".join(archive_targets) if archive_targets else npc_name
    state.dialogue_summary = {"npc_name": npc_label, "summary_text": summary_text, "candidates": candidates}
    await send_narrator_card(chat_id, build_dialogue_summary_card(npc_label, summary_text, candidates))
    _append_history(state, "assistant", f"与{npc_label}谈话纪要：{summary_text}", "narrator")
    if not candidate_items:
        state.dialogue_summary = {}
        await _start_turn_visits_or_panel(chat_id, state)


async def _summarize_talk(state: GameState, talk_history: list[dict], *, npc_names: list[str] | None = None) -> dict[str, Any]:
    """总结谈话（单人或朝会），并提取玩家明确决策。"""

    if len(talk_history) < 2:
        return {}

    talking_label = "、".join(npc_names) if npc_names else state.talking_to
    payload = {
        "talking_to": talking_label,
        "npc_name": talking_label,
        "turn": state.turn,
        "scene": state.current_scene,
        "talk_history": talk_history,
        "dimension_description": get_state_manager().get_dimension_description(state),
    }
    try:
        result = await get_llm_client().chat_json(TALK_SUMMARY_PROMPT, json.dumps(payload, ensure_ascii=False), temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("对话总结失败，跳过决策收集: %s", exc)
        return {}
    if not isinstance(result, dict):
        return {}
    raw_candidates = result.get("candidate_items") or result.get("decisions") or result.get("actions") or result.get("candidates") or []
    candidate_items = [item for item in _as_list(raw_candidates) if isinstance(item, dict)]
    key_facts = [item for item in _as_list(result.get("key_facts")) if isinstance(item, dict)]
    public_info = [item for item in _as_list(result.get("public_info")) if isinstance(item, dict)]
    return {
        "summary": str(result.get("summary", "")).strip(),
        "key_facts": key_facts,
        "candidate_items": candidate_items,
        "public_info": public_info,
    }


def _archive_talk_memory(state: GameState, npc_name: str, talk_history: list[dict], summary: Mapping[str, Any]) -> None:
    """把一次单独谈话归档到该NPC三层记忆和世界公共记忆。"""

    if not talk_history:
        return
    npc_id = _npc_memory_id(state, npc_name)
    npc_mem = _ensure_npc_memory(state, npc_id)
    summary_text = str(summary.get("summary", "")).strip() if isinstance(summary, Mapping) else ""
    round_number = state.turn
    for fact in _as_list(summary.get("key_facts") if isinstance(summary, Mapping) else []):
        if not isinstance(fact, Mapping):
            continue
        content = str(fact.get("content", "")).strip()
        if not content:
            continue
        fact_type = str(fact.get("type") or "information").strip()
        if fact_type not in {"promise", "attitude", "information", "request", "secret", "conflict"}:
            fact_type = "information"
        record: dict[str, Any] = {"id": str(fact.get("id") or uuid.uuid4()), "type": fact_type, "content": content, "round": round_number}
        if fact_type == "promise":
            record["fulfilled"] = bool(fact.get("fulfilled", False))
        npc_mem["key_facts"].append(record)
    if summary_text:
        npc_mem["talk_summaries"].append({"round": round_number, "summary": summary_text})
    npc_mem["recent_talks"].append({"round": round_number, "messages": [dict(item) for item in talk_history if isinstance(item, dict)]})
    npc_mem["recent_talks"] = npc_mem["recent_talks"][-2:]
    if isinstance(summary, Mapping):
        _append_world_memory(state, summary.get("public_info"), round_number)


def _ensure_npc_memory(state: GameState, npc_id: str) -> dict[str, list[dict]]:
    """确保某个NPC拥有完整三层记忆容器。"""

    memory = state.npc_memory.get(npc_id)
    if not isinstance(memory, dict):
        memory = {}
        state.npc_memory[npc_id] = memory
    for key in ("key_facts", "talk_summaries", "recent_talks"):
        if not isinstance(memory.get(key), list):
            memory[key] = []
    return memory  # type: ignore[return-value]


def _append_world_memory(state: GameState, items: Any, round_number: int | None = None) -> None:
    """追加公开世界记忆，忽略空内容。"""

    for item in _as_list(items):
        if not isinstance(item, Mapping):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        state.world_memory.append(
            {
                "round": _safe_int(item.get("round"), state.turn if round_number is None else round_number),
                "type": str(item.get("type") or "event"),
                "content": content,
            }
        )


def _npc_memory_id(state: GameState, npc_name: str) -> str:
    """将NPC展示名稳定映射为记忆键。"""

    npc_name = str(npc_name).strip()
    if npc_name in state.cast:
        return str(state.cast[npc_name])
    if npc_name in state.active_npcs:
        return npc_name
    for npc_id, npc in state.active_npcs.items():
        if str(npc.get("name", "")).strip() == npc_name:
            return str(npc_id)
    return npc_name


async def _update_memory_after_settlement(state: GameState, results: list[dict[str, Any]], interferences: list[dict[str, Any]]) -> None:
    """结算后更新承诺兑现、公开结果和NPC冲突记忆。"""

    _record_interference_memories(state, interferences)
    if not results:
        return
    promises = _pending_promise_records(state)
    payload = {
        "turn": state.turn,
        "promises": promises,
        "results": [
            {
                "item": result.get("item"),
                "outcome": result.get("outcome"),
                "narrative": result.get("narrative"),
                "effects": result.get("effects"),
                "public_info": result.get("public_info"),
                "interference": result.get("interference"),
            }
            for result in results
        ],
    }
    try:
        data = await get_llm_client().chat_json(SETTLEMENT_MEMORY_PROMPT, json.dumps(payload, ensure_ascii=False), temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("结算记忆更新失败，跳过公开结果和承诺兑现: %s", exc)
        return
    if not isinstance(data, dict):
        return
    _mark_fulfilled_promises(state, _as_list(data.get("fulfilled_promise_ids")))
    _append_world_memory(state, data.get("public_info"), state.turn)
    for result in results:
        _append_world_memory(state, result.get("public_info"), state.turn)


def _pending_promise_records(state: GameState) -> list[dict[str, Any]]:
    """收集所有未兑现承诺，供结算记忆裁决。"""

    promises: list[dict[str, Any]] = []
    for npc_id, memory in state.npc_memory.items():
        if not isinstance(memory, Mapping):
            continue
        for fact in _as_list(memory.get("key_facts")):
            if not isinstance(fact, Mapping):
                continue
            if fact.get("type") == "promise" and not bool(fact.get("fulfilled", False)):
                promises.append({"npc_id": npc_id, "id": str(fact.get("id", "")), "content": str(fact.get("content", "")), "round": fact.get("round")})
    return promises


def _mark_fulfilled_promises(state: GameState, fulfilled_ids: list[Any]) -> None:
    """按事实id标记承诺已兑现。"""

    id_set = {str(item) for item in fulfilled_ids if str(item).strip()}
    if not id_set:
        return
    for memory in state.npc_memory.values():
        if not isinstance(memory, Mapping):
            continue
        for fact in _as_list(memory.get("key_facts")):
            if isinstance(fact, dict) and str(fact.get("id")) in id_set and fact.get("type") == "promise":
                fact["fulfilled"] = True


def _record_interference_memories(state: GameState, interferences: list[dict[str, Any]]) -> None:
    """把NPC干扰写成该NPC的冲突事实。"""

    for interference in interferences:
        if not isinstance(interference, Mapping):
            continue
        npc_name = str(interference.get("npc", "")).strip()
        if not npc_name:
            continue
        npc_mem = _ensure_npc_memory(state, _npc_memory_id(state, npc_name))
        reason = str(interference.get("reason") or "干扰了本次结算行动").strip()
        npc_mem["key_facts"].append({"id": str(uuid.uuid4()), "type": "conflict", "content": reason, "round": state.turn})


async def _abandon_action(chat_id: str, state: GameState, proposal_id: str) -> None:
    """放弃待确认行动。"""

    if proposal_id:
        state.pending_actions = [item for item in state.pending_actions if str(item.get("proposal_id", "")) != proposal_id]
    else:
        state.pending_actions = []
    msg = "行动已放弃，局势暂无实质变化。"
    await send_narrator_card(chat_id, build_narration_card("行动放弃", msg))
    _append_history(state, "assistant", msg, "narrator")
    await _start_turn_visits_or_panel(chat_id, state)


def _build_dimension_summary_with_desc(state: GameState) -> str:
    """构建带定义解释的国势维度摘要。"""
    state_manager = get_state_manager()
    trends = state_manager.get_dimension_trends(state)
    weak_names = {str(item.get("name")) for item in state_manager.get_weak_dimensions(state)}
    lines = []
    for name, value in state.dimensions.world.items():
        clamped = max(0, min(10, int(value)))
        bar = "█" * clamped + "░" * (10 - clamped)
        trend = trends.get(name, "━")
        warn = " ⚠" if name in weak_names else ""
        info = DIMENSION_IMPACT.get(name)
        desc = f"({info['desc']})" if info else ""
        low_hint = ""
        if info and clamped <= 3:
            low_hint = f" → {info['low']}"
        elif info and clamped >= 8:
            high_val = info.get("high", "")
            if high_val:
                low_hint = f" → {high_val}"
        lines.append(f"{name}{desc} {bar} {value}/10 {trend}{warn}{low_hint}")
    return "\n".join(lines) or "世界维度未明"


async def _send_status_report(chat_id: str, state: GameState, first_time: bool = False) -> None:
    """发送棋盘总览。"""
    dimension_summary = _build_dimension_summary_with_desc(state)
    core_goal = str(state.player_goal.get("core_goal", "")).strip() if state.player_goal else ""
    await send_narrator_card(chat_id, build_board_card(
        _format_state_time(state),
        dimension_summary,
        state.board_regions,
        state.delayed_queue,
        state.turn_agenda if hasattr(state, 'turn_agenda') else [],
        state.discovered_clues if hasattr(state, 'discovered_clues') else [],
        first_time=first_time,
        campaign_goal=core_goal,
    ))


async def _check_board_fog_gates(state: GameState) -> list[dict]:
    """解锁达到世界维度门槛的棋盘迷雾。"""

    newly_unlocked: list[dict] = []
    regions = state.board_regions if isinstance(state.board_regions, dict) else {}
    for region_id, region in regions.items():
        if not isinstance(region, dict):
            continue
        region_name = str(region.get("name", region_id)).strip()
        layers = region.get("layers") if isinstance(region.get("layers"), list) else []
        news = region.setdefault("news", [])
        if not isinstance(news, list):
            news = []
            region["news"] = news
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            fog = layer.get("fog")
            if not isinstance(fog, dict) or fog.get("unlocked") is not False:
                continue
            condition = fog.get("unlock_condition")
            if not isinstance(condition, dict) or "dimension" not in condition or "threshold" not in condition:
                continue
            dim_name = str(condition.get("dimension", "")).strip()
            if not dim_name:
                continue
            threshold = _safe_int(condition.get("threshold"), 0)
            if state.dimensions.world.get(dim_name, 0) < threshold:
                continue

            fog["unlocked"] = True
            layer_label = str(layer.get("label", "")).strip()
            unlock_text = str(fog.get("unlock_text") or "").strip()
            # E3: LLM动态生成
            if not unlock_text:
                try:
                    state_mgr = get_state_manager()
                    dim_desc = state_mgr.get_dimension_description(state)
                    prompt_text = FOG_UNLOCK_GENERATE_PROMPT.format(
                        region_name=region_name,
                        layer_label=layer_label,
                        fog_hint=str(fog.get("hint", "")),
                        game_time=_format_state_time(state),
                        dimension_summary=dim_desc,
                    )
                    unlock_text = await get_llm_client().chat(prompt_text, "", temperature=0.5)
                    unlock_text = unlock_text.strip()
                except Exception:
                    unlock_text = str(fog.get("hint", "情报解锁，详情待查"))
            if unlock_text:
                known_text = str(layer.get("known_text") or "").strip()
                layer["known_text"] = f"{known_text}\n🔓 {unlock_text}" if known_text else f"🔓 {unlock_text}"
            news.append(f"🔓 {region_name}·{layer_label}：迷雾已揭开（{dim_name}≥{threshold}）")
            newly_unlocked.append(
                {
                    "region": region_name,
                    "layer": layer_label,
                    "content": unlock_text or str(fog.get("hint", "")),
                    "condition": f"{dim_name}≥{threshold}",
                    "turn": state.turn,
                }
            )

    state.discovered_clues.extend(newly_unlocked)
    return newly_unlocked


def _update_board_from_key_facts(state: GameState, key_facts: list[dict]) -> None:
    """把对话提炼出的区域事实写入棋盘信息流。"""

    for fact in key_facts:
        if not isinstance(fact, dict):
            continue
        content = str(fact.get("content") or "").strip()
        if not content:
            continue
        regions = state.board_regions if isinstance(state.board_regions, dict) else {}
        for region_id, region in regions.items():
            if not isinstance(region, dict):
                continue
            region_name = str(region.get("name", region_id)).strip()
            if not region_name or region_name not in content:
                continue
            layers = region.get("layers") if isinstance(region.get("layers"), list) else []
            target_layer = next((layer for layer in layers if isinstance(layer, dict)), None)
            if target_layer is None:
                continue
            events = target_layer.setdefault("events", [])
            if not isinstance(events, list):
                events = []
                target_layer["events"] = events
            events.append({"content": content, "turn": state.turn, "type": fact.get("type", "information")})

            news = region.setdefault("news", [])
            if not isinstance(news, list):
                news = []
                region["news"] = news
            news_content = content if len(content) <= 30 else f"{content[:30]}..."
            news.append(f"⚡ {news_content}")

            for layer in layers:
                if not isinstance(layer, dict):
                    continue
                fog = layer.get("fog")
                if not isinstance(fog, dict) or fog.get("unlocked") is not False:
                    continue
                condition = fog.get("unlock_condition")
                if isinstance(condition, dict) and "action" in condition and "dimension" not in condition:
                    continue


def _inject_events_to_board(state: GameState, events: list[dict]) -> None:
    """把被动事件注入对应棋盘区域的 news。"""
    regions = state.board_regions if isinstance(state.board_regions, dict) else {}
    if not regions or not events:
        return
    for event in events:
        if not isinstance(event, dict):
            continue
        desc = str(event.get("description") or event.get("name") or "").strip()
        if not desc:
            continue
        for region_id, region in regions.items():
            if not isinstance(region, dict):
                continue
            region_name = str(region.get("name", region_id)).strip()
            if not region_name or region_name not in desc:
                continue
            news = region.setdefault("news", [])
            if not isinstance(news, list):
                news = []
                region["news"] = news
            news_text = desc if len(desc) <= 40 else f"{desc[:40]}..."
            news.append(f"⚡ {news_text}")
            # 也写入第一个 layer 的 events
            layers = region.get("layers") if isinstance(region.get("layers"), list) else []
            target = next((l for l in layers if isinstance(l, dict)), None)
            if target is not None:
                ev_list = target.setdefault("events", [])
                if not isinstance(ev_list, list):
                    ev_list = []
                    target["events"] = ev_list
                ev_list.append({"content": desc, "turn": state.turn, "type": "passive_event"})


_DIMENSION_REGION_LAYER_MAP: dict[str, list[tuple[str, str]]] = {
    "兵力": [("辽东", "军情")],
    "士气": [("辽东", "军情")],
    "财政": [("京师", "财务"), ("辽东", "经济")],
    "民心": [("陕西", "民情")],
    "情报": [],
    "派系势力": [("京师", "朝堂")],
}


def _evolve_board_from_dimension_changes(state: GameState, changes: dict[str, int]) -> None:
    """维度变化后更新对应区域的事件流。"""
    regions = state.board_regions if isinstance(state.board_regions, dict) else {}
    if not regions or not changes:
        return
    for dim_name, delta in changes.items():
        if not isinstance(delta, int) or delta == 0:
            continue
        current_val = state.dimensions.world.get(str(dim_name), 5)
        mappings = _DIMENSION_REGION_LAYER_MAP.get(str(dim_name), [])
        for target_region_name, target_layer_label in mappings:
            for region_id, region in regions.items():
                if not isinstance(region, dict):
                    continue
                if str(region.get("name", "")) != target_region_name:
                    continue
                layers = region.get("layers") if isinstance(region.get("layers"), list) else []
                for layer in layers:
                    if not isinstance(layer, dict):
                        continue
                    if str(layer.get("label", "")) != target_layer_label:
                        continue
                    direction = "恶化" if delta < 0 else "好转"
                    ev_list = layer.setdefault("events", [])
                    if not isinstance(ev_list, list):
                        ev_list = []
                        layer["events"] = ev_list
                    ev_list.append({
                        "content": f"{dim_name}{direction}（{current_val}/10）",
                        "turn": state.turn,
                        "type": "dimension_change",
                    })
                    news = region.setdefault("news", [])
                    if not isinstance(news, list):
                        news = []
                        region["news"] = news
                    arrow = "▼" if delta < 0 else "▲"
                    news.append(f"{arrow} {target_region_name}·{target_layer_label}：{dim_name}{direction}")


DIMENSION_IMPACT = {
    "财政": {"desc": "国库存银", "low": "欠饷无法拨付、赈灾无钱可用", "high": "军饷充裕、可启动大型工程", "hint": "可召见户部尚书毕自严核查账目"},
    "兵力": {"desc": "可调动兵力", "low": "边防空虚、无力平叛", "high": "可主动出击、威慑藩镇", "hint": "可召见兵部尚书商议军务"},
    "民心": {"desc": "百姓拥护程度", "low": "流民增加、可能被裹挟为流寇", "high": "征税征兵阻力小、政令畅通", "hint": "可考虑减赋或赈灾安民"},
    "情报": {"desc": "对局势的掌控度", "low": "无法辨别军报真伪、无法识别朝臣暗中串联", "high": "提前预知威胁、看穿NPC隐瞒", "hint": "可召见锦衣卫指挥使骆养性"},
    "士气": {"desc": "军队战斗意志", "low": "可能哗变或溃逃", "high": "守城坚决、野战有胜算", "hint": "可犒赏将士或亲自慰问"},
    "补给": {"desc": "军粮物资储备", "low": "前线断粮、士兵以草根充饥", "high": "可支撑长期作战", "hint": "可调配粮草或开辟补给线"},
    "派系势力": {"desc": "朝中派系力量", "low": "无人执行政令、效率低下", "high": "权臣架空皇权、政令出不了内阁", "hint": "需平衡或打压强势派系"},
}


def _build_dimension_hints(state: GameState) -> str:
    """根据薄弱世界维度生成行动提示（含后果说明）。"""

    state_manager = get_state_manager()
    hints = []
    for item in state_manager.get_weak_dimensions(state, threshold=3):
        name = str(item.get("name") or "")
        if not name or name == "派系势力":
            continue
        info = DIMENSION_IMPACT.get(name)
        if info:
            value = _safe_int(state.dimensions.world.get(name), 5)
            hints.append(f"⚠ {name}不足({value}/10)：{info['low']}。→ {info['hint']}")
        else:
            hints.append(f"⚠ {name}偏低，建议关注")

    faction_power = state.dimensions.world.get("派系势力")
    if faction_power is not None and faction_power >= 8:
        info = DIMENSION_IMPACT.get("派系势力", {})
        hints.append(f"⚠ 派系势力过高({faction_power}/10)：{info.get('high', '权臣坐大')}。→ {info.get('hint', '需打压')}")

    return "\n".join(hints)


def _calc_remaining_turns(state: GameState) -> int:
    """计算距离崇祯十七年（year=17）还剩多少回合（月）。"""

    year = state.game_time.get("year", 1)
    month = state.game_time.get("month", 1)
    return max(0, (17 - year) * 12 - month + 1)


async def _send_action_panel(chat_id: str, state: GameState) -> None:
    """发送新架构行动面板。"""

    agenda_items = state.turn_agenda if hasattr(state, "turn_agenda") else []
    npc_list = []
    for npc_id in state.accessible_npcs:
        if str(npc_id) not in state.active_npcs:
            continue
        name = _npc_display_name(state, npc_id)
        loc_data = state.npc_locations.get(name, state.npc_locations.get(str(npc_id), {}))
        if loc_data.get("reachability", "present") in ("present", "nearby"):
            npc_list.append(name)

    state_manager = get_state_manager()
    trends = state_manager.get_dimension_trends(state)
    weak_names = {str(item.get("name")) for item in state_manager.get_weak_dimensions(state)}

    dim_lines = []
    for name, value in state.dimensions.world.items():
        value_int = _safe_int(value, 5)
        trend = trends.get(name, "━")
        warn = " ⚠" if name in weak_names else ""
        info = DIMENSION_IMPACT.get(name)
        impact = f"（{info['desc']}）" if info else ""
        status = ""
        if info:
            if value_int <= 3:
                status = f" → {info['low']}"
            elif value_int >= 8:
                status = f" → {info['high']}"
        dim_lines.append(f"{name}{impact}:{value}/10 {trend}{warn}{status}")
    dimension_summary = "\n".join(dim_lines) or "世界维度未明"

    remaining = _calc_remaining_turns(state)
    time_display = f"**{_format_state_time(state)}** · 每回合=1个月 · 距崇祯十七年还剩{remaining}回合"

    core_goal = str(state.player_goal.get("core_goal", "")).strip() if state.player_goal else ""

    hints = _build_dimension_hints(state)
    situation = hints if hints else ""

    panel = build_action_panel(
        time_display,
        dimension_summary,
        situation,
        npc_list,
        None,
        set(),
        len(state.backlog),
        agenda=agenda_items,
        campaign_goal=core_goal,
    )
    await send_narrator_card(chat_id, panel)


async def _start_turn_visits_or_panel(chat_id: str, state: GameState) -> None:
    """回合开始时先展示行动面板，然后呈现NPC主动来访。"""

    await _send_action_panel(chat_id, state)
    await _build_visit_queue(state)
    if state.visit_queue:
        visit_text = await _present_next_visit(state)
        if visit_text:
            message, gm_note, npc_name = visit_text
            await send_npc_card(chat_id, build_npc_reply_card(npc_name, message, npc_title=_npc_title(state, npc_name), gm_note=gm_note))
            state.current_talk_history.append({"role": "assistant", "speaker": npc_name, "content": message, "type": "visit_entry"})
            _append_history(state, "assistant", message, state.talking_to or "narrator")


async def _build_visit_queue(state: GameState) -> list[dict[str, Any]]:
    """根据本回合议程和低位维度生成NPC主动来访队列。"""

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_entry(npc_id: str, reason: str, urgency: int, source_type: str, visit_form: str = "") -> None:
        npc_name = _resolve_npc_name_from_text(state, npc_id) or npc_id
        if not npc_name or not reason:
            return
        expected_form = _visit_form_for_npc(state, npc_name)
        requested_form = str(visit_form or "").strip()
        if expected_form == "in_person":
            resolved_form = "in_person"
        else:
            resolved_form = requested_form if requested_form in {"messenger", "letter"} else expected_form
        key = (npc_name, source_type, reason[:40])
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "npc_id": npc_name,
                "reason": reason,
                "urgency": max(0, min(100, urgency)),
                "source_type": source_type,
                "visit_form": resolved_form,
            }
        )

    npc_catalog = _visit_npc_catalog(state)
    agenda_items = [item for item in state.turn_agenda if isinstance(item, Mapping) and not item.get("resolved")]
    if agenda_items and npc_catalog:
        payload = {
            "turn": state.turn,
            "agenda": agenda_items,
            "npcs": npc_catalog,
            "dimensions": _world_state_payload(state),
            "recent_history": state.conversation_history[-8:],
        }
        prompt = """
你是文字RPG的GM排程器。请把当前回合每个议程项匹配给最相关的NPC，生成NPC来访队列候选。

规则：
- npc_id 必须从给定npcs中的name或id选择，不要自创NPC
- 每个议程项最多匹配1个NPC；不相关可跳过
- urgency 为0-100，urgent约80-95，normal约45-65
- source_type 使用 agenda
- visit_form 必须根据NPC位置决定：NPC的location和玩家所在地相同或location为空时使用in_person；NPC在远方时使用messenger或letter
- 只输出JSON：{"visits":[{"npc_id":"...","reason":"...","urgency":70,"source_type":"agenda","visit_form":"in_person"}]}
""".strip()
        try:
            data = await get_llm_client().chat_json(prompt, json.dumps(payload, ensure_ascii=False), temperature=0.2)
            for visit in _as_list(data.get("visits") if isinstance(data, Mapping) else []):
                if not isinstance(visit, Mapping):
                    continue
                npc_name = _resolve_npc_name_from_text(state, str(visit.get("npc_id", "")))
                reason = str(visit.get("reason", "")).strip()
                if npc_name and reason:
                    add_entry(
                        npc_name,
                        reason,
                        _safe_int(visit.get("urgency"), 50),
                        str(visit.get("source_type") or "agenda"),
                        str(visit.get("visit_form") or ""),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("来访队列LLM匹配失败，使用规则兜底: %s", exc)

    for item in agenda_items:
        reason = str(item.get("description") or item.get("consequence") or "").strip()
        if not reason:
            continue
        npc_name = _infer_related_npc(state, reason, item)
        if npc_name:
            add_entry(npc_name, reason, _urgency_to_score(item.get("urgency"), 50), "agenda")

    threshold_value = 2
    for dim_name, value in state.dimensions.world.items():
        if _safe_int(value, 5) <= threshold_value:
            npc_name = _infer_dimension_npc(state, dim_name)
            if npc_name:
                severity = int((threshold_value - _safe_int(value, 5) + 1) * 5)
                add_entry(npc_name, f"{dim_name}跌至{value}/10，已低于警戒线", 90 + severity, "threshold")

    for item in state.consequence_seeds[-8:]:
        if not isinstance(item, Mapping):
            continue
        reason = str(item.get("content") or item.get("description") or item.get("name") or "").strip()
        if reason:
            npc_name = _infer_related_npc(state, reason, item)
            if npc_name:
                add_entry(npc_name, f"后果反馈：{reason}", _urgency_to_score(item.get("urgency") or item.get("severity"), 60), "settlement")

    for event in state.active_events[-8:]:
        if not isinstance(event, Mapping):
            continue
        reason = str(event.get("description") or event.get("name") or "").strip()
        if reason:
            npc_name = _infer_related_npc(state, reason, event)
            if npc_name:
                add_entry(npc_name, reason, _urgency_to_score(event.get("urgency") or event.get("severity"), 70), "event")

    entries.sort(key=lambda entry: _safe_int(entry.get("urgency"), 0), reverse=True)
    state.visit_queue = entries[:8]
    return state.visit_queue


async def _present_next_visit(state: GameState) -> tuple[str, str, str] | None:
    """弹出下一条来访，设置当前谈话状态并返回入场文本。"""

    while state.visit_queue:
        visit = state.visit_queue.pop(0)
        npc_name = _resolve_npc_name_from_text(state, str(visit.get("npc_id", ""))) or str(visit.get("npc_id", "")).strip()
        if not npc_name:
            continue
        visit_form = str(visit.get("visit_form") or _visit_form_for_npc(state, npc_name))
        if visit_form == "in_person" and not await _can_npc_join_now(state, npc_name):
            continue
        if visit_form == "in_person":
            state.talking_to = npc_name
            if npc_name not in state.present_npcs:
                state.present_npcs.append(npc_name)
            state.conversation_initiator = f"npc:{npc_name}"
            state.npc_join_round = {npc_name: len(state.current_talk_history)}
        else:
            state.talking_to = ""
            state.conversation_initiator = f"{visit_form}:{npc_name}"
            state.npc_join_round = {}
        state.current_talk_history = []
        state.phase = "talking" if visit_form == "in_person" else "free"
        reason = str(visit.get("reason") or "有事求见").strip()
        narration = await _generate_visit_entry_text(state, npc_name, reason, visit_form, _safe_int(visit.get("urgency"), 50))
        hint = await _generate_visit_opening_hint(state, npc_name, reason)
        return narration, hint, npc_name
    return None


async def _generate_visit_entry_text(state: GameState, npc_name: str, reason: str, visit_form: str, urgency: int) -> str:
    profile = await _resolve_npc_profile(state, npc_name)
    npc_title = str(profile.get("title") or profile.get("position") or _npc_title(state, npc_name) or "无头衔")
    player_role, player_title = _player_role_and_title(state)
    prompt = VISIT_NARRATION_PROMPT.format(
        npc_name=npc_name,
        npc_title=npc_title,
        player_role=player_role,
        player_title=player_title,
        reason=reason,
        visit_form=visit_form,
    )
    try:
        text = await get_llm_client().chat(prompt, "", temperature=0.55)
    except Exception as exc:  # noqa: BLE001
        logger.warning("来访入场叙事生成失败，使用兜底文本: %s", exc)
        form_text = {"messenger": "遣人来报", "letter": "递来书信"}.get(visit_form, "求见")
        text = f"{npc_name}{form_text}：{reason}"
    return text.strip() or f"{npc_name}求见：{reason}"


async def _generate_visit_opening_hint(state: GameState, npc_name: str, reason: str) -> str:
    agenda_match = ""
    for ag in (state.turn_agenda or []):
        desc = str(ag.get("description", ""))
        if any(kw in reason for kw in desc[:10].split("—")[:1]) or any(kw in desc for kw in reason[:10].split("—")[:1]):
            urgency = "🔴 紧急" if ag.get("urgency") == "urgent" else "🟡"
            agenda_match = f"{urgency} 回合议程触发：{desc}"
            break

    npc_hints = []
    for loc_name, loc_data in state.npc_locations.items():
        if loc_name == npc_name:
            continue
        title = loc_data.get("title", "")
        if not title:
            continue
        reason_lower = reason.lower()
        if any(kw in reason_lower for kw in ["饷", "钱", "银", "财", "粮", "税"]) and any(kw in title for kw in ["户部", "财"]):
            npc_hints.append(f"提示：可召集{loc_name}（{title}）询问国库详情")
        elif any(kw in reason_lower for kw in ["兵", "军", "战", "边"]) and any(kw in title for kw in ["兵部", "军"]):
            npc_hints.append(f"提示：可召集{loc_name}（{title}）商议军务")
        elif any(kw in reason_lower for kw in ["情报", "密", "探", "查"]) and any(kw in title for kw in ["锦衣卫", "情报"]):
            npc_hints.append(f"提示：可召集{loc_name}（{title}）调查详情")

    parts = []
    if agenda_match:
        parts.append(agenda_match)
    if npc_hints:
        parts.extend(npc_hints[:2])

    if parts:
        return "\n".join(parts)

    if not state.gm_hints_enabled:
        return ""
    context = {"source": "visit", "npc": npc_name, "reason": reason}
    hint = await _generate_gm_hints(state, reason, context)
    hint_text = str(hint.get("text", "")).strip() if state.turn == 0 else _apply_side_quest_hint(state, hint)
    if hint_text:
        return hint_text
    return _italic_hint(f"💡 此事涉及{reason[:28]}，可追问缘由、代价或可行方案")


async def _gm_decision_scan(state: GameState, player_text: str, npc_reply: str, npc_name: str) -> list[dict]:
    """轻量GM扫描：检测本轮对话中玩家是否做出了明确决策。"""
    recent = state.current_talk_history[-6:]
    recent_text = '\n'.join(f"{h.get('speaker','?')}：{h.get('content','')[:80]}" for h in recent)
    confirmed_text = '；'.join(d.get('summary', '') for d in state.confirmed_decisions[-10:]) or '无'
    prompt = GM_DECISION_SCAN_PROMPT.format(
        player_text=player_text,
        npc_name=npc_name,
        npc_reply=npc_reply[:300],
        recent_context=recent_text,
        confirmed_decisions=confirmed_text,
    )
    try:
        data = await get_llm_client().chat_json(prompt, player_text, temperature=0.1)
    except Exception as exc:
        logger.warning('GM决策扫描失败: %s', exc)
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get('decisions', [])
    if not isinstance(raw, list):
        return []
    results = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        summary = str(item.get('summary', '')).strip()
        if not summary:
            continue
        results.append({
            'id': str(item.get('id', f'dec_{uuid.uuid4().hex[:6]}')),
            'summary': summary,
            'category': str(item.get('category', '')).strip(),
            'npc_name': npc_name,
            'turn': state.turn,
        })
    _cid = getattr(state, '_runtime_chat_id', '')
    if _cid and results:
        get_tracer(_cid, enabled=state.trace_enabled).record('gm_decision_scan', '_gm_decision_scan', {'player_text': player_text[:200], 'decisions': results}, GameTracer.state_summary(state))
    return results


async def _generate_gm_hints(state: GameState, npc_message: str, context: Mapping[str, Any]) -> dict[str, Any]:
    """生成GM提示层文本和可点击方向。"""

    if not state.gm_hints_enabled:
        return {"text": "", "frames": []}
    payload = {
        "context": dict(context),
        "npc_message": npc_message,
        "talk_history": state.current_talk_history[-10:],
        "present_npcs": state.present_npcs or ([state.talking_to] if state.talking_to else []),
        "dimensions": _world_state_payload(state),
    }
    prompt = GM_HINT_PROMPT.format(
        context=json.dumps(payload.get("context", {}), ensure_ascii=False),
        npc_message=npc_message,
        dimensions=json.dumps(payload.get("dimensions", {}), ensure_ascii=False),
    )
    try:
        data = await get_llm_client().chat_json(prompt, json.dumps(payload, ensure_ascii=False), temperature=0.35)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GM提示生成失败，使用兜底提示: %s", exc)
        return _fallback_gm_hint(state, npc_message)
    if not isinstance(data, Mapping):
        return {"text": "", "frames": []}
    lines: list[str] = []
    frames = [str(item).strip() for item in _as_list(data.get("frame")) if str(item).strip()]
    for item in _as_list(data.get("annotate")):
        if not isinstance(item, Mapping):
            continue
        dimension = str(item.get("dimension", "")).strip()
        threshold = _safe_int(item.get("threshold"), 0)
        actual_value = _dimension_value_for_hint(state, dimension)
        if actual_value > threshold:
            text = str(item.get("text", "")).strip()
            if text:
                lines.append(_italic_hint(f"💡 你的{dimension}({actual_value}/10)使你察觉：{text}"))
    for alert in _as_list(data.get("alert"))[:2]:
        text = str(alert).strip()
        if text:
            lines.append(_italic_hint(f"⚠ {text}" if not text.startswith("⚠") else text))
    result: dict[str, Any] = {"text": "\n".join(lines), "frames": frames[:3]}
    side_quest = data.get("side_quest")
    if isinstance(side_quest, Mapping):
        result["side_quest"] = dict(side_quest)
    return result


def _register_side_quest(state: GameState, quest_data: dict) -> dict:
    """注册一个由对话触发的支线任务。"""

    name = str(quest_data.get("name", "")).strip()
    description = str(quest_data.get("description", "")).strip()
    if not name or name == "未命名支线" or not description:
        logger.warning("跳过无效支线任务：name=%s description=%s", name, description[:50])
        return {}

    for existing in state.side_quests:
        if not isinstance(existing, dict):
            continue
        if str(existing.get("name", "")).strip() == name:
            logger.info("跳过重复支线任务：%s", name)
            return existing

    quest = {
        "id": f"sq_{len(state.side_quests) + 1}",
        "name": name,
        "description": description,
        "status": "active",
        "rewards": str(quest_data.get("rewards") or "").strip(),
        "penalties": str(quest_data.get("penalties") or "").strip(),
        "source": str(quest_data.get("source") or "dialogue").strip() or "dialogue",
        "triggered_turn": state.turn,
        "completed_turn": None,
    }
    quest_name = quest["name"]
    for existing in state.side_quests:
        if existing.get("status") != "active":
            continue
        existing_name = str(existing.get("name") or "").strip()
        if existing_name and (
            existing_name == quest_name
            or existing_name in quest_name
            or quest_name in existing_name
        ):
            return existing
    state.side_quests.append(quest)
    return quest


def _apply_side_quest_hint(state: GameState, hints: Mapping[str, Any]) -> str:
    """注册GM识别出的支线任务，并返回包含支线展示的提示文本。"""

    hint_text = str(hints.get("text", "")).strip()
    side_quest = hints.get("side_quest")
    if not isinstance(side_quest, dict):
        return hint_text

    quest = _register_side_quest(state, side_quest)
    if not quest:
        return hint_text
    quest_text = (
        f"⚡ 触发支线：{quest['name']}\n"
        f"· 成功：{quest['rewards'] or '待定'}\n"
        f"· 风险：{quest['penalties'] or '待定'}"
    )
    return f"{hint_text}\n{quest_text}" if hint_text else quest_text


def _fallback_gm_hint(state: GameState, npc_message: str) -> dict[str, Any]:
    if not npc_message:
        return {"text": "", "frames": []}
    risky_words = ("立刻", "开战", "处死", "加税", "抄家", "调兵", "拨银", "兵变", "崩溃")
    frames = ["追问细节", "询问代价", "召人参议"]
    lines: list[str] = []
    if any(word in npc_message for word in risky_words):
        lines.append(_italic_hint("⚠ 此事可能牵动全局，贸然拍板会留下后果"))
    return {"text": "\n".join(lines), "frames": frames}


def _hint_frame_options(hints: Mapping[str, Any]) -> list[str]:
    return [str(item).strip() for item in _as_list(hints.get("frames")) if str(item).strip()][:3]


def _italic_hint(text: str) -> str:
    escaped = text.replace("\n", " ").strip()
    return f"*{escaped}*" if escaped else ""


def _dimension_value_for_hint(state: GameState, dimension: str) -> int:
    if dimension in state.dimensions.character:
        return _safe_int(state.dimensions.character.get(dimension), 0) * 10
    if dimension in state.dimensions.world:
        return _safe_int(state.dimensions.world.get(dimension), 0) * 10
    aliases = {"军事": "兵力", "财务": "财政", "政治": "派系势力"}
    alias = aliases.get(dimension)
    if alias and alias in state.dimensions.world:
        return _safe_int(state.dimensions.world.get(alias), 0) * 10
    return 0


async def _handle_talk_summon(chat_id: str, state: GameState, text: str) -> bool:
    if not state.talking_to:
        return False
    targets = await _detect_summon_target(state, text)
    if not targets:
        return False
    present = _normalize_present_npcs(state)
    for target in targets:
        if target in present:
            msg = f"{target}已在场。"
            await send_narrator_card(chat_id, build_narration_card("召见", msg))
            _append_history(state, "assistant", msg, "narrator")
            continue
        actual = await _ensure_npc_exists(state, target, chat_id)
        if actual:
            target = actual
        loc = state.npc_locations.get(target, {})
        reachability = str(loc.get("reachability", "present"))
        if reachability in {"distant", "unreachable"}:
            msg = f"*{target}不在附近，无法当场召来。可以书信传达。*"
            await send_narrator_card(chat_id, build_narration_card("无法即时召见", msg))
            _append_history(state, "assistant", msg, "narrator")
            continue
        await _join_npc_to_talk(chat_id, state, target, text)
        present = _normalize_present_npcs(state)
    return True


async def _join_npc_to_talk(chat_id: str, state: GameState, npc_name: str, player_text: str) -> None:
    brief_data = await _generate_join_brief(state, npc_name, player_text)
    narration = str(brief_data.get("narration") or f"{npc_name}入内，听过简要说明后向你行礼。").strip()
    brief = str(brief_data.get("brief") or "刚被召入当前对话。").strip()
    if npc_name not in state.present_npcs:
        state.present_npcs.append(npc_name)
    state.npc_join_round[npc_name] = len(state.current_talk_history) // 2
    state.current_talk_history.append({"role": "system", "speaker": npc_name, "content": f"{npc_name}加入对话。简报：{brief}", "type": "join_entry"})
    state.talking_to = npc_name
    npc_profile = await _resolve_npc_profile(state, npc_name)
    reply = await generate_npc_reply(state, npc_name, npc_profile, f"（你刚被召入，入场动作已由旁白描述完毕，你无需再行礼或自我介绍，直接说事。）\n背景简报：{brief}\n玩家原话：{player_text}", court_context=_talk_context_for_npc(state, npc_name, mode="join"))
    message = f"{narration}\n\n{npc_name}：{reply}"
    hints = await _generate_gm_hints(state, reply, {"source": "npc_join", "npc": npc_name, "brief": brief})
    hint_text = _apply_side_quest_hint(state, hints)
    frame_options = _hint_frame_options(hints)
    state.current_talk_history.append({"role": "assistant", "speaker": npc_name, "content": reply})
    await send_npc_card(chat_id, build_npc_reply_card(npc_name, message, frame_options=frame_options, npc_title=_npc_title(state, npc_name), gm_note=hint_text))
    _append_history(state, "assistant", message, npc_name)


async def _generate_join_brief(state: GameState, npc_name: str, player_text: str) -> dict[str, str]:
    payload = {
        "new_npc": npc_name,
        "present_npcs": state.present_npcs,
        "player_text": player_text,
        "talk_history": state.current_talk_history[-8:],
        "scene": state.current_scene,
    }
    try:
        data = await get_llm_client().chat_json(JOIN_BRIEF_PROMPT, json.dumps(payload, ensure_ascii=False), temperature=0.45)
    except Exception as exc:  # noqa: BLE001
        logger.warning("召见入场简报生成失败，使用兜底: %s", exc)
        data = {}
    return {
        "narration": str(data.get("narration") or f"{npc_name}被召入当前议谈。").strip(),
        "brief": str(data.get("brief") or _summarize_recent_talk_for_join(state)).strip(),
    }


def _summarize_recent_talk_for_join(state: GameState) -> str:
    recent = []
    for item in state.current_talk_history[-6:]:
        if not isinstance(item, Mapping):
            continue
        speaker = str(item.get("speaker") or ("玩家" if item.get("role") == "user" else state.talking_to or "在场者"))
        content = str(item.get("content", "")).strip()
        if content:
            recent.append(f"{speaker}：{content[:30]}")
    return "；".join(recent)[:120] or "当前正在议事。"


async def _handle_talk_dismiss(chat_id: str, state: GameState, text: str) -> bool:
    _normalize_present_npcs(state)
    target = _detect_dismiss_target(state, text)
    if not target:
        return False
    if target not in state.present_npcs:
        return False
    state.present_npcs = [npc for npc in state.present_npcs if npc != target]
    state.npc_join_round.pop(target, None)
    if state.talking_to == target:
        state.talking_to = state.present_npcs[0] if state.present_npcs else ""
    msg = f"*{target}行礼告退。*"
    state.current_talk_history.append({"role": "system", "speaker": target, "content": msg})
    await send_narrator_card(chat_id, build_narration_card("离场", msg))
    _append_history(state, "assistant", msg, "narrator")
    if not state.present_npcs:
        await _end_talk(chat_id, state)
    return True


async def _route_multi_talking_message(chat_id: str, state: GameState, text: str) -> None:
    present = _normalize_present_npcs(state)
    state.current_talk_history.append({"role": "user", "speaker": "玩家", "content": text})
    target = _detect_addressed_npc(state, text)
    if target:
        primary = target
    else:
        primary = await _choose_multi_talk_primary(state, text, present)
    if primary not in present:
        primary = state.talking_to if state.talking_to in present else present[0]

    responses: list[str] = []
    prior: list[dict[str, str]] = []
    profile = await _resolve_npc_profile(state, primary)
    context = _talk_context_for_npc(state, primary, mode="active")
    reply = await generate_npc_reply(state, primary, profile, text, court_context=context)
    state.current_talk_history.append({"role": "assistant", "speaker": primary, "content": reply})
    _append_history(state, "assistant", reply, primary)
    responses.append(f"**{primary}**：{reply}")
    prior.append({"npc": primary, "content": reply})

    for npc_name in [npc for npc in present if npc != primary]:
        if random.random() >= 0.3:
            continue
        profile = await _resolve_npc_profile(state, npc_name)
        context = _talk_context_for_npc(state, npc_name, mode="react", prior_speakers=prior)
        try:
            reply = await generate_npc_reply(state, npc_name, profile, text, court_context=context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("多人谈话补充发言失败，跳过 npc=%s: %s", npc_name, exc)
            continue
        state.current_talk_history.append({"role": "assistant", "speaker": npc_name, "content": reply})
        _append_history(state, "assistant", reply, npc_name)
        responses.append(f"**{npc_name}**：{reply}")
        prior.append({"npc": npc_name, "content": reply})
    state.talking_to = primary
    combined = "\n\n".join(responses)
    hints = await _generate_gm_hints(state, combined, {"source": "multi_talk", "present_npcs": present})
    hint_text = _apply_side_quest_hint(state, hints)
    multi_title = "、".join(title for title in (_npc_title(state, npc) for npc in present) if title)
    await send_npc_card(chat_id, build_npc_reply_card("、".join(present), combined, frame_options=_hint_frame_options(hints), npc_title=multi_title, gm_note=hint_text))
    state.turn += 1
    _cid = getattr(state, "_runtime_chat_id", "")
    if _cid:
        get_tracer(_cid, enabled=state.trace_enabled).record_turn_snapshot(state)
    state.turns_without_action += 1


async def _choose_multi_talk_primary(state: GameState, text: str, present: list[str]) -> str:
    speakers, _supplements = await _choose_multi_talk_speakers(state, text, present)
    primary = speakers[0] if speakers else ""
    if primary in present:
        return primary
    return state.talking_to if state.talking_to in present else (present[0] if present else "")


async def _choose_multi_talk_speakers(state: GameState, text: str, present: list[str]) -> tuple[list[str], list[str]]:
    payload = {
        "player_text": text,
        "present_npcs": [_npc_brief_for_arbiter(state, npc) for npc in present],
        "talk_history": state.current_talk_history[-8:],
        "dimensions": _world_state_payload(state),
    }
    try:
        data = await get_llm_client().chat_json(MULTI_TALK_ARBITER_PROMPT, json.dumps(payload, ensure_ascii=False), temperature=0.25)
    except Exception as exc:  # noqa: BLE001
        logger.warning("多人谈话裁定失败，使用关键词兜底: %s", exc)
        primary = _infer_related_npc_from_candidates(state, text, present) or state.talking_to or present[0]
        return [primary], [npc for npc in present if npc != primary][:1]
    primary = str(data.get("primary") or "").strip()
    if primary not in present:
        primary = _infer_related_npc_from_candidates(state, text, present) or state.talking_to or present[0]
    supplements = [str(item).strip() for item in _as_list(data.get("supplements")) if str(item).strip() in present and str(item).strip() != primary]
    return [primary], supplements[:2]


def _talk_context_for_npc(
    state: GameState,
    npc_name: str,
    *,
    mode: str = "active",
    prior_speakers: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    join_round = max(0, _safe_int(state.npc_join_round.get(npc_name), 0))
    join_index = min(len(state.current_talk_history), join_round * 2)
    visible_history = state.current_talk_history[join_index:]
    return {
        "scene_type": "private",
        "topic": _summarize_recent_talk_for_join(state),
        "present_npcs": list(state.present_npcs or [npc_name]),
        "mode": mode,
        "reason": "连续多人谈话",
        "prior_speakers": prior_speakers or [],
        "visible_history": visible_history[-8:],
    }


def _normalize_present_npcs(state: GameState) -> list[str]:
    present = [npc for npc in state.present_npcs if npc]
    if state.talking_to and state.talking_to not in present:
        present.insert(0, state.talking_to)
    state.present_npcs = present
    for npc in present:
        state.npc_join_round.setdefault(npc, 0)
    return present


async def _detect_summon_target(state: GameState, text: str) -> list[str]:
    direct = _summon_name_verb_match(state, text)
    if direct:
        return [direct]
    if not (
        re.search(r"(叫|召|宣|请|喊|找|把|让).{0,4}(来|过来|入殿|进来|到这)", text)
        or any(keyword in text for keyword in SUMMON_KEYWORDS)
    ):
        return []
    resolved = _resolve_npc_name_from_text(state, text)
    if resolved:
        return [resolved]
    return await _llm_resolve_summon_targets(text, _all_known_npc_names(state))


async def _llm_resolve_summon_targets(player_text: str, npc_names: list[str]) -> list[str]:
    known = [name for name in dict.fromkeys(str(item).strip() for item in npc_names) if name]
    if not known:
        return []
    payload = {"player_text": player_text, "npc_names": known}
    try:
        data = await get_llm_client().chat_json(SUMMON_TARGET_RESOLVE_PROMPT, json.dumps(payload, ensure_ascii=False), temperature=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("召见目标LLM解析失败，忽略兜底: %s", exc)
        return []
    targets: list[str] = []
    for item in _as_list(data.get("targets")):
        name = str(item).strip()
        if name in known and name not in targets:
            targets.append(name)
    return targets


def _detect_dismiss_target(state: GameState, text: str) -> str:
    target = _dismiss_name_match(state, text)
    if target:
        return target
    if re.search(r"(退下|先走|不用了|你先去|告退|下去|离开).{0,6}", text) and len(state.present_npcs) == 1:
        return state.present_npcs[0]
    return ""


def _detect_addressed_npc(state: GameState, text: str) -> str:
    stripped = text.strip()
    for npc in state.present_npcs:
        if not npc:
            continue
        for alias in _npc_aliases_for_text(state, npc):
            if f"@{alias}" in text or re.search(rf"^{re.escape(alias)}[，,：:\s、]", stripped):
                return npc
    return ""


def _resolve_npc_name_from_text(state: GameState, text: str) -> str:
    direct = _summon_name_verb_match(state, text)
    if direct:
        return direct
    candidates: list[tuple[str, str]] = []
    for npc_id, info in state.active_npcs.items():
        name = str(info.get("name") or npc_id).strip()
        title = str(info.get("title") or info.get("position") or "").strip()
        candidates.append((name, title))
        candidates.append((str(npc_id), title))
    for npc_name, loc in state.npc_locations.items():
        title = str(loc.get("title", "")) if isinstance(loc, Mapping) else ""
        candidates.append((str(npc_name), title))
    candidates.sort(key=lambda pair: max(len(pair[0]), len(pair[1])), reverse=True)
    for name, title in candidates:
        if name and name in text:
            return name
        if title and title in text:
            return name
    surname_candidates = re.findall(r"[一-鿿](?=[、，,\s和还有])", text)
    if surname_candidates:
        for surname in surname_candidates:
            for name, title in candidates:
                if name and len(name) >= 2 and name[0] == surname:
                    return name
    return ""


def _npc_aliases_for_text(state: GameState, npc_name: str) -> list[str]:
    aliases = [npc_name]
    title = _npc_title(state, npc_name)
    if title:
        aliases.append(title)
    profile = state.active_npcs.get(state.cast.get(npc_name, npc_name), {})
    if isinstance(profile, Mapping):
        for key in ("name", "title", "position"):
            value = str(profile.get(key) or "").strip()
            if value and value not in aliases:
                aliases.append(value)
    return sorted(aliases, key=len, reverse=True)


def _summon_name_verb_match(state: GameState, text: str) -> str:
    summon_verbs = r"(叫|召|宣|请|喊|找|让).{0,4}(来|过来|入殿|进来|到这)"
    for npc_name in _all_known_npc_names(state):
        for alias in _npc_aliases_for_text(state, npc_name):
            if re.search(rf"{re.escape(alias)}.{0,6}{summon_verbs}", text):
                return npc_name
            if re.search(rf"(叫|召|宣|请|喊|找|把|让).{{0,6}}{re.escape(alias)}", text):
                return npc_name
    return ""


def _dismiss_name_match(state: GameState, text: str) -> str:
    for npc in state.present_npcs:
        for alias in _npc_aliases_for_text(state, npc):
            if re.search(rf"{re.escape(alias)}.{0,6}(退下|先走|不用了|告退|下去|离开)", text):
                return npc
            if re.search(rf"(退下|先走|不用了|你先去|告退|下去|离开).{{0,6}}{re.escape(alias)}", text):
                return npc
    return ""


def _npc_brief_for_arbiter(state: GameState, npc_name: str) -> dict[str, str]:
    profile = state.active_npcs.get(state.cast.get(npc_name, npc_name), {})
    if not profile:
        profile = next((info for info in state.active_npcs.values() if str(info.get("name", "")) == npc_name), {})
    return {
        "name": npc_name,
        "title": str(profile.get("title") or profile.get("position") or _npc_title(state, npc_name)),
        "faction": str(profile.get("faction") or ""),
        "traits": "、".join(_as_str_list(profile.get("personality_traits") or profile.get("values"))),
    }


async def _can_npc_join_now(state: GameState, npc_name: str) -> bool:
    loc = state.npc_locations.get(npc_name, {})
    reachability = str(loc.get("reachability", "present"))
    return reachability not in {"distant", "unreachable"}


def _visit_form_for_npc(state: GameState, npc_name: str) -> str:
    loc = _npc_location_data(state, npc_name)
    npc_location = str(loc.get("location") or "").strip()
    if not npc_location:
        return "in_person"
    if _same_region(_player_location(state), npc_location):
        return "in_person"
    return "messenger" if random.random() < 0.5 else "letter"


def _visit_threshold_value(state: GameState) -> int:
    world = state.storylines.get("world", {}) if isinstance(state.storylines.get("world"), Mapping) else {}
    raw = world.get("visit_threshold") or world.get("threshold_low")
    if raw is not None:
        return max(0, min(10, _safe_int(raw, 2)))
    return 2


def _urgency_to_score(value: Any, default: int = 45) -> int:
    if isinstance(value, (int, float)):
        return max(0, min(100, int(value)))
    text = str(value or "").strip().lower()
    return URGENCY_SCORE.get(text, default)


def _infer_related_npc(state: GameState, reason: str, item: Mapping[str, Any] | None = None) -> str:
    if item:
        for key in ("npc_id", "npc", "target_npc", "source_npc", "related_npc"):
            value = str(item.get(key) or "").strip()
            if value:
                found = _resolve_npc_name_from_text(state, value)
                if found:
                    return found
    found = _resolve_npc_name_from_text(state, reason)
    if found:
        return found
    return _infer_related_npc_from_candidates(state, reason, list(_all_known_npc_names(state)))


def _infer_related_npc_from_candidates(state: GameState, text: str, candidates: list[str]) -> str:
    dimension_keywords = {
        "财政": ("钱", "银", "饷", "税", "库", "粮"),
        "兵力": ("兵", "军", "辽", "边", "战", "剿"),
        "民心": ("民", "灾", "饥", "赈", "流寇"),
        "士气": ("士气", "军心", "兵变"),
        "派系势力": ("朝", "党", "弹劾", "阁", "官"),
        "情报": ("密", "查", "探", "锦衣卫"),
        "补给": ("粮", "草", "运", "工"),
    }
    for dim, keywords in dimension_keywords.items():
        if any(keyword in text for keyword in keywords):
            npc = _infer_dimension_npc(state, dim, candidates=candidates)
            if npc:
                return npc
    return candidates[0] if candidates else ""


def _infer_dimension_npc(state: GameState, dim_name: str, *, candidates: list[str] | None = None) -> str:
    wanted = DIMENSION_RESPONSIBLE_NPCS.get(dim_name, ())
    pool = candidates or list(_all_known_npc_names(state))
    for keyword in wanted:
        for npc_name in pool:
            title = _npc_title(state, npc_name)
            if keyword and (keyword in npc_name or keyword in title):
                return npc_name
    return pool[0] if pool else ""


def _all_known_npc_names(state: GameState) -> list[str]:
    names: list[str] = []
    for npc_id, info in state.active_npcs.items():
        _append_unique_str(names, str(info.get("name") or npc_id))
    for npc_name in state.npc_locations:
        _append_unique_str(names, str(npc_name))
    return names


def _visit_npc_catalog(state: GameState) -> list[dict[str, str]]:
    """构建来访排程可选NPC清单。"""

    catalog: list[dict[str, str]] = []
    seen: set[str] = set()
    for npc_id, info in state.active_npcs.items():
        if not isinstance(info, Mapping):
            continue
        name = str(info.get("name") or npc_id).strip()
        if not name or name in seen:
            continue
        loc = state.npc_locations.get(name, state.npc_locations.get(str(npc_id), {}))
        if isinstance(loc, Mapping) and str(loc.get("reachability", "present")) in {"unreachable"}:
            continue
        seen.add(name)
        catalog.append(
            {
                "id": str(npc_id),
                "name": name,
                "title": str(info.get("title") or info.get("position") or _npc_title(state, name)),
                "faction": str(info.get("faction") or (loc.get("faction", "") if isinstance(loc, Mapping) else "")),
                "location": str(loc.get("location", "") if isinstance(loc, Mapping) else ""),
                "reachability": str(loc.get("reachability", "present") if isinstance(loc, Mapping) else "present"),
            }
        )
    for npc_name, loc in state.npc_locations.items():
        name = str(npc_name).strip()
        if not name or name in seen or not isinstance(loc, Mapping):
            continue
        if str(loc.get("reachability", "present")) in {"unreachable"}:
            continue
        seen.add(name)
        catalog.append(
            {
                "id": name,
                "name": name,
                "title": str(loc.get("title", "")),
                "faction": str(loc.get("faction", "")),
                "location": str(loc.get("location", "")),
                "reachability": str(loc.get("reachability", "present")),
            }
        )
    return catalog


def _dimension_state_from_role(parsed_dimensions: Mapping[str, Any]) -> DimensionState:
    """根据角色层解析结果创建维度状态。"""

    relations: dict[str, RelationDimensions] = {}
    raw_relations = parsed_dimensions.get("relations") if isinstance(parsed_dimensions.get("relations"), dict) else {}
    for npc_id, relation in raw_relations.items():
        if not isinstance(relation, dict):
            continue
        values = relation.get("values") if isinstance(relation.get("values"), dict) else {}
        tags = relation.get("tags") if isinstance(relation.get("tags"), list) else []
        relations[str(npc_id)] = RelationDimensions(
            values={str(k): _safe_int(v, 5) for k, v in values.items() if str(k) in RELATION_NUMERIC_DIMENSIONS},
            tags=[str(tag) for tag in tags if str(tag).strip()],
        )
    return DimensionState(
        character={str(k): _safe_int(v, 5) for k, v in dict(parsed_dimensions.get("character", {})).items() if str(k) in CHARACTER_DIMENSIONS},
        world={str(k): _safe_int(v, 5) for k, v in dict(parsed_dimensions.get("world", {})).items() if str(k) in WORLD_STATE_DIMENSIONS},
        extensions={str(k): _safe_int(v, 5) for k, v in dict(parsed_dimensions.get("extensions", {})).items() if str(k) in EXTENSION_DIMENSIONS},
        relations=relations,
    )


def _merge_pressure_sources(world: Mapping[str, Any], role: Mapping[str, Any]) -> list[dict[str, Any]]:
    """合并世界层事件源与角色层压力源。"""

    sources: list[dict[str, Any]] = []
    parsed = role.get("parsed_pressure_sources") if isinstance(role.get("parsed_pressure_sources"), dict) else {}
    for source_type, items in parsed.items():
        for item in _as_list(items):
            if isinstance(item, dict):
                source = dict(item)
                source.setdefault("source_type", source_type)
                source.setdefault("type", source_type)
                sources.append(source)
    for key, source_type in (("timeline_milestones", "milestones"), ("condition_events", "reactions")):
        for item in _as_list(world.get(key)):
            if isinstance(item, dict):
                source = dict(item)
                source.setdefault("source_type", source_type)
                source.setdefault("type", source_type)
                sources.append(source)
    random_pool = world.get("random_event_pool") if isinstance(world.get("random_event_pool"), dict) else {}
    for item in _as_list(random_pool.get("entries")):
        if isinstance(item, dict):
            source = dict(item)
            source.setdefault("source_type", "random")
            source.setdefault("type", "random")
            sources.append(source)
    return sources


def _active_npcs_from_world(world: Mapping[str, Any], role: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """用世界NPC池与角色关系初始化活跃NPC。"""

    related_ids = {
        str(item.get("npc_id"))
        for item in _as_list(role.get("npc_relationships"))
        if isinstance(item, dict) and item.get("npc_id")
    }
    scene = _initial_scene(role.get("role") if isinstance(role.get("role"), dict) else {})
    related_ids.update(str(item) for item in _as_list(scene.get("present_npcs")) if str(item).strip())
    npcs: dict[str, dict[str, Any]] = {}
    for npc in _as_list(world.get("global_npc_pool")):
        if not isinstance(npc, dict):
            continue
        npc_id = str(npc.get("npc_id") or npc.get("name") or "").strip()
        if not npc_id:
            continue
        name = str(npc.get("name") or npc_id).strip()
        if related_ids and npc_id not in related_ids and name not in related_ids:
            continue
        npcs[npc_id] = dict(npc)
    if not npcs:
        for npc in _as_list(world.get("global_npc_pool"))[:5]:
            if isinstance(npc, dict):
                npc_id = str(npc.get("npc_id") or npc.get("name") or "").strip()
                if npc_id:
                    npcs[npc_id] = dict(npc)
    return npcs


def _accessible_npcs_from_role(role: Mapping[str, Any], role_info: Mapping[str, Any]) -> list[str]:
    """初始化当前可直接对话的NPC列表。"""

    accessible: list[str] = []
    scene = _initial_scene(role_info)
    for npc_id in scene.get("present_npcs", []):
        _append_unique_str(accessible, npc_id)

    stances: dict[str, str] = {}
    for item in _as_list(role.get("npc_relationships")):
        if not isinstance(item, dict):
            continue
        npc_id = str(item.get("npc_id", "")).strip()
        if not npc_id:
            continue
        stance = str(item.get("stance", "")).strip().lower()
        stances[npc_id] = stance
        if stance == "ally":
            _append_unique_str(accessible, npc_id)

    return [npc_id for npc_id in accessible if stances.get(npc_id) not in {"enemy", "unknown"}]


def _npc_display_name(state: GameState, npc_id: str) -> str:
    """按NPC ID取得中文展示名。"""

    info = state.active_npcs.get(str(npc_id), {})
    return str(info.get("name") or npc_id)


def _append_unique_str(target: list[str], value: str) -> None:
    """向列表追加非空且不重复的字符串。"""

    item = str(value).strip()
    if item and item not in target:
        target.append(item)


def _build_cast(active_npcs: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """建立NPC展示名到ID的映射。"""

    cast: dict[str, str] = {}
    for npc_id, npc in active_npcs.items():
        name = str(npc.get("name") or npc_id).strip()
        cast[name] = str(npc_id)
    return cast


def _initial_scene(role_info: Mapping[str, Any]) -> dict[str, Any]:
    """解析角色层开局场景。"""

    scene = role_info.get("initial_scene") if isinstance(role_info.get("initial_scene"), dict) else {}
    return {
        "location": str(role_info.get("start_location") or scene.get("location") or "未明"),
        "context": str(scene.get("context") or role_info.get("identity") or "你站在局势的开端。"),
        "present_npcs": [str(item) for item in _as_list(scene.get("present_npcs"))],
        "immediate_choices_hint": [str(item) for item in _as_list(scene.get("immediate_choices_hint"))],
    }


def _compact_world(world: Mapping[str, Any]) -> dict[str, Any]:
    """保存世界层摘要，避免存档过大。"""

    return {
        "name": world.get("name") or world.get("title"),
        "summary": world.get("summary") or world.get("description"),
        "time_system": world.get("time_system", {}),
        "optional_roles": world.get("optional_roles", []),
    }


def _compact_role(role: Mapping[str, Any]) -> dict[str, Any]:
    """保存角色层摘要。"""

    role_info = role.get("role") if isinstance(role.get("role"), dict) else {}
    return {
        "role": role_info,
        "player_goal": role.get("player_goal", {}),
        "narration_style": role.get("narration_style", {}),
    }


def _time_config_from_world(world: Mapping[str, Any]) -> dict[str, Any]:
    """读取新格式时间配置。"""

    config = world.get("time_system") if isinstance(world.get("time_system"), dict) else {}
    return {
        "unit": str(config.get("unit", "月")),
        "display_format": str(config.get("display_format", "第{year}年{month}月")),
        "start": config.get("start", {"year": 1, "month": 1}),
        "action_time_cost": config.get("action_time_cost", {}),
    }


def _time_config_from_state(state: GameState) -> dict[str, Any]:
    """从存档摘要读取时间配置。"""

    world = state.storylines.get("world", {}) if isinstance(state.storylines.get("world"), dict) else {}
    return _time_config_from_world(world)


def _sync_time_from_config(state: GameState, config: Mapping[str, Any]) -> None:
    """按配置同步游戏时间。"""

    if isinstance(config.get("start"), dict):
        state.game_time = _safe_int_dict(config.get("start"), state.game_time)
    state.game_date = format_game_time(state.game_time, str(config.get("display_format", "第{year}年{month}月")))


def _format_state_time(state: GameState) -> str:
    """格式化当前游戏时间。"""

    config = _time_config_from_state(state)
    return format_game_time(state.game_time, str(config.get("display_format") or "第{year}年{month}月"))


def _objectives_from_goal(goal: Mapping[str, Any]) -> list[dict[str, Any]]:
    """从玩家目标生成行动面板目标。"""

    objectives: list[dict[str, Any]] = []
    core_goal = str(goal.get("core_goal", "")).strip()
    if core_goal:
        objectives.append({"name": core_goal, "completed": False})
    for item in _as_list(goal.get("milestones") or goal.get("objectives")):
        if isinstance(item, dict):
            objectives.append({"name": str(item.get("name") or item.get("description") or item.get("id") or "目标"), "completed": bool(item.get("completed"))})
        elif str(item).strip():
            objectives.append({"name": str(item).strip(), "completed": False})
    return objectives[:6]


def _prologue_dimensions(state: GameState) -> dict[str, dict[str, Any]]:
    """构建序幕卡片使用的世界维度面板数据。"""

    return {
        str(name): {
            "value": _safe_int(value, 0),
            "max": 10,
            "label": str(name),
        }
        for name, value in state.dimensions.world.items()
    }


def _prologue_pressure_warnings(state: GameState) -> list[str]:
    """构建序幕压力提示，优先展示即将按时间自动变化的维度。"""

    warnings: list[str] = []
    for source in state.pressure_sources:
        if not isinstance(source, Mapping):
            continue
        source_type = str(source.get("source_type", source.get("type", ""))).strip()
        if source_type != "decay":
            continue

        dim_name = str(source.get("dimension", source.get("metric", ""))).strip()
        if not dim_name:
            continue

        current_value = _safe_int(state.dimensions.world.get(dim_name), 5)
        rate = _safe_int(source.get("rate", source.get("delta", 0)), 0)
        if rate == 0:
            continue

        next_value = max(0, min(10, current_value + rate))
        direction = "恶化" if rate < 0 else "上升"
        warning = f"{dim_name}持续{direction}中，若本回合不处理，下回合{dim_name}将至{next_value}"
        narrative = str(source.get("narrative", "")).strip()
        if narrative:
            warning += f"（{narrative}）"
        warnings.append(warning)

    if warnings:
        return warnings[:3]
    return ["财政持续恶化中，若本回合不处理，下回合军事将降至4"]


def _prologue_main_quest(state: GameState) -> str:
    """构建序幕主线进度文本。"""

    objectives = _objectives_from_goal(state.player_goal)
    if not objectives:
        return "暂无明确目标"

    core_goal = str(state.player_goal.get("core_goal", "")).strip()
    if core_goal:
        milestones = [
            obj
            for obj in objectives
            if str(obj.get("name", "")).strip() and str(obj.get("name", "")).strip() != core_goal
        ]
        total = len(milestones)
        completed = sum(1 for obj in milestones if obj.get("completed"))
        if total > 0:
            return f"{core_goal}（{completed}/{total}）"

    total = len(objectives)
    completed = sum(1 for obj in objectives if obj.get("completed"))
    names = [str(obj.get("name", "")).strip() for obj in objectives if str(obj.get("name", "")).strip()]
    title = names[0] if names else "主线目标"
    return f"{title}（{completed}/{total}）"


async def _resolve_npc_profile(state: GameState, npc_name: str) -> dict[str, Any]:
    """按新旧剧本加载NPC资料。"""

    npc_id = state.cast.get(npc_name, npc_name)
    if npc_id in state.active_npcs:
        return dict(state.active_npcs[npc_id])
    for candidate_id, info in state.active_npcs.items():
        if str(info.get("name", "")) == npc_name:
            return dict(info)
    if is_new_format_script(state.script_id):
        try:
            return await load_npc_from_world(state.script_id, npc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载世界层NPC失败 npc=%s err=%s", npc_name, exc)
    try:
        return await load_npc_profile(state.script_id, npc_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("加载旧NPC失败 npc=%s err=%s", npc_name, exc)
        return {"npc_id": npc_id, "name": npc_name, "title": "未明", "personality": "谨慎"}


def _pick_target_npc(state: GameState, text: str, target_npc: str | None) -> str:
    """根据目标、文本和场景选择对话NPC。"""

    if target_npc:
        return target_npc
    for npc_id, npc in state.active_npcs.items():
        name = str(npc.get("name") or npc_id)
        if name and name in text:
            return name
        if npc_id in text:
            return name
    scene_npcs = _as_list(state.current_scene.get("present_npcs") if isinstance(state.current_scene, dict) else [])
    if scene_npcs:
        first = str(scene_npcs[0])
        return str(state.active_npcs.get(first, {}).get("name") or first)
    if state.active_npcs:
        npc_id, npc = next(iter(state.active_npcs.items()))
        return str(npc.get("name") or npc_id)
    return "旁人"


def _proposal_payload(proposal: ActionProposal) -> dict[str, Any]:
    """序列化行动提案。"""

    payload = asdict(proposal)
    payload["action_type"] = str(proposal.action_type.value if hasattr(proposal.action_type, "value") else proposal.action_type)
    payload["proposal_id"] = _proposal_id(proposal)
    return payload


def _backlog_item_from_proposal(proposal: ActionProposal, source: str) -> dict[str, Any]:
    """把行动提案转为当前回合待办项。"""

    duration = "delayed" if proposal.time_cost > 0 else "instant"
    return {
        "id": str(uuid.uuid4()),
        "description": proposal.description,
        "source": source,
        "duration": duration,
        "delay_count": proposal.time_cost if duration == "delayed" else 0,
        "removable": True,
        "severity_warning": "",
        "deterministic": False,
        "deterministic_effects": {},
        "dc": proposal.dc,
        "modifier": proposal.modifier,
        "main_dimension": proposal.main_dimension,
        "auxiliary_dimensions": proposal.auxiliary_dimensions,
        "tags": proposal.tags,
        "npc_id": proposal.npc_id,
        "success_rate": proposal.success_rate,
        "predicted_effects": {
            "success_rate": proposal.success_rate,
            "success": "目标按预期推进",
            "partial": "目标推进但产生代价",
            "failure": "目标受阻并可能留下隐患",
        },
    }


def _proposal_from_backlog_item(item: Mapping[str, Any]) -> ActionProposal:
    """把待办项还原为行动提案供骰子与后果系统使用。"""

    return ActionProposal(
        action_type="action",
        description=str(item.get("description") or "未命名待办"),
        dc=_safe_int(item.get("dc"), 8),
        modifier=_safe_int(item.get("modifier"), 0),
        main_dimension=str(item.get("main_dimension") or "意志"),
        auxiliary_dimensions=_as_str_list(item.get("auxiliary_dimensions")),
        tags=_as_str_list(item.get("tags")),
        time_cost=0,
        success_rate=float(item.get("success_rate", 0.0) or 0.0),
        npc_id=str(item.get("npc_id", "") or ""),
    )


def _normalize_candidate_item(state: GameState, item: Mapping[str, Any], source: str) -> dict[str, Any]:
    """把LLM或事件候选项规范化为待办项。"""

    description = str(item.get("description") or item.get("name") or item.get("event") or "未命名待办").strip()
    main_dimension = str(item.get("main_dimension") or _default_main_dimension(state)).strip()
    dc = _safe_int(item.get("dc"), 8)
    modifier = _safe_int(item.get("modifier"), 0)
    # 公式化结算：用维度驱动的公式覆盖 LLM 给出的 dc/modifier
    category = classify_action(description, main_dimension)
    formula_dc = calculate_dc(category, state.dimensions.world, state.dimensions.character)
    formula_modifier = calculate_modifier(category, state.dimensions.world, state.dimensions.character)
    dc = formula_dc
    modifier = formula_modifier
    _formula_cid = getattr(state, '_runtime_chat_id', '')
    if _formula_cid:
        get_tracer(_formula_cid, enabled=getattr(state, 'trace_enabled', True)).record(
            'formula_calc', '_normalize_candidate_item',
            {'description': description, 'category': category, 'formula_dc': formula_dc, 'formula_modifier': formula_modifier,
             'world_dims': dict(state.dimensions.world), 'char_dims': dict(state.dimensions.character)},
            GameTracer.state_summary(state))
    nature_info = classify_action_nature(description, category)
    if nature_info["nature"] == "command":
        duration = "instant"
        delay_count = 0
        deterministic = True
        deterministic_effects = predict_effects(category, "成功")
        dc = 4
        modifier = formula_modifier
    else:
        duration = str(item.get("duration", "instant")).strip()
        if duration not in {"instant", "delayed"}:
            duration = "delayed" if nature_info["default_delay"] > 0 else "instant"
        raw_delay = max(0, _safe_int(item.get("delay_count", item.get("time_cost")), 0))
        delay_count = max(raw_delay, nature_info["default_delay"])
        if delay_count > 0:
            duration = "delayed"
        deterministic = bool(item.get("deterministic", False))
        deterministic_effects = item.get("deterministic_effects", {}) if isinstance(item.get("deterministic_effects"), dict) else {}
    return {
        "id": str(item.get("id") or uuid.uuid4()),
        "description": description,
        "source": source,
        "duration": duration,
        "delay_count": delay_count,
        "removable": True,
        "severity_warning": str(item.get("severity_warning") or "").strip(),
        "deterministic": deterministic,
        "deterministic_effects": deterministic_effects,
        "dc": dc,
        "modifier": modifier,
        "main_dimension": main_dimension,
        "auxiliary_dimensions": _as_str_list(item.get("auxiliary_dimensions")),
        "tags": _as_str_list(item.get("tags")),
        "npc_id": str(item.get("npc_id", "") or ""),
        "success_rate": calculate_success_rate(formula_modifier, formula_dc),
        "predicted_effects": get_predicted_effects_all_tiers(category),
        "category": category,
        "nature": nature_info["nature"],
    }


def _backlog_item_from_event(state: GameState, event: Mapping[str, Any]) -> dict[str, Any]:
    """把被动事件转为下回合可移除待办。"""

    description = str(event.get("description") or event.get("name") or "突发事件需要回应。")
    severity = str(event.get("severity") or event.get("payload", {}).get("severity") if isinstance(event.get("payload"), dict) else "").strip()
    warning = f"若不处理，{description}可能继续恶化。" if not severity else f"若不处理，{description}可能造成{severity}级后果。"
    return _normalize_candidate_item(
        state,
        {
            "description": f"应对：{description}",
            "source": "passive_event",
            "severity_warning": warning,
            "dc": 8,
            "main_dimension": "意志",
        },
        "passive_event",
    )


def _world_state_payload(state: GameState) -> dict[str, Any]:
    """提取世界状态用于卡片预览。"""

    return {
        "time": _format_state_time(state),
        "world": dict(state.dimensions.world),
        "character": dict(state.dimensions.character),
        "extensions": dict(state.dimensions.extensions),
    }


def _weaken_dimension_effects(effects: Any) -> dict[str, Any]:
    """削弱干扰：维度增减折半，其他效果保留。"""

    if not isinstance(effects, dict):
        return {}
    weakened = json.loads(json.dumps(effects, ensure_ascii=False))
    dimensions = weakened.get("dimensions")
    if isinstance(dimensions, dict):
        for category_value in dimensions.values():
            if isinstance(category_value, dict):
                for key, value in list(category_value.items()):
                    if isinstance(value, int):
                        category_value[key] = int(value / 2)
    return weakened


def _proposal_from_payload(payload: Any) -> ActionProposal:
    """反序列化行动提案。"""

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    return ActionProposal(
        action_type=payload.get("action_type", "action"),
        description=str(payload.get("description") or payload.get("summary") or "未命名行动"),
        dc=_safe_int(payload.get("dc"), 8),
        modifier=_safe_int(payload.get("modifier"), 0),
        main_dimension=str(payload.get("main_dimension") or payload.get("main_dim") or "意志"),
        auxiliary_dimensions=_as_str_list(payload.get("auxiliary_dimensions") or payload.get("aux_dims")),
        tags=_as_str_list(payload.get("tags")),
        time_cost=max(0, _safe_int(payload.get("time_cost"), 0)),
        success_rate=float(payload.get("success_rate", 0.0) or 0.0),
        npc_id=str(payload.get("npc_id", "") or ""),
    )


def _proposal_id(proposal: ActionProposal) -> str:
    """生成稳定提案ID。"""

    raw = f"{proposal.description}|{proposal.main_dimension}|{proposal.dc}|{proposal.time_cost}|{proposal.npc_id}"
    return str(abs(hash(raw)))


def _remove_pending_proposal(state: GameState, proposal: ActionProposal) -> None:
    """从待确认列表移除已执行提案。"""

    proposal_id = _proposal_id(proposal)
    state.pending_actions = [
        item for item in state.pending_actions
        if str(item.get("proposal_id", "")) != proposal_id and item.get("description") != proposal.description
    ]


def _find_partial_success(state: GameState, proposal: ActionProposal) -> dict[str, Any] | None:
    """查找部分成功待选代价。"""

    proposal_id = _proposal_id(proposal)
    for item in state.pending_actions:
        if item.get("kind") != "partial_success":
            continue
        payload = item.get("proposal") if isinstance(item.get("proposal"), dict) else {}
        if str(payload.get("proposal_id")) == proposal_id or payload.get("description") == proposal.description:
            return item
    return None


def _queue_action_if_needed(state: GameState, proposal: ActionProposal, effects: Mapping[str, Any], passive: bool = False) -> None:
    """把耗时行动写入推进队列。"""

    if proposal.time_cost <= 0:
        return
    queued = {
        "description": proposal.description,
        "proposal": _proposal_payload(proposal),
        "time_cost": proposal.time_cost,
        "remaining_time": proposal.time_cost,
        "source": "passive_response" if passive else "action",
    }
    if isinstance(effects.get("advance_action"), dict):
        queued.update(effects["advance_action"])
    state.advance_queue.append(queued)


def _default_main_dimension(state: GameState) -> str:
    """获取默认主维度。"""

    if state.dimensions.character:
        return next(iter(state.dimensions.character.keys()))
    return "意志"


def _asks_for_numbers(text: str) -> bool:
    """判断是否显式查询数值。"""

    return any(keyword in text for keyword in ["多少", "几", "数值", "维度", "状态", "查看", "情况"])


def _serialize_record(value: Any) -> dict[str, Any] | None:
    """把dataclass或对象转为可展示dict。"""

    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": str(value)}


def _safe_int(value: Any, default: int = 0) -> int:
    """安全转换整数。"""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _safe_int_dict(value: Any, default: dict[str, int]) -> dict[str, int]:
    """安全转换整数字典。"""

    if not isinstance(value, dict):
        return dict(default)
    return {str(key): _safe_int(item) for key, item in value.items()}


def _as_list(value: Any) -> list[Any]:
    """安全转换列表。"""

    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _as_str_list(value: Any) -> list[str]:
    """安全转换字符串列表。"""

    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


__all__ = ["handle_message", "handle_card_action"]
