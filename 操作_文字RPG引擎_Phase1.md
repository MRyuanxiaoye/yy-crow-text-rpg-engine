# 操作_文字RPG引擎_Phase1

> 目标：搭建文字RPG游戏引擎框架，跑通从飞书接收消息→GameMaster路由→旁白/NPC回复的完整链路。
> 不含剧本内容，用硬编码的测试场景验证引擎。

---

## 全局约束

- 新项目，独立仓库，项目名 `text-rpg-engine`，放在 `/Users/yuanye/Documents/text-rpg-engine/`
- Python 3.12，FastAPI + uvicorn，async/await 全程异步
- LLM 用 DeepSeek（OpenAI SDK 兼容，base_url: `https://api.deepseek.com`，model: `deepseek-chat`）
- 飞书集成模式参考数码生物项目（`/Users/yuanye/Documents/情感陪伴数码生物/src/feishu/`），但完全重写，不直接复制
- 两个飞书Bot应用共用一个后端服务，通过不同的 app_id/app_secret 区分
- 代码注释用中文，变量名函数名用英文
- 配置通过 `.env` 文件管理，用 pydantic-settings 解析
- JSON 文件存储（MVP阶段不用数据库）

---

## 步骤 1：项目初始化

创建项目目录结构：

```
text-rpg-engine/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── config.py             # 配置管理（pydantic-settings）
│   ├── feishu/
│   │   ├── __init__.py
│   │   ├── receiver.py       # 飞书事件接收
│   │   ├── sender.py         # 飞书消息发送（文本+卡片）
│   │   └── card_builder.py   # 飞书卡片JSON构建器
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── game_master.py    # 游戏主控
│   │   ├── narrator.py       # 旁白生成引擎
│   │   ├── npc_engine.py     # NPC对话引擎
│   │   ├── judge.py          # 决策判定器
│   │   └── state.py          # 游戏状态管理+存档
│   ├── casting/
│   │   ├── __init__.py
│   │   ├── matcher.py        # 角色槽位匹配
│   │   └── adapter.py        # 历史人物适配
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── loader.py         # 剧本加载器
│   └── llm/
│       ├── __init__.py
│       └── client.py         # DeepSeek调用封装
├── scripts/                   # 剧本目录（Phase 2 填充）
├── characters_db/             # 历史人物库（Phase 2 填充）
├── data/
│   └── saves/                 # 存档目录
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore
```

**requirements.txt** 包含：
- fastapi, uvicorn[standard]
- openai (DeepSeek SDK)
- pydantic-settings
- pyyaml
- httpx (飞书API调用)
- python-dotenv

**config.py** 需要管理的配置项：
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`
- `NARRATOR_APP_ID`, `NARRATOR_APP_SECRET`（旁白Bot的飞书应用凭证）
- `NPC_APP_ID`, `NPC_APP_SECRET`（NPC Bot的飞书应用凭证）
- `FEISHU_ENCRYPT_KEY`, `FEISHU_VERIFICATION_TOKEN`（飞书事件订阅验证，两个Bot可共用或各一套）
- `GAME_CHAT_ID`（游戏群的chat_id）
- `SERVER_PORT`（默认8001，避免和数码生物的8000冲突）

---

## 步骤 2：飞书双Bot接入

### 2.1 sender.py — 消息发送

实现两个发送通道：

- `send_narrator_card(chat_id, card_json)` — 用旁白Bot的token发送卡片消息
- `send_npc_text(chat_id, role_name, text)` — 用NPC Bot的token发送文本消息，格式为 `【{role_name}】{text}`

每个Bot独立管理自己的 tenant_access_token，自动刷新（参考数码生物的实现逻辑）。

内部维护两套token：`_narrator_token` 和 `_npc_token`，各自独立过期和刷新。

### 2.2 receiver.py — 事件接收

一个 FastAPI endpoint `/feishu/event` 统一接收两个Bot的事件回调。

处理逻辑：
1. 飞书事件验证（URL verification challenge）
2. 事件去重（event_id 缓存）
3. 过滤掉Bot自己发的消息（通过 sender_id 判断，排除两个Bot的 open_id）
4. 提取用户消息内容
5. 转发给 GameMaster 处理

注意：两个飞书Bot应用需要在飞书开放平台各自配置事件订阅，但回调URL可以指向同一个endpoint。通过请求中的 app_id 或事件内容区分来源（实际上用户消息只会从一个入口进来，不需要区分）。

### 2.3 card_builder.py — 卡片消息构建

提供几种卡片模板函数：

- `build_narration_card(title, content, choices=None)` — 叙事卡片，标题+正文+可选的选项列表
- `build_report_card(title, metrics_text)` — 国情奏报卡片
- `build_transition_card(text)` — 过渡叙事卡片（时间推进）

卡片使用飞书交互卡片JSON格式。选项列表用文本展示（如 `▸ 选项A`），不用飞书按钮交互（MVP阶段用户直接打字选择）。

---

## 步骤 3：LLM 客户端封装

### llm/client.py

封装 DeepSeek 调用，提供统一接口：

- `async def chat(system_prompt, user_content, temperature=0.7) -> str` — 基础对话
- `async def chat_json(system_prompt, user_content, temperature=0.3) -> dict` — 返回JSON的对话（自动解析，处理markdown代码块包裹的情况）

全局复用一个 AsyncOpenAI 实例。

---

## 步骤 4：游戏状态管理

### engine/state.py

**GameState 数据类**，包含：
- `script_id: str` — 当前剧本ID
- `chapter_id: str` — 当前章节
- `phase: str` — 当前阶段（"free_dialogue" / "decision" / "transition"）
- `turn: int` — 回合数
- `game_date: str` — 游戏内日期（如"崇祯元年三月"）
- `metrics: dict[str, int]` — 数值系统（国库/军力/民心/朝廷稳定/边防）
- `storylines: dict[str, dict]` — 故事线状态
- `active_npcs: dict[str, dict]` — 当前活跃NPC及其状态（trust/alive/position）
- `cast: dict[str, str]` — 动态选角结果
- `decisions: list[dict]` — 决策历史
- `current_scene: dict` — 当前场景（location/present_npcs/context）
- `conversation_history: list[dict]` — 最近对话记录（用于LLM上下文）

**StateManager 类**：
- `load(chat_id) -> GameState` — 从 `data/saves/{chat_id}.json` 加载
- `save(chat_id, state)` — 保存到JSON文件
- `apply_effects(state, effects)` — 应用决策后果到状态（修改metrics/npc/storyline）
- `get_metrics_description(state) -> str` — 将数值转为文字描述（0-20极差/21-40差/41-60一般/61-80好/81-100极好）

conversation_history 只保留最近30条，更早的消息在存档中保留但不进LLM上下文。

---

## 步骤 5：GameMaster 游戏主控

### engine/game_master.py

这是引擎的核心路由器。

**`async def handle_message(chat_id, user_text) -> None`**

处理流程：
1. 加载当前 GameState
2. 将用户消息记入 conversation_history
3. 用 LLM 判断用户消息意图（intent classification）：
   - `"dialogue"` — 想和某个NPC对话（如"召见袁崇焕"、直接对话内容）
   - `"action"` — 想执行某个行动（如"派人打探"、"微服出巡"）
   - `"decision"` — 做出决策（如"全权委任袁崇焕"、"传旨"、"朕决定了"）
   - `"query"` — 查询信息（如"国库还有多少"、"边防情况如何"）
   - `"meta"` — 游戏外操作（如"存档"、"帮助"、"退出"）
4. 根据意图路由：
   - dialogue → 调用 NPC Engine，NPC Bot 发送回复
   - action → 调用 Narrator 描述行动过程和结果，旁白Bot发送
   - decision → 调用 Judge 判定后果 → 更新状态 → 调用 Narrator 叙述结果 → 旁白Bot发送 → 检查是否触发新事件
   - query → 调用 Narrator 生成信息回复（可能包含具体数值），旁白Bot发送
   - meta → 直接处理（存档确认/帮助信息等）
5. 保存状态（自动存档）

**意图分类 prompt**（传给LLM）：

```
你是一个文字RPG游戏的意图识别器。玩家扮演皇帝，判断玩家消息的意图。

当前场景：{scene_description}
当前阶段：{phase}
在场NPC：{present_npcs}

玩家消息：{user_text}

判断意图类型，输出JSON：
{"intent": "dialogue/action/decision/query/meta", "target_npc": "NPC名或null", "detail": "一句话说明"}
```

---

## 步骤 6：旁白引擎

### engine/narrator.py

旁白引擎负责所有叙事性内容的生成。

**核心函数：**

- `async def narrate_scene(state, scene_setup) -> str` — 场景描述（章节开场、场景切换）
- `async def narrate_action(state, action_desc) -> str` — 行动叙述（用户执行行动的过程和结果）
- `async def narrate_decision_result(state, decision, effects) -> str` — 决策后果叙述
- `async def narrate_transition(state, time_skip, events_during) -> str` — 时间推进过渡叙事
- `async def narrate_query(state, query) -> str` — 信息查询回复（可包含数值）
- `async def present_decision(state, decision_point) -> str` — 呈现决策选项

**旁白 system prompt 核心要素：**
- 你是一个历史文字RPG的旁白
- 叙事风格：古典但不晦涩，有画面感，适度使用文言词汇增加氛围
- 保持客观中立，不替玩家做决定
- 描述时注入当前状态的文字描述（国力/民心等，不直接说数字除非玩家问）
- 了解真实历史但不拘泥，玩家的决策可以改变历史走向

---

## 步骤 7：NPC对话引擎

### engine/npc_engine.py

NPC引擎负责扮演具体历史人物与玩家对话。

**核心函数：**

- `async def generate_npc_reply(state, npc_name, npc_profile, user_text) -> str` — 生成NPC回复

**NPC system prompt 构建逻辑：**
1. 注入人格卡信息（性格、说话风格、立场、当前职位）
2. 注入当前场景和对话上下文
3. 注入该NPC对玩家的信任度和关系
4. 注入该NPC知道的信息（基于故事线状态，NPC不应该知道自己不该知道的事）

**NPC的信息边界**（重要）：
- 每个NPC只知道自己职责范围内的事
- 袁崇焕知道辽东军事细节，不知道朝廷内部的党争内幕
- NPC的"知识范围"由其角色定位+当前剧情阶段决定
- 这需要在 prompt 中明确告知 LLM 该角色知道什么、不知道什么

---

## 步骤 8：决策判定器

### engine/judge.py

处理用户的决策（包括自由输入），判定合理性和后果。

**核心函数：**

- `async def judge_decision(state, user_decision, decision_point=None) -> JudgeResult`

**JudgeResult 包含：**
- `valid: bool` — 决策是否合理可执行
- `rejection_reason: str` — 如果不合理，拒绝原因（旁白会用这个劝谏玩家）
- `effects: dict` — 对 metrics/npc/storyline 的影响
- `narrative_hint: str` — 给旁白的叙事提示（后果概要）
- `triggered_events: list[str]` — 触发的新事件ID

**判定逻辑：**
1. 如果有预设决策点（decision_point），先检查用户输入是否匹配某个预设选项
2. 匹配到预设选项 → 直接返回预设的 effects
3. 未匹配（自由输入）→ 调用 LLM 判定：
   - 基于当前状态、历史背景、合理性，判断这个决策能否执行
   - 如果可执行，生成 effects（数值变化、NPC态度变化等）
   - 如果不合理，返回拒绝原因（如"大臣们群起反对"、"国库不足以支持此行动"）

**Judge 的 LLM prompt 要点：**
- 你是一个历史合理性判定器
- 基于明末的真实国力、制度、社会状况来判断决策可行性
- 对合理的创新决策持开放态度（改变历史是游戏目的）
- 对明显荒谬的决策给出合理的拒绝（如"把皇位让给李自成"→大臣不可能同意）
- 输出结构化JSON：effects 的格式要和 StateManager.apply_effects 兼容

---

## 步骤 9：剧本加载器

### scripts/loader.py

加载 YAML 格式的剧本文件。

**核心函数：**

- `load_manifest(script_id) -> dict` — 加载剧本元信息
- `load_chapter(script_id, chapter_id) -> dict` — 加载章节定义
- `load_npc_profile(script_id, npc_name) -> dict` — 加载NPC人格卡
- `load_decision_point(script_id, decision_id) -> dict` — 加载决策点定义
- `load_timeline(script_id) -> list[dict]` — 加载时间线

剧本文件路径：`scripts/{script_id}/`

MVP阶段先创建一个极简的测试剧本 `scripts/test/`，用于验证引擎：
- 一个章节、一个NPC、一个决策点
- 验证完整链路能跑通

---

## 步骤 10：测试验证场景

创建 `scripts/test/` 测试剧本：

**manifest.yaml**：
```yaml
id: test
name: 测试剧本
player_role: 国王
description: 用于引擎验证的最小剧本
```

**chapters/ch01_test.yaml**：
```yaml
id: ch01_test
title: 测试章节
game_date_start: "第一天"
opening_narration: |
  你是一个小国的国王。今天，将军来觐见，说边境有敌人入侵。
initial_scene:
  location: 王座大厅
  present_npcs: [将军]
  context: 将军来报告边境战事
```

**fixed_npcs/将军.yaml**：
```yaml
name: 将军
personality: 忠诚、直率、有经验
speaking_style: 军人作风，简洁有力
stance: 忠于国王，主战
```

**decisions/D001_test.yaml**：
```yaml
id: D001_test
chapter: ch01_test
narrator_setup: "将军问你：是战是和？"
preset_choices:
  - label: 出兵迎战
    effects:
      metrics: {军力: -10, 民心: +5}
  - label: 议和
    effects:
      metrics: {国库: -20, 民心: -5}
free_input: true
```

---

## 步骤 11：main.py 入口和启动

**main.py** 做的事：
1. 创建 FastAPI app
2. 注册飞书事件回调路由 `/feishu/event`
3. 启动时初始化：加载配置、创建LLM客户端、初始化StateManager
4. 健康检查 endpoint `/health`

**启动命令**：`uvicorn src.main:app --host 0.0.0.0 --port 8001`

---

## 步骤 12：Docker 部署配置

**Dockerfile** 和 **docker-compose.yml** 参考数码生物项目的模式，但：
- 端口改为 8001
- 容器名改为 `text-rpg-engine`
- 环境变量对应新的配置项

---

## 部署与飞书配置顺序（重要）

飞书事件订阅需要验证回调URL，所以必须**先部署服务再配飞书**。正确顺序：

### 前置准备（用户手动，代码执行前即可做）
1. 去飞书开放平台创建两个企业自建应用（旁白Bot + NPC Bot）
2. 拿到各自的 App ID 和 App Secret
3. 开通权限：`im:message:send_as_bot`（发消息）
4. **暂不配置**事件订阅回调URL（服务还没部署）

### 代码完成后的部署流程
1. 用户填入 `.env` 的 DeepSeek 和飞书 App 凭证
2. Docker 部署到服务器（159.75.246.98:8001）
3. 验证 `/health` 端点可访问
4. 回飞书开放平台 → 旁白Bot → 事件订阅 → 填回调URL：`http://159.75.246.98:8001/feishu/event`
   - 飞书会发送 URL verification challenge，服务端 receiver.py 必须正确响应
   - 事件类型勾选：`im.message.receive_v1`（接收消息）
5. NPC Bot 不需要配事件订阅（只发不收）
6. 创建飞书群 → 把两个Bot都拉进群 → 拿到 chat_id
7. 将 chat_id 填入 `.env` 的 `GAME_CHAT_ID`，重启服务
8. 在群里发消息测试

### .env 配置模板
```env
# DeepSeek
DEEPSEEK_API_KEY=sk-515edc5dd5754bc78efd4b20217c3837
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# 旁白Bot（飞书应用凭证）
NARRATOR_APP_ID=cli_a977204f24f09bdd
NARRATOR_APP_SECRET=II2blmWIZ1GBD3rR1dVyHGbDNZgl6zb3

# NPC Bot（飞书应用凭证）
NPC_APP_ID=cli_a9772162eefc5bd7
NPC_APP_SECRET=qSEayf4u0zORnu799NG2rhMjVWXJyDK4

# 飞书事件验证（旁白Bot的加密策略，部署前在飞书后台设置）
FEISHU_ENCRYPT_KEY=ZO8KRAsJvMIfwutKXVvN3ggdRZ3DPrnm
FEISHU_VERIFICATION_TOKEN=ZBbl0dXG9YJXpsuoXAUcleGoNaLspnit

# 游戏群
GAME_CHAT_ID=（部署后创建群再填）

# 服务
SERVER_PORT=8001
```

---

## 完成检查清单

- [ ] 项目目录结构完整，所有 `__init__.py` 就位
- [ ] `.env.example` 包含所有必要配置项
- [ ] 飞书双Bot的token独立管理和刷新
- [ ] 飞书 URL verification challenge 能正确响应（`receiver.py` 处理 `"challenge"` 字段并原样返回）
- [ ] 用户消息 → GameMaster → 意图识别 → 路由到对应引擎 完整链路通畅
- [ ] 旁白Bot发送卡片消息、NPC Bot发送文本消息，格式正确
- [ ] GameState 能正确序列化/反序列化到JSON
- [ ] 自动存档在每次决策后触发
- [ ] 决策判定器能处理预设选项和自由输入
- [ ] 测试剧本加载正常，能跑通一个完整的：开场叙事→NPC对话→决策→后果叙述 流程
- [ ] 日志输出关键节点信息（意图识别结果、NPC选择、决策判定等）
