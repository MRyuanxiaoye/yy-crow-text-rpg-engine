# Phase 3：朝会模式——多NPC同场对话

## 前置依赖

Phase 2 完成（GM解读层 + NPC位置/可达性）。

## 目标

实现多NPC同场对话模式，NPC之间可互动争执，模拟真实多人对话规则。

## 1. 对话模式扩展

### 1.1 GameState 新增字段

**`src/engine/state.py`**：

```python
# 朝会/多人对话模式
court_session: dict[str, Any] = field(default_factory=dict)
# 格式：
# {
#   "active": True/False,
#   "npcs": ["孙承宗", "温体仁"],
#   "topic": "辽东军饷",
#   "scene_type": "formal" / "private",  # 正式朝会 vs 私下议事
#   "round": 1,
#   "history": [
#     {"round": 1, "speaker": "孙承宗", "mode": "active", "content": "..."},
#     {"round": 1, "speaker": "温体仁", "mode": "react", "content": "..."},
#   ]
# }
```

### 1.2 发起方式

通过 Phase 2 的 GM 解读层自然触发：

- 玩家输入"把温体仁和孙承宗叫来商议辽东军饷"
- GM 解读返回 `type: "multi_talk"`，`target_npcs: ["温体仁", "孙承宗"]`
- 确认卡片展示可参与NPC（含位置/可达性），不在场的标注需召回
- 玩家确认后，初始化 `court_session`

### 1.3 与一对一对话的关系

- 一对一对话（`talking_to`）和朝会模式（`court_session`）互斥
- 进入朝会时清空 `talking_to`；进入一对一时清空 `court_session`
- 朝会中玩家可"单独拉某人到一边"→ 退出朝会，进入一对一

## 2. 发言序列裁定

### 2.1 每轮流程

```
玩家发言（泛问/点名/指示）
    ↓
发言序列裁定（轻量LLM调用）
    ↓
按序生成NPC发言（串行，后面的NPC看到前面的话）
    ↓
组合成一张卡片返回
    ↓
玩家下一轮输入
```

### 2.2 裁定 Prompt

```
你是朝会裁判，决定这一轮哪些NPC发言、按什么顺序。

【在场NPC】
{npc_list_with_brief}  // 每个NPC：名字、职位、阵营、性格关键词

【议题】
{topic}

【场景类型】
{scene_type}  // formal（正式朝会，官阶影响顺序）/ private（私下议事，性格影响顺序）

【本轮上下文】
玩家说：{player_text}
上一轮发言记录：{last_round_summary}

【是否点名】
{addressed_npc or "未点名"}

请输出JSON：
[
  {"npc": "孙承宗", "mode": "active", "reason": "话题直接涉及其职责"},
  {"npc": "温体仁", "mode": "react", "reason": "被点名的政敌话题，会借机发难"},
  {"npc": "王承恩", "mode": "silent", "expression": "微微颔首但不开口", "reason": "此议题非其所长，且场合正式"}
]
```

mode 类型：
- `"active"` — 完整发言（50-160字），由 NPC 引擎生成
- `"react"` — 简短回应（20-60字），由 NPC 引擎生成，prompt 注明"简短回应"
- `"silent"` — 不发言，旁白补一句非语言反应（expression 字段）

### 2.3 点名规则

- 玩家明确点名（"孙老，你怎么看"）→ 被点名NPC必须 mode=active，其他默认 silent（除非有强烈动机）
- 玩家泛问（"辽东军饷怎么办"）→ 按裁定排序多人发言
- NPC自发互动（如被暗讽后反击）→ 裁定中 reason 说明动机

## 3. 串行生成NPC发言

### 3.1 关键设计

后面的NPC**能看到**前面NPC说的话。这是朝会模式的核心价值——产生碰撞。

实现：在 `npc_engine.py` 的 `generate_npc_reply()` 调用中，将前面NPC的发言追加到 user_content：

```python
# 朝会模式下的 user_content 扩展
court_context = "【本轮其他人已说的话】\n"
for prior in prior_speakers:
    court_context += f"{prior['npc']}：{prior['content']}\n"
court_context += f"\n玩家对{'所有人' if not addressed else addressed}说：{player_text}"
```

### 3.2 NPC感知差异

每个NPC的 system prompt 增加场景信息：

```
【当前场景：多人议事】
- 在场人物：{npc_list}
- 议题：{topic}
- 场合：{formal/private}
- 你的发言模式：{active/react}（{reason}）
- 注意：这是公开场合，你的发言所有在场人都能听到。私下不想说的话不要在这里说。
```

公开场合 vs 私下的差异由 `scene_type` 控制：
- formal：NPC更注重面子、官阶、言辞分寸
- private：NPC更直接，可能说出公开场合不说的话

## 4. 飞书组合卡片

### 4.1 卡片结构

**`src/feishu/card_builder.py`** 新增：

```python
def build_court_session_card(
    topic: str,
    round_num: int,
    responses: list[dict],  # [{"npc": "...", "mode": "active/react/silent", "content": "...", "expression": "..."}]
    present_npcs: list[str],
    pending_npcs: list[dict],  # 召回中的NPC [{"name": "...", "eta": 1}]
) -> dict:
```

卡片布局：

```
🏛 朝会议事 · {topic} · 第{round}轮

【{npc_name}】（{title}）
"{content}"

【{npc_name2}】（{title2}）
"{content}"

💭 {silent_npc_expression}

---
💭 在场：{present_npcs}
📍 {pending_npc}（召回中，{eta}回合后到达）

[对{npc1}说...]  [对{npc2}说...]  [对所有人说...]
[结束议事]  [自由输入 ▾]
```

### 4.2 按钮回调

- `[对X说...]` → 设置 addressed_npc，等待玩家输入内容
- `[对所有人说...]` → 不设置 addressed_npc，等待泛问输入
- `[结束议事]` → 清空 court_session，回到自由模式
- `[自由输入]` → 打开自由文本输入框

## 5. 朝会结束处理

朝会结束时：
1. 对话历史（court_session.history）压缩为摘要存入各NPC记忆
2. 公开场合发言作为公共信息，所有在场NPC都"知道"
3. 清空 court_session，恢复自由模式
4. 旁白生成一段朝会结束叙事

## 6. 验收标准

1. **发起朝会**：自由文本"叫温体仁和孙承宗来商议" → GM解读 → 确认卡片 → 进入朝会
2. **泛问多人回应**：玩家说"辽东怎么办" → 多NPC依序发言 → 后者能回应前者
3. **点名提问**：玩家说"孙老你说说" → 孙承宗完整回答 → 其他人沉默+非语言反应
4. **NPC自发互动**：温体仁被暗讽 → 系统裁定温反击 → 不需要玩家触发
5. **组合卡片**：一张卡片包含所有发言 + 底部按钮选回应对象
6. **公私差异**：正式朝会语气更克制，私下议事更直接

## 7. 执行顺序

1. `state.py`：新增 court_session 字段 + 序列化
2. `game_master.py`：朝会流程主控（发起、每轮循环、结束）
3. 发言裁定：新增 `court_arbiter.py`（或在 game_master 内）
4. `npc_engine.py`：支持朝会上下文注入（前面NPC发言 + 场景信息）
5. `card_builder.py`：朝会组合卡片
6. 朝会记忆：结束时压缩存入NPC记忆
7. 验证：完整朝会流程端到端测试
