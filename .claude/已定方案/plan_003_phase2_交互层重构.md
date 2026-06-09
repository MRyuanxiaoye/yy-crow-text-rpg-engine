# Phase 2：交互层重构——GM解读层 + NPC位置/可达性

## 前置依赖

Phase 1 完成（NPC有 character_seed + timeline + initial_state.location）。

## 目标

1. 用 GM 解读层统一处理所有玩家自由文本，替代当前的意图分类 + 行动提炼两步流程
2. NPC 有物理位置和可达性，影响对话和行动
3. 时间成本操作（如召回远处NPC）有明确的游戏机制

## 1. GM 解读层

### 1.1 设计

所有非按钮操作的玩家自由文本，统一经过一次 LLM 调用，以游戏主持人视角解读。

替换 `game_master.py` 中的 `INTENT_SYSTEM_PROMPT`（第66行）和 `ACTION_ANALYSIS_PROMPT`（第79行），合并为一个 GM 解读调用。

### 1.2 GM 解读 Prompt

```
你是这个文字RPG的游戏主持人（GM）。玩家发来了一段话，请以GM视角解读：

【当前世界状态】
{world_brief}

【玩家角色】
{player_role_brief}

【在场/可联系的NPC】
{npc_list_with_locations}

【玩家输入】
{player_text}

请判断玩家想做什么，评估可行性和代价，输出JSON：
{
  "interpretation": "你对玩家意图的理解（一句话）",
  "feasibility": "可行/有条件/不可行",
  "options": [
    {
      "type": "talk/action/query/summon/multi_talk",
      "description": "选项描述",
      "target_npcs": ["NPC名"],
      "time_cost": 0,
      "prerequisites": "前置条件说明（如NPC不在身边需N回合召回）",
      "risk_hint": "风险提示（可选）"
    }
  ],
  "gm_note": "给玩家的GM补充说明（如为什么某个选项不可行）"
}
```

### 1.3 流程

```
玩家自由文本
    ↓
GM解读（一次LLM调用）
    ↓
生成确认卡片（展示选项+代价+可行性）
    ↓
玩家点选确认
    ↓
路由执行：
  - talk → 发起对话（Phase 1 已有）
  - action → 行动提案+掷骰（现有流程）
  - query → 旁白回答查询（现有流程）
  - summon → 加入召回队列（新增）
  - multi_talk → 进入朝会模式（Phase 3）
```

### 1.4 代码改造

**`src/engine/game_master.py`**：
- 新增 `async def gm_interpret(state, player_text) -> dict`
- 调用 LLM 生成上述 JSON
- 现有 `_classify_intent()` 或类似函数被此替代
- 按钮操作（已有明确语义）跳过 GM 解读，直接路由

**`src/feishu/card_builder.py`**：
- 新增 `build_gm_interpretation_card(interpretation_result)` 
- 卡片展示：GM解读 + 选项列表（每个选项一个按钮）+ 代价/风险提示
- 如果只有一个明确选项（如"找身边的NPC说话"），可简化卡片

## 2. NPC 位置与可达性

### 2.1 GameState 扩展

**`src/engine/state.py`**，GameState 新增字段：

```python
# NPC 运行时位置追踪
npc_locations: dict[str, dict] = field(default_factory=dict)
# 格式：{"袁崇焕": {"location": "辽东", "reachability": "distant", "recall_cost": 2}}
```

可达性枚举：
- `"present"` — 在玩家身边，可即时对话
- `"nearby"` — 同城/同区域，1回合内可召见
- `"distant"` — 远处，需N回合召回
- `"unreachable"` — 当前不可联系（如已死亡、被囚、敌方阵营）

### 2.2 位置初始化

游戏开局时，从每个 NPC 的 `initial_state.location` 初始化 `npc_locations`。

**`src/engine/state.py` 或 `src/engine/seed.py`**：在游戏初始化时加载 NPC 初始位置，并根据玩家角色的起始位置计算 reachability。

规则：
- 玩家是皇帝，起始位置京师 → 京师NPC = present，辽东NPC = distant
- 玩家是边关将领，起始位置辽东 → 辽东NPC = present，京师NPC = distant

### 2.3 位置随时间更新

当 timeline 事件触发且包含 `location` 字段时，自动更新 `npc_locations`。

**`src/engine/event_scheduler.py` 或时间推进流程中**：事件触发后检查是否有 NPC 位置变化。

### 2.4 召回机制

玩家通过 GM 解读层发起"召见远处NPC"时：

1. GM 解读返回 `type: "summon"`，`time_cost: N`
2. 玩家确认后，创建一个 delayed_queue 条目：
   ```python
   {"type": "summon", "npc": "袁崇焕", "remaining_advances": 2, "on_complete": "set_reachability_present"}
   ```
3. 时间推进N回合后，NPC可达性变为 present，系统通知玩家
4. 玩家可在NPC到达后发起对话

### 2.5 对话前可达性检查

**`src/engine/game_master.py`** 中发起对话前：
- 检查 `npc_locations[npc_name]["reachability"]`
- present → 直接对话
- nearby → 提示需1回合，卡片确认
- distant → 提示需N回合召回，卡片确认
- unreachable → 卡片说明原因，拒绝

## 3. 验收标准

1. **自由文本正确路由**：输入"召见户部尚书"→ GM解读为对话意图 → 展示确认卡片 → 确认后进入对话
2. **位置约束生效**：辽东的袁崇焕不能在京师即时对话，需要召回
3. **复合意图处理**：输入"下江南找美人" → GM解读为多步行动 → 展示可行性和代价
4. **永远有反馈**：任何自由文本都会得到GM解读卡片回复，不再静默
5. **按钮不受影响**：现有按钮操作（对话按钮、决策选项等）正常工作

## 4. 执行顺序

1. `state.py`：新增 npc_locations 字段 + 序列化/反序列化
2. 游戏初始化流程：从 NPC initial_state 加载位置
3. `game_master.py`：实现 `gm_interpret()` + 路由逻辑
4. `card_builder.py`：GM解读卡片
5. 召回机制：delayed_queue 集成
6. 验证：自由文本路由 + 位置约束
