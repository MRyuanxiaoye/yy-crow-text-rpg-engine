# Phase 1：数据层重构——时间线意识

## 目标

让引擎具备时间线意识：NPC的上下文信息严格截止到当前游戏时间点，AI不接触未来事件。

## 1. NPC YAML 格式重构

### 1.1 当前格式（以袁崇焕为例）

```yaml
name: 袁崇焕
title: 辽东督师
faction: 朝廷方
personality: 忠烈刚愎、自信过人、急于求成
speaking_style: 语速急而断句硬，常以军令口吻陈词
knowledge_boundary: 熟悉辽东军情...不掌握深宫党争细节
initial_stats:
  trust: 58
  loyalty: 90
  influence: 76
events:
  - id: yuan_promoted
    chapter: ch02_jisi
    trigger: ...
    effect: ...
```

**问题**：personality 是终身评价（"忠烈刚愎"对崇祯元年和下狱后都适用，但含义不同）；events 用 chapter 而非时间锚定；没有初始状态和位置。

### 1.2 新格式

```yaml
# 性格底色：时间无关，永远注入NPC上下文
character_seed:
  name: 袁崇焕
  personality_traits: ["忠烈", "刚愎自信", "急于求成"]
  values: ["辽东防线", "军令专一", "战功自证"]
  speaking_style: "语速急而断句硬，常以军令口吻陈词，少铺垫多结论"
  behavioral_tendencies: "遇质疑倾向强硬回击而非退让；高估战功可平息朝议"
  knowledge_domain: "辽东军情、宁锦防线、关宁军务"
  knowledge_blind_spots: "深宫党争细节、后金高层私议"

# 初始状态：游戏开始时NPC的快照
initial_state:
  time: { year: 1, month: 8 }
  title: "兵部右侍郎（候任辽东）"
  faction: 朝廷方
  location: 京师
  status: 候命
  situation: "崇祯即位不久，袁崇焕因辽东战功获召入京，正等待新帝接见与任命"
  public_profile: "兵部右侍郎，辽东战功卓著，朝野寄予厚望"
  initial_relations:
    trust: 58
    loyalty: 90
    influence: 76

# 时间线事件：按时间排列，引擎按当前时间过滤
timeline:
  - id: yuan_promoted
    time: { year: 1, month: 12 }
    name: "升任辽东督师"
    trigger:
      type: timeline  # 默认按时间触发
    situation_update: "受命督师辽东，誓守宁锦"
    location: 辽东
    status: 在任
    title_update: "辽东督师"
    public_profile_update: "新任辽东督师，统领关宁军务"
    effect:
      dimensions: { world: { "边防": 1 } }

  - id: yuan_impeached
    time: { year: 2, month: 12 }
    name: "言官群劾"
    trigger:
      type: condition
      conditions:
        - "朝廷稳定 < 3"
    situation_update: "弹章如雪，从柱石变为众矢之的"
    status: 被弹劾
    effect:
      dimensions: { world: { "朝廷稳定": -1 } }
      relations: { trust: -1 }

  - id: yuan_imprisoned
    time: { year: 3, month: 1 }
    name: "下狱候审"
    trigger:
      type: decision
      decision_id: D_yuan_fate
      choice: imprison
    situation_update: "被押入诏狱，辽东军心大震"
    location: 京师诏狱
    status: 下狱
    effect:
      dimensions: { world: { "边防": -2, "军力": -1 } }

  - id: yuan_pardoned
    time: { year: 3, month: 1 }
    name: "留任戴罪"
    trigger:
      type: decision
      decision_id: D_yuan_fate
      choice: pardon
    situation_update: "诏书留人却难平众议"
    location: 辽东
    status: 复职
    effect:
      dimensions: { world: { "边防": 1, "朝廷稳定": -1 } }
      relations: { trust: 1 }
```

### 1.3 格式规则

- `character_seed`：时间无关的性格底色，**始终**注入NPC上下文
- `initial_state`：游戏时间起点（manifest.yaml 的 time.start）的NPC快照
- `timeline`：按时间排列的事件节点
  - `trigger.type: timeline` = 到时间自动触发（历史既定事件）
  - `trigger.type: condition` = 满足条件才触发（可被玩家行为改变）
  - `trigger.type: decision` = 取决于玩家特定决策
  - 互斥事件（如 imprisoned / pardoned）通过 decision 分支处理，只有一个会发生
- `situation_update`：该事件发生后NPC"知道发生了什么"的一句话描述
- `location`、`status`、`title_update`、`public_profile_update`：可选，事件改变了NPC状态时填写

### 1.4 所有 11 个 NPC YAML 都需要按此格式重构

文件列表：
- `scripts/mingmo/fixed_npcs/袁崇焕.yaml`
- `scripts/mingmo/fixed_npcs/温体仁.yaml`
- `scripts/mingmo/fixed_npcs/孙承宗.yaml`
- `scripts/mingmo/fixed_npcs/吴三桂.yaml`
- `scripts/mingmo/fixed_npcs/王承恩.yaml`
- `scripts/mingmo/fixed_npcs/李自成.yaml`
- `scripts/mingmo/fixed_npcs/皇太极.yaml`
- `scripts/mingmo/fixed_npcs/多尔衮.yaml`
- `scripts/mingmo/fixed_npcs/周延儒.yaml`
- `scripts/mingmo/fixed_npcs/张献忠.yaml`
- `scripts/mingmo/fixed_npcs/陈圆圆.yaml`

重构时参考两个数据源合并：
1. `fixed_npcs/*.yaml` 的 events、personality、speaking_style
2. `world.yaml` 的 `global_npc_pool` 对应条目的 personality_layer、goal_layer、knowledge_layer

`world.yaml` 的 `global_npc_pool` 保持不变（它服务于角色选择、casting等流程），`fixed_npcs/` 才是运行时NPC引擎的数据源。

## 2. 引擎改造：按时间点拼接NPC上下文

### 2.1 `src/scripts/loader.py`

新增函数：

```python
def load_npc_profile_at_time(script_id: str, npc_name: str, game_time: dict) -> dict:
    """加载NPC在指定时间点的完整profile。
    
    逻辑：
    1. 读取 fixed_npcs/{npc_name}.yaml
    2. 提取 character_seed（始终包含）
    3. 从 initial_state 开始，遍历 timeline 中 time <= game_time 且已触发的事件
    4. 逐个应用 situation_update、location、status 等字段
    5. 返回组装好的 profile dict
    """
```

返回的 profile 结构：

```python
{
    "character_seed": { ... },  # 原样传递
    "current_state": {
        "title": "辽东督师",          # initial_state + timeline updates
        "faction": "朝廷方",
        "location": "辽东",
        "status": "在任",
        "situation": "受命督师辽东...", # 最近一次 situation_update
        "public_profile": "...",
    },
    "experiences": [                   # 到当前时间点的所有已发生事件
        {"id": "yuan_promoted", "name": "升任辽东督师", "situation": "..."},
    ],
    "initial_relations": { ... },
}
```

关键：**timeline 中时间晚于 game_time 的事件完全不出现在返回值中**。

### 2.2 判断事件是否已触发

需要一个辅助函数，接收 GameState 判断 timeline 事件的触发状态：

```python
def _is_event_triggered(event: dict, state: GameState) -> bool:
    """判断时间线事件在当前游戏状态下是否已触发。
    
    规则：
    - timeline 类型：time <= state.game_time 即触发
    - condition 类型：time <= state.game_time 且所有 conditions 满足
    - decision 类型：time <= state.game_time 且对应决策已做出且选择匹配
    
    已触发事件ID记录在 state.event_history 中。
    """
```

### 2.3 `src/engine/npc_engine.py` 改造

#### generate_npc_reply() 函数改造

当前（第646行起）直接用完整 npc_profile 构建 system prompt。

改为：

```python
# 第656行附近，替换 npc_profile 获取方式
from src.scripts.loader import load_npc_profile_at_time

time_filtered_profile = await load_npc_profile_at_time(
    state.script_id, npc_name, state.game_time
)
# 用 time_filtered_profile 替代原 npc_profile 构建 prompt
```

#### NPC System Prompt 模板调整

当前模板（第12行 NPC_SYSTEM_PROMPT_TEMPLATE）需要新增一个段落：

```
【经历层：到目前为止你经历了什么】
{experiences_text}

【当前处境】
- 当前职位：{current_title}
- 当前位置：{current_location}
- 当前状况：{current_situation}
```

**删除或重新定义的内容**：
- `personality` 字段从 character_seed.personality_traits 生成，不再是终身评价
- 知识边界从 character_seed.knowledge_domain + knowledge_blind_spots 生成
- 新增 experiences 段落替代原来的隐式全知视角

### 2.4 `src/engine/narrator.py` 改造

#### 核心原则：旁白不该看到NPC性格底色的隐藏面

当前 `_state_brief()` (第118行) 通过 state 传递了 present_npcs 等信息，但旁白的 system prompt 会通过 `_chat_narrator()` 接收完整状态。

改造点：

1. 新增 `_public_npc_summary(state)` 函数：
   - 只返回在场NPC的**公开信息**：名字、职位、阵营、公开状态
   - 不包含：personality_traits、behavioral_tendencies、values、goals
   - 例："袁崇焕（辽东督师，朝廷方，当前在任）"

2. `_state_brief()` 中，将 `present_npcs` 替换为 `_public_npc_summary()` 的输出

3. `NARRATOR_SYSTEM_PROMPT`（第29行）强化禁忌：
   - 新增："不要描述NPC的性格特征、内心想法或行为动机。只描述可观察到的言行举止。"

## 3. 向后兼容

### 3.1 格式检测

在 `loader.py` 中检测 YAML 是否包含 `character_seed` 字段：
- 有 → 新格式，走 `load_npc_profile_at_time()`
- 无 → 旧格式，走原有 `load_npc_profile()` 逻辑（保持不变）

### 3.2 存档兼容

`GameState` 无新字段（Phase 1 不加 npc_locations，那是 Phase 2 的事）。存档格式不变。

### 3.3 world.yaml 不改动

`global_npc_pool` 保持现有结构。它服务于角色选择和全局配置，不直接影响运行时NPC对话。`fixed_npcs/` 是运行时引擎的唯一NPC数据源。

## 4. 验收标准

1. **不泄露未来**：在崇祯元年开局，和袁崇焕对话，NPC不应提及己巳之变、下狱等后续事件。温体仁不应表现出"后来成为弄权者"的特征。
2. **初始状态一致**：多次开局，袁崇焕始终处于 initial_state 定义的状态（候任辽东），不再随机出现在诏狱。
3. **旁白中立**：旁白介绍温体仁时只说"内阁大学士"，不说"阴柔善谋"、"弄臣"等带有上帝视角的评价。
4. **旧格式兼容**：test 剧本（`scripts/test/`）使用旧格式，引擎仍能正常加载运行。
5. **时间推进后更新**：当游戏时间推进到袁崇焕升任督师的时间点，NPC的 title、location、situation 自动更新。

## 5. 执行顺序

1. 重构 11 个 NPC YAML 文件（数据层，不涉及代码）
2. 修改 `src/scripts/loader.py`：新增 `load_npc_profile_at_time()` + 格式检测
3. 修改 `src/engine/npc_engine.py`：用时间过滤后的 profile 构建 prompt
4. 修改 `src/engine/narrator.py`：旁白只拿公开NPC信息
5. 验证：运行游戏，检查上述4条验收标准
