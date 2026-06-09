# 引擎重构总览

## 背景

游戏体验暴露5个核心问题，根因分析指向引擎缺少**时间线意识**和**统一的玩家意图处理**：

1. NPC初始状态不一致 + 旁白泄露NPC本性 → AI拿到全知视角终身评价，而非当前时间点状态
2. NPC对话内容与游戏机制时间脱节 → NPC没有位置/可达性概念
3. 自由文本召见NPC未触发对话 → 缺少统一的意图解读层
4. 多NPC对话（朝会模式）→ 引擎只支持一对一对话

## 核心设计决策

| 决策 | 内容 | 确认轮次 |
|------|------|---------|
| NPC模型 | 性格底色(固定) + 已发生经历(随时间变化) → AI自行演绎 | Q6 |
| 数据结构 | NPC YAML拆为 character_seed + timeline + 事件条件 | Q7 |
| 信息边界 | AI上下文只含当前时间点之前的信息，未来信息根本不入上下文 | Q4 |
| 玩家意图 | GM解读层：自由文本统一经LLM以GM视角解读→卡片确认→路由执行 | Q13 |
| NPC可达性 | NPC有物理位置属性，位置决定可达性（即时/N回合/不可达） | Q10 |
| 朝会模式 | 发言序列裁定+串行生成+组合卡片+自发互动+非语言反应 | Q14-15 |
| 决策实时捕获 | 每轮GM扫描检测决策→决策卡片确认→写入state→注入NPC上下文 | Q33 |

## 三阶段重构路线

```
Phase 1（数据层）→ Phase 2（交互层）→ Phase 3（朝会模式）
每阶段独立可部署，前一阶段是后一阶段的基础
```

- **Phase 1**：NPC数据重构 + 时间线过滤 + 旁白信息边界。验证：NPC不泄露未来信息。
- **Phase 2**：GM解读层 + NPC位置/可达性 + 时间成本操作。验证：自由文本正确路由。
- **Phase 3**：多人朝会对话 + 发言裁定 + 场景属性。验证：多NPC互动。

## 涉及文件

### 数据文件
- `scripts/mingmo/fixed_npcs/*.yaml` — NPC定义重构
- `scripts/mingmo/world.yaml` — 全局NPC池补充
- `scripts/mingmo/events/timeline.yaml` — 已有，格式兼容
- `scripts/template/` — 模板更新

### 引擎代码
- `src/engine/npc_engine.py` — NPC上下文按时间过滤
- `src/engine/narrator.py` — 旁白信息边界控制
- `src/engine/state.py` — 新增NPC位置/朝会状态
- `src/engine/game_master.py` — GM解读层替换意图分类
- `src/scripts/loader.py` — 新YAML格式支持
- `src/feishu/card_builder.py` — 朝会组合卡片

## 对话提炼机制：逐轮扫描 + Chain-of-Thought

### 问题
结束NPC/朝会对话后，TALK_SUMMARY_PROMPT 让 LLM 从完整对话中提取 candidate_items，但 LLM 存在近因偏差——注意力集中在最近几轮，前半段的重大决策（人事任命、政策批准、制度安排等）被遗漏。

### 方案
不拆多次调用，在一次调用内强制逐轮扫描（chain-of-thought）：
- Step 1（逐轮扫描）：按轮次逐一列出玩家发言要点，标记是否包含决策/指令/批准。纯事实性罗列，不做判断。
- Step 2（结构化提取）：从 Step 1 清单中，将每个标记为有决策的轮次转为 candidate_item。

### 保证
- Step 1 必须覆盖每一轮，不得跳过
- Step 1 标记为有决策的轮次，Step 2 必须有对应 candidate_item
- '准奏''照办''就这么定'等批准类也是决策

### 改动文件
- game_master.py：TALK_SUMMARY_PROMPT 重写为两步结构

## 决策实时捕获系统

### 问题
对话提炼(TALK_SUMMARY_PROMPT)只在对话结束时执行。长对话中玩家做的决策不被捕获——backlog为空、agenda不更新、NPC重复已决话题。

### 方案
每轮NPC回复后执行轻量GM扫描（~100-200 token），检测玩家是否做了决策。
- 检测到 → 决策卡片反馈 + state更新 + NPC上下文注入
- 未检测到 → 无操作
详见 plan_009_交互框架.md 第二章第4节。

### 改动文件
- game_master.py：新增_gm_decision_scan函数、决策卡片发送、已决事项注入NPC prompt
- card_builder.py：新增build_decision_card
- state.py：新增confirmed_decisions字段
- npc_engine.py：system prompt注入已决事项列表

## 结算双层结构（2026-05-21 确认）

### 核心原则
- 每个行动 = 决定层（永远即时）+ 执行层（可能即时/delayed）
- 代价即时，收益看距离
- 皇帝直接命令自动成功，不掷骰

### 行动性质分类
- command（命令）：准奏/拨银/任免 → 即时，无判定
- local_exec（本地执行）：抄家/审讯 → 0-1回合，有判定
- remote_exec（远程执行）：调兵/催收 → 2-3回合，有判定
- contested（对抗）：说服/改革 → 1回合，有判定

### 判定规则
- 皇帝命令 + 资源足够 = 自动成功
- 判定只判执行质量（抄到多少/效率如何），不判命令能否下达
- 3d6 + modifier vs dc，维度驱动

### 议程联动
- 决策确认后立刻标记对应议程为已处理
- 已处理议程不再触发NPC来访
