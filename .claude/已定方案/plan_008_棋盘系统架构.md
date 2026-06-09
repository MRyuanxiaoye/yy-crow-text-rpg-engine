# 棋盘系统架构

## 一、棋盘的定义

棋盘是'当前局面'的完整可见表达。玩家看一眼回答三个问题：
- 我在哪（当前处境）
- 发生了什么（变化和趋势）
- 我能做什么（可选行动方向）

在线性对话流中，棋盘提供结构化、持久的、可随时调出的当前局面。

## 二、展示原则

1. **一屏之内**——不翻页、不折叠，一张卡片一眼扫完
2. **结构固定**——信息位置一致，玩家形成肌肉记忆
3. **变化可见**——趋势箭头(▲▼━)、新事件(⚡)、新解锁(🔓)、迷雾(❓)
4. **剧本定义内容，引擎定义结构**——框架通用，内容随剧本
5. **有'不知道'的区域**——迷雾不是空白，是'你知道那里有东西但看不到'

## 三、棋盘数据模型

### 3.1 引擎通用框架（6个区块）

| 区块 | 内容 | 来源 |
|------|------|------|
| 时间标识 | 当前时间+回合数 | state.game_date + state.turn |
| 维度面板 | 世界维度+角色维度+趋势+告警 | state.dimensions + trends |
| 区域板块 | 剧本定义的若干区域，每个区域多维度信息+迷雾 | 新增 state.board_regions |
| 进行中 | delayed待办+剩余回合数 | state.delayed_queue |
| 回合议程 | 本回合紧急/一般事项 | state.turn_agenda |
| 线索/记忆 | 累积的关键发现 | 新增 state.discovered_clues |

### 3.2 区域（Region）数据结构

每个区域是棋盘的核心信息单元。

```yaml
# 剧本YAML定义初始区域
board_regions:
  liaodong:
    name: 辽东
    category: geographic   # geographic / political / spatial（盗墓用）
    info_layers:
      - id: military
        label: 军情
        initial_known: "宁远守军欠饷4月，边防勉强维持"
        initial_fog:
          text: "敌军动向不明"
          unlock_condition: {dimension: 情报, threshold: 4}
          unlock_text: "皇太极近期在集结兵力，目标不明"
      - id: personnel
        label: 人物
        initial_known: "袁崇焕待命，可召回约2回合"
        initial_fog:
          text: "边将之间关系未知"
          unlock_condition: {action: "召见知情者或派人调查"}
          unlock_text: null  # 由LLM根据情节动态生成
      - id: economy
        label: 经济
        initial_known: "需拨银20万两"
        initial_fog: null  # 无迷雾
      - id: undercurrent
        label: 暗线
        initial_fog:
          text: "有密报称边将与商人勾结"
          unlock_condition: {dimension: 情报, threshold: 5}
          unlock_text: null  # LLM动态生成
```

### 3.3 运行时区域状态

```python
# state.board_regions 运行时结构
{
  "liaodong": {
    "name": "辽东",
    "category": "geographic",
    "layers": [
      {
        "id": "military",
        "label": "军情",
        "known_text": "宁远守军欠饷4月，边防勉强维持",
        "fog": {
          "hint": "敌军动向不明",
          "unlock_condition": {"dimension": "情报", "threshold": 4},
          "unlocked": false
        },
        "events": []  # 该层新发生的事件
      },
      ...
    ],
    "news": ["⚡ 皇太极集结兵力（第2回合解锁）"]  # 该区域新闻
  }
}
```

### 3.4 迷雾解锁机制

三种解锁触发：
1. **维度门控**——玩家某维度达到阈值时自动解锁（如情报≥4）
2. **行动触发**——玩家执行特定行动后解锁（如派锦衣卫调查）
3. **事件驱动**——游戏事件发生时解锁（如NPC对话中透露信息）

解锁后：
- 如果有预设 unlock_text，直接显示
- 如果 unlock_text 为 null，调用LLM根据当前游戏状态动态生成
- 解锁的信息标记为🔓新（当回合），下回合变为普通已知信息

### 3.5 区域更新时机

| 时机 | 更新内容 |
|------|---------|
| 回合开始 | 检查所有维度门控，自动解锁达标的迷雾 |
| 对话结束 | 从talk_summary的key_facts中提取区域相关信息，更新对应区域 |
| 行动结算 | 结算效果映射到区域状态（如拨银→辽东经济层更新） |
| 被动事件 | 新事件注入对应区域的events和news |
| 世界推进 | 衰减/里程碑/阈值事件更新区域状态 |

## 四、棋盘渲染

### 4.1 飞书卡片结构

棋盘渲染为一张飞书消息卡片，分区用markdown + 分隔线。

区块顺序：
1. 标题行：时间 + 回合数
2. 维度面板：进度条 + 趋势箭头 + 告警
3. 各区域板块：每个区域一个section，内含已知信息+迷雾提示
4. 进行中：delayed项 + 剩余回合
5. 回合议程：紧急🔴 / 一般🟡
6. 线索汇总：累积的关键发现（可选，过长时折叠为'共N条线索'）

### 4.2 棋盘展示时机

- 每回合开始时自动展示
- 玩家点击'查看局势'按钮时展示
- 重大事件发生后（如解锁新区域）主动推送

## 五、与现有系统的对接

### 5.1 与四维度架构的关系

| 四维度 | 棋盘中的体现 |
|--------|------------|
| 维度一·状态模型 | 维度面板区块 |
| 维度二·回合结构 | 回合议程区块 + 每回合自动展示棋盘 |
| 维度三·行动→结果 | 区域状态因行动结算而更新 |
| 维度四·驱动力 | 推力=议程中的🔴紧急项；引力=区域迷雾中的好奇钩子 |

### 5.2 需要修改的文件

| 文件 | 改动 |
|------|------|
| state.py | 新增 board_regions: dict / discovered_clues: list 字段 |
| loader.py | 新增 load_board_regions() 从剧本YAML加载区域定义 |
| game_master.py | 回合开始检查迷雾解锁 / 对话结束更新区域 / 结算更新区域 |
| card_builder.py | 新增 build_board_card() 渲染棋盘卡片，替换现有 build_situation_card |
| formula.py | 行动结算效果映射到区域（可选，Phase 2） |

### 5.3 剧本数据新增

| 文件 | 新增内容 |
|------|---------|
| world.yaml | board_regions 定义（区域列表+初始信息层+迷雾） |
| emperor.yaml | 无变化（角色维度已有） |
| NPC YAML | 无变化（NPC位置已有） |

## 六、实施分期

### Phase E1：数据层+渲染（核心可用）
- state.py 新增 board_regions / discovered_clues
- loader.py 新增 load_board_regions
- 明末 world.yaml 新增 board_regions（京师/辽东/陕西/江南 4个区域）
- card_builder.py 新增 build_board_card
- game_master.py 回合开始展示棋盘 + 查看局势改为展示棋盘
- 验证：开局看到带迷雾的棋盘

### Phase E2：迷雾解锁（信息流动）
- game_master.py 回合开始检查维度门控自动解锁
- game_master.py 对话结束从talk_summary提取区域信息更新
- game_master.py 行动结算更新区域状态
- 验证：情报提升后迷雾揭开，对话内容反映到棋盘

### Phase E3：动态内容（LLM驱动）
- unlock_text为null时调用LLM生成解锁内容
- 被动事件注入区域news
- 区域状态随世界推进自动演化
- 验证：棋盘每回合都有变化，不需要玩家记忆对话内容
