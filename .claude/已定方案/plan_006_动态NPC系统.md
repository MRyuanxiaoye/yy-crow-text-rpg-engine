# Phase 4: 功能性NPC + 动态NPC生成系统

## 背景
当前系统只有11个预定义固定NPC，玩家提到的合理角色（如户部尚书）不在系统中会被跳过。需要：
1. 预定义常见职位NPC（轻量版）
2. 支持动态生成未预见角色

## 设计原则
- 动态NPC与固定NPC引擎一视同仁，走相同的对话/朝会/记忆/维度系统
- 动态NPC = 完整character_seed（LLM生成）+ initial_state + 持久化
- 预定义功能性NPC比固定NPC更轻（有seed+initial_state，无timeline）

## Step 1: 预定义功能性NPC

### 1.1 新增数据文件
目录：scripts/mingmo/functional_npcs/
为明末剧本新增以下轻量NPC文件（每个一个YAML），格式与fixed_npcs一致但无timeline：

| 文件名 | 角色 | title | 位置 | 阵营 |
|--------|------|-------|------|------|
| 户部尚书.yaml | 毕自严 | 户部尚书 | 京师 | 朝廷方 |
| 兵部尚书.yaml | 王在晋 | 兵部尚书 | 京师 | 朝廷方 |
| 吏部尚书.yaml | 王永光 | 吏部尚书 | 京师 | 朝廷方 |
| 工部尚书.yaml | 曹珖 | 工部尚书 | 京师 | 朝廷方 |
| 刑部尚书.yaml | 乔允升 | 刑部尚书 | 京师 | 朝廷方 |
| 礼部尚书.yaml | 何如宠 | 礼部尚书 | 京师 | 朝廷方 |
| 锦衣卫指挥使.yaml | 骆养性 | 锦衣卫指挥使 | 京师 | 朝廷方 |
| 左都御史.yaml | 曹于汴 | 左都御史 | 京师 | 朝廷方 |

每个文件格式示例：
```yaml
character_seed:
  name: 毕自严
  personality_traits: ["务实稳重", "谨慎保守", "精于账目"]
  values: ["财政稳健", "社稷安定"]
  speaking_style: "条理清晰、引经据典，常以'臣以为'起首，善用数据说服。"
  behavioral_tendencies: "倾向紧缩开支、反对加派、重视税源维护。遇到军费请求时会权衡财政承受力。"
  knowledge_domain: "国库收支、各省税赋、盐铁专卖、钱粮调拨。"
  knowledge_blind_spots: "军事战略细节、边镇实际军情、地方民变深层原因。"

initial_state:
  time: { year: 1, month: 8 }
  title: 户部尚书
  faction: 朝廷方
  location: 京师
  status: 在朝
  situation: "新朝初立，国库亏空，各方催款不断。我正在清理前朝积欠，试图在军饷和赈灾之间找到平衡。"
  public_profile: "户部尚书，掌管天下钱粮，以务实著称。"
  initial_relations:
    trust: 60
    loyalty: 65
    influence: 55
```

每个NPC的性格都要有明确个性和内在张力，不能泛泛写'忠厚老实'。参考史实人物性格但不必完全还原。

### 1.2 修改 loader.py
新增函数 load_all_functional_npc_initial_locations(script_id) ，从 functional_npcs/ 目录加载所有功能性NPC的初始位置，返回格式与 load_all_fixed_npc_initial_locations 一致（含title字段）。

### 1.3 修改 game_master.py
在 _choose_role 函数中（NPC 位置初始化部分，约第597行），除了加载 fixed_npc 位置外，同时加载 functional_npcs 位置。也就是在 load_all_fixed_npc_initial_locations 之后，再调用 load_all_functional_npc_initial_locations 并合并到 npc_locations。

同时，functional_npcs 也需要被加入 state.active_npcs。在 _choose_role 中 state.active_npcs 赋值之后，遍历 functional_npcs/ 下所有 YAML（用 loader 加载），将每个NPC添加到 active_npcs（key = npc_name），格式与现有 active_npcs 中的条目兼容。

### 1.4 修改 GM_INTERPRET_PROMPT 中的 NPC 列表
确保 _build_npc_location_text 包含 functional_npcs（因为它们已经在 npc_locations 中，这步应该自动生效，但需确认）。

## Step 2: 动态NPC生成

### 2.1 新增函数 _generate_dynamic_npc
在 game_master.py 中新增异步函数：

```python
async def _generate_dynamic_npc(state: GameState, npc_description: str, chat_id: str) -> dict[str, Any]:
    """动态生成一个新NPC，持久化到state中。"""
```

功能：
- 构建生成 prompt，包含：世界背景、当前时间点、玩家角色、已有NPC列表（避免重名）、npc_description（如'户部尚书'）
- 调用 LLM 生成完整 character_seed + initial_state（JSON格式）
- 解析结果，补全必要字段（确保有 name, title, location, faction, status, situation, public_profile, personality_traits, values, speaking_style, behavioral_tendencies, knowledge_domain, knowledge_blind_spots）
- 将生成的NPC添加到 state.active_npcs（key=name）
- 计算可达性并添加到 state.npc_locations
- 添加 is_dynamic: true 标记
- 记录 trace
- 返回生成的 profile

生成 prompt 示例要点：
- 告诉LLM当前是什么剧本、什么时间点、什么世界背景
- 告诉LLM玩家想要什么角色（npc_description）
- 要求生成有个性有张力的角色，不要泛泛
- 要求输出严格JSON格式
- 已有NPC名单（避免重名或矛盾设定）

### 2.2 新增辅助函数 _ensure_npc_exists
在 game_master.py 中新增：

```python
async def _ensure_npc_exists(state: GameState, npc_name: str, chat_id: str) -> bool:
    """确保NPC存在于系统中，不存在则动态生成。返回是否成功。"""
```

逻辑：
1. 检查 npc_name 是否在 state.npc_locations 中 → 存在则返回 True
2. 检查 npc_name 是否能匹配 active_npcs 中某个NPC的 title → 找到则返回 True（同时用真实name替换）
3. 都找不到 → 调用 _generate_dynamic_npc 生成 → 成功返回 True，失败返回 False

### 2.3 修改 _init_court_session
将当前的 skipped 逻辑替换：遇到未知NPC时不再跳过，而是调用 _ensure_npc_exists。如果生成成功，继续处理该NPC的可达性；如果生成失败，才跳过。

### 2.4 修改 _handle_gm_selection
在路由到 talk/multi_talk 之前，对 targets 中的每个NPC调用 _ensure_npc_exists。

### 2.5 修改 _start_talk
在开始对话前，如果目标NPC不在 npc_locations 中，调用 _ensure_npc_exists。

## 验收标准
1. 玩家说'召见户部尚书'→ 系统找到预定义的毕自严 → 正常对话
2. 玩家说'召见都察院左副都御史'（未预定义）→ 系统动态生成NPC → 正常对话
3. 动态NPC在后续交互中保持一致（name、性格不变）
4. 动态NPC可参加朝会、一对一对话
5. py_compile 通过
