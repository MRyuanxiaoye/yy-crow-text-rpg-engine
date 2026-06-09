# Trace 测试回溯系统

## 目标

新增独立的交互追踪系统，记录游戏运行时的完整因果链（玩家输入→路由决策→LLM prompt/response→state变更），支持事后回溯诊断问题。

## 1. 新建 src/engine/tracer.py

核心模块，包含：

### GameTracer 类

```python
class GameTracer:
    def __init__(self, chat_id: str, enabled: bool = True)
    def record(self, entry_type: str, function: str, data: dict, state_summary: dict | None = None)
    def record_llm_call(self, function: str, template_name: str, template_text: str, variables: dict, raw_response: str, parsed_response: Any, state_summary: dict)
    def record_turn_snapshot(self, state_dict: dict)
    def clear(self)
    
    @staticmethod
    def state_summary(state: GameState) -> dict
```

### 存储格式

- 目录：`data/traces/{chat_id}/`（chat_id中的/替换为_）
- 文件：`{YYYYMMDD}.jsonl`，每天一个文件
- 每行一条JSON记录

### 记录结构

```json
{
  "ts": "2026-05-18T14:32:01",
  "chat_id": "xxx",
  "type": "llm_call | route | state_change | card_action | turn_snapshot | prompt_template",
  "function": "_gm_interpret",
  "data": { ... },
  "state": {
    "turn": 5,
    "game_time": {"year": 1, "month": 8},
    "phase": "free",
    "talking_to": "",
    "court_active": false,
    "settlement_phase": "free",
    "npc_locations_keys": ["温体仁", "孙承宗"]
  }
}
```

### Prompt 模板去重

- 维护 _template_registry: dict[str, str]（template_name → template_id）
- 首次出现某模板时，写一条 type=prompt_template 记录（含完整模板文本）
- 后续同模板的 llm_call 只记 template_id + variables + response

### Turn 快照

- 每次 state.turn 递增时，写一条 type=turn_snapshot
- 包含完整 state（但剥离 conversation_history 节省空间）
- 这是「锚点」，回溯时先定位到这里看全貌

### 全局实例管理

```python
_tracers: dict[str, GameTracer] = {}

def get_tracer(chat_id: str, enabled: bool = True) -> GameTracer:
    ...
```

## 2. 修改 state.py

GameState 新增字段：
```python
trace_enabled: bool = True
```

默认开启，跟存档一起持久化。_state_from_payload 中反序列化。

## 3. 修改 game_master.py

### 3.1 /trace 指令

在 handle_message 的 _is_restart_command 检查之前，加入 trace 指令检测：

```python
# /trace 指令处理
trace_cmd = _parse_trace_command(clean_text)
if trace_cmd:
    await _handle_trace_command(chat_id, state, trace_cmd)
    return
```

辅助函数：
```python
def _parse_trace_command(text: str) -> str | None:
    t = text.strip().lower()
    if t in ("/trace on", "/trace off", "/trace clear"):
        return t.split()[-1]  # "on" / "off" / "clear"
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
    await send_narrator_card(chat_id, build_narration_card("Trace", msg))
```

### 3.2 埋点位置（约10处）

| 位置 | type | 记录内容 |
|------|------|----------|
| handle_message 入口 | route | player_text, current phase |
| _gm_interpret | llm_call | GM prompt variables + LLM result |
| _handle_gm_selection | route | option_index, opt_type, targets |
| _start_talk | route | npc_name, reachability |
| _route_talking_message | route | npc_name, player text |
| _handle_court_message（裁定） | llm_call | arbiter prompt + speaking_order |
| _handle_court_message（每个NPC） | llm_call | npc_name, court_context, reply |
| _advance_world_phase | state_change | advance_result, triggered events |
| handle_card_action 入口 | card_action | action type, value |
| turn 递增处 | turn_snapshot | full state |

### 3.3 Tracer 初始化

在 handle_message 和 handle_card_action 开头，获取 tracer 并同步 enabled 状态：
```python
tracer = get_tracer(chat_id, enabled=state.trace_enabled)
```

## 4. 修改 npc_engine.py

generate_npc_reply 末尾加 trace：
```python
# 需要接收 chat_id 参数或通过其他方式获取 tracer
# 记录：npc_name, system_prompt摘要, user_content, reply
```

注意：generate_npc_reply 当前不接收 chat_id，需要新增参数或通过 state 传递。

## 5. 修改 narrator.py

各 narrate_* 函数末尾加 trace：记录旁白 prompt 摘要 + 生成的叙述文本。
同样需要 chat_id 参数传递问题。

## 6. chat_id 传递策略

npc_engine 和 narrator 当前不接收 chat_id。两种方案：
- A：给相关函数加 chat_id 参数（侵入性大但明确）
- B：在 state 上挂 _chat_id 运行时属性（不序列化，每次 handle_message 开头设置）

推荐方案B：state._runtime_chat_id = chat_id，tracer 从 state 获取。

## 7. 回溯工作流

1. 用户报告「第5回合温体仁对话有问题」
2. Claude 读取 data/traces/{chat_id}/{日期}.jsonl
3. grep turn=5 的 turn_snapshot 看当时完整 state
4. 找同 turn 的 llm_call 看 NPC prompt + response
5. 找 route 记录看路由决策
6. 定位根因

## 8. 验收标准

1. /trace off → 后续操作不产生新 trace 记录
2. /trace on → 恢复记录
3. /trace clear → trace 目录下文件被清空
4. 正常游戏流程中，每个 LLM 调用都有对应 trace 记录
5. 每个回合有完整 state 快照
6. trace 文件可被 Claude 直接读取并用于问题诊断
