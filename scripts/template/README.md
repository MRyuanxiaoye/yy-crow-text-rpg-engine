# scripts/template 新剧本格式模板

本目录提供“世界层 + 角色层”双层剧本 YAML 模板，用于替代旧的章节制剧本组织方式。模板只定义数据结构，不修改现有 `scripts/mingmo/` 旧格式内容，也不依赖 Python 代码变更。

## 文件用途

- `world.yaml`：世界层模板，适合在用户确定题材后生成一次。包含世界观背景、玩法类型、世界观类型、世界维度定义与初始值、时间系统、时间线里程碑、条件事件、随机事件池、全局 NPC 池和可选角色列表。
- `role_template.yaml`：角色层模板，适合在用户选择角色后为该角色生成。包含玩家目标、默认失败轨迹、三类压力源、角色维度、NPC 初始关系、前主角 NPC 目标线、角色专属事件、结局条件、旁白风格和信息池。
- `README.md`：说明新旧格式差异、模板用途和 loader 解析建议。

## 新旧格式对比

| 维度 | 旧格式 `scripts/mingmo/` | 新格式 `scripts/template/` |
| --- | --- | --- |
| 剧本组织 | `manifest.yaml` + `chapters/` + `events/` + `fixed_npcs/` + `endings/` | `world.yaml` + 每个角色一个角色层 YAML |
| 推进方式 | 章节驱动，依赖 `advance_conditions` 和 `first_chapter` | 目标驱动，依赖时间推进、压力源和事件触发 |
| 世界状态 | 多处文件分散定义指标和事件 | 世界层集中定义维度、时间、事件池和 NPC 骨架 |
| 角色差异 | 多角色共享主线，角色差异常由结局或分支体现 | 每个角色拥有独立目标、压力、关系、事件和结局 |
| NPC 数据 | 固定 NPC 文件包含性格、关系数值和事件 | 世界层定义人格/目标/知识，角色层定义与玩家的关系数值/标签 |
| 结局系统 | 按角色文件列出条件字符串 | 角色层使用底线条件、目标达成检查和多层级结局目录 |
| 信息传递 | 主要依赖事件叙事和 NPC 对话生成 | 角色层显式维护 `information_pool`，支持 fact/persistent/escalating |

## 解析建议

1. 先 `yaml.safe_load` 读取 `world.yaml`，校验 `schema_version`、`script_id`、稳定 ID 和维度范围。
2. 用户选择 `optional_roles.role_id` 后，读取对应角色层 YAML，并检查 `world_script_id` 与世界层匹配。
3. 合并时先加载世界维度初始值，再应用 `role_dimensions.world_dimension_overrides`。
4. NPC 以 `npc_id` 为主键：世界层提供人格、目标、知识范围；角色层补充数值型关系、标签和开局立场。
5. 事件可分池保存，但触发后统一转成“被动响应事件卡片”，便于复用行动确认、掷骰和结果叙事流程。
6. 条件建议统一解析 `all` / `any` / `not` 结构，避免继续依赖不可解析的自然语言条件字符串。

## 编写约定

- 所有维度数值默认使用 `0-10`，除非字段内 `range` 明确声明其他范围。
- 所有被存档、事件、关系引用的对象必须提供稳定 ID，例如 `event_id`、`npc_id`、`role_id`、`info_id`、`ending_id`。
- YAML 中保留中文注释说明字段用途和取值规范；注释不影响 `yaml.safe_load`。
- 模板使用占位符和示例值，不绑定任何具体剧本内容。
