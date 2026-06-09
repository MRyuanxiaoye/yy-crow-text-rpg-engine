# Writer 项目

这是一个专门用于写作的项目，支持通过角色切换实现不同文风的写作输出。

## 项目说明
- 本项目位于 `深度搜索/Writer` 目录下
- 主要用途：写作相关的内容创作与管理

## 项目结构
- `roles/` — 角色定义目录，每个角色一个子目录
- `references/` — 文风参考库，按角色分类存放参考文档
- `stories/` — 故事参考库（索引+按需拉取模式）
  - `index.json` — 故事索引（标题、标签、URL、摘要）
  - `cache/` — 临时缓存（用完即删）
  - `custom/` — 用户手动添加的故事
- `scripts/` — 工具脚本
  - `fetcher.py` — 通用作品拉取脚本
  - `story_manager.py` — 故事库管理脚本
  - `novel_analyzer.py` — 网文分析器（多模型分层压缩）
- `output/` — 写作输出目录
- `novels/` — 网文分析库
  - `index.json` — 网文索引
  - `{id}_{slug}/` — 每本书一个目录，含章节、摘要、索引

## 角色切换指令

| 指令 | 角色 | 说明 |
|------|------|------|
| `/role 黑乌鸦` | 黑乌鸦 | 激活黑乌鸦写手角色 |
| `/role 审稿人` | 审稿人 | 激活审稿人审查角色 |
| `/role 架构师` | 架构师 | 激活架构师角色（卷级结构、世界观、体验线） |
| `/role 逻辑` | 逻辑审查 | 激活逻辑审查角色（逻辑、悬念、钩子、铺垫） |
| `/role 人物` | 人物设计 | 激活人物设计角色（人物弧光、配角、NPC） |
| `/role off` | 无 | 退出角色模式，恢复默认 |

**默认角色**：用户未指定角色时，写作任务默认使用黑乌鸦角色。

### 角色激活流程
1. 用户输入 `/role 黑乌鸦`
2. Claude 读取 `roles/black_crow/prompt.md` 加载角色设定
3. Claude 读取 `references/black_crow/` 下的参考文档学习文风
4. 回复开头确认：`【当前角色：黑乌鸦】`
5. 后续所有写作输出遵循该角色的文风设定

### 审稿人激活流程
1. 用户输入 `/role 审稿人`
2. Claude 读取 `roles/reviewer/prompt.md` 加载角色设定
3. Claude 读取 `roles/reviewer/contrast_list.md` 加载正反对比列表
4. Claude 读取当前写作角色的语感片段（用于判断语感是否符合）
5. 回复开头确认：`【当前角色：审稿人】`
6. 后续按审查流程工作

### 写作-审查协作流程
```
【前置工作】
架构师：卷级结构 + 世界观 + 当前卷章节规划
人物设计：主要人物档案 + 配角储备

【写某一章时】
用户给出方向
    ↓
架构师：设计本章体验线
    ↓
人物设计：确认本章人物状态和行为动机
    ↓
逻辑审查：审查体验线 + 人物设计（9项清单）
    ↓ 通过 → 补充注入悬念/钩子/铺垫
    ↓ 打回 → 架构师/人物设计修改 → 重审
    ↓
黑乌鸦：分段写 + 自审 → 输出 v1
    ↓
审稿人：扫描AI味 → 输出打回清单
    ↓
黑乌鸦：根据打回清单修改 → 输出 v2 + 版本记录
    ↓
最终稿 → _原稿.md + _修改.md
    ↓
用户在 _修改.md 中用 <!-- --> 标注反馈
    ↓
审稿人读取标注 → 更新 contrast_list.md
逻辑审查 → 更新 story_threads.md
人物设计 → 更新 characters.md
```

### 写作输出规则
- 角色激活后，写作内容保存到 `output/` 目录
- 文件命名格式：`YYYY-MM-DD_标题.md`
- 每次写作前先阅读参考库，确保文风一致

### 分段写作规则（强制）
- 重写或新写内容超过单段长度（约500-800字）时，必须分段出稿
- 每段单独控制语感和冷热属性
- 每段出完后做一次AI范式扫描，修完再接下一段
- 禁止整章一次性输出

## 网文分析器

用于将长篇网文分层压缩为结构化参考资料，写作时参考情节逻辑而非文风。

### 基本用法

```bash
# 导入小说
python3 scripts/novel_analyzer.py import --file /path/to/novel.txt --title "书名" --author "作者"

# 查看列表和进度
python3 scripts/novel_analyzer.py list
python3 scripts/novel_analyzer.py status --id 1

# 全自动分析（章节摘要→卷摘要→全书摘要→索引）
python3 scripts/novel_analyzer.py analyze --id 1

# 分步分析
python3 scripts/novel_analyzer.py analyze --id 1 --step chapter
python3 scripts/novel_analyzer.py analyze --id 1 --step chapter --detail --range 51-100
python3 scripts/novel_analyzer.py analyze --id 1 --step volume
python3 scripts/novel_analyzer.py analyze --id 1 --step book
python3 scripts/novel_analyzer.py analyze --id 1 --step indexes

# 删除
python3 scripts/novel_analyzer.py remove --id 1
```

### 写作时加载网文参考

- 参考骨架：读 `book.md` + `indexes/` + 对应卷/章摘要
- 照搬情节：额外读 `chapter_detail/` 获取场景序列和节奏
