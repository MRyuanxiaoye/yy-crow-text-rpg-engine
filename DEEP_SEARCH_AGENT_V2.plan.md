# Deep Search Agent V2 升级计划

## 1. 核心目标
- **全领域适应性**: 让智能体能够动态调整角色和策略，胜任从“技术实操”到“历史人文”的各类话题。
- **深度全书阅读**: 实现“混合分级阅读器”，在不消耗巨额 Token 的前提下，高效读取和分析长文/整书。
- **数据获取增强**: 集成策略以寻找付费内容的开源替代品。

## 2. 架构升级

### A. 自适应规划器 (Adaptive Planner) - “变形者”
- **逻辑**: 不再使用固定的 Prompt，而是先使用 **Meta-Prompt** 分析查询的领域。
- **输出**:
  - `domain`: 例如 "技术/软件" 或 "历史/二战"。
  - `role`: 例如 "资深 UX 设计师" 或 "军事历史研究员"（强调准确性，而非文学性）。
  - `search_strategy`: 例如 "优先查阅官方文档" 或 "优先查阅原始史料"。
  - `output_format`: 针对该领域的最佳输出范式指导。
    - *技术类*: 分步指南 (Step-by-step)、表格对比、参数说明。
    - *人文类*: 深度叙事、时间线梳理、因果分析、详略得当的总结。
  - `sub_topics`: 标准的子问题拆解。
- **核心原则**: 文风以**准确展示元数据**为主，拒绝过度的文学修辞，确保信息的纯度和密度。
- **行动**: 重写 `src/agents/planner.py`。

### B. 混合分级阅读器 (Hybrid Hierarchical Reader) - “潜水员”
- **新模块**: `src/tools/reader_engine.py`
- **工作流**:
  1.  **爬虫 (Level 1)**: 使用 `Trafilatura` 抓取 URL 的纯文本（长文或书籍章节）。
  2.  **BM25 初筛 (Level 2)**: 本地关键词索引，筛选出前 20 个相关片段（免费）。
  3.  **向量精筛 (Level 3)**:
      - 使用 **OpenAI `text-embedding-3-small`** 对片段进行 Embedding。
      - 存入 **ChromaDB** (临时)。
      - 检索语义最相关的 5-10 个片段。
  4.  **LLM 总结 (Level 4)**: 发送筛选后的精华片段给 DeepSeek 进行总结。

### C. 智能信源路由 (Intelligent Source Routing)
- **逻辑**: 在 `Researcher` 中，根据 Planner 确定的 `search_strategy` 选择搜索域。
- **白名单**: 维护 "领域 -> 可信站点" 映射表 (例如: 技术 -> StackOverflow, GitHub; 历史 -> Wikipedia, Ctext)。

## 3. 实施步骤

1.  **依赖升级**:
    - 添加 `rank_bm25`, `jieba` (中文分词), `trafilatura`。
    - 配置 `OpenAIEmbeddings`。

2.  **重构 Planner**:
    - 实现领域/角色检测逻辑。
    - 将 `role` 状态在 Graph 中向下传递。

3.  **构建阅读器引擎 (Reader Engine)**:
    - 实现 4 级阅读流水线。
    - 在 LangGraph 中创建专门的 `Reader` 节点（并行或串行于 Researcher 之后）。

4.  **更新工作流 (Workflow)**:
    - **Graph**: `Planner` -> `Researcher` -> `Decision` (是否深读?) -> `Reader` -> `Writer`。
    - 添加条件边: 如果 `Researcher` 发现“高价值长文” (例如 > 5k 字)，触发 `Reader`。

5.  **UI 增强**:
    - 显示“阅读进度” (例如: "正在索引 50 页...", "正在扫描关键词...").

## 4. 成本控制策略
- **Embedding**: 使用 OpenAI `text-embedding-3-small` (极度便宜)。
- **过滤**: 在 Embedding 之前激进地使用 BM25 (免费) 过滤，减少 API 调用。

