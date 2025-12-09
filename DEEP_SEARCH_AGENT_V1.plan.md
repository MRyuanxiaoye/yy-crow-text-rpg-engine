# Deep Search Agent V1 实施计划

## 1. 环境与配置
- **目标**: 搭建项目基础结构并完成 API 集成。
- **行动**:
  - 创建 `.env` 文件配置 DeepSeek & Tavily 密钥。
  - 安装 `langgraph`, `langchain`, `langchain_openai` (用于 DeepSeek), `tavily-python`, `chromadb`。
  - 配置 `ChatOpenAI` 客户端以适配 DeepSeek API。

## 2. 核心智能体模块 (Nodes)

### A. 规划器 (Planner Agent) - “大脑”
- **逻辑**: 接收用户问题 -> 生成结构化的“研究大纲” (DAG 或 列表)。
- **Prompt**: 指导 DeepSeek 扮演“资深研究员”，将主题拆解为 3-5 个核心维度（如：历史、仪式、层级）。
- **输出**: 子主题 JSON 列表。

### B. 研究员 (Researcher Agent) - “眼睛” (深度增强版)
- **核心逻辑**: 获取子主题 -> 生成查询词 -> 调用 Tavily。
- **解决用户痛点 5 (数据量与深度)**:
  - **循环迭代**: 如果搜索结果少于 5 个有效来源，生成同义词/相关术语重试（最多 3 次）。
  - **信源优先**: 提升 `ctext.org` (中国哲学书电子化计划), `zh.wikisource.org` (维基文库), `guoxuedashi.net` (国学大师) 的权重。
  - **内容获取**: 使用 Tavily 的 `include_raw_content=True` 或针对已知数字图书馆的自定义抓取器，尽可能获取全文。

### C. 筛选与精炼 (Selector & Refiner) - “过滤器”
- **逻辑**: LLM 阅读搜索结果摘要。
- **筛选**: 决定哪些 URL 值得进行全文抓取。
- **精炼**: 如果识别出特定书籍（如“道藏”）但内容缺失，生成特定查询：`site:ctext.org "道藏" [具体术语]`。

### D. 撰稿人 (Writer Agent) - “笔杆子”
- **逻辑**: 将所有收集的笔记综合成连贯的报告。
- **风格**: “资深研究员”人设——信息密集、引用典故、解释概念。
- **结构**: 清晰层级的 Markdown。

## 3. 工作流架构 (LangGraph)
- **状态 (State)**: `query`, `sub_topics`, `gathered_data`, `draft`, `critique`.
- **日志系统 (Logging)**: 
  - 在每个 Node 的入口和出口植入结构化日志。
  - 使用中文记录关键决策（例如：“搜索结果不足，正在尝试关键词 [X]”）。
  - 将日志流实时推送到前端，方便用户“看见”思考过程。
- **图流向 (Graph Flow)**:
  1. `Start` -> `Planner`
  2. `Planner` -> `Researcher` (各子主题并行执行)
  3. `Researcher` -> `Selector` -> `Scraper` -> `Researcher` (深度循环)
  4. `Researcher` (全部完成) -> `Writer`
  5. `Writer` -> `End` (第一阶段暂不加入 Reviewer 循环，先确保线性深度挖掘)。

## 4. 数据处理策略 (解决痛点 5)
- **临时向量库**: 
  - 当发现大篇幅文本（如道藏的一章）时，下载并索引到临时 ChromaDB 集合。
  - 使用 RAG 从这本“书”中提取具体答案，而不是把整本书塞进上下文。
- **兜底策略**: 如果找不到特定书籍原文，寻找该书的“学术解析”或“注疏”。

## 5. 界面 (第一阶段)
- **CLI / 简单脚本**: 验证逻辑日志和输出。
- **Streamlit**: 迁移之前的 UI 以支持基于 Graph 的流式输出。

