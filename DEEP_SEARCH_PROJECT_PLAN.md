# 深度搜索智能体 (Deep Search Agent) 项目实施计划

本计划旨在从零构建一个基于 **LangGraph** + **云端大模型 API** 的深度文化搜索智能体，并将其服务化以便远程调用。

## 一、 API 准备指南

为了构建高智商、低成本的搜索系统，我们需要“大脑”和“眼睛”。

### 1. “大脑”：大模型 API
**策略**：混合使用 DeepSeek 和 OpenAI 以达到最佳性价比。

*   **DeepSeek API (必选)**
    *   **角色**：主力模型（Workhorse）。
    *   **用途**：负责绝大多数的文本生成、长文写作、初步资料分析。
    *   **理由**：DeepSeek-V3 的中文写作能力极强，且 API 成本极低（约 GPT-4 的 1/10），适合大量 Token 消耗的场景。
    *   **申请**：访问 DeepSeek 开放平台。

*   **OpenAI API (强烈建议可选)**
    *   **角色**：逻辑控制（Controller）。
    *   **用途**：负责 LangGraph 的复杂逻辑路由、Function Calling（工具调用）、意图识别。
    *   **理由**：GPT-4o 在遵循复杂 JSON 格式指令和工具调用稳定性上目前仍是业界标杆。如果 DeepSeek 在某些复杂逻辑上偶尔“掉链子”，GPT-4o 是完美的兜底。
    *   **申请**：访问 OpenAI Platform。

### 2. “眼睛”：搜索 API
**策略**：使用专为 AI 设计的搜索接口，避免反爬虫和验证码。

*   **Tavily Search API (推荐)**
    *   **用途**：作为 `Researcher Agent` 的核心工具。
    *   **理由**：它不仅返回搜索结果，还会自动爬取网页内容并清洗（去掉广告、导航栏），直接给大模型返回干净的文本。这比自己写爬虫稳定得多。
    *   **申请**：访问 Tavily 官网。

---

## 二、 服务器与部署架构

### 1. 服务器选型
*   **类型**：海外 VPS (Virtual Private Server)。
*   **配置**：
    *   **CPU**: 2 核
    *   **内存**: 2GB - 4GB
    *   **带宽**: 建议 10Mbps 以上
    *   **系统**: Ubuntu 22.04 LTS
*   **推荐厂商**：
    *   **腾讯云/阿里云（香港轻量应用服务器）**：最推荐。支持支付宝，自带外网访问能力，连接国内速度快，方便调试。
    *   **AWS / Vultr**：如果手头有外币卡，可选这些国际大厂。

### 2. 技术栈架构
*   **后端框架**：**FastAPI** (高性能 Python Web 框架)。
*   **智能体框架**：**LangGraph** (用于构建循环、反思工作流)。
*   **容器化**：**Docker** (确保本地和服务器环境一致，一键部署)。

---

## 三、 开发路线图 (Roadmap)

### 阶段 1：本地原型验证 (Local Prototype)
**目标**：在本地跑通“规划->搜索->反思->写作”的闭环。

1.  **环境搭建**：
    *   初始化 Python 项目，安装 `langgraph`, `langchain`, `fastapi`。
    *   配置 `.env` 文件（填入 DeepSeek/OpenAI/Tavily Key）。
2.  **构建 Planner Node**：
    *   编写 Prompt，让大模型将问题拆解为 DAG（有向无环图）或线性大纲。
3.  **构建 Researcher Node**：
    *   集成 Tavily API。
    *   实现“自我修正”逻辑（如果搜不到，尝试换关键词）。
4.  **构建 Writer Node**：
    *   编写“深度研究员”风格的 Prompt。
5.  **组装 Graph**：
    *   使用 LangGraph 串联各节点，实现循环工作流。

### 阶段 2：后端服务化 (Backend API)
**目标**：将 Python 脚本转化为可远程调用的 HTTP 接口。

1.  **API 设计**：
    *   `POST /search`: 启动任务。
    *   `GET /stream/{task_id}`: 建立 SSE 连接，实时推送“正在思考...”、“正在阅读...”等日志。
2.  **状态管理**：
    *   引入简单的内存存储或 Redis，保存任务状态。
3.  **Docker 封装**：
    *   编写 `Dockerfile` 和 `docker-compose.yml`。

### 阶段 3：服务器部署 (Deployment)
**目标**：在云服务器上上线。

1.  **服务器初始化**：安装 Docker, Git。
2.  **代码部署**：拉取代码，配置环境变量。
3.  **服务启动**：`docker-compose up -d`。
4.  **域名配置** (可选)：配置 Nginx 反向代理和 HTTPS。

### 阶段 4：客户端接入 (Frontend/App)
**目标**：用户界面。

*   **Web**：部署一个简单的 React/Vue 页面或继续使用 Streamlit（改造成调用后端 API 模式）。
*   **App**：开发 iOS/Android 客户端对接 API。

---

## 四、 核心升级策略 (V2 & V3)

### 1. 全领域自适应规划 (Adaptive Domain Planner)
**目标**：解决“冷门领域”无法匹配预设角色的问题。

*   **技术方案**：使用 Meta Prompting (元提示)。
*   **逻辑**：Planner 不再选择分类，而是先让 LLM 分析问题，**动态生成**一个最适合该问题的“专家角色设定”和“搜索指导原则”。
    *   *例子*：遇到“桌游规则”，动态生成“资深桌游裁判”角色，并制定“优先搜索 BGG 论坛和官方 Rulebook”的策略。

### 2. 混合资料获取策略 (Hybrid Data Access)
**目标**：解决“付费墙”和“数据孤岛”问题。

*   **策略 A：寻找平替 (Open Alternatives)**
    *   对于付费论文，优先检索 ArXiv 预印本或作者个人主页的 PDF。
    *   对于付费教程，检索 YouTube 字幕或开源社区文档。
*   **策略 B：影子索引 (Shadow Indexing)**
    *   接入 Google Scholar, Z-Library (Metadata) 等索引库。即使无法下载原文，也能提供精确的文献元数据，指引用户自行获取。
*   **策略 C：诚实降级 (Graceful Degradation)**
    *   明确告知用户哪些数据在付费墙内，并基于公开摘要进行最大程度的推演。

### 3. 分级阅读器 (Hybrid Hierarchical Reader)
**目标**：解决“全书深度阅读”与“Token 成本”的矛盾。

*   **Level 1: 爬虫与清洗 (Crawler)**
    *   针对 `ctext.org` 等权威古籍站，使用 `Trafilatura` 抓取纯文本章节。
*   **Level 2: 关键词初筛 (BM25)**
    *   建立本地倒排索引，使用 BM25 算法筛选出含有查询关键词的高相关章节（无需 API 成本）。
*   **Level 3: 向量精筛 (Embedding)**
    *   对筛选出的章节进行切片并向量化（使用 OpenAI `text-embedding-3-small`，成本极低）。
    *   在临时 ChromaDB 中进行语义检索，提取最相关的 5-10 个片段。
*   **Level 4: 深度阅读 (LLM)**
    *   将精筛后的片段（约 3k-5k token）发送给 DeepSeek 进行深度总结。

---

## 5. 界面 (第一阶段)
- **Streamlit**: 支持基于 LangGraph 的流式输出，实时展示思考过程。
- **功能增强**:
  - 思考日志折叠/展开。
  - 一键复制/下载 Markdown 报告。
  - 参考文献智能链接引用。

## 六、 下一步行动

1.  **您**：去申请 DeepSeek, OpenAI (可选), Tavily 的 API Key。
2.  **您**：去租一台腾讯云香港轻量服务器（或类似 VPS）。
3.  **我**：在您准备好后，开始在本地为您搭建 **LangGraph** 的基础代码骨架。
