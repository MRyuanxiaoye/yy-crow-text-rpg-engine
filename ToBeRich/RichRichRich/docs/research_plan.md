# 号码规律深度研究方案

> 核心原则：不预设任何规律形式，不做任何简化假设，穷举一切可能的条件组合，让数据自己说话。
> 量变引起质变——用算力暴力覆盖所有可能的规律空间。

---

## 一、目标

从历史开奖数据中发现**统计显著的排除规则**：

- 号码与号码之间的关联
- 同一位置号码的时序规律
- 相邻位置号码的联动规律
- 组合层面的约束规律
- 任意条件组合下的号码分布规律

所有发现的规律最终输出为排除规则，经过严格回测验证后接入现有方法链。

---

## 二、搜索空间估算

| 维度 | 规模 |
|------|------|
| 单位置方向模式（窗口3-15） | ~265万种 |
| 跨位置联合方向模式（2-5位置组合） | ~数千万种 |
| 条件排除规则（1-4个原子条件组合） | ~数亿种 |
| 差分空间模式 | ~数百万种 |
| 滑动窗口统计量 | ~数十万种 |
| 深度学习特征空间 | 连续空间，由模型自动搜索 |

**保守估计总搜索空间：数十亿种条件组合。**

---

## 三、硬件环境

- 芯片：Apple M3（4性能核 + 4效率核）
- 内存：16 GB
- GPU：M3 集成 GPU，支持 Metal Performance Shaders (MPS)
- PyTorch MPS 加速：`torch.device("mps")`

---

## 四、模块设计

### 模块A：穷举式统计规律搜索（CPU 8核并行）

纯统计穷举，不用任何机器学习。用 `multiprocessing` 利用全部 CPU 核心。

#### A1. 全维度方向模式扫描 (`direction_scanner.py`)

对每期红球按位置拆分为5条时间序列，相邻期变化编码为 U(升)/D(降)/E(平)。

**扫描范围：**
- 单位置：5个位置 × 窗口3-15 × 所有方向模式
- 双位置联合：C(5,2)=10种组合 × 窗口3-10
- 三位置联合：C(5,3)=10种组合 × 窗口3-8
- 四位置联合：C(5,4)=5种组合 × 窗口3-6
- 五位置联合：窗口3-5

**对每种模式统计：**
- 出现次数（支持度）
- 后续每个方向的条件概率
- 后续值域分布（均值、标准差、分位数）
- 卡方检验 p 值
- 效应量（Cramér's V）

#### A2. 条件排除规则全量挖掘 (`rule_miner.py`)

定义原子条件库（约80-100种），穷举所有1-4个原子条件的AND组合。

**原子条件库：**
- 位置方向类：位置i最近1/2/3期方向为U/D/E
- 位置值域类：位置i值在区间[a,b]（按三等分/五等分）
- 差值类：位置i与位置j差值>N / <N
- 遗漏值类：号码X遗漏值>N期
- 组合类：奇偶比为X:Y、和值在区间[a,b]、AC值为N、连号组数为N
- 跨期类：上期和值与本期和值差>N

**对每种条件组合统计：**
- 满足条件的期数（支持度）
- 满足条件时每个号码的出现概率
- 与无条件概率的偏离度
- Fisher精确检验 p 值
- 提升度（lift）

#### A3. 差分空间全量分析 (`diff_analyzer.py`)

**分析维度：**
- 一阶到四阶差分
- 单位置差分序列
- 跨位置差分：位置i - 位置j 的差值序列（C(5,2)=10种）
- 差分的方向模式（复用A1的扫描方法）
- 差分绝对值的分布规律
- 差分序列的自相关函数（ACF）和偏自相关函数（PACF）
- 差分序列的游程检验（Runs Test）

#### A4. 滑动窗口统计量扫描 (`sliding_stats.py`)

**统计量列表：**
- 和值、跨度、AC值
- 奇偶比、大小比
- 连号组数
- 每个位置的值
- 位置间差值

**窗口大小：** 3-50

**对每个统计量的每个窗口：**
- 均值回复检测：偏离均值后多少期回归
- 突变检测：统计量突然变化后的规律
- 趋势持续性：上升/下降趋势最多持续几期
- 波动率变化：波动率高/低时期的号码分布差异

### 模块B：深度学习规律发现（MPS GPU 加速）

用 PyTorch + MPS 加速，从原始数据中自动学习特征。

#### B1. Transformer 序列预测模型 (`transformer_model.py`)

**架构：**
- 输入：过去 N 期的完整开奖数据（每期7维：5红+2蓝）
- 位置编码：可学习的位置编码
- Transformer Encoder：4-8层，128-256维，4-8头注意力
- 输出头：每个位置的值分布（分类头，softmax over 号码范围）

**训练：**
- 滚动窗口：用前 T 期预测第 T+1 期
- 损失函数：交叉熵
- 优化器：AdamW，余弦退火学习率
- 训练 500-2000 个 epoch

**规律提取：**
- 提取 attention 权重矩阵 → 哪些历史期对预测最重要
- 提取 attention 的位置模式 → 哪些位置间有关联
- 梯度归因（Integrated Gradients）→ 哪些输入特征最重要

#### B2. 对比学习 + 最近邻检索 (`contrastive_model.py`)

**架构：**
- Encoder：CNN 或 Transformer，将连续 N 期编码为 128 维向量
- 对比损失：NT-Xent（SimCLR风格）
- 正样本：时间上接近的片段
- 负样本：时间上远离的片段

**应用：**
- 编码当前最近 N 期
- 在历史中检索最相似的 K 个片段
- 统计这 K 个片段之后的号码分布 → 预测

#### B3. 变分自编码器 VAE (`vae_model.py`)

**架构：**
- Encoder：MLP 或 CNN，输入一期开奖数据，输出潜在分布 (μ, σ)
- Decoder：从潜在空间重建开奖数据
- 潜在空间维度：16-64

**应用：**
- 重建概率低的组合 = 异常组合 = 不太可能出现 → 排除
- 在潜在空间中采样 → 生成"最可能的组合"
- 潜在空间聚类 → 发现号码组合的隐藏类别

#### B4. 图神经网络 GNN (`gnn_model.py`)

**架构：**
- 节点：35个红球号码（大乐透）
- 边：共现频率、转移概率、位置关联
- GNN层：GraphSAGE 或 GAT，3-4层
- 输出：每个节点的"激活概率"

**应用：**
- 给定上一期的激活节点，预测下一期的激活概率
- 图结构本身就是号码关系的可视化

### 模块C：评估与验证框架

#### C1. 滚动窗口回测 (`backtester.py`)

- 训练集：前 N 期
- 验证集：第 N+1 到 N+M 期
- 滚动步长：1期
- 统计每条规则/模型在样本外的命中率、排除准确率

#### C2. 多重比较校正 (`significance.py`)

- Bonferroni 校正：p 值 × 测试次数
- Benjamini-Hochberg FDR 控制
- Permutation test：打乱时间序列，重新计算统计量，比较真实值是否显著

#### C3. 规则稳定性检验 (`stability.py`)

- 将历史数据分为 5 个时间段
- 规则在 ≥4 个时间段都显著 → 稳定规则
- 只在 1-2 个时间段显著 → 不稳定，标记为低可信度

#### C4. 规则排名与输出 (`rule_ranker.py`)

- 综合评分 = 效应量 × 稳定性 × 置信度 × 样本外命中率
- 输出格式：可直接接入 stage1 排除引擎
- 生成人类可读的规则报告

---

## 五、文件结构

```
src/research/
├── config.py                      # 全局研究配置
├── data_loader.py                 # 统一数据加载 + 预处理
│
├── brute_force/                   # 模块A：穷举搜索
│   ├── __init__.py
│   ├── direction_scanner.py       # A1: 全维度方向模式扫描
│   ├── rule_miner.py              # A2: 条件排除规则挖掘
│   ├── diff_analyzer.py           # A3: 差分空间全量分析
│   ├── sliding_stats.py           # A4: 滑动窗口统计量扫描
│   └── parallel_engine.py         # 多核并行引擎
│
├── deep_learning/                 # 模块B：深度学习
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset
│   ├── transformer_model.py       # B1: Transformer
│   ├── contrastive_model.py       # B2: 对比学习
│   ├── vae_model.py               # B3: VAE
│   ├── gnn_model.py               # B4: GNN
│   ├── trainer.py                 # 统一训练器（MPS加速）
│   └── attention_extractor.py     # 从训练好的模型提取规律
│
├── evaluation/                    # 模块C：评估验证
│   ├── __init__.py
│   ├── backtester.py              # C1: 滚动窗口回测
│   ├── significance.py            # C2: 多重比较校正
│   ├── stability.py               # C3: 规则稳定性检验
│   └── rule_ranker.py             # C4: 规则排名输出
│
├── runner.py                      # 主运行器（支持后台运行）
└── results/                       # 结果输出目录
    ├── rules/                     # 发现的规则（JSON）
    ├── models/                    # 训练好的模型（.pt）
    └── reports/                   # 分析报告（JSON + 可读文本）
```

---

## 六、运行方式

### 息屏后台运行

```bash
# 方式1：nohup（最简单）
nohup python3 src/research/runner.py --module all 2>&1 > research.log &

# 方式2：caffeinate 防止休眠 + nohup
caffeinate -i nohup python3 src/research/runner.py --module all 2>&1 > research.log &

# 方式3：tmux 会话（推荐，可随时查看进度）
tmux new -s research
python3 src/research/runner.py --module all
# Ctrl+B D 脱离会话，息屏后继续跑
# tmux attach -t research 重新连接查看进度
```

### 分模块运行

```bash
# 只跑模块A（CPU穷举）
python3 src/research/runner.py --module brute_force

# 只跑模块B（深度学习）
python3 src/research/runner.py --module deep_learning

# 只跑模块C（评估验证）
python3 src/research/runner.py --module evaluation

# 跑单个子模块
python3 src/research/runner.py --module A1  # 方向模式扫描
python3 src/research/runner.py --module B1  # Transformer
```

### 进度监控

```bash
# 查看日志
tail -f research.log

# 查看已发现的规则数量
ls -la src/research/results/rules/

# 查看 GPU 使用情况（MPS）
sudo powermetrics --samplers gpu_power -i 1000
```

---

## 七、预期产出

1. **规则库**：数百到数千条经过验证的排除规则，按效应量排名
2. **训练好的模型**：4个深度学习模型，可直接用于预测
3. **分析报告**：
   - 每个位置的最强规律是什么
   - 位置间的关联强度排名
   - 数据中可利用信息量的精确评估
   - 哪些维度的规律最强、哪些最弱
4. **可接入的排除引擎**：规则格式化后直接替换/增强现有 stage1

---

## 八、实施顺序

1. **基础设施**：config.py、data_loader.py、parallel_engine.py、runner.py
2. **模块A 全部实现**：A1 → A2 → A3 → A4（纯CPU，可立即开跑）
3. **模块C 回测框架**：backtester.py、significance.py（A跑完后立即验证）
4. **模块B 深度学习**：安装 PyTorch → dataset.py → B1 → B2 → B3 → B4
5. **模块C 完整验证**：对所有规则和模型做统一验证排名
6. **接入方法链**：将验证通过的规则接入 stage1 排除引擎

---

*文档创建时间：2026-02-07*
