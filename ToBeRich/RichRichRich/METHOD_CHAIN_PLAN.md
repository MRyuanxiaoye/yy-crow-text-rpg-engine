# 方法链计划：从排除到购买的完整算法管道

## 设计理念

现有计划的问题在于：各分析方法（频率分析、马尔可夫链、LSTM等）各自独立运行，各自产出独立结论，缺乏一条**从数据到购买方案**的贯通链路。

本方案设计一条**方法链（Method Chain）**，也叫算法管道（Pipeline），所有算法不再独立存在，而是作为管道中的一个环节，数据从头流到尾，最终输出一个可执行的购买方案。

---

## 管道总览

```
历史数据
  │
  ▼
┌─────────────────────────────────────────────┐
│  阶段0：数据预处理与特征工程                    │
│  输入：原始开奖记录                             │
│  输出：特征矩阵 + 统计摘要                      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  阶段1：排除引擎（Exclusion Engine）            │
│  输入：特征矩阵                                │
│  输出：排除号码集合 + 每个号码的排除置信度        │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  阶段2：权重引擎（Weight Engine）               │
│  输入：未排除号码 + 特征矩阵                     │
│  输出：每个候选号码的综合权重分数                 │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  阶段3：组合生成器（Combination Generator）     │
│  输入：号码权重表                               │
│  输出：候选号码组合列表（按期望得分排序）          │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  阶段4：购买优化器（Purchase Optimizer）        │
│  输入：候选组合列表 + 预算约束                   │
│  输出：最终购买方案（最少钱覆盖最多号码）          │
└─────────────────────────────────────────────┘
```

**关键设计原则**：
- 每个阶段的输出是下一个阶段的输入，形成严格的数据流
- 每个阶段内部可以并行运行多个算法，但最终必须融合为一个统一输出
- 管道可配置：可以替换、增删任何阶段内的算法，不影响整体流程

---

## 阶段0：数据预处理与特征工程

### 目标
将原始开奖记录转化为可供后续所有阶段使用的**统一特征矩阵**。

### 输入
```
原始数据：[{period, date, red_balls, blue_balls}, ...]
```

### 处理流程

```
原始数据
  │
  ├──▶ 基础特征提取
  │     ├── 每个号码的历史出现频率
  │     ├── 每个号码的最近出现期数（遗漏值）
  │     ├── 每个号码的平均间隔期
  │     └── 每个号码的最大间隔期
  │
  ├──▶ 组合特征提取
  │     ├── 奇偶比（如 3:2）
  │     ├── 大小比（如 2:3）
  │     ├── 区间分布（三区或四区各多少个）
  │     ├── 和值
  │     ├── 跨度（最大值 - 最小值）
  │     ├── 连号组数
  │     └── AC值（号码复杂度指标）
  │
  ├──▶ 时序特征提取
  │     ├── 滑动窗口统计（近5/10/20/50期）
  │     ├── 号码的出现趋势（上升/下降/平稳）
  │     ├── 差分特征（相邻期的变化量）
  │     └── 周期性特征（FFT频谱分析）
  │
  └──▶ 关联特征提取
        ├── 号码共现矩阵（哪些号码经常一起出现）
        ├── 位置间相关性矩阵
        └── 一阶马尔可夫转移矩阵
```

### 输出
```python
{
    "number_features": {
        # 每个号码的特征向量
        1: {"freq": 0.15, "missing": 3, "avg_gap": 6.2, "trend": 0.8, ...},
        2: {"freq": 0.12, "missing": 7, "avg_gap": 7.1, "trend": -0.3, ...},
        ...
    },
    "combo_stats": {
        # 组合层面的统计摘要
        "odd_even_distribution": {...},
        "sum_range": {"mean": 85, "std": 15},
        ...
    },
    "correlation_matrix": [...],  # 号码共现矩阵
    "transition_matrix": [...],   # 马尔可夫转移矩阵
}
```

### 实现要点
- 所有后续阶段共享这一份特征数据，避免重复计算
- 特征矩阵在每次新数据到来时增量更新
- 提供缓存机制，避免每次全量重算

---

## 阶段1：排除引擎（Exclusion Engine）

### 目标
通过多种算法**并行投票**，计算每个号码在下一期**不出现的置信度**，将高置信度的号码排除。

### 核心思想
不是单一算法说了算，而是多个排除算法各自给出"排除建议"，最终通过**加权投票**决定哪些号码被排除。

### 排除算法集合

#### 算法1.1：连续重复排除器
```
规则：最近N期内连续出现的号码，下一期继续出现的概率降低
输入：每个号码的连续出现次数
输出：排除置信度（连续次数越多，置信度越高）

公式：
  confidence = 1 - (1/base_prob)^consecutive_count
  其中 base_prob = 该号码的历史出现概率
```

#### 算法1.2：遗漏值异常排除器
```
规则：遗漏值远低于平均间隔的号码（刚出现不久），短期内再次出现概率较低
输入：每个号码的当前遗漏值、平均间隔期
输出：排除置信度

公式：
  若 missing < avg_gap * 0.3:
    confidence = 0.6 * (1 - missing / (avg_gap * 0.3))
  否则:
    confidence = 0（不排除）
```

#### 算法1.3：极端组合约束排除器
```
规则：基于历史数据，某些组合特征几乎不出现，违反这些特征的号码被排除
约束条件（以大乐透前区为例）：
  - 全奇或全偶的概率 < 2%，排除导致全奇/全偶的号码
  - 全大或全小的概率 < 2%，排除导致全大/全小的号码
  - 和值超出 [历史均值 ± 2σ] 范围的组合
  - 连号超过3组的组合
  - AC值低于3的组合

输出：不是排除单个号码，而是标记"约束违反风险分数"
```

#### 算法1.4：马尔可夫转移排除器
```
规则：基于转移矩阵，从上一期号码出发，转移概率极低的号码被排除
输入：转移矩阵 + 上一期号码
输出：排除置信度

公式：
  对于号码 v:
    transition_prob = avg(P(v | last_number_i)) for i in last_draw
    若 transition_prob < threshold:
      confidence = 1 - transition_prob / threshold
```

#### 算法1.5：周期性排除器
```
规则：通过FFT或自相关分析，识别号码的出现周期，当前不在周期窗口内的号码被排除
输入：每个号码的出现时间序列
输出：排除置信度

方法：
  1. 对每个号码的出现序列做自相关分析
  2. 找到主周期 T
  3. 计算当前期距上次出现的期数 d
  4. 若 d 远离 T 的整数倍，排除置信度升高
```

#### 算法1.6：聚类异常排除器
```
规则：将历史开奖号码组合做聚类，距离所有聚类中心都很远的号码组合被排除
输入：历史组合的聚类结果
输出：号码的异常分数

方法：
  1. 对历史组合做 K-means 聚类
  2. 计算每个候选号码加入后，组合到最近聚类中心的距离
  3. 距离过大的号码获得较高排除置信度
```

### 融合策略：加权投票

```python
# 每个排除算法有一个权重（通过历史回测确定）
algorithm_weights = {
    "consecutive":   0.15,  # 连续重复排除器
    "missing_value": 0.20,  # 遗漏值异常排除器
    "extreme_combo": 0.25,  # 极端组合约束排除器
    "markov":        0.15,  # 马尔可夫转移排除器
    "periodicity":   0.10,  # 周期性排除器
    "cluster":       0.15,  # 聚类异常排除器
}

# 融合公式
for number in all_numbers:
    final_exclusion_confidence = Σ(weight_i * confidence_i(number))

# 排除阈值（可调参数）
EXCLUSION_THRESHOLD = 0.6

excluded_numbers = {n for n in all_numbers if final_exclusion_confidence[n] > EXCLUSION_THRESHOLD}
```

### 输出
```python
{
    "excluded_numbers": [3, 7, 12, 18, ...],       # 被排除的号码列表
    "exclusion_details": {
        3:  {"confidence": 0.78, "reasons": ["consecutive:0.9", "missing:0.7", ...]},
        7:  {"confidence": 0.65, "reasons": ["markov:0.8", "cluster:0.6", ...]},
        ...
    },
    "remaining_numbers": [1, 2, 4, 5, 6, 8, ...],  # 未被排除的号码
    "exclusion_rate": 0.35,                          # 排除比例（建议控制在30%-50%）
}
```

### 关键参数
- **排除阈值**：控制排除的激进程度，阈值越低排除越多，覆盖率下降但精度可能上升
- **算法权重**：通过历史回测自动调优，表现好的算法获得更高权重
- **排除比例上限**：硬性约束，防止排除过多导致候选池太小（建议不超过50%）

---

## 阶段2：权重引擎（Weight Engine）

### 目标
对阶段1输出的**未排除号码**，通过多种算法计算每个号码在下一期出现的**综合权重分数**。分数越高，该号码出现的可能性越大。

### 核心思想
与排除引擎类似，多个权重算法并行计算，最终通过**归一化加权融合**得到每个号码的综合分数。

### 权重算法集合

#### 算法2.1：频率回归权重
```
思路：号码的出现频率会向理论概率回归（大数定律）
      长期低于理论频率的号码，未来出现的概率倾向于升高

公式：
  theoretical_prob = 选取个数 / 号码总数
  actual_freq = 该号码历史出现次数 / 总期数
  deviation = theoretical_prob - actual_freq
  weight = sigmoid(deviation * scale_factor)

说明：偏差越大（实际频率越低于理论值），权重越高
```

#### 算法2.2：遗漏值回归权重
```
思路：遗漏值越大（越久没出现），出现的"压力"越大
      但不是简单线性关系，而是符合几何分布的累积概率

公式：
  p = 1 / avg_gap  （单期出现概率的估计）
  weight = 1 - (1-p)^current_missing  （累积概率）

说明：遗漏值越接近或超过平均间隔，权重越高
      但超过最大历史间隔后权重增长放缓（可能是冷号）
```

#### 算法2.3：时序衰减权重
```
思路：近期数据比远期数据更有参考价值
      使用指数衰减对历史数据加权

公式：
  对于号码 v，扫描最近 W 期：
    weight = Σ (出现标记_i * exp(-λ * i))  # i=0是最近一期
    其中 λ 是衰减系数（可调参数）

说明：最近出现过的号码获得较高的时序权重
      但这与排除引擎的"连续重复排除"形成制衡
```

#### 算法2.4：马尔可夫转移权重
```
思路：基于转移矩阵，从上一期号码出发，转移概率高的号码获得高权重

公式：
  对于候选号码 v：
    weight = avg(P(v | last_number_i)) for i in last_draw

  可扩展为二阶马尔可夫：
    weight = avg(P(v | last_draw, second_last_draw))

说明：与排除引擎的马尔可夫排除器共享转移矩阵，但方向相反
      排除器关注低概率，权重器关注高概率
```

#### 算法2.5：共现关联权重
```
思路：某些号码经常一起出现，如果其中一些号码已经被选为高权重，
      与它们共现频率高的号码也应获得加分

公式：
  对于候选号码 v：
    co_occurrence_score = Σ (共现次数(v, u) * weight(u)) / Σ weight(u)
    其中 u 遍历当前权重 Top-K 的号码

说明：这是一个迭代过程，可以运行2-3轮收敛
```

#### 算法2.6：深度学习预测权重（必选，使用 Apple M3 CPU）
```
思路：使用 LSTM/Transformer 模型，输入历史序列特征，
      输出每个号码在下一期出现的概率分布

架构：
  输入：最近 N 期的特征矩阵 (N, feature_dim)
  模型：Transformer Encoder (4层, 8头注意力)
  输出：每个号码的出现概率 (num_range,)

训练：
  使用历史数据的滑动窗口做监督学习
  损失函数：多标签交叉熵

硬件：Apple M3 芯片（CPU + Metal GPU 加速）
  - 使用 PyTorch 的 MPS (Metal Performance Shaders) 后端加速
  - M3 的统一内存架构适合中小规模模型训练
  - 模型规模控制在合理范围内，确保 M3 可流畅运行

说明：这是计算量最大的算法，也是管道中不可缺少的核心算法
      通过 M3 的 MPS 加速，训练和推理性能可满足需求
```

### 融合策略：归一化加权求和

```python
# 权重算法的融合权重（通过回测调优）
weight_algorithm_weights = {
    "freq_regression":  0.20,  # 频率回归
    "missing_regression": 0.20,  # 遗漏值回归
    "time_decay":       0.15,  # 时序衰减
    "markov":           0.15,  # 马尔可夫转移
    "co_occurrence":    0.15,  # 共现关联
    "deep_learning":    0.15,  # 深度学习（必选，M3 MPS加速）
}

# 融合流程
for number in remaining_numbers:
    # 1. 每个算法输出原始分数
    raw_scores = {algo: algo.compute(number) for algo in algorithms}

    # 2. 每个算法的分数做 Min-Max 归一化（映射到 0-1）
    normalized = {algo: min_max_normalize(score, algo) for algo, score in raw_scores.items()}

    # 3. 加权求和
    final_weight = Σ(algorithm_weight_i * normalized_score_i)

# 4. 最终分数再做一次全局归一化，使所有号码权重之和为1
final_weights = softmax(final_weights, temperature=T)
```

### 输出
```python
{
    "number_weights": {
        # 号码: 综合权重（已归一化，所有号码权重之和为1）
        1:  0.045,
        2:  0.038,
        4:  0.052,
        5:  0.061,
        ...
    },
    "weight_details": {
        1: {
            "final": 0.045,
            "breakdown": {
                "freq_regression": 0.5,
                "missing_regression": 0.3,
                "time_decay": 0.4,
                "markov": 0.6,
                "co_occurrence": 0.2,
            }
        },
        ...
    },
    "top_numbers": [5, 14, 21, 8, 29, ...],  # 按权重降序排列
}
```

---

## 阶段3：组合生成器（Combination Generator）

### 目标
根据阶段2输出的号码权重表，生成**多组候选号码组合**，每组组合都满足彩票规则约束，并按**期望得分**排序。

### 彩票规则约束（硬约束）

```
大乐透：
  前区：从 1-35 中选 5 个不重复号码（升序排列）
  后区：从 1-12 中选 2 个不重复号码（升序排列）
  单注价格：2元

双色球：
  红球：从 1-33 中选 6 个不重复号码（升序排列）
  蓝球：从 1-16 中选 1 个号码
  单注价格：2元
```

### 合理性约束（软约束，基于历史统计）

```
以大乐透前区为例：
  - 奇偶比：优选 2:3 或 3:2（历史占比约60%）
  - 大小比：优选 2:3 或 3:2（历史占比约55%）
  - 和值范围：[55, 120]（历史95%分位区间）
  - 跨度范围：[18, 34]（历史90%分位区间）
  - 连号：0-2组（历史95%分位）
  - AC值：≥ 4（历史90%分位）
  - 三区比：至少每区1个号码的概率约70%

以上参数均从历史数据动态计算，不硬编码
```

### 生成策略

#### 策略3.1：加权随机采样
```
方法：
  1. 将号码权重作为概率分布
  2. 按概率分布无放回采样，选出所需个数的号码
  3. 检查是否满足软约束，不满足则重新采样
  4. 重复 M 次，生成 M 组候选

优点：简单高效，天然倾向高权重号码
缺点：可能生成大量不满足软约束的组合，需要过滤
```

#### 策略3.2：贪心构造 + 约束修正
```
方法：
  1. 按权重降序排列号码
  2. 依次选入权重最高的号码
  3. 每选入一个号码，检查是否违反软约束
  4. 若违反，跳过该号码，选下一个
  5. 直到选满所需个数

优点：保证每组都满足约束
缺点：多样性不足，容易生成相似的组合
```

#### 策略3.3：遗传算法生成
```
方法：
  1. 初始种群：用策略3.1生成 P 组候选
  2. 适应度函数：
     fitness = w1 * 权重得分      # 号码权重之和
            + w2 * 约束满足度     # 满足软约束的程度
            + w3 * 多样性得分     # 与已有组合的差异度
  3. 选择：锦标赛选择
  4. 交叉：均匀交叉（交换部分号码）
  5. 变异：随机替换1-2个号码
  6. 迭代 G 代后输出最终种群

优点：兼顾质量和多样性
缺点：计算量较大
```

#### 策略3.4：蒙特卡洛树搜索（MCTS）
```
方法：
  1. 将号码选择建模为树搜索问题
  2. 每个节点代表"已选号码集合"
  3. 扩展：添加一个新号码
  4. 模拟：随机补全剩余号码
  5. 评估：计算组合的综合得分
  6. 回传：更新路径上的节点价值

优点：在大搜索空间中高效探索
缺点：实现复杂度较高
```

### 组合评分函数

```python
def score_combination(combo, number_weights, combo_stats):
    """计算一组号码的综合得分"""

    # 1. 权重得分：所有号码权重之和（越高越好）
    weight_score = sum(number_weights[n] for n in combo)

    # 2. 约束满足度：满足多少软约束（0-1之间）
    constraint_score = check_constraints(combo, combo_stats)

    # 3. 均衡性得分：号码分布是否均匀（避免扎堆）
    balance_score = compute_balance(combo)

    # 4. 历史相似度：与历史中奖组合的相似程度
    similarity_score = compute_historical_similarity(combo)

    # 综合得分
    final_score = (
        0.40 * weight_score +
        0.25 * constraint_score +
        0.20 * balance_score +
        0.15 * similarity_score
    )
    return final_score
```

### 输出
```python
{
    "candidates": [
        {
            "red_balls": [3, 8, 14, 25, 31],
            "blue_balls": [4, 9],
            "score": 0.87,
            "score_breakdown": {
                "weight": 0.92,
                "constraint": 1.0,
                "balance": 0.78,
                "similarity": 0.72
            }
        },
        {
            "red_balls": [5, 11, 19, 27, 33],
            "blue_balls": [2, 10],
            "score": 0.83,
            ...
        },
        # ... 生成 N 组候选（建议 N=50~200）
    ],
    "generation_stats": {
        "total_generated": 5000,    # 总共生成的组合数
        "after_filter": 200,        # 过滤后保留的组合数
        "avg_score": 0.75,
        "max_score": 0.87,
    }
}
```

---

## 阶段4：购买优化器（Purchase Optimizer）

### 目标
在给定预算下，从候选组合中选出**最少花费、覆盖最多号码**的购买方案。这是一个**集合覆盖问题（Set Cover Problem）**的变体。

### 问题建模

```
定义：
  U = 所有高权重候选号码的集合（从阶段2的 Top-K 号码中取）
  S = {s₁, s₂, ..., sₙ} 为阶段3生成的候选组合集合
  每个 sᵢ ⊆ U，且 cost(sᵢ) = 2元（单注价格）

目标：
  找到 S 的一个子集 S*，使得：
    1. ∪(sᵢ ∈ S*) 覆盖 U 中尽可能多的号码
    2. |S*| * 2 ≤ 预算 Budget
    3. 在满足预算的前提下，最大化覆盖率和期望得分
```

### 优化策略

#### 策略4.1：贪心集合覆盖
```
这是经典的贪心近似算法，近似比为 ln(n)+1

算法：
  covered = {}  # 已覆盖的号码
  selected = []  # 已选组合
  remaining = candidates.copy()

  while budget_remaining >= 2 and remaining:
      # 选择"性价比"最高的组合
      best = max(remaining, key=lambda s:
          len(s.numbers - covered) * s.score  # 新覆盖号码数 × 组合得分
      )
      selected.append(best)
      covered |= best.numbers
      remaining.remove(best)
      budget_remaining -= 2

  return selected
```

#### 策略4.2：分层覆盖策略
```
思路：将号码按权重分为三层，优先覆盖高权重层

  第一层（必覆盖）：权重 Top 30% 的号码 → 必须被至少2组覆盖
  第二层（应覆盖）：权重 30%-60% 的号码 → 至少被1组覆盖
  第三层（可覆盖）：权重 60%-100% 的号码 → 有预算余量时覆盖

算法：
  1. 先用贪心法选出覆盖第一层的最少组合
  2. 在剩余预算内，贪心覆盖第二层
  3. 若还有预算，覆盖第三层
```

#### 策略4.3：整数线性规划（ILP）
```
这是精确解法，适合候选组合数量不太大时（<1000）

变量：
  xᵢ ∈ {0, 1}  # 第 i 组是否购买
  yⱼ ∈ {0, 1}  # 第 j 个号码是否被覆盖

目标函数：
  maximize Σ(wⱼ * yⱼ)  # 最大化覆盖号码的权重之和

约束：
  Σ(xᵢ) * 2 ≤ Budget           # 预算约束
  yⱼ ≤ Σ(xᵢ : j ∈ sᵢ)         # 覆盖约束
  xᵢ ∈ {0,1}, yⱼ ∈ {0,1}      # 整数约束

求解：使用 OR-Tools 或 PuLP 求解器
```

#### 策略4.4：复式/胆拖投注优化
```
思路：利用彩票的复式和胆拖玩法，用更少的钱覆盖更多组合

大乐透复式规则：
  前区可选 6-20 个号码（至少5个）
  后区可选 3-12 个号码（至少2个）
  注数 = C(前区选号数, 5) × C(后区选号数, 2)
  价格 = 注数 × 2元

大乐透胆拖规则：
  胆码：必选号码（前区最多4个，后区最多1个）
  拖码：备选号码
  注数 = C(前区拖码数, 5-前区胆码数) × C(后区拖码数, 2-后区胆码数)

优化目标：
  找到最优的 胆码+拖码 组合，使得：
    1. 覆盖的高权重号码最多
    2. 花费最少

算法：
  1. 将权重最高的 K 个号码作为"胆码候选"
  2. 枚举不同的胆码组合（胆码数量少，枚举可行）
  3. 对每种胆码组合，贪心选择拖码
  4. 计算每种方案的 花费 和 覆盖权重
  5. 选择性价比最高的方案
```

### 购买方案对比评估

```python
def evaluate_plan(plan):
    """评估一个购买方案的综合指标"""
    return {
        "total_cost": plan.total_cost,                    # 总花费
        "total_combinations": plan.num_combinations,       # 覆盖的注数
        "unique_numbers_covered": len(plan.all_numbers),   # 覆盖的不同号码数
        "weight_coverage": plan.covered_weight_sum,        # 覆盖的权重总和
        "cost_efficiency": plan.covered_weight_sum / plan.total_cost,  # 性价比
        "top10_coverage": plan.top10_hit_rate,             # Top10号码覆盖率
        "diversity": plan.combination_diversity,           # 组合多样性
    }
```

### 输出（最终购买方案）

```python
{
    "plan_type": "compound",  # single(单式) / compound(复式) / drag(胆拖) / mixed(混合)
    "budget": 20,             # 预算（元）
    "total_cost": 18,         # 实际花费
    "total_combinations": 9,  # 总注数

    # 方案详情
    "tickets": [
        {
            "type": "胆拖",
            "dan_red": [5, 14],           # 前区胆码
            "tuo_red": [8, 21, 25, 29],   # 前区拖码
            "dan_blue": [4],              # 后区胆码
            "tuo_blue": [9, 11],          # 后区拖码
            "combinations": 6,            # 展开注数
            "cost": 12,                   # 花费
        },
        {
            "type": "单式",
            "red_balls": [3, 11, 19, 27, 33],
            "blue_balls": [2, 10],
            "combinations": 1,
            "cost": 2,
        },
        # ...
    ],

    # 覆盖分析
    "coverage": {
        "all_red_numbers": [3, 5, 8, 11, 14, 19, 21, 25, 27, 29, 33],
        "all_blue_numbers": [2, 4, 9, 10, 11],
        "top10_red_covered": 8,    # Top10权重红球覆盖了8个
        "top5_blue_covered": 4,    # Top5权重蓝球覆盖了4个
        "total_weight_covered": 0.72,  # 覆盖了72%的总权重
    },

    # 性价比分析
    "efficiency": {
        "cost_per_combination": 2.0,
        "weight_per_yuan": 0.04,
        "numbers_per_yuan": 0.89,
    }
}
```

---

## 回测验证体系

### 目标
验证整条方法链的有效性，并自动调优各阶段的参数。

### 回测方法

```
滚动窗口回测：

  历史数据：[期1, 期2, ..., 期N]

  第1轮：用 [期1 ~ 期N-50] 训练 → 预测期N-49 → 对比实际结果
  第2轮：用 [期1 ~ 期N-49] 训练 → 预测期N-48 → 对比实际结果
  ...
  第50轮：用 [期1 ~ 期N-1] 训练 → 预测期N → 对比实际结果

  共50轮回测，统计各项指标的平均值
```

### 评估指标

```python
metrics = {
    # ---- 排除引擎指标 ----
    "exclusion_precision":    "被排除号码中，确实未出现的比例（越高越好）",
    "exclusion_recall":       "实际未出现的号码中，被排除的比例",
    "exclusion_miss_rate":    "实际出现的号码被错误排除的比例（越低越好，核心指标）",

    # ---- 权重引擎指标 ----
    "weight_correlation":     "号码权重与实际出现的相关系数",
    "top5_hit_rate":          "权重Top5号码中，实际出现的个数",
    "top10_hit_rate":         "权重Top10号码中，实际出现的个数",
    "weighted_rank":          "实际出现号码的平均权重排名（越小越好）",

    # ---- 组合生成器指标 ----
    "combination_hit_count":  "候选组合中，命中号码最多的组合命中了几个",
    "any_prize_rate":         "候选组合中，至少中一个奖级的比例",

    # ---- 购买优化器指标 ----
    "coverage_efficiency":    "每元钱覆盖的权重值",
    "actual_prize":           "回测期间的实际中奖金额（如果有）",
    "roi":                    "投资回报率 = 中奖金额 / 投入金额",
}
```

### 参数自动调优

```
可调参数列表：
  - 排除阈值 EXCLUSION_THRESHOLD (0.4 ~ 0.8)
  - 各排除算法权重 (6个参数)
  - 各权重算法权重 (6个参数)
  - 组合评分函数权重 (4个参数)
  - softmax温度参数 T
  - 滑动窗口大小 W
  - 时序衰减系数 λ

调优方法：
  1. 贝叶斯优化（Optuna）
     - 目标函数：回测的 exclusion_miss_rate 最小化 + top10_hit_rate 最大化
     - 搜索空间：上述所有可调参数
     - 迭代次数：100-500次

  2. 网格搜索（用于关键参数的精细调优）
     - 对排除阈值和Top-K参数做细粒度网格搜索
```

---

## 实施路线图

### 第一步：基础设施（数据层）

```
目标：搭建数据管道，获取真实历史数据

任务：
  [1.1] 实现数据爬取/下载模块，获取大乐透和双色球全部历史数据
  [1.2] 实现数据清洗和验证模块
  [1.3] 实现阶段0的特征工程模块
  [1.4] 编写单元测试验证数据正确性

产出文件：
  src/data/crawler.py        # 数据获取
  src/data/loader.py         # 数据加载
  src/data/validator.py      # 数据验证
  src/features/engineer.py   # 特征工程
```

### 第二步：排除引擎

```
目标：实现阶段1的所有排除算法和融合逻辑

任务：
  [2.1] 实现6个排除算法（每个算法一个类，统一接口）
  [2.2] 实现加权投票融合器
  [2.3] 编写回测脚本，验证排除准确率
  [2.4] 调优排除阈值和算法权重

产出文件：
  src/engine/exclusion/consecutive.py
  src/engine/exclusion/missing_value.py
  src/engine/exclusion/extreme_combo.py
  src/engine/exclusion/markov.py
  src/engine/exclusion/periodicity.py
  src/engine/exclusion/cluster.py
  src/engine/exclusion/fusion.py       # 融合器
  src/engine/exclusion/__init__.py
```

### 第三步：权重引擎

```
目标：实现阶段2的所有权重算法和融合逻辑

任务：
  [3.1] 实现6个权重算法（统一接口）
  [3.2] 实现归一化加权融合器
  [3.3] 编写回测脚本，验证权重相关性
  [3.4] 调优算法权重和温度参数

产出文件：
  src/engine/weight/freq_regression.py
  src/engine/weight/missing_regression.py
  src/engine/weight/time_decay.py
  src/engine/weight/markov.py
  src/engine/weight/co_occurrence.py
  src/engine/weight/deep_learning.py    # 必选，M3 MPS加速
  src/engine/weight/fusion.py
  src/engine/weight/__init__.py
```

### 第四步：组合生成器

```
目标：实现阶段3的组合生成和评分逻辑

任务：
  [4.1] 实现加权随机采样生成器
  [4.2] 实现贪心构造生成器
  [4.3] 实现遗传算法生成器
  [4.4] 实现组合评分函数
  [4.5] 实现软约束检查器

产出文件：
  src/engine/generator/sampler.py
  src/engine/generator/greedy.py
  src/engine/generator/genetic.py
  src/engine/generator/scorer.py
  src/engine/generator/constraints.py
  src/engine/generator/__init__.py
```

### 第五步：购买优化器

```
目标：实现阶段4的购买方案优化

任务：
  [5.1] 实现贪心集合覆盖算法
  [5.2] 实现分层覆盖策略
  [5.3] 实现复式/胆拖投注计算器
  [5.4] 实现购买方案评估函数
  [5.5] 实现ILP精确求解（可选）

产出文件：
  src/engine/optimizer/greedy_cover.py
  src/engine/optimizer/layered_cover.py
  src/engine/optimizer/compound_calc.py   # 复式/胆拖计算
  src/engine/optimizer/evaluator.py
  src/engine/optimizer/__init__.py
```

### 第六步：管道串联与回测

```
目标：将所有阶段串联为完整管道，运行端到端回测

任务：
  [6.1] 实现 Pipeline 类，串联阶段0-4
  [6.2] 实现回测框架（滚动窗口）
  [6.3] 实现参数自动调优（Optuna）
  [6.4] 运行完整回测，输出评估报告
  [6.5] 根据回测结果调优各阶段参数

产出文件：
  src/pipeline.py             # 管道主类
  src/backtest/runner.py      # 回测运行器
  src/backtest/metrics.py     # 指标计算
  src/backtest/optimizer.py   # 参数调优
  src/main.py                 # 主入口
```

### 第七步：可视化与报告

```
目标：输出可读的分析报告和可视化图表

任务：
  [7.1] 排除引擎可视化（哪些号码被排除，为什么）
  [7.2] 权重分布可视化（号码权重热力图）
  [7.3] 购买方案可视化（覆盖分析图）
  [7.4] 回测结果可视化（各指标趋势图）
  [7.5] 生成综合分析报告（HTML或PDF）

产出文件：
  src/visualization/exclusion_viz.py
  src/visualization/weight_viz.py
  src/visualization/plan_viz.py
  src/visualization/report.py
```

---

## 项目目录结构（完整）

```
RichRichRich/
├── data/
│   ├── daletou_history.json
│   └── shuangseqiu_history.json
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── crawler.py
│   │   ├── loader.py
│   │   └── validator.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py
│   ├── engine/
│   │   ├── exclusion/
│   │   │   ├── __init__.py
│   │   │   ├── consecutive.py
│   │   │   ├── missing_value.py
│   │   │   ├── extreme_combo.py
│   │   │   ├── markov.py
│   │   │   ├── periodicity.py
│   │   │   ├── cluster.py
│   │   │   └── fusion.py
│   │   ├── weight/
│   │   │   ├── __init__.py
│   │   │   ├── freq_regression.py
│   │   │   ├── missing_regression.py
│   │   │   ├── time_decay.py
│   │   │   ├── markov.py
│   │   │   ├── co_occurrence.py
│   │   │   ├── deep_learning.py
│   │   │   └── fusion.py
│   │   ├── generator/
│   │   │   ├── __init__.py
│   │   │   ├── sampler.py
│   │   │   ├── greedy.py
│   │   │   ├── genetic.py
│   │   │   ├── scorer.py
│   │   │   └── constraints.py
│   │   └── optimizer/
│   │       ├── __init__.py
│   │       ├── greedy_cover.py
│   │       ├── layered_cover.py
│   │       ├── compound_calc.py
│   │       └── evaluator.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   └── optimizer.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── exclusion_viz.py
│   │   ├── weight_viz.py
│   │   ├── plan_viz.py
│   │   └── report.py
│   ├── pipeline.py
│   └── main.py
├── tests/
│   ├── test_data.py
│   ├── test_exclusion.py
│   ├── test_weight.py
│   ├── test_generator.py
│   ├── test_optimizer.py
│   └── test_pipeline.py
├── notebooks/
│   └── exploration.ipynb
├── results/
├── configs/
│   └── default.yaml          # 所有可调参数的默认配置
├── METHOD_CHAIN_PLAN.md      # 本文档
├── PROJECT_PLAN.md
├── DATA_ANALYSIS_METHODOLOGY.md
├── README.md
├── TASK_PROGRESS.md
└── requirements.txt
```

---

## 与现有计划的关系

| 现有文档 | 本方案中的对应 | 变化 |
|----------|---------------|------|
| PROJECT_PLAN.md 的排除法推理 | 阶段1：排除引擎 | 从单一方法升级为6算法投票融合 |
| PROJECT_PLAN.md 的权重计算 | 阶段2：权重引擎 | 从独立计算升级为6算法归一化融合 |
| PROJECT_PLAN.md 的组合生成 | 阶段3：组合生成器 | 新增遗传算法、MCTS、软约束检查 |
| DATA_ANALYSIS_METHODOLOGY.md 的统计学方法 | 分散在阶段0/1/2中 | 不再独立存在，而是作为管道环节 |
| DATA_ANALYSIS_METHODOLOGY.md 的机器学习方法 | 阶段2的深度学习权重算法 | 必选的权重算法，使用M3 MPS加速 |
| DATA_ANALYSIS_METHODOLOGY.md 的组合优化方法 | 阶段3和阶段4 | 遗传算法用于生成，ILP用于购买优化 |
| DATA_ANALYSIS_METHODOLOGY.md 的集成方法 | 贯穿阶段1-4的融合策略 | 不再是独立模块，而是每个阶段的内置能力 |
| **无对应** | **阶段4：购买优化器** | **全新模块，解决"怎么买最划算"的问题** |

---

## 总结

本方法链的核心价值：

1. **贯通性**：从原始数据到购买方案，一条管道走到底，不再有独立的分析孤岛
2. **融合性**：每个阶段内多算法并行+投票融合，比单一算法更稳健
3. **可验证性**：完整的回测体系，每个阶段都有明确的评估指标
4. **可调优性**：所有参数通过贝叶斯优化自动调优，减少人工干预
5. **实用性**：最终输出的不是抽象的"分析结论"，而是一个**可直接执行的购买方案**
6. **经济性**：购买优化器确保在预算内最大化号码覆盖，利用复式/胆拖玩法降低成本
