# RichRichRich 架构文档

> 本文档供 AI 助手快速理解项目结构，避免重复探索。每次架构变更后需同步更新。

## 项目目标

彩票数据分析研究框架，针对双色球和大乐透，通过多阶段数据驱动方法探索号码预测和投注策略。

## 目录结构

```
src/research/
├── data_loader.py              # 数据加载器 (LotteryData)
├── experiment/
│   ├── e5_framework.py         # E5 公共框架 (521行)
│   ├── e5_fusion.py            # E5 融合与缓存 (275行)
│   ├── e5a_matrix_profile.py   # Matrix Profile (224行)
│   ├── e5b_autoencoder.py      # 自编码器 (232行)
│   ├── e5c_sax.py              # SAX 符号化 (256行) ← 最强子模型
│   ├── e5d_shapelet.py         # Shapelet (279行)
│   ├── e5e_contrastive.py      # 对比学习 (306行)
│   ├── e5f_dictionary.py       # 字典学习 (202行)
│   ├── e5g_adaptive_features.py# 位置自适应特征KNN (162行) ← 全量效果差，已弃用
│   ├── e6_constraint_propagation.py  # 百分位约束 (被E7取代)
│   ├── e6_scan_percentiles.py        # 百分位扫描
│   ├── e7_constraint_discovery.py    # HDI约束发现+AC3传播 (1140行)
│   ├── e8_fusion_simulation.py       # E5+E7端到端投注模拟 (1509行)
│   └── e9_cross_position_mining.py   # 跨位置关联暴力穷举 (新)
├── results/
│   ├── e5_pattern_discovery/
│   │   └── weights_cache/      # E5各子模型预计算权重 (.pkl)
│   ├── e6_constraint/
│   ├── e7_discovery/
│   ├── e8_simulation/
│   │   └── cache/              # E8回测检查点
│   └── e9_cross_position/      # E9跨位置穷举结果
└── experiment/e7/              # E7子实验目录
```

## 数据层

### LotteryData (data_loader.py)

```python
data = LotteryData("shuangseqiu")  # 或 "daletou"
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `n_draws` | int | 总期数 (双色球3413, 大乐透2833) |
| `red_count` | int | 红球个数 (双色球6, 大乐透5) |
| `red_range` | int | 红球范围 (双色球33, 大乐透35) |
| `blue_count` | int | 蓝球个数 (双色球1, 大乐透2) |
| `blue_range` | int | 蓝球范围 (双色球16, 大乐透12) |
| `position_series[p]` | ndarray(n_draws,) | 第p个位置的历史值序列 |
| `direction_series[p]` | ndarray(n_draws,) | 方向序列 (-1/0/1) |
| `red_matrix` | ndarray(n_draws, red_count) | 红球矩阵 |
| `blue_matrix` | ndarray(n_draws, blue_count) | 蓝球矩阵 |

## E5 系列：数据驱动模式发现

### 方法链

```
各子模型独立预测 → 缓存权重 → 加权融合 → 输出每位置号码概率分布
```

### 子模型性能 (双色球512期全量)

| 模型 | 方法 | AUC | rank | 关键参数 |
|------|------|:---:|:---:|---------|
| e5c | SAX符号化+频繁子串 | 0.8056 | 7.2 | alphabet=3/5/7, min_freq=5 |
| e5f | 字典学习+稀疏编码 | 0.7619 | 8.6 | n_comp=50/100/200, alpha=0.5/1.0 |
| e5b | 自编码器+低维KNN | 0.7594 | 8.7 | latent=16, epochs=500 |
| e5d | Shapelet+信息增益 | 0.7592 | 8.7 | n_cand=1000, top_m=100 |
| e5a | Matrix Profile+KNN | 0.7580 | 8.7 | window=[5,8,10,15,20], k=20 |
| e5e | 对比学习+嵌入KNN | 0.7407 | 9.3 | embed=32, n_neg=5 |
| e5g | 位置自适应特征KNN | 0.7173 | 10.0 | 位置特定特征1-3个 |

### 融合配置 (e5_fusion.py)

当前最优6模型融合权重 (双色球)：
```python
E5_BEST_ALPHAS = {
    "shuangseqiu": {
        "e5a": 0.119, "e5b": 0.077, "e5c": 0.428,
        "e5d": 0.107, "e5e": 0.080, "e5f": 0.190,
    }
}
```

融合后 AUC ≈ 0.80, rank ≈ 7.0

### 统一接口

所有子模型遵循相同签名：
```python
def run_e5x(data: LotteryData, test_indices: np.ndarray,
            max_train_idx: int, **kwargs) -> List[List[Dict[int, float]]]:
    """
    Returns: all_weights[sample_idx][pos] = {value: weight}
    - sample_idx: 对应 test_indices 中的第几个测试样本
    - pos: 位置索引 0..red_count-1
    - value: 号码值
    - weight: 预测权重 (越大越可能)
    """
```

### 评估函数 (e5_framework.py)

```python
evaluate_weights(all_weights, data, test_indices) -> Dict
```
- Level 1: 方向级准确率
- Level 2: 区间级存活率
- Level 3: 号码级 AUC（核心指标）、平均rank、top-K命中率

### 数据分割

```python
SPLIT_CONFIG = {
    "shuangseqiu": {"train_end": 2900, "test_start": 2901},  # 512期测试
    "daletou":     {"train_end": 2400, "test_start": 2401},
}
```

## E7 系列：约束发现与传播

### 约束类型 (6大类, 75-114种)

| 类型 | 说明 | 示例 |
|------|------|------|
| A | 单位置同期差分 | pos_diff_P0: 当期P0与上期P0的差 |
| B | 同期内两两约束 | gap_P0_P1: P1-P0的间距 |
| C | 跨期跨位置约束 | cross_P0_lag_P1: P0(t)与P1(t-1)的差 |
| D | 三元组约束 | triple_span_P0_P1_P2: 三位置跨度 |
| E | 聚合统计约束 | agg_sum, agg_span, agg_odd_count等 |
| F | 二阶差分约束 | diff2_P0: P0的二阶差分 |

### HDI (Highest Density Interval)

- 用 KDE 估计分布，找最短的包含 credible_mass 概率的区间
- 三级：HDI-70 (紧) / HDI-80 (中) / HDI-90 (松)
- 约束筛选：按 usefulness = reliability × tightness 排名，取 top-30

### AC-3 弧一致性传播

约束之间互相收紧候选集，迭代直到收敛（max_iter=10）

### Phase 流程

1. 穷举所有约束 + 计算 HDI
2. 排名筛选 top-30
3. AC-3 传播 + 组合计数
4. 对比 E6 效果
5. 局面条件化（按波动率分 low/mid/high）

## E8 系列：端到端投注模拟

### 方法链

```
E5权重筛选 → E7约束传播 → 回溯枚举合法组合 → 蓝球预测 → 动态分配 → 奖金计算
```

### 关键参数

| 参数 | 值 | 说明 |
|------|:--:|------|
| E5_CUMPROB_CUTOFF | 0.60 | E5累积概率截断阈值 |
| E7_HDI_KEY | "hdi_90" | E7使用的HDI级别 |
| E7_CACHE_REFRESH_INTERVAL | 50 | E7约束缓存刷新间隔(期) |
| BUDGET_BETS | 50 | 每期投注数 |
| MIN_TRAIN_PERIODS | 500 | 前500期不回测 |
| CHECKPOINT_INTERVAL | 50 | 检查点保存间隔 |

### 诊断追踪阶段

`diagnose_one_period` 逐步追踪真实号码存活：
- E5-1: 各子模型单独排名
- E5-2: 融合后排名
- E5-3: 累积概率截断 ← **主要瓶颈：98%的期在此被排除**
- E7-1: A类约束初始化
- E7-2: 交集
- E7-3: 排序
- E7-4: AC-3传播
- E7-5: 排序
- COMBO: B/D/E类组合约束剪枝
- TOP: 排名截断

### 最新回测结果 (双色球512期, 10注/期)

- 总投入: 9216元, 总奖金: 2880元, ROI: -68.76%
- 一等奖: 0期, 二等奖: 0期, 三等奖: 2期
- 平均最佳红球命中: 1.68个
- 红球命中分布: 0红11.7%, 1红40.6%, 2红30.5%, 3红13.3%, 4红3.5%, 5红0.4%

## 已验证的结论

1. **e5c (SAX) 是最强单模型**，AUC=0.8056，因为离散化天然降噪
2. **特征工程路线在全量上不如形态匹配**：e5g 在50期窗口上 rank=8.2 但全量 AUC=0.7173 垫底
3. **动态特征搜索无效**：30期验证窗口太短，选出的特征过拟合局部噪声
4. **E5 累积概率截断是主要瓶颈**：98%的期真实号码在 E5-3 阶段被排除
5. **E7 单独过滤能力不弱**：能砍掉47.9%号码空间，70%的期真实号码全部存活
6. **6模型融合 AUC≈0.80**，加入 e5g 无增量贡献
7. **跨位置联合建模（Step7）无效**：多任务MLP和自回归MLP均未超过独立MLP（差异<0.5%），原因是模型架构变了但输入特征仍是单位置的，没有真正的跨位置关联特征
8. **跨位置关联信号真实存在**：E1.5发现条件预测比独立预测提升52%，但尚未被有效利用

## E9 系列：跨位置关联暴力穷举（进行中）

### 核心思路

**零预设、纯暴力穷举**。不假设任何关联形式（不预设"升拉升"或"升挤降"），把所有位置组合的所有局面状态全部统计，输出完整的频次分布，不做任何显著性筛选。

### 设计哲学

每个位置的下一期变化同时受两股力量：
1. **自身惯性**：该位置自己的历史走势规律
2. **其他位置的牵引**：跨位置的联动关系

两股力量的权重不固定，由当前局面决定。E9 的目标是通过暴力穷举，把这两股力量的完整分布图画出来，供后续模型查表使用。

### 穷举范围

**位置组合**（双色球6位置 / 大乐透5位置）：

| 组合阶数 | 双色球 C(6,k) | 大乐透 C(5,k) |
|----------|:---:|:---:|
| 二元组 | 15 | 10 |
| 三元组 | 20 | 10 |
| 四元组 | 15 | 5 |
| 五元组 | 6 | 1 |
| 六元组 | 1 | - |
| **合计** | **57** | **26** |

**局面描述粒度**（从粗到细，全部跑）：

| 粒度 | 每位置状态数 | 描述 |
|------|:---:|------|
| G1 | 3 | 纯方向（升/降/平） |
| G2 | 6 | 方向 × 幅度二档（小≤中位数/大>中位数） |
| G3 | 9 | 方向 × 幅度三档（小/中/大，按三分位） |
| G4 | ~15-20 | 精确差分值离散化 |

**统计内容**：

对每种（位置组合 × 粒度 × 局面状态），记录该局面出现后**每个位置**下一期的：
- 方向分布（升/降/平的频次）
- 差分值的完整频次分布（不截断、不筛选）

输出一张巨大的查找表，形成类似正态分布的频次图。

### 数据分割

| 集合 | 范围 | 用途 |
|------|------|------|
| 统计期 | 第1期 ~ 第(N-200)期 | 穷举统计频次分布 |
| 回测期 | 最后200期 | 验证分布的预测能力 |

### 回测评估

对回测期的每一期：
1. 用当前局面查表，得到每个位置的变化分布
2. 多种组合、多种粒度的分布加权融合
3. 评估指标：
   - 方向预测准确率（对比当前87%基线）
   - 幅度预测MAE（对比当前3.82基线）
   - 号码级AUC（对比当前0.806基线）
   - 组合级全对率（对比当前45%基线）

### 文件规划

```
src/research/experiment/
├── e9_cross_position_mining.py    # 主实验：穷举 + 统计 + 回测
src/research/results/
└── e9_cross_position/
    ├── e9_distributions_shuangseqiu.pkl  # 完整频次分布表
    ├── e9_distributions_daletou.pkl
    ├── e9_backtest_shuangseqiu.json      # 回测结果
    ├── e9_backtest_daletou.json
    └── e9_summary.json                   # 汇总对比
```

### 计算量估算

最大情况（双色球，四元组 × G4）：15组合 × 20⁴ = 240万种局面。
但绝大多数局面在3200期中出现0次，用稀疏字典存储。
预计总内存 < 2GB，运行时间可控。

## 待探索方向

1. ~~降噪增强~~（优先级降低，先做E9）
2. ~~融合权重优化~~（优先级降低）
3. ~~e5c 失败案例分析~~（优先级降低）
4. **E9 跨位置关联穷举**（当前重点）→ 如果E9发现强信号，后续可替代E5作为第一步过滤

---

*最后更新: 2026-02-11*
*更新原因: 新增E9跨位置关联暴力穷举实验计划；补充Step7和跨位置关联的已验证结论*
