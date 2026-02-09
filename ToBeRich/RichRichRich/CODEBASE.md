# 项目代码结构文档

> 最后更新：2026-02-07
> 每次代码变更后需同步更新此文档

## 目录结构

```
RichRichRich/
├── CLAUDE.md                      # 架构计划 + 实验进度
├── data/
│   ├── daletou_history.json       # 大乐透历史数据（~2833期）
│   └── shuangseqiu_history.json   # 双色球历史数据（~3413期）
├── src/
│   ├── research/                  # 研究模块
│   │   ├── runner.py              # 主运行器（214行）
│   │   ├── config.py              # 全局配置（121行）
│   │   ├── data_loader.py         # 数据加载器（160行）
│   │   ├── brute_force/           # 模块A：穷举搜索
│   │   │   ├── direction_scanner.py   # A1：方向模式扫描（298行）
│   │   │   ├── rule_miner.py          # A2：条件规则挖掘（503行）
│   │   │   ├── diff_analyzer.py       # A3：差分分析（392行）
│   │   │   ├── sliding_stats.py       # A4：滑动窗口统计（395行）
│   │   │   └── parallel_engine.py     # 并行计算引擎（121行）
│   │   ├── deep_learning/         # 模块B：深度学习
│   │   │   ├── transformer_model.py   # Transformer模型（140行）
│   │   │   ├── dataset.py             # 数据集封装（138行）
│   │   │   └── trainer.py             # 训练器（303行）
│   │   ├── evaluation/            # 模块C：评估验证
│   │   │   ├── backtester.py          # 回测验证（184行）
│   │   │   └── stability.py           # 稳定性检验（193行）
│   │   ├── experiment/            # 规则适配实验（待实现）
│   │   │   └── PLAN.md                # 实验计划文档
│   │   └── results/               # 输出目录
│   │       ├── rules/             # A模块规则文件
│   │       ├── models/            # B模块模型权重
│   │       └── reports/           # C模块评估报告
│   └── pipeline/                  # 预测流水线（已有，非研究模块）
│       ├── advanced/
│       │   ├── position_pattern.py
│       │   ├── sequence_mining.py
│       │   └── information_theory.py
│       ├── advanced_analysis.py
│       ├── config.py
│       ├── stage0_preprocess.py
│       ├── stage1_exclusion.py
│       └── stage2_weight.py
└── .claude/
    └── settings.local.json        # Claude Code权限配置
```

## 数据格式

### 历史数据 (data/*_history.json)

```json
{
  "lottery_type": "daletou",
  "rules": {
    "red_balls": { "count": 5, "range": [1, 35] },
    "blue_balls": { "count": 2, "range": [1, 12] }
  },
  "draws": [
    { "period": "07001", "date": "2007-05-30", "red_balls": [22,24,29,31,35], "blue_balls": [4,11] }
  ]
}
```

| 彩种 | 红球数 | 红球范围 | 蓝球数 | 蓝球范围 | 总期数 |
|------|--------|----------|--------|----------|--------|
| 大乐透 | 5 | 1-35 | 2 | 1-12 | ~2833 |
| 双色球 | 6 | 1-33 | 1 | 1-16 | ~3413 |

## 核心类和函数

### data_loader.py — LotteryData

```python
class LotteryData:
    # 属性
    lottery_type: str              # "daletou" / "shuangseqiu"
    draws: List[Dict]              # 原始开奖数据
    red_count: int                 # 红球数量（大乐透5，双色球6）
    red_range: int                 # 红球最大值（大乐透35，双色球33）
    n_draws: int                   # 总期数
    position_series: Dict[int, np.ndarray]   # {pos: 值序列}
    direction_series: Dict[int, np.ndarray]  # {pos: 方向序列}，1=升/0=平/-1=降
    red_matrix: np.ndarray         # (n_draws, red_count) 红球矩阵

    # 方法
    get_diff_series(pos, order) -> np.ndarray      # N阶差分序列
    get_cross_diff_series(pos_i, pos_j) -> np.ndarray  # 跨位置差分
    get_combo_stats_series() -> Dict[str, np.ndarray]  # 组合统计量序列
        # 返回: sum, span, odd_count, big_count, ac_value, consec_groups
```

### config.py — ResearchConfig

```python
class ResearchConfig:
    # 模块A参数
    direction_single_windows = (3, 30)   # 单位置方向窗口范围
    direction_dual_windows = (3, 20)
    direction_triple_windows = (3, 15)
    direction_min_support = 3
    direction_p_threshold = 0.01
    rule_max_conditions = 5              # A2最大条件组合数
    rule_min_support = 10
    rule_min_lift = 1.2
    rule_value_bins = 10                 # 值域分箱数
    diff_max_order = 5
    diff_window_range = (3, 20)
    sliding_window_range = (3, 100)

    # 模块B参数
    transformer_seq_len = 50
    transformer_d_model = 256
    transformer_n_heads = 8
    transformer_n_layers = 6
    transformer_epochs = 5000
    transformer_lr = 5e-4
    transformer_batch_size = 32
    transformer_patience = 300

    # 模块C参数
    backtest_train_ratio = 0.7
    backtest_min_train = 500
    stability_n_folds = 10

    # 并行参数
    n_workers = 7
    chunk_size = 500
```

### runner.py — 执行流程

```
main(lottery_type, module)
  → LotteryData(lottery_type)
  → run_module_a()  # A1→A2→A3→A4 顺序执行
  → run_module_b()  # B1 Transformer训练
  → run_module_c()  # C1回测 → C2稳定性
  → generate_report()
```

### brute_force/rule_miner.py — A2规则挖掘

```python
class RuleMiner:
    # 原子条件类型（约118个）:
    #   P{0-4}_val_{lo}_{hi}     位置值域分箱（每位置10箱）
    #   P{0-4}_dir_{U/D/E}       方向（每位置3个）
    #   P{0-4}_odd / _even       奇偶
    #   P{0-4}_big / _small      大小
    #   P{0-4}_diff_{lo}_{hi}    一阶差分区间（每位置3箱）
    #   combo_{stat}_{lo}_{hi}   组合统计量分箱（6统计量×3箱）

    # target格式: next_P{0-4}_{U/D/E}

    # 规则结构:
    # {
    #   "conditions": ["P1_val_6_8", "P0_odd", ...],
    #   "target": "next_P1_E",
    #   "support": 10,        # 前件出现次数
    #   "n_both": 6,          # 前件+后件同时出现次数
    #   "confidence": 0.6,    # n_both / support
    #   "expected_confidence": 0.0434,
    #   "lift": 13.82,        # confidence / expected_confidence
    #   "chi2": 71.35,
    #   "contingency": [6, 4, 117, 2706]  # 2x2列联表
    # }

    # 并行策略: 穷举1-5条件组合，约1.93亿个，7进程并行
    # JSON中保存top 10000条规则（按lift降序）
```

### brute_force/direction_scanner.py — A1方向扫描

```python
class DirectionScanner:
    # 方向编码: -1→0, 0→1, 1→2，3进制编码
    # 扫描: 所有位置组合(1-5位) × 所有窗口大小
    # 输出模式结构:
    # {
    #   "positions": [0,1],
    #   "window": 5,
    #   "pattern": "(UUDDE)(DDUUE)",
    #   "pattern_key": [242, 0, 242, 0, 242],
    #   "observed": 5,
    #   "expected": 0.0,
    #   "chi2": 7492824622.5,
    #   "next_distribution": {"(1,-1)": {"count":3, "prob":0.6}, ...},
    #   "prediction_confidence": 0.6,
    #   "total_windows": 2827
    # }
```

### brute_force/diff_analyzer.py — A3差分分析

```python
class DiffAnalyzer:
    # 分析内容:
    #   diff_distributions: 各位置各阶差分的统计分布
    #   acf_analysis: 自相关函数，找显著滞后期
    #   runs_tests: 游程检验，判断随机性
    #   cross_position_correlation: 跨位置差分相关系数
    #   diff_patterns: 差分模式扫描（离散化为大降/小降/平/小升/大升）
    #     结构: {pos: {order: {n_patterns, top_patterns: [{window, pattern, observed, expected, chi2, prediction, prediction_confidence}]}}}
```

### brute_force/sliding_stats.py — A4滑动窗口统计

```python
class SlidingStatsScanner:
    # 统计量: sum, span, odd_count, big_count, ac_value, consec_groups
    # 分析内容:
    #   sliding_stats: 窗口5/10/20/30/50的均值/标准差/趋势
    #   changepoints: 突变点检测（窗口20，阈值2σ）
    #   periodicity: FFT周期性检测
    #   cross_stat_correlation: 统计量间滞后相关（±10期）
    #   stat_patterns: 统计量模式扫描
    #     单统计量 + 两两联合统计量
    #     结构: {stat: {n_patterns, top_patterns: [{window, pattern, observed, expected, chi2, prediction, prediction_confidence}]}}
```

### deep_learning/dataset.py — 特征构建

```python
class LotterySequenceDataset:
    # 特征维度: red_count*5 + 6（大乐透31维，双色球36维）
    #   位置值归一化: red_count维
    #   方向one-hot: red_count*3维
    #   一阶差分归一化: red_count维
    #   组合统计量归一化: 6维
    # 序列长度: 50期
    # 目标: 下一期各位置方向（-1→0, 0→1, 1→2）
```

### deep_learning/transformer_model.py

```python
class LotteryTransformer(nn.Module):
    # 结构: InputProj → PositionalEncoding → TransformerEncoder(6层) → 每位置独立分类头
    # 参数: ~490万
    # 输入: (batch, seq_len=50, feature_dim)
    # 输出: 每位置3分类logits
```

### evaluation/backtester.py

```python
class Backtester:
    # 回测方式: 用A1方向模式在测试期逐期预测
    # 指标: 准确率（预测方向=实际方向）、覆盖率（有预测的期数占比）
    # 切分: train_ratio=0.7
```

### evaluation/stability.py

```python
class StabilityChecker:
    # 10折交叉验证
    # 检查规则在不同折上的一致性
    # 通过标准: 各折准确率标准差 < 阈值
```

### parallel_engine.py

```python
def parallel_map(func, tasks, n_workers=7, chunk_size=500):
    # 封装 multiprocessing.Pool.map
    # 任务太少时自动切换单进程
```

## 结果文件

### rules/ 目录

| 文件 | 内容 | 关键字段 |
|------|------|----------|
| a1_direction_patterns_*.json | 方向模式 | total_patterns, patterns[] |
| a2_exclusion_rules_*.json | 条件排除规则（top 10000） | total_rules, rules[] |
| a3_diff_analysis_*.json | 差分分析 | analysis.{diff_distributions, acf_analysis, runs_tests, cross_position_correlation, diff_patterns} |
| a4_sliding_stats_*.json | 滑动窗口统计 | analysis.{sliding_stats, changepoints, periodicity, cross_stat_correlation, stat_patterns} |

### 规则数量统计

| 模块 | 大乐透 | 双色球 |
|------|--------|--------|
| A1 方向模式 | 21,248 | 40,377 |
| A2 条件规则（总量/JSON中） | 6,095,591 / 10,000 | 19,751,636 / 10,000 |
| A3 差分模式 | 6,117 | - |
| A4 统计量模式 | 14,634 | - |

### A2规则分布（大乐透top 10000）

- 条件数: 3条42个, 4条981个, 5条8977个
- Target: next_P1_E(4926), next_P2_E(2750), next_P3_E(2268), next_P0_E(56)
- Support: 10-29, 平均11.6
- Confidence: 0.333-0.800, 平均0.373
- Lift: 7.55-13.82, 平均8.43
- 条件类型频率: val > diff > dir > odd > big > small > even > consec > sum > span > ac
