# 规则适配学习实验 - 实现计划

## 概述

本文档是实验脚本的详细实现计划。脚本为全自动运行，无需人工干预。
先读此文档了解设计，再看代码。

## 文件结构

```
src/research/experiment/
├── run_experiment.py          # 主入口，串联四个阶段
├── phase1_compress.py         # 阶段一：规则压缩
├── phase2_replay.py           # 阶段二：历史回放 + 训练数据构造
├── phase3_train.py            # 阶段三：适配模型训练（3A/3B/3C）
├── phase4_analyze.py          # 阶段四：结果分析
└── utils.py                   # 公共工具函数
```

输出目录：`src/research/results/experiment/`

## 依赖

现有依赖：numpy, scipy, torch, sklearn
新增依赖：xgboost（或 lightgbm）

安装命令：`pip3 install xgboost`

---

## 阶段一：规则压缩 (phase1_compress.py)

### 输入
- `results/rules/a2_exclusion_rules_daletou.json`（10000条，JSON中保存的top规则）
- `results/rules/a2_exclusion_rules_shuangseqiu.json`
- `results/rules/a1_direction_patterns_*.json`
- 注意：A2 JSON 中只保存了 top 10000 条规则，不是全部 2585 万条

### 处理逻辑

#### Step 1: A2 规则特征化

每条 A2 规则有 3-5 个 conditions 和 1 个 target。将其编码为固定维度向量：

```python
# 条件类型 one-hot（11类）：val/dir/odd/even/big/small/diff/sum/span/ac/consec
# 涉及位置 multi-hot（5位）：P0-P4
# target 方向（3类）：U/D/E
# target 位置（5位）：P0-P4
# 数值特征：support, confidence, lift, chi2, n_conditions

# 特征维度：11 + 5 + 3 + 5 + 5 = 29 维
```

具体编码函数：
```python
def encode_rule(rule):
    vec = np.zeros(29)
    for cond in rule['conditions']:
        # 解析条件类型
        cond_type = parse_condition_type(cond)  # 返回 0-10 的索引
        vec[cond_type] += 1  # 计数而非 one-hot，因为可能有多个同类条件
        # 解析涉及位置
        pos = parse_condition_position(cond)  # 返回 0-4 或 None（combo类）
        if pos is not None:
            vec[11 + pos] = 1
    # target 编码
    target_dir = rule['target'].split('_')[-1]  # U/D/E
    vec[16 + {'U':0, 'D':1, 'E':2}[target_dir]] = 1
    target_pos = int(rule['target'].split('_')[1][1])  # P0-P4
    vec[19 + target_pos] = 1
    # 数值特征（归一化）
    vec[24] = rule['support']
    vec[25] = rule['confidence']
    vec[26] = rule['lift']
    vec[27] = rule['chi2']
    vec[28] = len(rule['conditions'])
    return vec
```

#### Step 2: 聚类

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 归一化
scaler = StandardScaler()
X = scaler.fit_transform(feature_matrix)

# 搜索最优簇数
best_k, best_score = 0, -1
for k in [100, 200, 500, 1000, 1500, 2000]:
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels, sample_size=min(5000, len(X)))
    if score > best_score:
        best_k, best_score = k, score

# 用最优 k 聚类
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = km.fit_predict(X)
```

#### Step 3: 选簇代表

每个簇选 confidence 最高的规则作为代表：
```python
clusters = {}
for cluster_id in range(best_k):
    mask = labels == cluster_id
    cluster_rules = [rules[i] for i in range(len(rules)) if mask[i]]
    # 按 confidence 降序，取第一条
    representative = max(cluster_rules, key=lambda r: r['confidence'])
    clusters[cluster_id] = {
        'representative': representative,
        'size': int(mask.sum()),
        'avg_confidence': np.mean([r['confidence'] for r in cluster_rules]),
        'avg_lift': np.mean([r['lift'] for r in cluster_rules]),
    }
```

#### Step 4: A1 规则筛选

A1 有 21,248 条（大乐透），数量可管理。按 prediction_confidence 筛选：
```python
# 只保留 confidence >= 0.4 的模式
a1_filtered = [p for p in a1_patterns if p['prediction_confidence'] >= 0.4]
# 按位置组合+窗口大小分组，每组保留 top 5
```

### 输出
```
results/experiment/
├── phase1_clusters_daletou.json
│   {
│     "n_clusters": 500,
│     "silhouette_score": 0.35,
│     "clusters": { "0": { "representative": {...}, "size": 20, ... }, ... },
│     "a1_filtered": [ ... ],
│     "a1_filtered_count": 8000
│   }
├── phase1_clusters_shuangseqiu.json
└── phase1_summary.json   # 两个彩种的压缩汇总
```

---

## 阶段二：历史回放 (phase2_replay.py)

### 输入
- 原始历史数据：`data/daletou_history.json`
- 阶段一输出：`results/experiment/phase1_clusters_*.json`

### 核心问题：数据泄露

现有 A2 规则是用全量数据挖掘的。严格来说应该用训练期重新挖掘。
但重新挖掘 A2 需要几十分钟且代码复杂。

**折中方案**：
- 不重新挖掘规则（太耗时）
- 但严格用时间切分做评估：规则虽然用全量数据挖掘，但适配模型只在后半段训练，测试集完全独立
- 在最终报告中标注这个局限性
- 如果实验结果有价值，后续再做严格的时间切分版本

### 时间切分

```python
n = len(draws)
# 前 60% 作为"规则已知期"（规则本身就是从这些数据挖出来的）
# 中间 25% 作为适配训练期
# 后 15% 作为测试期
split1 = int(n * 0.6)
split2 = int(n * 0.85)

rule_period = draws[:split1]       # 规则挖掘期（不用于训练适配模型）
train_period = draws[split1:split2] # 适配训练期
test_period = draws[split2:]        # 测试期
```

### 特征构造

对训练期和测试期的每一期 t，构造特征向量：

```python
def build_features(t, data, clusters, a1_filtered):
    features = {}

    # === 局面特征 ===
    # 1. 最近 5 期各位置的值（归一化）
    for lag in range(1, 6):
        for pos in range(red_count):
            features[f'val_lag{lag}_P{pos}'] = normalize(data.red_matrix[t-lag, pos])

    # 2. 最近 5 期各位置的方向
    for lag in range(1, 6):
        for pos in range(red_count):
            features[f'dir_lag{lag}_P{pos}'] = data.direction_series[pos][t-lag]

    # 3. 最近 5 期各位置的一阶差分（归一化）
    for lag in range(1, 6):
        for pos in range(red_count):
            diff = data.get_diff_series(pos, 1)
            features[f'diff_lag{lag}_P{pos}'] = normalize(diff[t-lag])

    # 4. 最近 3 期的组合统计量
    combo_stats = data.get_combo_stats_series()
    for lag in range(1, 4):
        for stat_name in ['sum', 'span', 'odd_count', 'big_count', 'ac_value', 'consec_groups']:
            features[f'stat_lag{lag}_{stat_name}'] = normalize(combo_stats[stat_name][t-lag])

    # === 规则簇触发特征 ===
    # 5. 每个 A2 规则簇的代表规则是否被触发
    for cluster_id, cluster in clusters.items():
        rule = cluster['representative']
        triggered = check_rule_conditions(rule['conditions'], data, t)
        features[f'a2_cluster_{cluster_id}'] = 1 if triggered else 0

    # 6. A1 方向模式匹配
    for i, pattern in enumerate(a1_filtered):
        matched = check_direction_pattern(pattern, data, t)
        features[f'a1_pattern_{i}'] = 1 if matched else 0

    # === 触发元信息 ===
    # 7. 触发的 A2 簇数量、平均 confidence
    triggered_clusters = [c for cid, c in clusters.items()
                          if features[f'a2_cluster_{cid}'] == 1]
    features['n_triggered_a2'] = len(triggered_clusters)
    features['avg_conf_triggered'] = (
        np.mean([c['avg_confidence'] for c in triggered_clusters])
        if triggered_clusters else 0
    )
    features['avg_lift_triggered'] = (
        np.mean([c['avg_lift'] for c in triggered_clusters])
        if triggered_clusters else 0
    )

    return features
```

### 标签构造

```python
def build_labels(t, data):
    labels = {}
    for pos in range(data.red_count):
        # 方向标签：下一期该位置的方向
        direction = data.direction_series[pos][t]  # -1/0/1
        labels[f'dir_P{pos}'] = direction + 1  # 映射到 0/1/2

        # 值标签：下一期该位置的实际值
        labels[f'val_P{pos}'] = data.red_matrix[t, pos]
    return labels
```

### 条件匹配函数

```python
def check_rule_conditions(conditions, data, t):
    """检查 A2 规则的所有条件是否在第 t 期满足"""
    for cond in conditions:
        if not check_single_condition(cond, data, t):
            return False
    return True

def check_single_condition(cond, data, t):
    """解析并检查单个原子条件"""
    # 解析条件字符串，如 "P1_val_6_8", "P0_odd", "P3_dir_U", "combo_sum_27_83"
    parts = cond.split('_')

    if parts[0].startswith('P'):
        pos = int(parts[0][1])
        val = data.red_matrix[t, pos]

        if parts[1] == 'val':
            lo, hi = int(parts[2]), int(parts[3])
            return lo <= val <= hi
        elif parts[1] == 'dir':
            dir_map = {'U': 1, 'D': -1, 'E': 0}
            return data.direction_series[pos][t] == dir_map[parts[2]]
        elif parts[1] == 'odd':
            return val % 2 == 1
        elif parts[1] == 'even':
            return val % 2 == 0
        elif parts[1] == 'big':
            return val > data.red_range / 2
        elif parts[1] == 'small':
            return val <= data.red_range / 2
        elif parts[1] == 'diff':
            lo, hi = int(parts[2]), int(parts[3])
            diff = data.get_diff_series(pos, 1)
            return lo <= diff[t] <= hi

    elif parts[0] == 'combo':
        stats = data.get_combo_stats_series()
        stat_name = parts[1]
        lo, hi = int(parts[2]), int(parts[3])
        return lo <= stats[stat_name][t] <= hi

    return False
```

### 输出

```
results/experiment/
├── phase2_train_daletou.npz      # 训练集特征矩阵 + 标签
├── phase2_test_daletou.npz       # 测试集特征矩阵 + 标签
├── phase2_train_shuangseqiu.npz
├── phase2_test_shuangseqiu.npz
├── phase2_feature_names.json     # 特征名列表
└── phase2_summary.json           # 数据集统计信息
```

---

## 阶段三：适配模型训练 (phase3_train.py)

### 输入
- 阶段二输出的训练/测试数据

### 任务定义

对每个位置，训练一个二分类模型：
- 输入：局面特征 + 规则触发特征
- 输出：每个候选号码是否应该被排除

但候选号码太多（35个位置），直接做号码级排除样本量不够。

**简化方案：方向预测 + 规则加权**

分两步：
1. 预测每个位置的方向（升/降/平，3分类）
2. 根据方向预测 + 触发规则的投票，排除候选号码

模型只需要做方向预测（3分类），样本量 = 期数 × 位置数，足够。

### 实验 3A：统计基线

```python
def experiment_3a(train_data, test_data):
    """纯统计方法：对每个规则簇，统计其在训练期的方向预测准确率"""
    # 对每个规则簇 × 每个位置 × 每个方向
    # 统计：当该簇触发时，该位置实际方向的分布
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        triggered_mask = train_X[:, cluster_feature_idx[cluster_id]] == 1
        if triggered_mask.sum() < 5:
            continue
        for pos in range(red_count):
            dir_counts = np.bincount(train_Y[triggered_mask, pos], minlength=3)
            cluster_stats[(cluster_id, pos)] = dir_counts / dir_counts.sum()

    # 预测：对测试期每一期，汇总所有触发簇的方向概率
    predictions = []
    for i in range(len(test_X)):
        for pos in range(red_count):
            vote = np.zeros(3)
            n_vote = 0
            for cluster_id in range(n_clusters):
                if test_X[i, cluster_feature_idx[cluster_id]] == 1:
                    if (cluster_id, pos) in cluster_stats:
                        vote += cluster_stats[(cluster_id, pos)]
                        n_vote += 1
            if n_vote > 0:
                vote /= n_vote
            else:
                vote = np.array([1/3, 1/3, 1/3])  # 无规则触发，均匀分布
            predictions.append(vote)

    return evaluate(predictions, test_Y)
```

### 实验 3B：XGBoost

```python
import xgboost as xgb

def experiment_3b(train_X, train_Y, test_X, test_Y):
    """XGBoost 多分类"""
    results = {}
    for pos in range(red_count):
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            train_X, train_Y[:, pos],
            eval_set=[(test_X, test_Y[:, pos])],
            verbose=False,
        )
        probs = model.predict_proba(test_X)
        results[f'P{pos}'] = {
            'model': model,
            'probs': probs,
            'feature_importance': dict(zip(feature_names,
                                           model.feature_importances_)),
        }
    return results
```

### 实验 3C：轻量 MLP（仅在 3B 不够好时执行）

```python
import torch
import torch.nn as nn

class RuleAdapterMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)
```

### 评估函数（统一）

```python
def evaluate(probs, true_labels, data, test_indices):
    """
    probs: (n_test, n_pos, 3) 方向概率
    true_labels: (n_test, n_pos) 实际方向 0/1/2

    评估流程：
    1. 根据方向概率确定候选区间
    2. 计算缩减率和存活率
    """
    results = {'per_position': {}, 'overall': {}}

    for pos in range(n_pos):
        reductions = []
        survivals = []

        for i in range(len(test_indices)):
            t = test_indices[i]
            current_val = data.red_matrix[t - 1, pos]
            true_val = data.red_matrix[t, pos]

            # 方向概率
            p_up, p_equal, p_down = probs[i, pos, 2], probs[i, pos, 1], probs[i, pos, 0]

            # 全候选空间
            full_candidates = set(range(1, data.red_range + 1))

            # 根据方向概率排除候选
            # 概率 < 0.15 的方向，排除对应候选
            candidates = set(full_candidates)
            if p_up < 0.15:
                candidates -= {v for v in full_candidates if v > current_val}
            if p_down < 0.15:
                candidates -= {v for v in full_candidates if v < current_val}
            if p_equal < 0.15:
                candidates -= {current_val}

            reduction = 1 - len(candidates) / len(full_candidates)
            survival = 1 if true_val in candidates else 0

            reductions.append(reduction)
            survivals.append(survival)

        results['per_position'][f'P{pos}'] = {
            'avg_reduction': np.mean(reductions),
            'avg_survival': np.mean(survivals),
            'efficiency': np.mean(reductions) * np.mean(survivals),
        }

    # 汇总
    all_red = [v['avg_reduction'] for v in results['per_position'].values()]
    all_surv = [v['avg_survival'] for v in results['per_position'].values()]
    results['overall'] = {
        'avg_reduction': np.mean(all_red),
        'avg_survival': np.mean(all_surv),
        'efficiency': np.mean(all_red) * np.mean(all_surv),
    }
    return results
```

### 输出

```
results/experiment/
├── phase3_exp3a_daletou.json     # 3A 结果
├── phase3_exp3b_daletou.json     # 3B 结果 + feature importance
├── phase3_exp3c_daletou.json     # 3C 结果（可选）
├── phase3_exp3a_shuangseqiu.json
├── phase3_exp3b_shuangseqiu.json
├── phase3_models/                # 保存的模型文件
│   ├── xgb_daletou_P0.json
│   ├── xgb_daletou_P1.json
│   └── ...
└── phase3_summary.json           # 三个实验的对比汇总
```

---

## 阶段四：结果分析 (phase4_analyze.py)

### 分析内容

1. **方案对比表**：3A vs 3B vs 3C 的缩减率、存活率、综合效率
2. **Feature importance 分析**（来自 3B）：
   - Top 20 最重要特征
   - 规则簇特征 vs 局面特征的重要性占比
   - 哪些规则簇贡献最大
3. **失败案例分析**：
   - 正确号码被误排除的案例
   - 分析误排除时的局面特征和触发规则
4. **规则簇有效性**：
   - 每个簇的触发频率 vs 预测准确率散点图数据
   - 高频高准 / 高频低准 / 低频高准 / 低频低准 四象限分布
5. **结论和建议**：
   - 推荐方案
   - 是否需要重新做严格时间切分
   - v2 Step 3 的实现建议

### 输出

```
results/experiment/
├── phase4_comparison.json        # 方案对比
├── phase4_feature_analysis.json  # 特征重要性分析
├── phase4_failure_analysis.json  # 失败案例
├── phase4_cluster_effectiveness.json  # 规则簇有效性
└── phase4_final_report.json      # 最终报告（含结论和建议）
```

---

## 主入口 (run_experiment.py)

```python
def main():
    setup_logging()
    log("=" * 60)
    log("规则适配学习实验 - 开始")
    log("=" * 60)

    for lottery_type in ['daletou', 'shuangseqiu']:
        log(f"\n{'='*40}")
        log(f"彩种: {lottery_type}")
        log(f"{'='*40}")

        # 阶段一
        log("\n[阶段一] 规则压缩...")
        clusters, a1_filtered = phase1_compress(lottery_type)

        # 阶段二
        log("\n[阶段二] 历史回放 + 数据构造...")
        train_data, test_data, feature_names = phase2_replay(
            lottery_type, clusters, a1_filtered
        )

        # 阶段三
        log("\n[阶段三] 适配模型训练...")

        log("  实验 3A: 统计基线...")
        results_3a = experiment_3a(train_data, test_data)

        log("  实验 3B: XGBoost...")
        results_3b = experiment_3b(train_data, test_data, feature_names)

        # 3C 仅在 3B 效率 < 0.25 时执行
        if results_3b['overall']['efficiency'] < 0.25:
            log("  实验 3C: MLP（3B 效率不足，启动备选）...")
            results_3c = experiment_3c(train_data, test_data)
        else:
            log("  实验 3C: 跳过（3B 效率已达标）")
            results_3c = None

        # 阶段四
        log("\n[阶段四] 结果分析...")
        phase4_analyze(lottery_type, results_3a, results_3b, results_3c)

    log("\n" + "=" * 60)
    log("实验完成！结果保存在 results/experiment/")
    log("=" * 60)
```

---

## 特征维度估算

以大乐透为例（5个位置）：

| 特征类别 | 维度 | 说明 |
|----------|------|------|
| 位置值 lag1-5 | 25 | 5位置 × 5期 |
| 方向 lag1-5 | 25 | 5位置 × 5期 |
| 差分 lag1-5 | 25 | 5位置 × 5期 |
| 组合统计量 lag1-3 | 18 | 6统计量 × 3期 |
| A2 规则簇触发 | ~500-2000 | 取决于聚类结果 |
| A1 模式匹配 | ~100-500 | 筛选后的 A1 模式数 |
| 触发元信息 | 3 | 触发数/平均conf/平均lift |
| **总计** | **~700-2600** | |

样本量：
- 大乐透训练期：~700 期 × 5 位置 = 3500 条
- 大乐透测试期：~430 期 × 5 位置 = 2150 条

特征维度 vs 样本量比例偏高，XGBoost 的正则化能力在这种场景下比较关键。

---

## 风险与应对

| 风险 | 应对 |
|------|------|
| 数据泄露（规则用全量挖掘） | 在报告中标注；如果实验有价值，后续做严格版本 |
| 特征维度 >> 样本量 | XGBoost 正则化 + 特征选择；必要时减少规则簇数 |
| A2 JSON 只有 top 10000 条 | 先用这 10000 条做实验；如果效果好再考虑加载全量 |
| 聚类效果差 | 备选：不聚类，直接用 top 1000 条高 confidence 规则 |
| XGBoost 过拟合 | 早停 + 交叉验证 + 限制树深度 |

---

## 执行命令

```bash
cd /Users/yuanye/Documents/深度搜索/ToBeRich/RichRichRich
python3 -m src.research.experiment.run_experiment 2>&1 | tee experiment.log
```

全自动运行，无需人工干预。预计运行完成后查看：
- `results/experiment/phase4_final_report.json` — 最终结论
- `experiment.log` — 完整运行日志
