# -*- coding: utf-8 -*-
"""
方向3：信息论评估

核心思路：
  用信息论工具量化位置序列中的可利用信息量：
  - 条件熵：H(X_t | X_{t-1},...,X_{t-d})，评估历史深度的预测价值
  - 互信息矩阵：5×5位置间互信息，量化位置间信息共享
  - 传递熵：方向性因果关系 TE(i→j)，用 surrogate 检验显著性
  - 可利用信息评估：总结数据中有多少可利用的信息量

输出：
  - conditional_entropy: 各位置各深度的条件熵
  - mutual_information: 位置间互信息矩阵
  - transfer_entropy: 显著因果流
  - exploitable_information: 总体评估
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter


class InformationTheoryAnalyzer:
    """信息论分析器"""

    def __init__(self, draws: List[Dict], red_count: int, red_range: int):
        self.draws = draws
        self.red_count = red_count
        self.red_range = red_range

        # 按位置拆分时间序列
        self.position_series: Dict[int, List[int]] = {}
        self._extract_position_series()

        # 离散化后的序列
        self.discretized: Dict[int, List[int]] = {}
        self.n_bins = 5  # 等频分箱数

    def _extract_position_series(self):
        for pos in range(self.red_count):
            self.position_series[pos] = []
        for draw in self.draws:
            red = draw.get("red_balls", [])
            if len(red) < self.red_count:
                continue
            for pos in range(self.red_count):
                self.position_series[pos].append(red[pos])

    # ============================================================
    # 1. 离散化：等频分箱
    # ============================================================

    def discretize(self, n_bins: int = 5):
        """对每个位置的值序列进行等频分箱离散化"""
        self.n_bins = n_bins
        for pos in range(self.red_count):
            series = self.position_series.get(pos, [])
            if not series:
                self.discretized[pos] = []
                continue

            arr = np.array(series, dtype=float)
            # 计算分位数边界
            percentiles = np.linspace(0, 100, n_bins + 1)
            boundaries = np.percentile(arr, percentiles)

            # 分箱
            binned = np.digitize(arr, boundaries[1:-1])
            self.discretized[pos] = binned.tolist()

    # ============================================================
    # 2. 条件熵：H(X_t | X_{t-1},...,X_{t-d})
    # ============================================================

    def conditional_entropy(
        self, pos: int, max_depth: int = 5
    ) -> Dict[int, float]:
        """
        计算指定位置在不同历史深度下的条件熵。
        深度 d 表示用前 d 期的值来预测当前期。
        条件熵越低，说明历史信息对预测越有价值。
        """
        seq = self.discretized.get(pos, [])
        if len(seq) < max_depth + 10:
            return {}

        results = {}
        # 无条件熵 H(X)
        counter = Counter(seq)
        total = len(seq)
        h0 = -sum(
            (c / total) * np.log2(c / total)
            for c in counter.values() if c > 0
        )
        results[0] = round(h0, 4)

        for d in range(1, max_depth + 1):
            # 构建 (context, target) 对
            joint_counts = Counter()
            context_counts = Counter()

            for t in range(d, len(seq)):
                context = tuple(seq[t - d:t])
                target = seq[t]
                joint_counts[(context, target)] += 1
                context_counts[context] += 1

            # H(X_t | context) = H(X_t, context) - H(context)
            n_samples = len(seq) - d
            if n_samples <= 0:
                break

            h_joint = -sum(
                (c / n_samples) * np.log2(c / n_samples)
                for c in joint_counts.values() if c > 0
            )
            h_context = -sum(
                (c / n_samples) * np.log2(c / n_samples)
                for c in context_counts.values() if c > 0
            )
            h_cond = h_joint - h_context
            results[d] = round(max(h_cond, 0), 4)

        return results

    # ============================================================
    # 3. 互信息矩阵
    # ============================================================

    def mutual_information_matrix(self) -> Dict[str, Any]:
        """
        计算位置间的互信息矩阵 MI(i, j)。
        MI 越高，两个位置的信息共享越多。
        """
        n = self.red_count
        mi_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                mi = self._mutual_information(i, j)
                mi_matrix[i][j] = mi
                mi_matrix[j][i] = mi

        # 找出强关联对
        strong_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if mi_matrix[i][j] > 0.1:
                    strong_pairs.append({
                        "pos_i": i,
                        "pos_j": j,
                        "mi": round(float(mi_matrix[i][j]), 4),
                    })
        strong_pairs.sort(key=lambda x: x["mi"], reverse=True)

        return {
            "matrix": [[round(float(v), 4) for v in row] for row in mi_matrix],
            "strong_pairs": strong_pairs,
        }

    def _mutual_information(self, pos_i: int, pos_j: int) -> float:
        """计算两个位置序列的互信息"""
        seq_i = self.discretized.get(pos_i, [])
        seq_j = self.discretized.get(pos_j, [])

        min_len = min(len(seq_i), len(seq_j))
        if min_len < 10:
            return 0.0

        seq_i = seq_i[:min_len]
        seq_j = seq_j[:min_len]

        # 联合分布和边际分布
        joint = Counter(zip(seq_i, seq_j))
        margin_i = Counter(seq_i)
        margin_j = Counter(seq_j)

        mi = 0.0
        for (xi, xj), n_ij in joint.items():
            p_ij = n_ij / min_len
            p_i = margin_i[xi] / min_len
            p_j = margin_j[xj] / min_len
            if p_ij > 0 and p_i > 0 and p_j > 0:
                mi += p_ij * np.log2(p_ij / (p_i * p_j))

        return max(mi, 0.0)

    # ============================================================
    # 4. 传递熵：方向性因果关系
    # ============================================================

    def transfer_entropy(
        self, depth: int = 1, n_surrogates: int = 50
    ) -> Dict[str, Any]:
        """
        计算传递熵 TE(i→j)：位置 i 对位置 j 的方向性信息流。
        TE(i→j) = H(Y_t | Y_{t-1:t-d}) - H(Y_t | Y_{t-1:t-d}, X_{t-1:t-d})

        用 surrogate 数据（随机打乱 X）检验显著性。
        """
        n = self.red_count
        te_matrix = np.zeros((n, n))
        p_values = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                te, p_val = self._compute_te(i, j, depth, n_surrogates)
                te_matrix[i][j] = te
                p_values[i][j] = p_val

        # 找出显著的因果流（p < 0.05）
        significant_flows = []
        for i in range(n):
            for j in range(n):
                if i != j and p_values[i][j] < 0.05 and te_matrix[i][j] > 0.01:
                    significant_flows.append({
                        "source": i,
                        "target": j,
                        "te": round(float(te_matrix[i][j]), 4),
                        "p_value": round(float(p_values[i][j]), 4),
                    })
        significant_flows.sort(key=lambda x: x["te"], reverse=True)

        return {
            "matrix": [[round(float(v), 4) for v in row] for row in te_matrix],
            "significant_flows": significant_flows,
        }

    def _compute_te(
        self, source: int, target: int, depth: int, n_surrogates: int
    ) -> Tuple[float, float]:
        """计算单对传递熵及 surrogate p 值"""
        x = self.discretized.get(source, [])
        y = self.discretized.get(target, [])
        min_len = min(len(x), len(y))

        if min_len < depth + 20:
            return 0.0, 1.0

        x = x[:min_len]
        y = y[:min_len]

        # 真实 TE
        real_te = self._te_value(x, y, depth)

        # Surrogate 检验：打乱 x 序列
        surrogate_tes = []
        rng = np.random.RandomState(42)
        for _ in range(n_surrogates):
            x_shuffled = list(x)
            rng.shuffle(x_shuffled)
            s_te = self._te_value(x_shuffled, y, depth)
            surrogate_tes.append(s_te)

        # p 值：surrogate 中 >= real_te 的比例
        p_val = sum(1 for s in surrogate_tes if s >= real_te) / n_surrogates

        return real_te, p_val

    @staticmethod
    def _te_value(x: List[int], y: List[int], depth: int) -> float:
        """计算 TE(X→Y) 的值"""
        n = len(y)
        # H(Y_t | Y_past)
        joint_yy = Counter()
        ctx_y = Counter()
        # H(Y_t | Y_past, X_past)
        joint_yyx = Counter()
        ctx_yx = Counter()

        for t in range(depth, n):
            y_past = tuple(y[t - depth:t])
            x_past = tuple(x[t - depth:t])
            y_t = y[t]

            joint_yy[(y_past, y_t)] += 1
            ctx_y[y_past] += 1

            joint_yyx[(y_past, x_past, y_t)] += 1
            ctx_yx[(y_past, x_past)] += 1

        samples = n - depth
        if samples <= 0:
            return 0.0

        # H(Y_t | Y_past)
        h_y_given_ypast = 0.0
        for key, cnt in joint_yy.items():
            p_joint = cnt / samples
            p_ctx = ctx_y[key[0]] / samples
            if p_joint > 0 and p_ctx > 0:
                h_y_given_ypast -= p_joint * np.log2(p_joint / p_ctx)

        # H(Y_t | Y_past, X_past)
        h_y_given_yxpast = 0.0
        for key, cnt in joint_yyx.items():
            p_joint = cnt / samples
            p_ctx = ctx_yx[(key[0], key[1])] / samples
            if p_joint > 0 and p_ctx > 0:
                h_y_given_yxpast -= p_joint * np.log2(p_joint / p_ctx)

        te = h_y_given_ypast - h_y_given_yxpast
        return max(te, 0.0)

    # ============================================================
    # 5. 可利用信息评估
    # ============================================================

    def assess_exploitable_info(
        self, cond_entropies: Dict[int, Dict[int, float]],
        mi_result: Dict[str, Any],
        te_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        综合评估数据中可利用的信息量。
        """
        # 条件熵下降比例：(H(X) - H(X|past)) / H(X)
        entropy_reductions = {}
        for pos, ent_dict in cond_entropies.items():
            h0 = ent_dict.get(0, 0)
            if h0 <= 0:
                continue
            best_depth = 0
            best_reduction = 0.0
            for d, hd in ent_dict.items():
                if d == 0:
                    continue
                reduction = (h0 - hd) / h0
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_depth = d
            entropy_reductions[pos] = {
                "unconditional_entropy": h0,
                "best_depth": best_depth,
                "best_conditional_entropy": ent_dict.get(best_depth, h0),
                "reduction_ratio": round(best_reduction, 4),
            }

        # 平均熵下降
        avg_reduction = np.mean([
            v["reduction_ratio"] for v in entropy_reductions.values()
        ]) if entropy_reductions else 0.0

        # 显著因果流数量
        n_significant = len(te_result.get("significant_flows", []))

        # 强互信息对数量
        n_strong_mi = len(mi_result.get("strong_pairs", []))

        # 综合评级
        score = avg_reduction * 0.4 + min(n_significant / 10, 1.0) * 0.3 + min(n_strong_mi / 5, 1.0) * 0.3

        if score > 0.5:
            level = "高"
            description = "数据中存在较多可利用的结构性信息，位置间存在显著的信息流和关联"
        elif score > 0.25:
            level = "中等"
            description = "数据中存在一定的可利用信息，部分位置的历史对预测有帮助"
        else:
            level = "低"
            description = "数据接近随机，可利用的结构性信息有限"

        return {
            "level": level,
            "score": round(float(score), 4),
            "description": description,
            "entropy_reductions": entropy_reductions,
            "avg_entropy_reduction": round(float(avg_reduction), 4),
            "significant_causal_flows": n_significant,
            "strong_mi_pairs": n_strong_mi,
        }

    # ============================================================
    # 6. 主入口
    # ============================================================

    def analyze(self) -> Dict[str, Any]:
        """运行完整的信息论分析"""
        if not self.position_series or not self.position_series.get(0):
            return {
                "conditional_entropy": {},
                "mutual_information": {"matrix": [], "strong_pairs": []},
                "transfer_entropy": {"matrix": [], "significant_flows": []},
                "exploitable_information": {
                    "level": "低", "score": 0, "description": "数据不足"
                },
            }

        # 离散化
        self.discretize(n_bins=5)

        # 条件熵
        cond_entropies = {}
        for pos in range(self.red_count):
            cond_entropies[pos] = self.conditional_entropy(pos, max_depth=5)

        # 互信息矩阵
        mi_result = self.mutual_information_matrix()

        # 传递熵（用较少的 surrogate 加速）
        te_result = self.transfer_entropy(depth=1, n_surrogates=30)

        # 可利用信息评估
        exploit = self.assess_exploitable_info(
            cond_entropies, mi_result, te_result
        )

        return {
            "conditional_entropy": cond_entropies,
            "mutual_information": mi_result,
            "transfer_entropy": te_result,
            "exploitable_information": exploit,
        }
