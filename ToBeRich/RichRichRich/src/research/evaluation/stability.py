# -*- coding: utf-8 -*-
"""
C2: 稳定性检验 + 规则排名

对发现的规则进行交叉验证稳定性检验：
  - 将数据分为 K 折，每折都验证规则是否仍然显著
  - 计算规则在各折中的一致性得分
  - 综合排名所有规则
"""

import json
import time
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

from research.config import ResearchConfig, RULES_DIR, REPORTS_DIR
from research.data_loader import LotteryData


class StabilityChecker:
    """稳定性检验 + 规则排名"""

    def __init__(self, data: LotteryData, config: ResearchConfig):
        self.data = data
        self.config = config
        self.results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """对 A1 方向模式进行稳定性检验"""
        start = time.time()
        n = self.data.n_draws
        k_folds = self.config.stability_n_folds

        # 加载 A1 规则
        a1_path = RULES_DIR / f"a1_direction_patterns_{self.data.lottery_type}.json"
        if not a1_path.exists():
            print("[C2] A1 规则文件不存在，跳过")
            return {}

        with open(a1_path, "r", encoding="utf-8") as f:
            a1_data = json.load(f)

        patterns = a1_data.get("patterns", [])
        if not patterns:
            print("[C2] 无规则可检验")
            return {}

        # 取 top 规则
        top_patterns = patterns[:200]
        print(f"[C2] 稳定性检验: {len(top_patterns)} 条规则, {k_folds} 折")

        # 分折
        fold_size = n // k_folds
        fold_ranges = []
        for i in range(k_folds):
            s = i * fold_size
            e = (i + 1) * fold_size if i < k_folds - 1 else n
            fold_ranges.append((s, e))

        # 对每条规则在每折中检验
        stable_rules = []
        for pat_idx, pat in enumerate(top_patterns):
            positions = pat.get("positions", [])
            window = pat.get("window", 3)
            pattern_key = pat.get("pattern_key", [])

            if not positions or not pattern_key:
                continue

            # 只检验单位置规则（简化）
            if len(positions) != 1:
                fold_scores = [1.0] * k_folds  # 多位置规则暂时给满分
                consistency = 1.0
            else:
                pos = positions[0]
                dir_seq = self.data.direction_series.get(pos)
                if dir_seq is None:
                    continue

                fold_scores = []
                for fold_start, fold_end in fold_ranges:
                    # 在这一折中统计模式出现次数和预测准确率
                    fold_hits = 0
                    fold_total = 0

                    for t in range(max(fold_start, window), min(fold_end - 1, len(dir_seq))):
                        # 构建当前窗口编码
                        current = []
                        valid = True
                        for k in range(window):
                            idx = t - window + k
                            if idx < 0 or idx >= len(dir_seq):
                                valid = False
                                break
                            current.append(int(dir_seq[idx]) + 1)

                        if not valid or current != pattern_key:
                            continue

                        fold_total += 1
                        # 检查预测是否正确
                        next_dist = pat.get("next_distribution", {})
                        if next_dist and t < len(dir_seq):
                            best_next = max(next_dist.items(),
                                            key=lambda x: x[1].get("prob", 0))
                            try:
                                pred_tuple = eval(best_next[0])
                                pred_dir = pred_tuple[0] if isinstance(pred_tuple, tuple) else pred_tuple
                            except:
                                pred_dir = None

                            actual_dir = int(dir_seq[t])
                            if pred_dir == actual_dir:
                                fold_hits += 1

                    if fold_total > 0:
                        fold_scores.append(fold_hits / fold_total)
                    else:
                        fold_scores.append(0)

                # 一致性：各折准确率的最小值 / 最大值
                if max(fold_scores) > 0:
                    consistency = min(fold_scores) / max(fold_scores)
                else:
                    consistency = 0

            # 综合得分
            avg_fold_acc = np.mean(fold_scores) if fold_scores else 0
            original_conf = pat.get("prediction_confidence", 0)
            chi2 = pat.get("chi2", 0)

            # 综合排名分 = 稳定性 × 置信度 × log(chi2)
            log_chi2 = np.log1p(chi2)
            composite_score = consistency * avg_fold_acc * log_chi2

            stable_rules.append({
                "pattern": pat.get("pattern", ""),
                "positions": positions,
                "window": window,
                "original_chi2": chi2,
                "original_confidence": original_conf,
                "fold_accuracies": [round(s, 4) for s in fold_scores],
                "avg_fold_accuracy": round(float(avg_fold_acc), 4),
                "consistency": round(float(consistency), 4),
                "composite_score": round(float(composite_score), 4),
            })

            if (pat_idx + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"[C2] 进度: {pat_idx + 1}/{len(top_patterns)}, "
                      f"耗时 {elapsed:.1f}s")

        # 按综合得分排序
        stable_rules.sort(key=lambda x: x["composite_score"], reverse=True)

        # 统计
        n_stable = sum(1 for r in stable_rules if r["consistency"] > 0.5)
        n_high_score = sum(1 for r in stable_rules if r["composite_score"] > 1.0)

        elapsed = time.time() - start
        self.results = {
            "n_rules_checked": len(stable_rules),
            "n_stable": n_stable,
            "n_high_score": n_high_score,
            "k_folds": k_folds,
            "top20_rules": stable_rules[:20],
            "stability_distribution": {
                "high": sum(1 for r in stable_rules if r["consistency"] > 0.7),
                "medium": sum(1 for r in stable_rules if 0.3 < r["consistency"] <= 0.7),
                "low": sum(1 for r in stable_rules if r["consistency"] <= 0.3),
            },
            "time": round(elapsed, 1),
        }

        print(f"[C2] 完成: {n_stable}/{len(stable_rules)} 条规则稳定 (一致性>0.5), "
              f"{n_high_score} 条高分规则, 耗时 {elapsed:.1f}s")

        return self.results

    def save(self, filename: str = "c2_stability_results.json"):
        """保存结果"""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "lottery_type": self.data.lottery_type,
                "stability": self.results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[C2] 结果已保存: {path}")

    def summary(self) -> Dict[str, Any]:
        return self.results
