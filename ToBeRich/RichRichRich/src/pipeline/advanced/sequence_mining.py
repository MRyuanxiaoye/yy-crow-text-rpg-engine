# -*- coding: utf-8 -*-
"""
方向2：序列模式挖掘（PrefixSpan）

核心思路：
  将每期开奖编码为多维事件（位置×属性），
  使用 PrefixSpan 算法自动发现频繁子序列，
  计算关联规则的置信度和提升度，
  检查最近几期是否触发某些规则的前件并输出预测。

输出：
  - frequent_patterns: 频繁子序列列表
  - cross_position_rules: 跨位置关联规则
  - temporal_rules: 跨时间关联规则
  - active_rules: 当前期触发的规则及预测
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter, defaultdict


class SequencePatternMiner:
    """序列模式挖掘器"""

    def __init__(self, draws: List[Dict], red_count: int, red_range: int):
        self.draws = draws
        self.red_count = red_count
        self.red_range = red_range
        self.mid = (1 + red_range) / 2

        # 编码后的事件序列
        self.event_sequences: List[List[str]] = []
        self._encode_events()

    # ============================================================
    # 1. 事件编码
    # ============================================================

    def _encode_events(self):
        """
        每期编码为多维事件列表。
        每个位置生成属性标签：
          - 大小: B(大)/S(小)
          - 奇偶: O(奇)/E(偶)
          - 方向: U(升)/D(降)/F(平)（与上期同位置比较）
          - 区间: Z1/Z2/Z3（三等分区间）
        格式: "P{pos}_{attr}{val}"，如 "P0_BU" 表示位置0大号且上升
        """
        prev_red = None
        for draw in self.draws:
            red = draw.get("red_balls", [])
            if len(red) < self.red_count:
                prev_red = red
                continue

            events = []
            for pos in range(self.red_count):
                val = red[pos]
                # 大小
                size = "B" if val > self.mid else "S"
                # 奇偶
                parity = "O" if val % 2 == 1 else "E"
                # 区间（三等分）
                zone_size = self.red_range / 3
                zone = min(int((val - 1) / zone_size), 2)
                zone_label = f"Z{zone}"
                # 方向
                if prev_red and len(prev_red) > pos:
                    diff = val - prev_red[pos]
                    direction = "U" if diff > 0 else ("D" if diff < 0 else "F")
                else:
                    direction = "F"

                # 组合编码：位置+大小+方向
                events.append(f"P{pos}_{size}{direction}")
                # 位置+奇偶
                events.append(f"P{pos}_{parity}")
                # 位置+区间
                events.append(f"P{pos}_{zone_label}")

            self.event_sequences.append(events)
            prev_red = red

    # ============================================================
    # 2. PrefixSpan 实现（简化版，支持 gap 约束）
    # ============================================================

    def prefixspan(
        self,
        min_support: float = 0.02,
        max_len: int = 4,
        max_gap: int = 3,
    ) -> List[Tuple[List[str], int]]:
        """
        简化版 PrefixSpan：发现频繁子序列。

        参数:
            min_support: 最小支持度（占总序列数的比例）
            max_len: 最大模式长度
            max_gap: 最大间隔期数

        返回:
            [(pattern, support_count), ...]
        """
        n = len(self.event_sequences)
        min_count = max(int(n * min_support), 3)

        # 收集所有单事件的频率
        item_counts = Counter()
        for seq in self.event_sequences:
            for item in set(seq):  # 每个序列中去重
                item_counts[item] += 1

        # 频繁单项
        freq_items = {
            item for item, cnt in item_counts.items() if cnt >= min_count
        }

        results = []

        # 递归挖掘
        def _mine(prefix: List[str], projected_db: List[Tuple[int, int]]):
            """
            prefix: 当前前缀模式
            projected_db: [(seq_idx, start_pos), ...] 投影数据库
            """
            if len(prefix) >= max_len:
                return

            # 统计投影数据库中下一个可能的事件
            next_counts: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

            for seq_idx, start_pos in projected_db:
                if seq_idx >= n:
                    continue
                # 在后续 max_gap 个时间步内搜索
                seen_items: Set[str] = set()
                for gap in range(max_gap + 1):
                    t = start_pos + gap
                    if t >= n:
                        break
                    for item in self.event_sequences[t]:
                        if item in freq_items and item not in seen_items:
                            seen_items.add(item)
                            next_counts[item].append((seq_idx, t + 1))

            for item, positions in next_counts.items():
                # 去重：同一序列只算一次
                unique_seqs = set(p[0] for p in positions)
                support = len(unique_seqs)
                if support >= min_count:
                    new_prefix = prefix + [item]
                    results.append((new_prefix, support))
                    # 递归：每个序列只保留第一个匹配位置
                    deduped = {}
                    for seq_idx, pos in positions:
                        if seq_idx not in deduped:
                            deduped[seq_idx] = pos
                    new_db = list(deduped.items())
                    _mine(new_prefix, new_db)

        # 从每个频繁单项开始
        for item in sorted(freq_items):
            positions = []
            for i, seq in enumerate(self.event_sequences):
                if item in seq:
                    positions.append((i, i + 1))
            unique_seqs = set(p[0] for p in positions)
            support = len(unique_seqs)
            if support >= min_count:
                results.append(([item], support))
                deduped = {}
                for seq_idx, pos in positions:
                    if seq_idx not in deduped:
                        deduped[seq_idx] = pos
                _mine([item], list(deduped.items()))

        return results

    # ============================================================
    # 3. 关联规则：置信度和提升度
    # ============================================================

    def compute_rules(
        self,
        frequent_patterns: List[Tuple[List[str], int]],
        min_confidence: float = 0.5,
        min_lift: float = 1.2,
    ) -> List[Dict[str, Any]]:
        """
        从频繁模式中提取关联规则 A -> B。
        对长度>=2的模式，前缀为前件A，最后一个事件为后件B。
        """
        n = len(self.event_sequences)
        if n == 0:
            return []

        # 建立模式支持度索引
        pattern_support = {}
        for pattern, support in frequent_patterns:
            pattern_support[tuple(pattern)] = support

        rules = []
        for pattern, support in frequent_patterns:
            if len(pattern) < 2:
                continue

            # 前件 = pattern[:-1]，后件 = pattern[-1]
            antecedent = tuple(pattern[:-1])
            consequent = pattern[-1]

            ant_support = pattern_support.get(antecedent, 0)
            if ant_support == 0:
                continue

            confidence = support / ant_support

            # 后件的基础支持度
            cons_support = pattern_support.get((consequent,), 0)
            cons_prob = cons_support / n if n > 0 else 0

            lift = confidence / cons_prob if cons_prob > 0 else 0

            if confidence >= min_confidence and lift >= min_lift:
                rules.append({
                    "antecedent": list(antecedent),
                    "consequent": consequent,
                    "support": support,
                    "confidence": round(confidence, 3),
                    "lift": round(lift, 3),
                    "ant_support": ant_support,
                })

        # 按提升度降序
        rules.sort(key=lambda r: r["lift"], reverse=True)
        return rules

    # ============================================================
    # 4. 分类规则：跨位置 vs 跨时间
    # ============================================================

    @staticmethod
    def classify_rules(
        rules: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        将规则分为跨位置规则和跨时间规则。
        跨位置：前件和后件涉及不同位置（同一时间步）
        跨时间：前件和后件涉及相同或不同位置（不同时间步）
        """
        cross_position = []
        temporal = []

        for rule in rules:
            ant_positions = set()
            for item in rule["antecedent"]:
                if item.startswith("P"):
                    pos = item.split("_")[0]
                    ant_positions.add(pos)

            cons_pos = ""
            if rule["consequent"].startswith("P"):
                cons_pos = rule["consequent"].split("_")[0]

            # 如果前件和后件位置不同，归为跨位置
            if cons_pos and cons_pos not in ant_positions:
                cross_position.append(rule)
            else:
                temporal.append(rule)

        return cross_position, temporal

    # ============================================================
    # 5. 当前匹配：检查最近几期是否触发规则
    # ============================================================

    def find_active_rules(
        self, rules: List[Dict[str, Any]], lookback: int = 3
    ) -> List[Dict[str, Any]]:
        """
        检查最近 lookback 期的事件是否匹配某些规则的前件。
        返回被触发的规则及其预测。
        """
        if not self.event_sequences:
            return []

        # 收集最近 lookback 期的所有事件
        recent_start = max(0, len(self.event_sequences) - lookback)
        recent_events = set()
        for t in range(recent_start, len(self.event_sequences)):
            for item in self.event_sequences[t]:
                recent_events.add(item)

        active = []
        for rule in rules:
            # 检查前件是否都在最近事件中
            ant_matched = all(
                item in recent_events for item in rule["antecedent"]
            )
            if ant_matched:
                active.append({
                    "rule": rule,
                    "prediction": rule["consequent"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                })

        # 按置信度降序
        active.sort(key=lambda x: x["confidence"], reverse=True)
        return active[:30]  # 最多返回30条

    # ============================================================
    # 6. 从预测事件解码为号码范围
    # ============================================================

    def decode_prediction(self, event: str) -> Dict[str, Any]:
        """
        将事件编码解码为号码范围信息。
        如 "P0_BU" -> 位置0，大号，上升
        """
        parts = event.split("_")
        if len(parts) != 2:
            return {"event": event, "position": -1, "description": event}

        pos_str = parts[0]
        attr = parts[1]

        pos = int(pos_str[1:]) if pos_str.startswith("P") else -1

        description_parts = []
        number_filter = {}

        if pos >= 0:
            description_parts.append(f"位置{pos}")
            number_filter["position"] = pos

        if "B" in attr:
            description_parts.append("大号")
            number_filter["size"] = "big"
        elif "S" in attr:
            description_parts.append("小号")
            number_filter["size"] = "small"

        if "O" in attr:
            description_parts.append("奇数")
            number_filter["parity"] = "odd"
        elif "E" in attr:
            description_parts.append("偶数")
            number_filter["parity"] = "even"

        if "U" in attr:
            description_parts.append("上升")
            number_filter["direction"] = "up"
        elif "D" in attr:
            description_parts.append("下降")
            number_filter["direction"] = "down"
        elif "F" in attr:
            description_parts.append("持平")
            number_filter["direction"] = "flat"

        if "Z0" in attr:
            description_parts.append("低区")
            number_filter["zone"] = 0
        elif "Z1" in attr:
            description_parts.append("中区")
            number_filter["zone"] = 1
        elif "Z2" in attr:
            description_parts.append("高区")
            number_filter["zone"] = 2

        return {
            "event": event,
            "position": pos,
            "description": "→".join(description_parts),
            "filter": number_filter,
        }

    # ============================================================
    # 7. 主入口
    # ============================================================

    def analyze(self) -> Dict[str, Any]:
        """运行完整的序列模式挖掘"""
        if len(self.event_sequences) < 20:
            return {
                "frequent_patterns": [],
                "cross_position_rules": [],
                "temporal_rules": [],
                "active_rules": [],
            }

        # PrefixSpan 挖掘频繁模式
        freq_patterns = self.prefixspan(
            min_support=0.02, max_len=3, max_gap=2
        )

        # 只保留长度>=2的模式用于规则提取
        multi_patterns = [
            (p, s) for p, s in freq_patterns if len(p) >= 2
        ]

        # 提取关联规则
        rules = self.compute_rules(
            freq_patterns, min_confidence=0.4, min_lift=1.1
        )

        # 分类规则
        cross_pos_rules, temp_rules = self.classify_rules(rules)

        # 查找当前触发的规则
        active = self.find_active_rules(rules, lookback=3)

        # 解码活跃规则的预测
        for item in active:
            item["decoded"] = self.decode_prediction(item["prediction"])

        # 格式化频繁模式输出（只保留 top 50）
        freq_output = []
        freq_patterns.sort(key=lambda x: x[1], reverse=True)
        for pattern, support in freq_patterns[:50]:
            if len(pattern) >= 2:
                freq_output.append({
                    "pattern": pattern,
                    "support": support,
                    "support_ratio": round(
                        support / len(self.event_sequences), 3
                    ),
                })

        return {
            "frequent_patterns": freq_output,
            "cross_position_rules": cross_pos_rules[:20],
            "temporal_rules": temp_rules[:20],
            "active_rules": active,
        }
