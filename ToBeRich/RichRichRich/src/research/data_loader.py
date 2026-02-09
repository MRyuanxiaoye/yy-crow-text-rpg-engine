# -*- coding: utf-8 -*-
"""
统一数据加载与预处理

将原始开奖数据转换为研究模块所需的各种格式：
  - 位置序列
  - 方向序列
  - 差分序列
  - 特征矩阵
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

from research.config import DATA_DIR


class LotteryData:
    """彩票历史数据的统一封装"""

    def __init__(self, lottery_type: str, data_path: str = None):
        self.lottery_type = lottery_type
        self._data_path = data_path
        self.draws: List[Dict] = []
        self.red_count: int = 0
        self.red_range: int = 0
        self.blue_count: int = 0
        self.blue_range: int = 0

        self._load()

        # 预计算常用数据
        self.n_draws = len(self.draws)
        self.position_series = self._build_position_series()
        self.direction_series = self._build_direction_series()
        self.red_matrix = self._build_red_matrix()

        # 蓝球序列
        self.blue_position_series = self._build_blue_position_series()
        self.blue_direction_series = self._build_blue_direction_series()
        self.blue_matrix = self._build_blue_matrix()

    def _load(self):
        """加载原始数据"""
        if self._data_path:
            path = Path(self._data_path)
        else:
            path = DATA_DIR / f"{self.lottery_type}_history.json"
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.draws = data.get("draws", [])
        rules = data.get("rules", {})
        red_rules = rules.get("red_balls", {})
        # 兼容 blue_balls（大乐透）和 blue_ball（双色球）两种字段名
        blue_rules = rules.get("blue_balls", rules.get("blue_ball", {}))
        self.red_count = red_rules.get("count", 5)
        self.red_range = red_rules.get("range", [1, 35])[1]
        self.blue_count = blue_rules.get("count", 2)
        self.blue_range = blue_rules.get("range", [1, 12])[1]

        print(f"[数据] 加载 {self.lottery_type}: {len(self.draws)} 期, "
              f"红球 {self.red_count}/{self.red_range}, "
              f"蓝球 {self.blue_count}/{self.blue_range}")

    def _build_position_series(self) -> Dict[int, np.ndarray]:
        """按位置拆分为时间序列，返回 {pos: array}"""
        series = {pos: [] for pos in range(self.red_count)}
        for draw in self.draws:
            red = draw.get("red_balls", [])
            if len(red) < self.red_count:
                continue
            for pos in range(self.red_count):
                series[pos].append(red[pos])
        return {pos: np.array(vals, dtype=np.int32) for pos, vals in series.items()}

    def _build_direction_series(self) -> Dict[int, np.ndarray]:
        """
        按位置构建方向序列。
        0=E(平), 1=U(升), -1=D(降)
        """
        directions = {}
        for pos, series in self.position_series.items():
            diff = np.diff(series)
            d = np.zeros(len(diff), dtype=np.int8)
            d[diff > 0] = 1
            d[diff < 0] = -1
            directions[pos] = d
        return directions

    def _build_red_matrix(self) -> np.ndarray:
        """
        构建红球矩阵 (n_draws, red_count)
        每行是一期的红球号码
        """
        rows = []
        for draw in self.draws:
            red = draw.get("red_balls", [])
            if len(red) >= self.red_count:
                rows.append(red[:self.red_count])
        return np.array(rows, dtype=np.int32)

    def get_diff_series(self, pos: int, order: int = 1) -> np.ndarray:
        """获取指定位置的N阶差分序列"""
        series = self.position_series[pos].astype(np.float64)
        for _ in range(order):
            series = np.diff(series)
        return series

    def get_cross_diff_series(self, pos_i: int, pos_j: int) -> np.ndarray:
        """获取跨位置差分序列：位置i - 位置j"""
        return self.position_series[pos_i].astype(np.float64) - \
               self.position_series[pos_j].astype(np.float64)

    def get_combo_stats_series(self) -> Dict[str, np.ndarray]:
        """获取每期的组合统计量序列"""
        sums = []
        spans = []
        odd_counts = []
        big_counts = []
        ac_values = []
        consec_groups = []
        mid = (1 + self.red_range) / 2

        for row in self.red_matrix:
            sums.append(int(np.sum(row)))
            spans.append(int(np.max(row) - np.min(row)))
            odd_counts.append(int(np.sum(row % 2 == 1)))
            big_counts.append(int(np.sum(row > mid)))

            # AC值
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(abs(int(row[i]) - int(row[j])))
            ac_values.append(len(diffs) - (len(row) - 1))

            # 连号组数
            sorted_r = sorted(row)
            groups = 0
            in_group = False
            for k in range(1, len(sorted_r)):
                if sorted_r[k] - sorted_r[k - 1] == 1:
                    if not in_group:
                        groups += 1
                        in_group = True
                else:
                    in_group = False
            consec_groups.append(groups)

        return {
            "sum": np.array(sums, dtype=np.int32),
            "span": np.array(spans, dtype=np.int32),
            "odd_count": np.array(odd_counts, dtype=np.int32),
            "big_count": np.array(big_counts, dtype=np.int32),
            "ac_value": np.array(ac_values, dtype=np.int32),
            "consec_groups": np.array(consec_groups, dtype=np.int32),
        }

    def _build_blue_position_series(self) -> Dict[int, np.ndarray]:
        """按位置拆分蓝球时间序列，返回 {pos: array}"""
        series = {pos: [] for pos in range(self.blue_count)}
        for draw in self.draws:
            # 兼容 blue_balls（数组）和 blue_ball（单数字）
            blue = draw.get("blue_balls", None)
            if blue is None:
                bb = draw.get("blue_ball", None)
                if bb is not None:
                    blue = [bb]
                else:
                    continue
            if len(blue) < self.blue_count:
                continue
            for pos in range(self.blue_count):
                series[pos].append(blue[pos])
        return {pos: np.array(vals, dtype=np.int32) for pos, vals in series.items()}

    def _build_blue_direction_series(self) -> Dict[int, np.ndarray]:
        """按位置构建蓝球方向序列"""
        directions = {}
        for pos, series in self.blue_position_series.items():
            diff = np.diff(series)
            d = np.zeros(len(diff), dtype=np.int8)
            d[diff > 0] = 1
            d[diff < 0] = -1
            directions[pos] = d
        return directions

    def _build_blue_matrix(self) -> np.ndarray:
        """构建蓝球矩阵 (n_draws, blue_count)"""
        rows = []
        for draw in self.draws:
            blue = draw.get("blue_balls", None)
            if blue is None:
                bb = draw.get("blue_ball", None)
                if bb is not None:
                    blue = [bb]
                else:
                    continue
            if len(blue) >= self.blue_count:
                rows.append(blue[:self.blue_count])
        return np.array(rows, dtype=np.int32)

    def get_blue_diff_series(self, pos: int, order: int = 1) -> np.ndarray:
        """获取蓝球指定位置的N阶差分序列"""
        series = self.blue_position_series[pos].astype(np.float64)
        for _ in range(order):
            series = np.diff(series)
        return series

    def get_blue_combo_stats_series(self) -> Dict[str, np.ndarray]:
        """获取蓝球每期的组合统计量序列（仅大乐透后区有意义，双色球返回空dict）"""
        if self.blue_count < 2:
            return {}

        sums = []
        spans = []
        odd_counts = []
        mid = (1 + self.blue_range) / 2

        for row in self.blue_matrix:
            sums.append(int(np.sum(row)))
            spans.append(int(np.max(row) - np.min(row)))
            odd_counts.append(int(np.sum(row % 2 == 1)))

        return {
            "blue_sum": np.array(sums, dtype=np.int32),
            "blue_span": np.array(spans, dtype=np.int32),
            "blue_odd_count": np.array(odd_counts, dtype=np.int32),
        }

    def direction_to_str(self, d: int) -> str:
        """方向数值转字符"""
        if d == 1:
            return "U"
        elif d == -1:
            return "D"
        return "E"
