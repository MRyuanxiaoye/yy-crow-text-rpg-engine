# -*- coding: utf-8 -*-
"""
彩种配置模块

定义大乐透和双色球的参数配置，供方法链各阶段共享使用。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class LotteryConfig:
    """彩种配置"""
    name: str                    # 彩种名称
    key: str                     # 彩种标识
    red_range: int               # 红球/前区号码范围上限
    red_count: int               # 每期选取红球个数
    blue_range: int              # 蓝球/后区号码范围上限
    blue_count: int              # 每期选取蓝球个数

    # 排除引擎参数
    exclusion_threshold: float = 0.6       # 排除置信度阈值
    exclusion_target_ratio: float = 0.35   # 目标排除比例（30%-50%）

    # 排除算法权重
    exclusion_weights: Dict[str, float] = field(default_factory=dict)

    # 权重引擎参数
    weight_algo_weights: Dict[str, float] = field(default_factory=dict)

    # 组合生成参数
    combo_candidate_count: int = 100       # 候选组合数量
    combo_raw_count: int = 5000            # 原始采样数量

    # 购买优化参数
    budget: int = 20                       # 默认预算（元）
    ticket_price: int = 2                  # 单注价格

    @property
    def red_prob(self) -> float:
        """红球单个号码的理论出现概率"""
        return self.red_count / self.red_range

    @property
    def blue_prob(self) -> float:
        """蓝球单个号码的理论出现概率"""
        return self.blue_count / self.blue_range

    @property
    def red_numbers(self) -> List[int]:
        """红球号码列表"""
        return list(range(1, self.red_range + 1))

    @property
    def blue_numbers(self) -> List[int]:
        """蓝球号码列表"""
        return list(range(1, self.blue_range + 1))

    @property
    def red_midpoint(self) -> float:
        """红球大小分界点"""
        return (1 + self.red_range) / 2


# ============================================================
# 预定义配置
# ============================================================

DALETOU = LotteryConfig(
    name="大乐透",
    key="daletou",
    red_range=35,
    red_count=5,
    blue_range=12,
    blue_count=2,
    exclusion_weights={
        "consecutive": 0.12,
        "missing_value": 0.18,
        "extreme_combo": 0.20,
        "markov": 0.13,
        "periodicity": 0.10,
        "cluster": 0.12,
        "position_pattern": 0.15,
    },
    weight_algo_weights={
        "frequency_regression": 0.17,
        "missing_regression": 0.17,
        "time_decay": 0.13,
        "markov_transition": 0.13,
        "co_occurrence": 0.12,
        "deep_learning": 0.13,
        "position_pattern": 0.15,
    },
)

SHUANGSEQIU = LotteryConfig(
    name="双色球",
    key="shuangseqiu",
    red_range=33,
    red_count=6,
    blue_range=16,
    blue_count=1,
    exclusion_weights={
        "consecutive": 0.12,
        "missing_value": 0.18,
        "extreme_combo": 0.20,
        "markov": 0.13,
        "periodicity": 0.10,
        "cluster": 0.12,
        "position_pattern": 0.15,
    },
    weight_algo_weights={
        "frequency_regression": 0.17,
        "missing_regression": 0.17,
        "time_decay": 0.13,
        "markov_transition": 0.13,
        "co_occurrence": 0.12,
        "deep_learning": 0.13,
        "position_pattern": 0.15,
    },
)

# 配置字典，方便按 key 查找
CONFIGS: Dict[str, LotteryConfig] = {
    "daletou": DALETOU,
    "shuangseqiu": SHUANGSEQIU,
}


def get_config(lottery_type: str) -> LotteryConfig:
    """根据彩种标识获取配置"""
    if lottery_type not in CONFIGS:
        raise ValueError(f"未知彩种: {lottery_type}，可选: {list(CONFIGS.keys())}")
    return CONFIGS[lottery_type]
