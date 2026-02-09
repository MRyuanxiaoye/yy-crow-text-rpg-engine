# -*- coding: utf-8 -*-
"""
研究模块全局配置

量大管饱版：大幅扩大搜索空间，预计运行数小时。
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"

# 结果输出目录
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RULES_DIR = RESULTS_DIR / "rules"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "reports"


@dataclass
class ResearchConfig:
    """研究参数配置"""

    # 彩种
    lottery_type: str = "daletou"

    # === 模块A：穷举搜索参数 ===

    # A1: 方向模式扫描 — 大幅扩大窗口
    direction_single_windows: Tuple[int, int] = (3, 30)       # 单位置窗口 3-30
    direction_dual_windows: Tuple[int, int] = (3, 20)          # 双位置窗口 3-20
    direction_triple_windows: Tuple[int, int] = (3, 15)        # 三位置窗口 3-15
    direction_quad_windows: Tuple[int, int] = (3, 10)          # 四位置窗口 3-10
    direction_quint_windows: Tuple[int, int] = (3, 8)          # 五位置窗口 3-8
    direction_min_support: int = 3                              # 最小支持度降低，发现更多稀有模式
    direction_p_threshold: float = 0.01                         # 卡方检验 p 值阈值

    # A2: 条件排除规则挖掘 — 大幅扩大条件空间
    rule_max_conditions: int = 5           # 最大5个条件组合（AND）
    rule_min_support: int = 10             # 降低支持度门槛
    rule_min_lift: float = 1.2             # 降低提升度门槛
    rule_p_threshold: float = 0.01         # Fisher检验 p 值阈值
    rule_value_bins: int = 10              # 值域分箱数翻倍，更精细
    rule_max_rules: int = 0               # 0=不限制规则数量上限

    # A3: 差分分析 — 扩大分析深度
    diff_max_order: int = 5                # 最大5阶差分
    diff_window_range: Tuple[int, int] = (3, 20)   # 差分模式窗口扩大
    diff_acf_max_lag: int = 100            # ACF 最大滞后期扩大

    # A4: 滑动窗口统计量 — 扩大窗口和精度
    sliding_window_range: Tuple[int, int] = (3, 100)  # 窗口扩大到100
    sliding_window_step: int = 1                        # 窗口步长
    sliding_n_levels: int = 5                            # 离散化5级（原来3级）

    # === 模块B：深度学习参数 — 加大模型 ===

    # B1: Transformer
    transformer_seq_len: int = 50          # 输入序列长度加长
    transformer_d_model: int = 256         # 模型维度翻倍
    transformer_n_heads: int = 8           # 注意力头数翻倍
    transformer_n_layers: int = 6          # 层数增加
    transformer_epochs: int = 5000         # 训练轮数大幅增加
    transformer_lr: float = 5e-4           # 学习率降低
    transformer_batch_size: int = 32       # 批大小减小（更多梯度更新）
    transformer_patience: int = 300        # 早停耐心大幅增加

    # B2: 对比学习
    contrastive_embed_dim: int = 256       # 嵌入维度翻倍
    contrastive_epochs: int = 2000
    contrastive_k_neighbors: int = 50      # 检索近邻数增加

    # B3: VAE
    vae_latent_dim: int = 64               # 潜在空间维度翻倍
    vae_epochs: int = 2000

    # B4: GNN
    gnn_hidden_dim: int = 128
    gnn_n_layers: int = 5
    gnn_epochs: int = 2000

    # === 模块C：评估参数 ===
    backtest_train_ratio: float = 0.7      # 训练集比例
    backtest_min_train: int = 500          # 最小训练期数
    stability_n_folds: int = 10            # 稳定性检验折数翻倍
    fdr_alpha: float = 0.05               # FDR 控制水平

    # === 并行参数 ===
    n_workers: int = 7                     # 并行进程数（留1核给系统）
    chunk_size: int = 500                  # 每个任务块大小

    # === 日志 ===
    log_interval: int = 50000              # 每处理N个模式打印一次进度


# 大乐透配置
DALETOU_RESEARCH = ResearchConfig(
    lottery_type="daletou",
)

# 双色球配置
SHUANGSEQIU_RESEARCH = ResearchConfig(
    lottery_type="shuangseqiu",
)

RESEARCH_CONFIGS = {
    "daletou": DALETOU_RESEARCH,
    "shuangseqiu": SHUANGSEQIU_RESEARCH,
}


def get_research_config(lottery_type: str) -> ResearchConfig:
    if lottery_type not in RESEARCH_CONFIGS:
        raise ValueError(f"未知彩种: {lottery_type}")
    return RESEARCH_CONFIGS[lottery_type]
