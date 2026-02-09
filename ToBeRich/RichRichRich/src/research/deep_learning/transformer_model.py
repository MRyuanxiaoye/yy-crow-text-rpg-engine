# -*- coding: utf-8 -*-
"""
B1: Transformer 序列预测模型

用多头自注意力捕捉历史序列中的长程依赖，
预测下一期各位置的方向（D/E/U 三分类）。
训练完成后提取注意力权重，分析模型关注的时间步和特征。
"""

import numpy as np
import torch
import torch.nn as nn
import math
from typing import Dict, List, Any, Optional


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class LotteryTransformer(nn.Module):
    """
    彩票序列 Transformer。

    输入: (batch, seq_len, feature_dim)
    输出: (batch, n_positions, 3)  每个位置的方向概率
    """

    def __init__(
        self,
        feature_dim: int,
        n_positions: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出头：每个位置一个3分类
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 3),
            )
            for _ in range(n_positions)
        ])

        # 存储注意力权重（推理时使用）
        self._attention_weights = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, feature_dim)
        返回: (batch, n_positions, 3)
        """
        # 投影 + 位置编码
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.input_norm(h)

        # Transformer 编码
        h = self.transformer(h)  # (batch, seq_len, d_model)

        # 取最后一个时间步的表示
        last = h[:, -1, :]  # (batch, d_model)

        # 各位置分类
        outputs = []
        for head in self.output_heads:
            outputs.append(head(last))  # (batch, 3)

        return torch.stack(outputs, dim=1)  # (batch, n_positions, 3)

    def get_attention_weights(self, x: torch.Tensor) -> List[np.ndarray]:
        """提取各层注意力权重（用于分析）"""
        self.eval()
        weights = []

        # 注册 hook 捕获注意力
        hooks = []
        attn_maps = []

        def hook_fn(module, input, output):
            # TransformerEncoderLayer 内部的 self_attn
            if hasattr(module, 'self_attn'):
                # 重新计算注意力
                with torch.no_grad():
                    src = input[0]
                    _, attn = module.self_attn(src, src, src, need_weights=True)
                    attn_maps.append(attn.detach().cpu().numpy())

        for layer in self.transformer.layers:
            hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            h = self.input_proj(x)
            h = self.pos_encoding(h)
            h = self.input_norm(h)
            self.transformer(h)

        for hook in hooks:
            hook.remove()

        return attn_maps
