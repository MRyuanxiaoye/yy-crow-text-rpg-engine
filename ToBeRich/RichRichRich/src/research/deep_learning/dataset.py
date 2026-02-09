# -*- coding: utf-8 -*-
"""
深度学习数据集

将彩票历史数据转换为 PyTorch Dataset，
支持多种特征编码方式。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

from research.data_loader import LotteryData


class LotterySequenceDataset(Dataset):
    """
    彩票序列数据集。
    每个样本是连续 seq_len 期的特征，目标是下一期的方向。
    """

    def __init__(
        self,
        data: LotteryData,
        seq_len: int = 30,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ):
        self.data = data
        self.seq_len = seq_len
        self.rc = data.red_count

        # 构建特征矩阵
        self.features, self.targets = self._build_features(start_idx, end_idx)

    def _build_features(
        self, start_idx: int, end_idx: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建特征和目标。

        特征（每期）：
          - 各位置归一化值 (rc 维)
          - 各位置方向 one-hot (rc * 3 维)
          - 各位置一阶差分归一化 (rc 维)
          - 组合统计量归一化 (6 维)
        总维度: rc + rc*3 + rc + 6 = rc*5 + 6

        目标：
          - 各位置下一期方向 (rc 维, 值为 0/1/2 对应 D/E/U)
        """
        n = self.data.n_draws
        rc = self.rc

        # 位置值归一化
        pos_vals = np.zeros((n, rc), dtype=np.float32)
        for pos in range(rc):
            series = self.data.position_series[pos].astype(np.float32)
            mean, std = np.mean(series), np.std(series)
            if std > 0:
                pos_vals[:, pos] = (series - mean) / std

        # 方向 one-hot (n, rc*3)，第一期无方向设为 [0,0,0]
        dir_onehot = np.zeros((n, rc * 3), dtype=np.float32)
        for pos in range(rc):
            dir_seq = self.data.direction_series[pos]  # 长度 n-1
            for t in range(len(dir_seq)):
                d = dir_seq[t]  # -1, 0, 1
                dir_onehot[t + 1, pos * 3 + (d + 1)] = 1.0

        # 一阶差分归一化
        diff_vals = np.zeros((n, rc), dtype=np.float32)
        for pos in range(rc):
            diff = self.data.get_diff_series(pos, 1)
            mean, std = np.mean(diff), np.std(diff)
            if std > 0:
                diff_vals[1:1 + len(diff), pos] = (diff - mean) / std

        # 组合统计量归一化
        combo = self.data.get_combo_stats_series()
        stat_names = ["sum", "span", "odd_count", "big_count", "ac_value", "consec_groups"]
        combo_vals = np.zeros((n, len(stat_names)), dtype=np.float32)
        for i, name in enumerate(stat_names):
            s = combo[name].astype(np.float32)
            mean, std = np.mean(s), np.std(s)
            if std > 0:
                combo_vals[:len(s), i] = (s - mean) / std

        # 拼接特征
        features = np.concatenate([pos_vals, dir_onehot, diff_vals, combo_vals], axis=1)

        # 目标：下一期方向 (D=-1→0, E=0→1, U=1→2)
        targets = np.zeros((n, rc), dtype=np.int64)
        for pos in range(rc):
            dir_seq = self.data.direction_series[pos]
            targets[:len(dir_seq), pos] = dir_seq + 1  # 映射到 0,1,2

        # 切片
        if end_idx is None:
            end_idx = n
        end_idx = min(end_idx, n)

        # 构建序列样本
        valid_start = max(start_idx, self.seq_len)
        valid_end = end_idx - 1  # 需要下一期作为目标

        n_samples = valid_end - valid_start
        if n_samples <= 0:
            return np.zeros((0, self.seq_len, features.shape[1]), dtype=np.float32), \
                   np.zeros((0, rc), dtype=np.int64)

        X = np.zeros((n_samples, self.seq_len, features.shape[1]), dtype=np.float32)
        Y = np.zeros((n_samples, rc), dtype=np.int64)

        for i in range(n_samples):
            t = valid_start + i
            X[i] = features[t - self.seq_len:t]
            Y[i] = targets[t]  # 第 t 期的方向（相对第 t-1 期）

        return X, Y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.targets[idx]),
        )

    @property
    def feature_dim(self) -> int:
        return self.features.shape[2] if len(self.features) > 0 else 0

    @property
    def n_classes(self) -> int:
        return 3  # D, E, U
