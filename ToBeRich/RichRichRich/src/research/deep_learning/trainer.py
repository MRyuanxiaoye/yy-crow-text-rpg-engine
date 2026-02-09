# -*- coding: utf-8 -*-
"""
深度学习训练器

统一的训练/评估/保存流程，支持 MPS 加速。
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path

from research.config import ResearchConfig, MODELS_DIR, RULES_DIR
from research.data_loader import LotteryData
from research.deep_learning.dataset import LotterySequenceDataset
from research.deep_learning.transformer_model import LotteryTransformer


def get_device() -> torch.device:
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        print("[设备] 使用 Apple MPS 加速")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[设备] 使用 CUDA GPU")
        return torch.device("cuda")
    else:
        print("[设备] 使用 CPU")
        return torch.device("cpu")


class TransformerTrainer:
    """Transformer 模型训练器"""

    def __init__(self, data: LotteryData, config: ResearchConfig):
        self.data = data
        self.config = config
        self.device = get_device()
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.results: Dict[str, Any] = {}

    def _build_datasets(self):
        """构建训练集和验证集"""
        n = self.data.n_draws
        split = int(n * self.config.backtest_train_ratio)
        split = max(split, self.config.backtest_min_train)

        train_ds = LotterySequenceDataset(
            self.data, seq_len=self.config.transformer_seq_len,
            start_idx=0, end_idx=split,
        )
        val_ds = LotterySequenceDataset(
            self.data, seq_len=self.config.transformer_seq_len,
            start_idx=split, end_idx=n,
        )
        print(f"[B1] 训练集: {len(train_ds)} 样本, 验证集: {len(val_ds)} 样本, "
              f"特征维度: {train_ds.feature_dim}")
        return train_ds, val_ds

    def run(self) -> Dict[str, Any]:
        """训练 Transformer 模型"""
        start = time.time()

        train_ds, val_ds = self._build_datasets()
        if len(train_ds) == 0:
            print("[B1] 训练集为空，跳过")
            return {}

        feature_dim = train_ds.feature_dim
        n_positions = self.data.red_count

        # 构建模型
        self.model = LotteryTransformer(
            feature_dim=feature_dim,
            n_positions=n_positions,
            d_model=self.config.transformer_d_model,
            n_heads=self.config.transformer_n_heads,
            n_layers=self.config.transformer_n_layers,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[B1] 模型参数量: {total_params:,}")

        # 数据加载器
        train_loader = DataLoader(
            train_ds, batch_size=self.config.transformer_batch_size,
            shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.transformer_batch_size,
            shuffle=False,
        )

        # 优化器和损失
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.transformer_lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.transformer_epochs,
        )
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_val_acc = 0
        best_epoch = 0
        patience = getattr(self.config, 'transformer_patience', 100)
        no_improve = 0

        for epoch in range(1, self.config.transformer_epochs + 1):
            # 训练
            self.model.train()
            train_loss = 0
            n_batches = 0
            for X, Y in train_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X)  # (batch, n_pos, 3)

                # 计算各位置的损失之和
                loss = 0
                for pos in range(n_positions):
                    loss += criterion(logits[:, pos, :], Y[:, pos])
                loss /= n_positions

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = train_loss / max(n_batches, 1)
            self.train_losses.append(avg_train_loss)

            # 验证
            val_loss, val_acc, per_pos_acc = self._evaluate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve = 0
                self._save_model("best")
            else:
                no_improve += 1

            # 日志
            if epoch % 50 == 0 or epoch == 1:
                elapsed = time.time() - start
                print(f"[B1] Epoch {epoch}/{self.config.transformer_epochs}: "
                      f"train_loss={avg_train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"val_acc={val_acc:.4f}, "
                      f"best={best_val_acc:.4f}@{best_epoch}, "
                      f"耗时 {elapsed:.1f}s")

            if no_improve >= patience:
                print(f"[B1] 早停: {patience} 轮无改善, 最佳 epoch={best_epoch}")
                break

        # 最终评估
        self._load_model("best")
        final_loss, final_acc, per_pos_acc = self._evaluate(val_loader, criterion)

        # 提取注意力权重分析
        attention_analysis = self._analyze_attention(val_ds)

        elapsed = time.time() - start
        self.results = {
            "total_epochs": epoch,
            "best_epoch": best_epoch,
            "best_val_acc": round(best_val_acc, 4),
            "final_val_acc": round(final_acc, 4),
            "final_val_loss": round(final_loss, 4),
            "per_position_accuracy": {
                f"P{i}": round(acc, 4) for i, acc in enumerate(per_pos_acc)
            },
            "total_params": total_params,
            "attention_analysis": attention_analysis,
            "training_time": round(elapsed, 1),
        }

        print(f"[B1] 训练完成: 最佳验证准确率 {best_val_acc:.4f}, "
              f"各位置: {per_pos_acc}, 耗时 {elapsed:.1f}s")

        return self.results

    def _evaluate(self, loader, criterion) -> tuple:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        all_preds = []
        all_targets = []
        n_positions = self.data.red_count

        with torch.no_grad():
            for X, Y in loader:
                X = X.to(self.device)
                Y = Y.to(self.device)

                logits = self.model(X)
                loss = 0
                for pos in range(n_positions):
                    loss += criterion(logits[:, pos, :], Y[:, pos])
                loss /= n_positions

                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=-1)  # (batch, n_pos)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(Y.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)

        if all_preds:
            preds = np.concatenate(all_preds, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            overall_acc = float(np.mean(preds == targets))
            per_pos_acc = [
                round(float(np.mean(preds[:, i] == targets[:, i])), 4)
                for i in range(n_positions)
            ]
        else:
            overall_acc = 0
            per_pos_acc = [0] * n_positions

        return avg_loss, overall_acc, per_pos_acc

    def _analyze_attention(self, val_ds) -> Dict[str, Any]:
        """分析注意力权重"""
        if self.model is None or len(val_ds) == 0:
            return {}

        # 取最后几个样本分析
        n_samples = min(10, len(val_ds))
        X = torch.from_numpy(val_ds.features[-n_samples:]).to(self.device)

        attn_maps = self.model.get_attention_weights(X)
        if not attn_maps:
            return {"status": "无法提取注意力"}

        # 平均注意力：哪些时间步被关注最多
        analysis = {}
        for layer_idx, attn in enumerate(attn_maps):
            # attn: (n_samples, seq_len, seq_len)
            # 最后一个时间步对其他步的注意力
            last_attn = attn[:, -1, :]  # (n_samples, seq_len)
            avg_attn = np.mean(last_attn, axis=0)  # (seq_len,)

            # 找最受关注的时间步
            top_indices = np.argsort(avg_attn)[::-1][:5]
            analysis[f"layer_{layer_idx}"] = {
                "top_attended_steps": [
                    {"step": int(idx), "weight": round(float(avg_attn[idx]), 4)}
                    for idx in top_indices
                ],
                "recent_bias": round(float(np.mean(avg_attn[-5:])), 4),
                "distant_weight": round(float(np.mean(avg_attn[:5])), 4),
            }

        return analysis

    def _save_model(self, tag: str):
        """保存模型"""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"transformer_{self.data.lottery_type}_{tag}.pt"
        torch.save(self.model.state_dict(), path)

    def _load_model(self, tag: str):
        """加载模型"""
        path = MODELS_DIR / f"transformer_{self.data.lottery_type}_{tag}.pt"
        if path.exists():
            self.model.load_state_dict(torch.load(path, weights_only=True))

    def save(self, filename: str = "b1_transformer_results.json"):
        """保存结果"""
        RULES_DIR.mkdir(parents=True, exist_ok=True)
        path = RULES_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "lottery_type": self.data.lottery_type,
                "results": self.results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[B1] 结果已保存: {path}")

    def summary(self) -> Dict[str, Any]:
        """生成摘要"""
        return self.results
