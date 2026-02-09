"""E0-Step7: 跨位置联合建模

验证多任务学习是否能利用位置间关联（E1.5发现的+52%潜力）。

三种方案对比：
  A) 基线：逐位置独立MLP（现有Phase3方案）
  B) 多任务MLP：共享底层 + 独立输出头
  C) 序列化预测MLP：自回归方式，P_i的预测依赖P_{i-1}的输出

评估指标：
  - 方向准确率（逐位置 + 整体）
  - 组合级准确率（所有位置同时正确的比例）
  - 缩减率、存活率、综合效率
  - 条件准确率（给定相邻位置方向的准确率）

用法: python3 -m src.research.experiment.e0_step7_joint_model
"""

import sys
import numpy as np
from pathlib import Path
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import research.experiment.utils as exp_utils
from research.experiment.utils import setup_logging, log, Timer, save_json
from research.experiment.phase1_compress import run_phase1
from research.experiment.phase2_replay import run_phase2
from research.experiment.phase3_train import evaluate_direction_predictions

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_RULES_DIR = RESULTS_DIR / "rules_strict"
STEP7_DIR = RESULTS_DIR / "e0_step7_joint"


# ============================================================
# 方案A：逐位置独立MLP（基线，复用Phase3逻辑）
# ============================================================

def train_independent_mlp(train_X, train_Y, test_X, test_Y, data, test_indices):
    """逐位置独立MLP基线"""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    log(f"    设备: {device}")

    n_pos = train_Y.shape[1]
    input_dim = train_X.shape[1]
    probs = np.zeros((test_X.shape[0], n_pos, 3))

    n_train = train_X.shape[0]
    n_val = max(50, int(n_train * 0.2))

    tr_X_t = torch.FloatTensor(train_X[:-n_val]).to(device)
    va_X_t = torch.FloatTensor(train_X[-n_val:]).to(device)
    te_X_t = torch.FloatTensor(test_X).to(device)

    for pos in range(n_pos):
        tr_Y_t = torch.LongTensor(train_Y[:-n_val, pos]).to(device)
        va_Y_t = torch.LongTensor(train_Y[-n_val:, pos]).to(device)

        model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        dataset = TensorDataset(tr_X_t, tr_Y_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(500):
            model.train()
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_Y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(va_X_t), va_Y_t).item()
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 50:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pos_probs = torch.softmax(model(te_X_t), dim=1).cpu().numpy()
            probs[:, pos, :] = pos_probs

        log(f"      P{pos}: epochs={epoch+1}, val_loss={best_val_loss:.4f}")

    return probs


# ============================================================
# 方案B：多任务MLP（共享底层 + 独立输出头）
# ============================================================

class MultiTaskMLP(object):
    """多任务MLP：共享特征提取层，每个位置独立输出头"""

    def __init__(self, input_dim, n_positions, hidden_dim=256, shared_dim=128):
        import torch.nn as nn
        import torch

        self.n_positions = n_positions

        # 共享底层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 每个位置的独立输出头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )
            for _ in range(n_positions)
        ])

        # 组合成完整模型
        self.model = nn.Module()
        self.model.shared = self.shared
        self.model.heads = self.heads

        # 手动收集参数
        self._params = list(self.shared.parameters()) + list(self.heads.parameters())

    def to(self, device):
        self.shared = self.shared.to(device)
        self.heads = self.heads.to(device)
        return self

    def parameters(self):
        return self._params

    def train(self):
        self.shared.train()
        self.heads.train()

    def eval(self):
        self.shared.eval()
        self.heads.eval()

    def forward(self, x):
        """返回 (batch, n_positions, 3) 的logits"""
        import torch
        shared_feat = self.shared(x)
        outputs = [head(shared_feat) for head in self.heads]
        return torch.stack(outputs, dim=1)

    def state_dict(self):
        state = {}
        for k, v in self.shared.state_dict().items():
            state[f'shared.{k}'] = v
        for i, head in enumerate(self.heads):
            for k, v in head.state_dict().items():
                state[f'head_{i}.{k}'] = v
        return state

    def load_state_dict(self, state):
        shared_state = {k.replace('shared.', ''): v for k, v in state.items() if k.startswith('shared.')}
        self.shared.load_state_dict(shared_state)
        for i, head in enumerate(self.heads):
            head_state = {k.replace(f'head_{i}.', ''): v for k, v in state.items() if k.startswith(f'head_{i}.')}
            head.load_state_dict(head_state)


def train_multitask_mlp(train_X, train_Y, test_X, test_Y, data, test_indices):
    """多任务MLP训练"""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    log(f"    设备: {device}")

    n_pos = train_Y.shape[1]
    input_dim = train_X.shape[1]
    n_train = train_X.shape[0]
    n_val = max(50, int(n_train * 0.2))

    tr_X = torch.FloatTensor(train_X[:-n_val]).to(device)
    tr_Y = torch.LongTensor(train_Y[:-n_val]).to(device)
    va_X = torch.FloatTensor(train_X[-n_val:]).to(device)
    va_Y = torch.LongTensor(train_Y[-n_val:]).to(device)
    te_X = torch.FloatTensor(test_X).to(device)

    model = MultiTaskMLP(input_dim, n_pos, hidden_dim=256, shared_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    dataset = TensorDataset(tr_X, tr_Y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(500):
        model.train()
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            logits = model.forward(batch_X)  # (batch, n_pos, 3)
            loss = sum(criterion(logits[:, p, :], batch_Y[:, p]) for p in range(n_pos)) / n_pos
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model.forward(va_X)
            val_loss = sum(criterion(val_logits[:, p, :], va_Y[:, p]) for p in range(n_pos)).item() / n_pos
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 50:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model.forward(te_X)
        probs = torch.softmax(test_logits, dim=2).cpu().numpy()

    log(f"    epochs={epoch+1}, val_loss={best_val_loss:.4f}")
    return probs


# ============================================================
# 方案C：序列化预测MLP（自回归，P_i依赖P_{i-1}的输出）
# ============================================================

class AutoregressiveMLP(object):
    """自回归MLP：每个位置的预测依赖前一个位置的预测结果"""

    def __init__(self, input_dim, n_positions, hidden_dim=256):
        import torch.nn as nn

        self.n_positions = n_positions

        # 第一个位置：只用原始特征
        self.head_0 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

        # 后续位置：原始特征 + 前一位置的3维概率
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim + 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3),
            )
            for _ in range(n_positions - 1)
        ])

        self._params = list(self.head_0.parameters()) + list(self.heads.parameters())

    def to(self, device):
        self.head_0 = self.head_0.to(device)
        self.heads = self.heads.to(device)
        return self

    def parameters(self):
        return self._params

    def train(self):
        self.head_0.train()
        self.heads.train()

    def eval(self):
        self.head_0.eval()
        self.heads.eval()

    def forward(self, x, teacher_labels=None):
        """
        Args:
            x: (batch, input_dim)
            teacher_labels: (batch, n_pos) 训练时用teacher forcing

        Returns:
            logits: (batch, n_positions, 3)
        """
        import torch

        all_logits = []

        # P0
        logits_0 = self.head_0(x)
        all_logits.append(logits_0)

        for i, head in enumerate(self.heads):
            if teacher_labels is not None:
                # Teacher forcing：用真实标签的one-hot
                prev_dir = torch.zeros(x.shape[0], 3, device=x.device)
                prev_dir.scatter_(1, teacher_labels[:, i:i+1], 1.0)
            else:
                # 推理：用前一步的softmax概率
                prev_dir = torch.softmax(all_logits[-1].detach(), dim=1)

            combined = torch.cat([x, prev_dir], dim=1)
            logits_i = head(combined)
            all_logits.append(logits_i)

        return torch.stack(all_logits, dim=1)

    def state_dict(self):
        state = {}
        for k, v in self.head_0.state_dict().items():
            state[f'head_0.{k}'] = v
        for i, head in enumerate(self.heads):
            for k, v in head.state_dict().items():
                state[f'head_{i+1}.{k}'] = v
        return state

    def load_state_dict(self, state):
        h0_state = {k.replace('head_0.', ''): v for k, v in state.items() if k.startswith('head_0.')}
        self.head_0.load_state_dict(h0_state)
        for i, head in enumerate(self.heads):
            hi_state = {k.replace(f'head_{i+1}.', ''): v for k, v in state.items() if k.startswith(f'head_{i+1}.')}
            head.load_state_dict(hi_state)


def train_autoregressive_mlp(train_X, train_Y, test_X, test_Y, data, test_indices):
    """自回归MLP训练"""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    log(f"    设备: {device}")

    n_pos = train_Y.shape[1]
    input_dim = train_X.shape[1]
    n_train = train_X.shape[0]
    n_val = max(50, int(n_train * 0.2))

    tr_X = torch.FloatTensor(train_X[:-n_val]).to(device)
    tr_Y = torch.LongTensor(train_Y[:-n_val]).to(device)
    va_X = torch.FloatTensor(train_X[-n_val:]).to(device)
    va_Y = torch.LongTensor(train_Y[-n_val:]).to(device)
    te_X = torch.FloatTensor(test_X).to(device)

    model = AutoregressiveMLP(input_dim, n_pos, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    dataset = TensorDataset(tr_X, tr_Y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(500):
        model.train()
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            # 训练时用 teacher forcing
            logits = model.forward(batch_X, teacher_labels=batch_Y)
            loss = sum(criterion(logits[:, p, :], batch_Y[:, p]) for p in range(n_pos)) / n_pos
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # 验证时不用 teacher forcing
            val_logits = model.forward(va_X, teacher_labels=None)
            val_loss = sum(criterion(val_logits[:, p, :], va_Y[:, p]) for p in range(n_pos)).item() / n_pos
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 50:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model.forward(te_X, teacher_labels=None)
        probs = torch.softmax(test_logits, dim=2).cpu().numpy()

    log(f"    epochs={epoch+1}, val_loss={best_val_loss:.4f}")
    return probs


# ============================================================
# 评估工具
# ============================================================

def compute_combo_accuracy(probs, true_Y):
    """计算组合级准确率（所有位置同时预测正确的比例）"""
    pred = np.argmax(probs, axis=2)
    all_correct = np.all(pred == true_Y, axis=1)
    return float(np.mean(all_correct))


def compute_conditional_accuracy(probs, true_Y):
    """计算条件准确率：给定相邻位置真实方向时的准确率提升"""
    pred = np.argmax(probs, axis=2)
    n_samples, n_pos = true_Y.shape

    results = {}
    for pos in range(1, n_pos):
        # 独立准确率
        ind_acc = float(np.mean(pred[:, pos] == true_Y[:, pos]))

        # 条件准确率：按前一位置的真实方向分组
        cond_accs = {}
        for d in range(3):
            mask = true_Y[:, pos-1] == d
            if mask.sum() > 0:
                cond_accs[d] = float(np.mean(pred[mask, pos] == true_Y[mask, pos]))

        avg_cond_acc = np.mean(list(cond_accs.values())) if cond_accs else ind_acc

        results[f'P{pos-1}->P{pos}'] = {
            'independent_accuracy': ind_acc,
            'conditional_accuracy': avg_cond_acc,
            'per_direction': cond_accs,
        }

    return results


def compute_direction_consistency(probs, true_Y, data):
    """计算方向一致性：相邻位置预测方向的合理性"""
    pred = np.argmax(probs, axis=2)
    n_samples, n_pos = true_Y.shape

    # 统计相邻位置预测方向的联合分布
    consistency_scores = []
    for i in range(n_samples):
        score = 0
        for pos in range(1, n_pos):
            # 如果相邻位置预测同向（都升或都降），加分
            if pred[i, pos] == pred[i, pos-1]:
                score += 1
        consistency_scores.append(score / (n_pos - 1))

    return float(np.mean(consistency_scores))


# ============================================================
# 主实验流程
# ============================================================

def run_joint_model_for_lottery(lottery_type):
    """对单个彩种运行联合建模实验"""
    log(f"\n{'═'*60}")
    log(f"  E0-Step7 跨位置联合建模: {lottery_type}")
    log(f"{'═'*60}")

    # Phase1 + Phase2
    with Timer(f"Phase1 [{lottery_type}]"):
        clusters, a1_filtered = run_phase1(lottery_type, rules_dir=STRICT_RULES_DIR)

    with Timer(f"Phase2 [{lottery_type}]"):
        (train_X, train_Y, test_X, test_Y,
         train_Y_val, test_Y_val,
         feature_names, train_indices, test_indices, data) = run_phase2(
            lottery_type, clusters, a1_filtered, rules_dir=STRICT_RULES_DIR)

    log(f"\n  特征维度: {train_X.shape[1]}, 训练样本: {train_X.shape[0]}, 测试样本: {test_X.shape[0]}")
    log(f"  位置数: {train_Y.shape[1]}")

    methods = OrderedDict([
        ("A_independent", ("逐位置独立MLP（基线）", train_independent_mlp)),
        ("B_multitask", ("多任务MLP（共享底层）", train_multitask_mlp)),
        ("C_autoregressive", ("自回归MLP（序列化预测）", train_autoregressive_mlp)),
    ])

    all_results = {}

    for method_id, (method_desc, train_fn) in methods.items():
        log(f"\n{'─'*50}")
        log(f"  方案 {method_id}: {method_desc}")
        log(f"{'─'*50}")

        with Timer(f"{method_id}"):
            probs = train_fn(train_X, train_Y, test_X, test_Y, data, test_indices)

        # 标准评估
        eval_results = evaluate_direction_predictions(probs, test_Y, data, test_indices)

        # 组合级准确率
        combo_acc = compute_combo_accuracy(probs, test_Y)

        # 条件准确率
        cond_acc = compute_conditional_accuracy(probs, test_Y)

        # 方向一致性
        consistency = compute_direction_consistency(probs, test_Y, data)

        result = {
            "method": method_id,
            "description": method_desc,
            "direction_accuracy": eval_results["overall"]["direction_accuracy"],
            "avg_reduction": eval_results["overall"]["avg_reduction"],
            "avg_survival": eval_results["overall"]["avg_survival"],
            "efficiency": eval_results["overall"]["efficiency"],
            "combo_accuracy": combo_acc,
            "direction_consistency": consistency,
            "per_position": eval_results["per_position"],
            "conditional_accuracy": cond_acc,
        }

        all_results[method_id] = result

        log(f"\n  结果:")
        log(f"    方向准确率: {result['direction_accuracy']:.4f}")
        log(f"    组合准确率: {result['combo_accuracy']:.4f}")
        log(f"    缩减率: {result['avg_reduction']:.4f}")
        log(f"    存活率: {result['avg_survival']:.4f}")
        log(f"    综合效率: {result['efficiency']:.4f}")
        log(f"    方向一致性: {result['direction_consistency']:.4f}")

    # 汇总对比
    log(f"\n{'═'*60}")
    log(f"  汇总对比: {lottery_type}")
    log(f"{'═'*60}")

    log(f"\n  {'方案':<20} {'方向准确率':<12} {'组合准确率':<12} {'综合效率':<10} {'一致性':<10}")
    log(f"  {'─'*64}")
    for mid, r in all_results.items():
        log(f"  {r['description']:<20} {r['direction_accuracy']:<12.4f} {r['combo_accuracy']:<12.4f} {r['efficiency']:<10.4f} {r['direction_consistency']:<10.4f}")

    # 增量分析
    baseline = all_results["A_independent"]
    log(f"\n  增量分析（相对基线）:")
    for mid in ["B_multitask", "C_autoregressive"]:
        r = all_results[mid]
        delta_acc = r["direction_accuracy"] - baseline["direction_accuracy"]
        delta_combo = r["combo_accuracy"] - baseline["combo_accuracy"]
        delta_eff = r["efficiency"] - baseline["efficiency"]
        log(f"    {r['description']}:")
        log(f"      方向准确率: {delta_acc:+.4f}")
        log(f"      组合准确率: {delta_combo:+.4f}")
        log(f"      综合效率:   {delta_eff:+.4f}")

    # 保存
    save_json(all_results, STEP7_DIR / f"e0_step7_joint_{lottery_type}.json")
    return all_results


def main():
    STEP7_DIR.mkdir(parents=True, exist_ok=True)
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STEP7_DIR
    logger = setup_logging()

    log("=" * 60)
    log("  E0-Step7: 跨位置联合建模实验")
    log("=" * 60)

    all_reports = {}
    try:
        for lottery_type in ["daletou", "shuangseqiu"]:
            report = run_joint_model_for_lottery(lottery_type)
            all_reports[lottery_type] = report

        # 跨彩种汇总
        log(f"\n{'═'*60}")
        log(f"  跨彩种汇总")
        log(f"{'═'*60}")

        for lt, rpt in all_reports.items():
            log(f"\n  {lt}:")
            baseline_acc = rpt["A_independent"]["direction_accuracy"]
            for mid in ["B_multitask", "C_autoregressive"]:
                r = rpt[mid]
                delta = r["direction_accuracy"] - baseline_acc
                log(f"    {r['description']}: 准确率 {r['direction_accuracy']:.4f} ({delta:+.4f}), 组合准确率 {r['combo_accuracy']:.4f}")

        # 最终结论
        log(f"\n  最终结论:")
        for lt, rpt in all_reports.items():
            best_method = max(rpt.keys(), key=lambda k: rpt[k]["direction_accuracy"])
            best = rpt[best_method]
            log(f"    {lt}: 最优方案 = {best['description']}, 准确率 = {best['direction_accuracy']:.4f}")

        save_json(all_reports, STEP7_DIR / "e0_step7_summary.json")

    finally:
        exp_utils.EXPERIMENT_DIR = original_exp_dir

    log(f"\n{'═'*60}")
    log(f"  E0-Step7 联合建模实验完成！")
    log(f"  结果目录: {STEP7_DIR}")
    log(f"{'═'*60}")


if __name__ == "__main__":
    main()
