"""阶段三：适配模型训练

实验3A：统计基线（规则簇投票）
实验3B：XGBoost
实验3C：轻量MLP（备选）
"""

import numpy as np

from .utils import (
    log, Timer, save_json,
    EXPERIMENT_DIR,
)


# === 统一评估 ===

def evaluate_direction_predictions(probs, true_Y, data, test_indices):
    """评估方向预测的缩号效果

    Args:
        probs: (n_samples, n_pos, 3) 方向概率 [P(D), P(E), P(U)]
        true_Y: (n_samples, n_pos) 实际方向 0=D, 1=E, 2=U
        data: LotteryData
        test_indices: 期数索引数组

    Returns:
        results: 评估结果字典
    """
    n_samples, n_pos = true_Y.shape
    results = {'per_position': {}, 'overall': {}}

    # 方向预测准确率
    pred_dirs = np.argmax(probs, axis=2)  # (n_samples, n_pos)
    for pos in range(n_pos):
        acc = float(np.mean(pred_dirs[:, pos] == true_Y[:, pos]))
        results['per_position'][f'P{pos}'] = {'direction_accuracy': acc}

    overall_acc = float(np.mean(pred_dirs == true_Y))
    results['overall']['direction_accuracy'] = overall_acc

    # 缩号评估：根据方向概率排除候选号码
    all_reductions = []
    all_survivals = []

    for pos in range(n_pos):
        reductions = []
        survivals = []

        for i in range(n_samples):
            t = int(test_indices[i])
            current_val = int(data.red_matrix[t - 1, pos]) if t > 0 else int(data.red_range / 2)
            true_val = int(data.red_matrix[t, pos])

            p_down, p_equal, p_up = probs[i, pos, 0], probs[i, pos, 1], probs[i, pos, 2]

            # 全候选空间
            full = set(range(1, data.red_range + 1))
            candidates = set(full)

            # 概率 < 0.15 的方向，排除对应候选
            if p_up < 0.15:
                candidates -= {v for v in full if v > current_val}
            if p_down < 0.15:
                candidates -= {v for v in full if v < current_val}
            if p_equal < 0.15:
                candidates.discard(current_val)

            # 确保至少保留1个候选
            if len(candidates) == 0:
                candidates = full

            reduction = 1.0 - len(candidates) / len(full)
            survival = 1.0 if true_val in candidates else 0.0

            reductions.append(reduction)
            survivals.append(survival)

        avg_red = float(np.mean(reductions))
        avg_surv = float(np.mean(survivals))
        results['per_position'][f'P{pos}'].update({
            'avg_reduction': avg_red,
            'avg_survival': avg_surv,
            'efficiency': avg_red * avg_surv,
        })
        all_reductions.extend(reductions)
        all_survivals.extend(survivals)

    results['overall'].update({
        'avg_reduction': float(np.mean(all_reductions)),
        'avg_survival': float(np.mean(all_survivals)),
        'efficiency': float(np.mean(all_reductions)) * float(np.mean(all_survivals)),
    })

    return results


# === 实验 3A：统计基线 ===

def experiment_3a(train_X, train_Y, test_X, test_Y, feature_names, data, test_indices):
    """统计基线：对每个规则簇，统计触发时的方向分布，投票预测"""
    log("\n  ── 实验 3A: 统计基线 ──")

    n_train, n_features = train_X.shape
    n_pos = train_Y.shape[1]

    # 找出 A2 簇特征和 A1 模式特征的索引
    a2_indices = [i for i, name in enumerate(feature_names) if name.startswith('a2_c')]
    a1_indices = [i for i, name in enumerate(feature_names) if name.startswith('a1_p')]
    rule_indices = a2_indices + a1_indices
    log(f"    规则特征数: A2={len(a2_indices)}, A1={len(a1_indices)}, 总={len(rule_indices)}")

    # 统计每个规则特征触发时的方向分布
    rule_stats = {}  # (rule_idx, pos) → [count_D, count_E, count_U]
    for ri in rule_indices:
        triggered = train_X[:, ri] == 1
        n_trig = int(triggered.sum())
        if n_trig < 5:
            continue
        for pos in range(n_pos):
            counts = np.bincount(train_Y[triggered, pos], minlength=3).astype(float)
            rule_stats[(ri, pos)] = counts / counts.sum()

    log(f"    有效规则-位置对: {len(rule_stats)}")

    # 预测
    n_test = test_X.shape[0]
    probs = np.full((n_test, n_pos, 3), 1.0 / 3.0)

    for i in range(n_test):
        for pos in range(n_pos):
            vote = np.zeros(3)
            n_vote = 0
            for ri in rule_indices:
                if test_X[i, ri] == 1 and (ri, pos) in rule_stats:
                    vote += rule_stats[(ri, pos)]
                    n_vote += 1
            if n_vote > 0:
                probs[i, pos] = vote / n_vote

    results = evaluate_direction_predictions(probs, test_Y, data, test_indices)
    results['method'] = '3A_statistical_baseline'
    results['n_rule_pos_pairs'] = len(rule_stats)

    log(f"    方向准确率: {results['overall']['direction_accuracy']:.4f}")
    log(f"    缩减率: {results['overall']['avg_reduction']:.4f}")
    log(f"    存活率: {results['overall']['avg_survival']:.4f}")
    log(f"    综合效率: {results['overall']['efficiency']:.4f}")

    save_json(results, EXPERIMENT_DIR / f"phase3_exp3a_{data.lottery_type}.json")
    return results, probs


# === 实验 3B：XGBoost ===

def experiment_3b(train_X, train_Y, test_X, test_Y, feature_names, data, test_indices):
    """梯度提升树多分类方向预测（XGBoost优先，sklearn GradientBoosting后备）"""
    log("\n  ── 实验 3B: 梯度提升树 ──")

    use_xgb = False
    try:
        import xgboost as xgb
        # 尝试实际加载库（触发 dylib 加载）
        xgb.XGBClassifier()
        use_xgb = True
        log("    使用 XGBoost")
    except Exception:
        log("    XGBoost 不可用，使用 sklearn GradientBoosting")

    from sklearn.ensemble import GradientBoostingClassifier

    n_pos = train_Y.shape[1]
    probs = np.zeros((test_X.shape[0], n_pos, 3))
    feature_importances = {}
    models_info = {}

    for pos in range(n_pos):
        log(f"    训练 P{pos}...")

        # 划分验证集（从训练集末尾取20%）
        n_train = train_X.shape[0]
        n_val = max(50, int(n_train * 0.2))
        tr_X, tr_Y = train_X[:-n_val], train_Y[:-n_val, pos]
        va_X, va_Y = train_X[-n_val:], train_Y[-n_val:, pos]

        if use_xgb:
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.6,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                early_stopping_rounds=30,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(tr_X, tr_Y, eval_set=[(va_X, va_Y)], verbose=False)
            pos_probs = model.predict_proba(test_X)
            imp = model.feature_importances_
            best_iter = model.best_iteration if hasattr(model, 'best_iteration') else -1
            val_acc = float(np.mean(model.predict(va_X) == va_Y))

            # 保存模型
            model_path = EXPERIMENT_DIR / "phase3_models"
            model_path.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_path / f"xgb_{data.lottery_type}_P{pos}.json"))
        else:
            model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=10,
                subsample=0.8,
                max_features=0.6,
                random_state=42,
                validation_fraction=0.2,
                n_iter_no_change=30,
                verbose=0,
            )
            model.fit(tr_X, tr_Y)
            pos_probs = model.predict_proba(test_X)
            imp = model.feature_importances_
            best_iter = model.n_estimators_ if hasattr(model, 'n_estimators_') else model.n_estimators
            val_acc = float(np.mean(model.predict(va_X) == va_Y))

        probs[:, pos, :] = pos_probs

        # feature importance
        top_indices = np.argsort(imp)[::-1][:20]
        top_features = [(feature_names[idx], float(imp[idx])) for idx in top_indices]
        feature_importances[f'P{pos}'] = top_features

        models_info[f'P{pos}'] = {
            'best_iteration': best_iter,
            'val_accuracy': val_acc,
        }
        log(f"      best_iter={best_iter}, val_acc={val_acc:.4f}")

    results = evaluate_direction_predictions(probs, test_Y, data, test_indices)
    results['method'] = '3B_xgboost'
    results['feature_importances'] = feature_importances
    results['models_info'] = models_info

    log(f"    方向准确率: {results['overall']['direction_accuracy']:.4f}")
    log(f"    缩减率: {results['overall']['avg_reduction']:.4f}")
    log(f"    存活率: {results['overall']['avg_survival']:.4f}")
    log(f"    综合效率: {results['overall']['efficiency']:.4f}")

    save_json(results, EXPERIMENT_DIR / f"phase3_exp3b_{data.lottery_type}.json")
    return results, probs


# === 实验 3C：轻量 MLP ===

def experiment_3c(train_X, train_Y, test_X, test_Y, feature_names, data, test_indices):
    """轻量MLP方向预测（备选方案）"""
    log("\n  ── 实验 3C: MLP ──")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        log("    [警告] torch 未安装，跳过实验 3C")
        return None, None

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    log(f"    设备: {device}")

    n_pos = train_Y.shape[1]
    input_dim = train_X.shape[1]
    probs = np.zeros((test_X.shape[0], n_pos, 3))

    # 划分验证集
    n_train = train_X.shape[0]
    n_val = max(50, int(n_train * 0.2))

    tr_X_t = torch.FloatTensor(train_X[:-n_val]).to(device)
    va_X_t = torch.FloatTensor(train_X[-n_val:]).to(device)
    te_X_t = torch.FloatTensor(test_X).to(device)

    for pos in range(n_pos):
        log(f"    训练 P{pos}...")

        tr_Y_t = torch.LongTensor(train_Y[:-n_val, pos]).to(device)
        va_Y_t = torch.LongTensor(train_Y[-n_val:, pos]).to(device)

        # 模型
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

        # 训练
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50

        dataset = TensorDataset(tr_X_t, tr_Y_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(500):
            model.train()
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_Y)
                loss.backward()
                optimizer.step()

            # 验证
            model.eval()
            with torch.no_grad():
                val_out = model(va_X_t)
                val_loss = criterion(val_out, va_Y_t).item()
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # 加载最优模型
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            test_out = model(te_X_t)
            pos_probs = torch.softmax(test_out, dim=1).cpu().numpy()
            probs[:, pos, :] = pos_probs

            val_pred = torch.argmax(model(va_X_t), dim=1).cpu().numpy()
            val_acc = float(np.mean(val_pred == train_Y[-n_val:, pos]))

        log(f"      epochs={epoch+1}, val_loss={best_val_loss:.4f}, val_acc={val_acc:.4f}")

    results = evaluate_direction_predictions(probs, test_Y, data, test_indices)
    results['method'] = '3C_mlp'

    log(f"    方向准确率: {results['overall']['direction_accuracy']:.4f}")
    log(f"    缩减率: {results['overall']['avg_reduction']:.4f}")
    log(f"    存活率: {results['overall']['avg_survival']:.4f}")
    log(f"    综合效率: {results['overall']['efficiency']:.4f}")

    save_json(results, EXPERIMENT_DIR / f"phase3_exp3c_{data.lottery_type}.json")
    return results, probs


# === 主入口 ===

def run_phase3(train_X, train_Y, test_X, test_Y, feature_names, data, test_indices):
    """执行阶段三全部实验

    Returns:
        results_3a, results_3b, results_3c: 各实验结果
    """
    log(f"\n{'─'*40}")
    log(f"阶段三：适配模型训练 [{data.lottery_type}]")
    log(f"{'─'*40}")

    with Timer("实验 3A"):
        results_3a, probs_3a = experiment_3a(
            train_X, train_Y, test_X, test_Y, feature_names, data, test_indices)

    with Timer("实验 3B"):
        results_3b, probs_3b = experiment_3b(
            train_X, train_Y, test_X, test_Y, feature_names, data, test_indices)

    # 3C 仅在 3B 效率不足时执行
    results_3c = None
    if results_3b is None or results_3b['overall']['efficiency'] < 0.25:
        reason = "3B 未执行" if results_3b is None else f"3B 效率={results_3b['overall']['efficiency']:.4f} < 0.25"
        log(f"\n  启动 3C（原因: {reason}）")
        with Timer("实验 3C"):
            results_3c, probs_3c = experiment_3c(
                train_X, train_Y, test_X, test_Y, feature_names, data, test_indices)
    else:
        log(f"\n  跳过 3C（3B 效率={results_3b['overall']['efficiency']:.4f} >= 0.25）")

    return results_3a, results_3b, results_3c
