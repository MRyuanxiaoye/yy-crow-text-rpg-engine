"""E0-Step4: B1 Transformer 融合验证

将已训练的 Transformer 模型的预测概率作为额外特征，
加入 Phase2 特征矩阵，验证深度学习与规则模型的协同效果。

方案：
  G0: 基线（A1+A2+局面，无A3/A4，无Transformer）
  G1: G0 + Transformer 方向概率特征（n_pos * 3 维）

注意：现有 Transformer 模型用全量数据训练，存在数据泄露风险。
本实验结果仅作参考，严格验证需要用前60%数据重新训练 Transformer。

用法: python3 -m src.research.experiment.e0_step4_transformer_fusion
"""

import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import research.experiment.utils as exp_utils
from research.experiment.utils import setup_logging, log, Timer, save_json
from research.experiment.phase1_compress import run_phase1
from research.experiment.phase2_replay import run_phase2
from research.experiment.phase3_train import experiment_3b, experiment_3c
from research.data_loader import LotteryData
from research.deep_learning.dataset import LotterySequenceDataset
from research.deep_learning.transformer_model import LotteryTransformer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STRICT_RULES_DIR = RESULTS_DIR / "rules_strict"
MODELS_DIR = RESULTS_DIR / "models"
STEP4_DIR = RESULTS_DIR / "e0_step4_transformer"


def extract_transformer_features(lottery_type, data, train_indices, test_indices):
    """从已训练的 Transformer 模型提取方向概率特征

    Args:
        lottery_type: 彩种
        data: LotteryData
        train_indices: 训练集期数索引
        test_indices: 测试集期数索引

    Returns:
        train_feats: (n_train, n_pos*3) Transformer概率特征
        test_feats: (n_test, n_pos*3) Transformer概率特征
        feat_names: 特征名列表
    """
    rc = data.red_count
    seq_len = 30  # Transformer 默认序列长度

    # 构建完整数据集用于特征提取
    full_ds = LotterySequenceDataset(data, seq_len=seq_len, start_idx=0, end_idx=data.n_draws)

    # 加载模型
    model_path = MODELS_DIR / f"transformer_{lottery_type}_best.pt"
    if not model_path.exists():
        log(f"    [警告] 模型文件不存在: {model_path}")
        return None, None, None

    state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)

    # 从 checkpoint 自动推断模型参数
    d_model = state_dict['input_proj.weight'].shape[0]
    ckpt_feature_dim = state_dict['input_proj.weight'].shape[1]
    n_layers = max(int(k.split('.')[2]) for k in state_dict if k.startswith('transformer.layers.')) + 1
    n_heads_3d = state_dict['transformer.layers.0.self_attn.in_proj_weight'].shape[0]
    n_heads = n_heads_3d // (3 * (d_model // (n_heads_3d // 3)))  # 推断 n_heads
    # 简化：in_proj_weight shape = (3*d_model, d_model)，n_heads 从训练配置推断
    # 尝试常见值
    for nh in [4, 8, 2, 16]:
        if d_model % nh == 0:
            n_heads = nh
            break

    log(f"    Checkpoint 参数: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, feature_dim={ckpt_feature_dim}")

    model = LotteryTransformer(
        feature_dim=ckpt_feature_dim,
        n_positions=rc,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    model.load_state_dict(state_dict, strict=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 提取特征：对每个期数索引，构造输入序列并推理
    def extract_for_indices(indices):
        feats = np.zeros((len(indices), rc * 3), dtype=np.float32)
        features_all = full_ds.features  # (n_samples, seq_len, feature_dim)
        # full_ds 的样本索引映射：样本 i 对应期数 seq_len + i
        ds_offset = seq_len  # 第一个有效样本对应的期数

        for i, t in enumerate(indices):
            ds_idx = t - ds_offset
            if ds_idx < 0 or ds_idx >= len(features_all):
                # 超出范围，用均匀分布
                feats[i] = 1.0 / 3.0
                continue

            x = torch.FloatTensor(features_all[ds_idx]).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)  # (1, n_pos, 3)
                probs = torch.softmax(logits, dim=2).cpu().numpy()[0]  # (n_pos, 3)
                feats[i] = probs.flatten()

        return feats

    log(f"    提取训练集 Transformer 特征 ({len(train_indices)} 样本)...")
    train_feats = extract_for_indices(train_indices)

    log(f"    提取测试集 Transformer 特征 ({len(test_indices)} 样本)...")
    test_feats = extract_for_indices(test_indices)

    # 特征名
    dir_names = ['D', 'E', 'U']
    feat_names = [f'transformer_P{pos}_prob_{d}' for pos in range(rc) for d in dir_names]

    log(f"    Transformer 特征维度: {len(feat_names)}")
    log(f"    训练集概率均值: {train_feats.mean(axis=0)[:6]}")
    log(f"    测试集概率均值: {test_feats.mean(axis=0)[:6]}")

    return train_feats, test_feats, feat_names


def filter_a3a4_features(feature_names, train_X, test_X):
    """移除 A3/A4 特征（Step3 已证明无用）"""
    keep = []
    for i, name in enumerate(feature_names):
        is_a3 = name.startswith('a3_')
        is_a4 = name.startswith('a4_')
        is_a3_meta = name in ('n_matched_a3', 'avg_chi2_a3', 'avg_conf_a3')
        is_a4_meta = name in ('n_matched_a4', 'avg_chi2_a4', 'avg_conf_a4')
        if not is_a3 and not is_a4 and not is_a3_meta and not is_a4_meta:
            keep.append(i)

    keep = np.array(keep)
    return ([feature_names[i] for i in keep],
            train_X[:, keep], test_X[:, keep])


def run_fusion_for_lottery(lottery_type):
    """对单个彩种运行 Transformer 融合实验"""
    log(f"\n{'═'*60}")
    log(f"  E0-Step4 Transformer 融合: {lottery_type}")
    log(f"{'═'*60}")

    # Phase1 + Phase2
    with Timer(f"Phase1 [{lottery_type}]"):
        clusters, a1_filtered = run_phase1(lottery_type, rules_dir=STRICT_RULES_DIR)

    with Timer(f"Phase2 [{lottery_type}]"):
        (train_X, train_Y, test_X, test_Y,
         train_Y_val, test_Y_val,
         feature_names, train_indices, test_indices, data) = run_phase2(
            lottery_type, clusters, a1_filtered, rules_dir=STRICT_RULES_DIR)

    # 移除 A3/A4（Step3 结论）
    g0_names, g0_train, g0_test = filter_a3a4_features(feature_names, train_X, test_X)
    log(f"\n  G0 基线特征: {len(g0_names)} 维（已移除 A3/A4）")

    # 提取 Transformer 特征
    with Timer("提取 Transformer 特征"):
        tf_train, tf_test, tf_names = extract_transformer_features(
            lottery_type, data, train_indices, test_indices)

    if tf_train is None:
        log("  [错误] 无法提取 Transformer 特征，跳过")
        return None

    # G1: G0 + Transformer
    g1_train = np.hstack([g0_train, tf_train])
    g1_test = np.hstack([g0_test, tf_test])
    g1_names = g0_names + tf_names
    log(f"  G1 融合特征: {len(g1_names)} 维（G0 + Transformer {len(tf_names)} 维）")

    all_results = {}

    # G0 基线
    log(f"\n{'─'*50}")
    log(f"  G0: 基线 (A1+A2+局面)")
    log(f"{'─'*50}")

    log(f"\n  --- XGBoost ---")
    res_g0_xgb, probs_g0_xgb = experiment_3b(
        g0_train, train_Y, g0_test, test_Y, g0_names, data, test_indices)

    log(f"\n  --- MLP ---")
    res_g0_mlp, probs_g0_mlp = experiment_3c(
        g0_train, train_Y, g0_test, test_Y, g0_names, data, test_indices)

    all_results["G0"] = {
        "description": "基线 (A1+A2+局面)",
        "n_features": len(g0_names),
        "xgboost": {
            "direction_accuracy": res_g0_xgb["overall"]["direction_accuracy"],
            "efficiency": res_g0_xgb["overall"]["efficiency"],
            "avg_reduction": res_g0_xgb["overall"]["avg_reduction"],
            "avg_survival": res_g0_xgb["overall"]["avg_survival"],
        },
        "mlp": {
            "direction_accuracy": res_g0_mlp["overall"]["direction_accuracy"],
            "efficiency": res_g0_mlp["overall"]["efficiency"],
            "avg_reduction": res_g0_mlp["overall"]["avg_reduction"],
            "avg_survival": res_g0_mlp["overall"]["avg_survival"],
        } if res_g0_mlp else None,
    }

    # G1 融合
    log(f"\n{'─'*50}")
    log(f"  G1: G0 + Transformer 概率特征")
    log(f"{'─'*50}")

    log(f"\n  --- XGBoost ---")
    res_g1_xgb, probs_g1_xgb = experiment_3b(
        g1_train, train_Y, g1_test, test_Y, g1_names, data, test_indices)

    log(f"\n  --- MLP ---")
    res_g1_mlp, probs_g1_mlp = experiment_3c(
        g1_train, train_Y, g1_test, test_Y, g1_names, data, test_indices)

    all_results["G1"] = {
        "description": "G0 + Transformer 概率特征",
        "n_features": len(g1_names),
        "xgboost": {
            "direction_accuracy": res_g1_xgb["overall"]["direction_accuracy"],
            "efficiency": res_g1_xgb["overall"]["efficiency"],
            "avg_reduction": res_g1_xgb["overall"]["avg_reduction"],
            "avg_survival": res_g1_xgb["overall"]["avg_survival"],
        },
        "mlp": {
            "direction_accuracy": res_g1_mlp["overall"]["direction_accuracy"],
            "efficiency": res_g1_mlp["overall"]["efficiency"],
            "avg_reduction": res_g1_mlp["overall"]["avg_reduction"],
            "avg_survival": res_g1_mlp["overall"]["avg_survival"],
        } if res_g1_mlp else None,
    }

    # Transformer 特征重要性（从 XGBoost 中提取）
    if "feature_importances" in res_g1_xgb:
        tf_importance = {}
        for pos_key, top_feats in res_g1_xgb["feature_importances"].items():
            tf_feats = [(name, imp) for name, imp in top_feats if name.startswith('transformer_')]
            if tf_feats:
                tf_importance[pos_key] = tf_feats
        all_results["transformer_importance"] = tf_importance

    # 增量分析
    log(f"\n{'═'*60}")
    log(f"  增量分析: {lottery_type}")
    log(f"{'═'*60}")

    for model_type in ["xgboost", "mlp"]:
        g0_r = all_results["G0"].get(model_type)
        g1_r = all_results["G1"].get(model_type)
        if g0_r and g1_r:
            delta_acc = g1_r["direction_accuracy"] - g0_r["direction_accuracy"]
            delta_eff = g1_r["efficiency"] - g0_r["efficiency"]
            log(f"\n  {model_type}:")
            log(f"    G0 准确率: {g0_r['direction_accuracy']:.4f}, 效率: {g0_r['efficiency']:.4f}")
            log(f"    G1 准确率: {g1_r['direction_accuracy']:.4f}, 效率: {g1_r['efficiency']:.4f}")
            log(f"    增量: 准确率 {delta_acc:+.4f}, 效率 {delta_eff:+.4f}")

    # 结论
    g0_best = max(
        all_results["G0"]["xgboost"]["direction_accuracy"],
        all_results["G0"]["mlp"]["direction_accuracy"] if all_results["G0"]["mlp"] else 0
    )
    g1_best = max(
        all_results["G1"]["xgboost"]["direction_accuracy"],
        all_results["G1"]["mlp"]["direction_accuracy"] if all_results["G1"]["mlp"] else 0
    )
    delta = g1_best - g0_best
    useful = delta > 0.005

    all_results["conclusion"] = {
        "transformer_useful": "有用" if useful else "无用",
        "best_accuracy_delta": delta,
        "caveat": "Transformer模型用全量数据训练，存在数据泄露风险，结果仅供参考",
    }

    log(f"\n  结论: Transformer 融合 {'有用' if useful else '无用'} (增量 {delta:+.4f})")
    log(f"  注意: {all_results['conclusion']['caveat']}")

    save_json(all_results, STEP4_DIR / f"e0_step4_fusion_{lottery_type}.json")
    return all_results


def main():
    STEP4_DIR.mkdir(parents=True, exist_ok=True)
    original_exp_dir = exp_utils.EXPERIMENT_DIR
    exp_utils.EXPERIMENT_DIR = STEP4_DIR
    logger = setup_logging()

    log("=" * 60)
    log("  E0-Step4: B1 Transformer 融合验证")
    log("=" * 60)

    all_reports = {}
    try:
        for lottery_type in ["daletou", "shuangseqiu"]:
            report = run_fusion_for_lottery(lottery_type)
            if report:
                all_reports[lottery_type] = report

        # 跨彩种汇总
        log(f"\n{'═'*60}")
        log(f"  跨彩种汇总")
        log(f"{'═'*60}")

        for lt, rpt in all_reports.items():
            c = rpt["conclusion"]
            log(f"\n  {lt}: Transformer 融合 {c['transformer_useful']} (增量 {c['best_accuracy_delta']:+.4f})")

        save_json(all_reports, STEP4_DIR / "e0_step4_summary.json")

    finally:
        exp_utils.EXPERIMENT_DIR = original_exp_dir

    log(f"\n{'═'*60}")
    log(f"  E0-Step4 Transformer 融合实验完成！")
    log(f"  结果目录: {STEP4_DIR}")
    log(f"{'═'*60}")


if __name__ == "__main__":
    main()
