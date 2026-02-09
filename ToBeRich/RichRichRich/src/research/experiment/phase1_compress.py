"""阶段一：规则压缩

将A2的10000条规则聚类压缩为规则簇，筛选A1高置信度模式。
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .utils import (
    log, Timer, save_json, load_json,
    RULES_DIR, EXPERIMENT_DIR,
    CONDITION_TYPES, CONDITION_TYPE_INDEX,
    parse_condition_type, parse_condition_position, parse_target,
)


def encode_rule(rule):
    """将A2规则编码为29维特征向量

    维度分配:
      [0:11]  条件类型计数 (val/dir/odd/even/big/small/diff/sum/span/ac/consec)
      [11:16] 涉及位置 multi-hot (P0-P4)
      [16:19] target方向 one-hot (U/D/E)
      [19:24] target位置 one-hot (P0-P4)
      [24:29] 数值特征 (support, confidence, lift, chi2, n_conditions)
    """
    vec = np.zeros(29, dtype=np.float32)

    for cond in rule['conditions']:
        ctype_idx = parse_condition_type(cond)
        vec[ctype_idx] += 1

        pos = parse_condition_position(cond)
        if pos is not None and pos < 5:
            vec[11 + pos] = 1

    target_pos, target_dir = parse_target(rule['target'])
    vec[16 + target_dir] = 1
    if target_pos < 5:
        vec[19 + target_pos] = 1

    vec[24] = rule['support']
    vec[25] = rule['confidence']
    vec[26] = rule['lift']
    vec[27] = rule['chi2']
    vec[28] = len(rule['conditions'])

    return vec


def compress_a2_rules(rules, candidate_ks=(100, 200, 500, 800, 1000)):
    """对A2规则做KMeans聚类压缩

    Args:
        rules: A2规则列表
        candidate_ks: 候选簇数列表

    Returns:
        clusters: {cluster_id: {representative, size, avg_confidence, avg_lift, rules_indices}}
        best_k: 最优簇数
        silhouette: 轮廓系数
    """
    log(f"  编码 {len(rules)} 条规则...")
    feature_matrix = np.array([encode_rule(r) for r in rules])

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    log(f"  搜索最优簇数，候选: {candidate_ks}")
    best_k, best_score = candidate_ks[0], -1
    best_labels = None

    for k in candidate_ks:
        if k >= len(rules):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        labels = km.fit_predict(X)
        sample_size = min(5000, len(X))
        score = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
        log(f"    k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
            best_labels = labels

    log(f"  最优簇数: {best_k}, silhouette={best_score:.4f}")

    # 用最优k重新聚类（更多迭代）
    if best_labels is None or len(set(best_labels)) != best_k:
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
        best_labels = km.fit_predict(X)

    # 构建簇信息
    clusters = {}
    for cid in range(best_k):
        mask = best_labels == cid
        indices = np.where(mask)[0].tolist()
        cluster_rules = [rules[i] for i in indices]

        representative = max(cluster_rules, key=lambda r: (r['confidence'], r['support']))

        clusters[str(cid)] = {
            'representative': representative,
            'size': len(indices),
            'avg_confidence': float(np.mean([r['confidence'] for r in cluster_rules])),
            'avg_lift': float(np.mean([r['lift'] for r in cluster_rules])),
            'rules_indices': indices,
        }

    return clusters, best_k, best_score


def filter_a1_patterns(patterns, min_confidence=0.4, top_per_group=5):
    """筛选A1方向模式

    按 positions+window 分组，每组保留 top_per_group 个高置信度模式。
    """
    # 先过滤低置信度
    filtered = [p for p in patterns if p['prediction_confidence'] >= min_confidence]
    log(f"  A1 confidence >= {min_confidence}: {len(filtered)}/{len(patterns)}")

    # 按 (positions, window) 分组
    groups = {}
    for p in filtered:
        key = (tuple(p['positions']), p['window'])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    # 每组取 top
    result = []
    for key, group in groups.items():
        group.sort(key=lambda x: x['prediction_confidence'], reverse=True)
        result.extend(group[:top_per_group])

    log(f"  A1 分组筛选后: {len(result)} 条模式")
    return result


def run_phase1(lottery_type, rules_dir=None):
    """执行阶段一：规则压缩

    Args:
        lottery_type: "daletou" 或 "shuangseqiu"
        rules_dir: 自定义规则目录，默认使用 RULES_DIR

    Returns:
        clusters: A2规则簇
        a1_filtered: 筛选后的A1模式
    """
    _rules_dir = rules_dir or RULES_DIR
    log(f"\n{'─'*40}")
    log(f"阶段一：规则压缩 [{lottery_type}]")
    log(f"{'─'*40}")

    # 加载A2规则
    with Timer("加载A2规则"):
        a2_path = _rules_dir / f"a2_exclusion_rules_{lottery_type}.json"
        a2_data = load_json(a2_path)
        rules = a2_data['rules']
        log(f"  A2规则数: {len(rules)} (JSON中保存的top规则)")
        log(f"  A2总规则数: {a2_data['total_rules']}")

    # 聚类压缩
    with Timer("A2规则聚类"):
        clusters, best_k, sil_score = compress_a2_rules(rules)

    # 加载并筛选A1模式
    with Timer("A1模式筛选"):
        a1_path = _rules_dir / f"a1_direction_patterns_{lottery_type}.json"
        a1_data = load_json(a1_path)
        a1_patterns = a1_data['patterns']
        log(f"  A1总模式数: {len(a1_patterns)}")
        a1_filtered = filter_a1_patterns(a1_patterns)

    # 保存结果
    output = {
        'lottery_type': lottery_type,
        'n_clusters': best_k,
        'silhouette_score': sil_score,
        'n_a2_rules_input': len(rules),
        'clusters': clusters,
        'a1_filtered': a1_filtered,
        'a1_filtered_count': len(a1_filtered),
    }

    # 保存时去掉 rules_indices（太大）
    output_save = dict(output)
    output_save['clusters'] = {}
    for cid, cinfo in clusters.items():
        output_save['clusters'][cid] = {
            'representative': cinfo['representative'],
            'size': cinfo['size'],
            'avg_confidence': cinfo['avg_confidence'],
            'avg_lift': cinfo['avg_lift'],
        }

    save_json(output_save, EXPERIMENT_DIR / f"phase1_clusters_{lottery_type}.json")

    log(f"\n  阶段一汇总:")
    log(f"    A2: {len(rules)} 条 → {best_k} 个簇 (silhouette={sil_score:.4f})")
    log(f"    A1: {len(a1_patterns)} 条 → {len(a1_filtered)} 条")

    return clusters, a1_filtered
