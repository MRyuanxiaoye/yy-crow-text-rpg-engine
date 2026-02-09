# -*- coding: utf-8 -*-
"""
彩票分析可视化报告 - Streamlit 主入口

启动方式：
  streamlit run src/visualization/app.py
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import streamlit as st
from visualization.components.ball_display import load_custom_css
from visualization.components.overview import render_overview, render_funnel
from visualization.components.stage0_view import render_stage0
from visualization.components.stage1_view import render_stage1
from visualization.components.stage2_view import render_stage2
from visualization.components.stage3_view import render_stage3
from visualization.components.stage4_view import render_stage4
from pipeline.pipeline import Pipeline

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="彩票分析报告",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_custom_css()


# ============================================================
# Mock 数据（后续接入真实方法链后替换）
# ============================================================

def generate_mock_data(lottery_type="daletou"):
    """生成模拟报告数据，用于 UI 开发测试"""
    import random
    random.seed(42)

    if lottery_type == "daletou":
        num_range = 35
        red_count = 5
        blue_range = 12
        blue_count = 2
    else:
        num_range = 33
        red_count = 6
        blue_range = 16
        blue_count = 1

    # 阶段0：特征数据
    number_features = {}
    for n in range(1, num_range + 1):
        freq = random.uniform(0.02, 0.06)
        number_features[str(n)] = {
            "frequency": freq,
            "missing_value": random.randint(0, 20),
            "avg_gap": random.uniform(4, 10),
            "max_gap": random.randint(15, 40),
            "consecutive": random.randint(0, 3),
            "trend": random.choice(["上升", "下降", "平稳"]),
        }

    # 奇偶比/大小比分布
    odd_even_dist = {"3:2": 35, "2:3": 30, "4:1": 12, "1:4": 10, "5:0": 7, "0:5": 6}
    big_small_dist = {"3:2": 32, "2:3": 33, "4:1": 14, "1:4": 11, "5:0": 5, "0:5": 5}
    sum_values = [random.randint(40, 140) for _ in range(200)]
    ac_values = [random.randint(2, 10) for _ in range(200)]

    # 共现矩阵
    co_matrix = [[random.randint(5, 50) if i != j else 0
                   for j in range(num_range)] for i in range(num_range)]

    stage0 = {
        "number_features": number_features,
        "appearance_matrix": [
            [random.choice([0, 0, 0, 1]) for _ in range(30)]
            for _ in range(num_range)
        ],
        "combo_stats": {
            "odd_even_dist": odd_even_dist,
            "big_small_dist": big_small_dist,
            "sum_values": sum_values,
            "ac_values": ac_values,
        },
        "time_features": {},
        "correlation_matrix": co_matrix,
        "transition_matrix": [
            [random.uniform(0, 0.1) for _ in range(num_range)]
            for _ in range(num_range)
        ],
    }

    # 阶段1：排除数据
    excluded = sorted(random.sample(range(1, num_range + 1), random.randint(10, 15)))
    algo_names = ["连续重复", "遗漏值异常", "极端组合", "马尔可夫", "周期性", "聚类异常"]
    exclusion_details = {}
    for n in excluded:
        reasons = {algo: random.uniform(0.3, 1.0) for algo in algo_names}
        exclusion_details[str(n)] = {
            "confidence": random.uniform(0.6, 0.95),
            "reasons": reasons,
        }

    stage1 = {
        "excluded_numbers": excluded,
        "exclusion_details": exclusion_details,
        "remaining_numbers": [n for n in range(1, num_range + 1) if n not in excluded],
    }

    # 阶段2：权重数据
    remaining = stage1["remaining_numbers"]
    w_algos = ["频率回归", "遗漏值回归", "时序衰减", "马尔可夫", "共现关联", "深度学习"]
    number_weights = {}
    weight_details = {}
    raw_weights = {str(n): random.uniform(0.2, 1.0) for n in remaining}
    total_w = sum(raw_weights.values())
    for n_str, w in raw_weights.items():
        number_weights[n_str] = w / total_w
        weight_details[n_str] = {
            "final": w / total_w,
            "breakdown": {algo: random.uniform(0.1, 0.9) for algo in w_algos},
        }

    sorted_by_weight = sorted(remaining, key=lambda n: number_weights[str(n)], reverse=True)
    stage2 = {
        "number_weights": number_weights,
        "weight_details": weight_details,
        "top_numbers": sorted_by_weight,
        "top_count": min(15, len(remaining)),
    }

    # 阶段3：候选组合
    candidates = []
    for _ in range(80):
        red = sorted(random.sample(remaining, min(red_count, len(remaining))))
        blue = sorted(random.sample(range(1, blue_range + 1), blue_count))
        score = random.uniform(0.5, 0.95)
        candidates.append({
            "red_balls": red,
            "blue_balls": blue,
            "score": score,
            "score_breakdown": {
                "weight": random.uniform(0.5, 1.0),
                "constraint": random.uniform(0.6, 1.0),
                "balance": random.uniform(0.4, 0.9),
                "similarity": random.uniform(0.3, 0.8),
            },
            "ac_value": random.randint(3, 9),
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)

    stage3 = {
        "candidates": candidates,
        "top_score": candidates[0]["score"],
        "generation_stats": {
            "total_generated": 5000,
            "after_filter": 200,
            "strategy_1": 120, "strategy_2": 30,
            "strategy_3": 100, "strategy_4": 20,
        },
    }

    # 阶段4：购买方案
    top5_red = sorted_by_weight[:5]
    top3_blue = random.sample(range(1, blue_range + 1), min(3, blue_range))
    all_covered_red = set()
    all_covered_blue = set()

    tickets = []
    # 胆拖票
    dan_red = sorted(top5_red[:2])
    tuo_red = sorted(top5_red[2:5] + random.sample(remaining, 1))
    dan_blue = sorted(top3_blue[:1])
    tuo_blue = sorted(top3_blue[1:3])
    all_covered_red.update(dan_red + tuo_red)
    all_covered_blue.update(dan_blue + tuo_blue)
    tickets.append({
        "type": "胆拖", "dan_red": dan_red, "tuo_red": tuo_red,
        "dan_blue": dan_blue, "tuo_blue": tuo_blue,
        "combinations": 6, "cost": 12,
    })
    # 单式票
    for i in range(3):
        red = sorted(random.sample(remaining, min(red_count, len(remaining))))
        blue = sorted(random.sample(range(1, blue_range + 1), blue_count))
        all_covered_red.update(red)
        all_covered_blue.update(blue)
        tickets.append({
            "type": "单式", "red_balls": red, "blue_balls": blue,
            "combinations": 1, "cost": 2,
        })

    stage4 = {
        "budget": 20,
        "total_cost": 18,
        "total_combinations": 9,
        "tickets": tickets,
        "coverage": {
            "all_red_numbers": sorted(all_covered_red),
            "all_blue_numbers": sorted(all_covered_blue),
            "top10_red_covered": min(10, len(all_covered_red)),
        },
        "coverage_pct": len(all_covered_red) / num_range,
        "weight_coverage": random.uniform(0.6, 0.85),
        "efficiency": {"weight_per_yuan": 0.04, "cost_per_combination": 2.0},
        "strategy_comparison": [
            {"name": "贪心覆盖", "score": 0.72, "cost": 18, "combinations": 9,
             "weight_coverage": 0.68, "top10_covered": 7},
            {"name": "分层覆盖", "score": 0.78, "cost": 20, "combinations": 10,
             "weight_coverage": 0.75, "top10_covered": 8},
            {"name": "ILP精确解", "score": 0.81, "cost": 18, "combinations": 9,
             "weight_coverage": 0.79, "top10_covered": 9},
            {"name": "复式/胆拖", "score": 0.85, "cost": 18, "combinations": 9,
             "weight_coverage": 0.82, "top10_covered": 9},
        ],
    }

    return {
        "metadata": {
            "lottery_type": lottery_type,
            "total_numbers": num_range,
            "blue_range": blue_range,
            "total_draws": 2833 if lottery_type == "daletou" else 3413,
            "latest_period": "2026015",
        },
        "stage0": stage0,
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
    }


# ============================================================
# 数据加载函数
# ============================================================

@st.cache_data(ttl=300)
def run_pipeline(lt_key: str, budget: int) -> dict:
    """运行方法链并缓存结果"""
    pipe = Pipeline(lt_key, budget=budget)
    return pipe.run()


def load_saved_report(lt_key: str) -> dict:
    """加载已保存的报告"""
    report_path = PROJECT_ROOT / "data" / f"{lt_key}_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.title("🎯 分析设置")

    lottery_type = st.radio("彩种选择", ["大乐透", "双色球"], index=0)
    lt_key = "daletou" if lottery_type == "大乐透" else "shuangseqiu"

    st.divider()
    data_source = st.radio("数据来源", ["实时运行方法链", "加载已保存报告", "Mock数据"], index=0)

    budget = st.slider("购买预算(元)", min_value=2, max_value=100, value=20, step=2)

    st.divider()

    if data_source == "实时运行方法链":
        if st.button("▶ 运行分析", type="primary", use_container_width=True):
            with st.spinner("方法链运行中..."):
                report = run_pipeline(lt_key, budget)
                st.session_state["report"] = report
                st.session_state["lt_key"] = lt_key
        report = st.session_state.get("report")
        if report and st.session_state.get("lt_key") == lt_key:
            st.subheader("运行状态")
            for i in range(5):
                st.markdown(f"✅ 阶段{i} 完成")
            timing = report.get("timing", {})
            if timing:
                st.caption(f"总耗时: {timing.get('total', 0):.2f}s")
        else:
            report = None
    elif data_source == "加载已保存报告":
        report = load_saved_report(lt_key)
        if report:
            st.subheader("运行状态")
            for i in range(5):
                st.markdown(f"✅ 阶段{i} 完成")
            st.caption("(从文件加载)")
        else:
            st.warning(f"未找到 {lt_key}_report.json")
    else:
        report = generate_mock_data(lt_key)
        st.subheader("运行状态")
        for i in range(5):
            st.markdown(f"✅ 阶段{i} 完成")
        st.caption("(Mock数据)")

    if report:
        st.divider()
        st.subheader("数据信息")
        meta = report.get("metadata", {})
        st.text(f"总期数: {meta.get('total_draws', '?')}")
        st.text(f"最新期: {meta.get('latest_period', '?')}")
        st.text(f"号码范围: 1-{meta.get('total_numbers', '?')}")


# ============================================================
# 主内容区
# ============================================================

st.title(f"🎯 {lottery_type}分析报告")

if not report:
    st.info("请在左侧选择数据来源并运行分析。")
    st.stop()

meta = report.get("metadata", {})
st.caption(f"数据期数: {meta.get('total_draws', '?')} | 最新期: {meta.get('latest_period', '?')}")

# 概览
render_overview(report)
render_funnel(report)

st.divider()

# 5个阶段 Tab
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📊 阶段0: 数据预处理",
    "🚫 阶段1: 排除引擎",
    "⚖️ 阶段2: 权重引擎",
    "🎲 阶段3: 组合生成",
    "🛒 阶段4: 购买方案",
])

with tab0:
    render_stage0(report["stage0"], meta)

with tab1:
    render_stage1(report["stage1"], meta)

with tab2:
    render_stage2(report["stage2"], meta)

with tab3:
    render_stage3(report["stage3"], meta)

with tab4:
    render_stage4(report["stage4"], meta)
