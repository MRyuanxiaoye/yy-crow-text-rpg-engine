# -*- coding: utf-8 -*-
"""阶段3视图：组合生成器"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from .ball_display import show_combination


STRATEGY_NAMES = ["加权随机采样", "贪心构造", "遗传算法", "MCTS"]


def render_stage3(stage3_data, meta):
    """渲染阶段3的全部内容"""
    candidates = stage3_data.get("candidates", [])
    stats = stage3_data.get("generation_stats", {})

    if not candidates:
        st.info("暂无候选组合数据")
        return

    # 策略贡献统计
    st.subheader("生成策略贡献")
    _render_strategy_stats(stats)

    # 评分分布
    st.subheader("候选组合评分分布")
    _render_score_distribution(candidates)

    # Top10 组合展示
    st.subheader("Top10 候选组合")
    _render_top_combinations(candidates[:10])

    # 评分明细对比
    st.subheader("Top10 评分分解")
    _render_score_breakdown(candidates[:10])

    # 组合特征统计
    st.subheader("组合特征统计")
    _render_combo_stats(candidates)

    # 完整列表
    st.subheader("全部候选组合")
    _render_full_table(candidates)


def _render_strategy_stats(stats):
    """4种策略各贡献了多少组合"""
    cols = st.columns(4)
    for i, name in enumerate(STRATEGY_NAMES):
        with cols[i]:
            count = stats.get(f"strategy_{i+1}", 0)
            st.metric(name, f"{count}组")


def _render_score_distribution(candidates):
    """评分分布直方图"""
    scores = [c.get("score", 0) for c in candidates]
    fig = go.Figure(go.Histogram(
        x=scores, nbinsx=25,
        marker_color='#0984e3',
    ))
    fig.update_layout(
        xaxis_title="综合评分", yaxis_title="组合数",
        height=300, margin=dict(l=40, r=20, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_top_combinations(top_candidates):
    """Top10 组合用号码球展示"""
    for i, c in enumerate(top_candidates):
        red = c.get("red_balls", [])
        blue = c.get("blue_balls", c.get("blue_ball", []))
        if isinstance(blue, int):
            blue = [blue]
        score = c.get("score", 0)
        col1, col2 = st.columns([1, 9])
        with col1:
            st.markdown(f"**#{i+1}**")
        with col2:
            show_combination(red, blue, score)


def _render_score_breakdown(top_candidates):
    """Top10 评分分解柱状图"""
    if not top_candidates:
        return

    labels = [f"#{i+1}" for i in range(len(top_candidates))]
    score_keys = ["weight", "constraint", "balance", "similarity"]
    score_labels = ["权重得分", "约束满足", "均衡性", "相似度"]
    colors = ['#e17055', '#00b894', '#0984e3', '#6c5ce7']

    fig = go.Figure()
    for j, (key, label) in enumerate(zip(score_keys, score_labels)):
        vals = []
        for c in top_candidates:
            bd = c.get("score_breakdown", {})
            vals.append(bd.get(key, 0))
        fig.add_trace(go.Bar(name=label, x=labels, y=vals, marker_color=colors[j]))

    fig.update_layout(
        barmode='group',
        xaxis_title="组合排名", yaxis_title="分项得分",
        height=350, margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_combo_stats(candidates):
    """组合特征箱线图：奇偶比/和值/跨度/AC值"""
    sums, spans, acs = [], [], []
    for c in candidates:
        red = c.get("red_balls", [])
        if red:
            sums.append(sum(red))
            spans.append(max(red) - min(red))
        acs.append(c.get("ac_value", 0))

    col1, col2 = st.columns(2)
    with col1:
        if sums:
            fig = go.Figure(go.Box(y=sums, name="和值", marker_color='#0984e3'))
            fig.update_layout(height=280, margin=dict(l=40, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if spans:
            fig = go.Figure(go.Box(y=spans, name="跨度", marker_color='#00b894'))
            fig.update_layout(height=280, margin=dict(l=40, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


def _render_full_table(candidates):
    """全部候选组合表"""
    rows = []
    for i, c in enumerate(candidates):
        red = c.get("red_balls", [])
        blue = c.get("blue_balls", c.get("blue_ball", []))
        if isinstance(blue, int):
            blue = [blue]
        rows.append({
            "排名": i + 1,
            "红球": " ".join(f"{n:02d}" for n in red),
            "蓝球": " ".join(f"{n:02d}" for n in blue),
            "评分": c.get("score", 0),
            "和值": sum(red),
            "跨度": max(red) - min(red) if red else 0,
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(subset=["评分"], cmap='YlGn'),
        use_container_width=True, hide_index=True,
    )
