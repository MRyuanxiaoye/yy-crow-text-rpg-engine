# -*- coding: utf-8 -*-
"""阶段0视图：数据预处理与特征工程"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_stage0(stage0_data, meta):
    """渲染阶段0的全部内容"""
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(
        ["基础特征", "组合特征", "时序特征", "关联特征"]
    )

    with sub_tab1:
        _render_basic_features(stage0_data, meta)
    with sub_tab2:
        _render_combo_features(stage0_data)
    with sub_tab3:
        _render_time_features(stage0_data)
    with sub_tab4:
        _render_correlation_features(stage0_data)


def _render_basic_features(data, meta):
    """基础特征：频率柱状图 + 遗漏值热力图 + 特征表"""
    features = data.get("number_features", {})
    num_range = meta.get("total_numbers", 35)
    theoretical = 1.0 / num_range

    # 频率柱状图
    st.subheader("号码出现频率")
    numbers = list(range(1, num_range + 1))
    freqs = [features.get(str(n), {}).get("frequency", 0) for n in numbers]
    colors = ['#e63946' if f > theoretical else '#74b9ff' for f in freqs]

    fig = go.Figure(go.Bar(x=numbers, y=freqs, marker_color=colors))
    fig.add_hline(y=theoretical, line_dash="dash", line_color="gray",
                  annotation_text=f"理论概率 {theoretical:.4f}")
    fig.update_layout(
        xaxis_title="号码", yaxis_title="出现频率",
        height=350, margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 遗漏值热力图
    st.subheader("最近30期出现情况")
    appearance = data.get("appearance_matrix", [])
    if appearance:
        fig2 = go.Figure(go.Heatmap(
            z=appearance,
            x=[f"第{i}期" for i in range(1, len(appearance[0]) + 1)] if appearance else [],
            y=[f"{n:02d}" for n in numbers],
            colorscale=[[0, '#dfe6e9'], [1, '#e63946']],
            showscale=False,
        ))
        fig2.update_layout(
            height=max(400, num_range * 14),
            margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 特征明细表
    st.subheader("号码特征明细")
    rows = []
    for n in numbers:
        f = features.get(str(n), {})
        rows.append({
            "号码": f"{n:02d}",
            "频率": f.get("frequency", 0),
            "遗漏值": f.get("missing_value", 0),
            "平均间隔": f.get("avg_gap", 0),
            "最大间隔": f.get("max_gap", 0),
            "连续次数": f.get("consecutive", 0),
            "趋势": f.get("trend", "平稳"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_combo_features(data):
    """组合特征：奇偶比/大小比饼图 + 和值/AC值分布"""
    combo = data.get("combo_stats", {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("奇偶比分布")
        odd_even = combo.get("odd_even_dist", {})
        if odd_even:
            fig = go.Figure(go.Pie(
                labels=list(odd_even.keys()),
                values=list(odd_even.values()),
                hole=0.3,
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("大小比分布")
        big_small = combo.get("big_small_dist", {})
        if big_small:
            fig = go.Figure(go.Pie(
                labels=list(big_small.keys()),
                values=list(big_small.values()),
                hole=0.3,
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("和值分布")
        sums = combo.get("sum_values", [])
        if sums:
            fig = go.Figure(go.Histogram(x=sums, nbinsx=30, marker_color='#0984e3'))
            fig.update_layout(
                xaxis_title="和值", yaxis_title="频次",
                height=300, margin=dict(l=40, r=20, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("AC值分布")
        ac_values = combo.get("ac_values", [])
        if ac_values:
            fig = go.Figure(go.Histogram(x=ac_values, nbinsx=15, marker_color='#00b894'))
            fig.update_layout(
                xaxis_title="AC值", yaxis_title="频次",
                height=300, margin=dict(l=40, r=20, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_time_features(data):
    """时序特征：滑动窗口统计"""
    time_data = data.get("time_features", {})
    if not time_data:
        st.info("时序特征数据暂无")
        return

    st.subheader("号码出现趋势（近50期滑动窗口）")
    trend_data = time_data.get("trend_lines", {})
    if trend_data:
        # 展示 Top5 热号和 Top5 冷号的趋势线
        fig = go.Figure()
        for label, values in list(trend_data.items())[:10]:
            fig.add_trace(go.Scatter(
                y=values, name=f"号码{label}", mode='lines',
            ))
        fig.update_layout(
            xaxis_title="期数（近→远）", yaxis_title="滑动窗口出现次数",
            height=400, margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_correlation_features(data):
    """关联特征：共现矩阵 + 转移矩阵"""
    st.subheader("号码共现矩阵")
    co_matrix = data.get("correlation_matrix", [])
    if co_matrix:
        num_range = len(co_matrix)
        labels = [f"{n:02d}" for n in range(1, num_range + 1)]
        fig = go.Figure(go.Heatmap(
            z=co_matrix, x=labels, y=labels,
            colorscale='YlOrRd',
        ))
        fig.update_layout(
            height=max(500, num_range * 16),
            margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("马尔可夫转移矩阵")
    trans_matrix = data.get("transition_matrix", [])
    if trans_matrix:
        num_range = len(trans_matrix)
        labels = [f"{n:02d}" for n in range(1, num_range + 1)]
        fig = go.Figure(go.Heatmap(
            z=trans_matrix, x=labels, y=labels,
            colorscale='Blues',
        ))
        fig.update_layout(
            height=max(500, num_range * 16),
            margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
