# -*- coding: utf-8 -*-
"""阶段2视图：权重引擎"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from .ball_display import render_ball_row


WEIGHT_ALGOS = [
    "频率回归", "遗漏值回归", "时序衰减",
    "马尔可夫", "共现关联", "深度学习",
]


def render_stage2(stage2_data, meta):
    """渲染阶段2的全部内容"""
    weights = stage2_data.get("number_weights", {})
    details = stage2_data.get("weight_details", {})

    if not weights:
        st.info("暂无权重数据")
        return

    # 号码球展示（Top5 金色边框）
    sorted_nums = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    top5 = set(int(n) for n in sorted_nums[:5])
    all_nums = sorted(int(n) for n in weights.keys())

    st.subheader("候选号码权重概览")
    html = render_ball_row(all_nums, "red", hot_set=top5)
    st.markdown(html, unsafe_allow_html=True)
    st.caption("金色边框 = Top5 高权重号码")

    # 权重分布柱状图
    st.subheader("权重分布")
    _render_weight_bar(weights)

    # 6算法贡献堆叠图
    st.subheader("算法贡献分解")
    _render_stacked_bar(details)

    # Top10 雷达图
    st.subheader("Top10 号码算法维度")
    _render_radar(details, sorted_nums[:10])

    # 权重明细表
    st.subheader("权重明细")
    _render_weight_table(weights, details)


def _render_weight_bar(weights):
    """权重分布柱状图"""
    nums = sorted(int(n) for n in weights.keys())
    vals = [weights[str(n)] for n in nums]
    max_w = max(vals) if vals else 1

    colors = [
        f'rgb({int(255 * v / max_w)}, {int(100 + 100 * (1 - v / max_w))}, 80)'
        for v in vals
    ]

    fig = go.Figure(go.Bar(
        x=[f"{n:02d}" for n in nums], y=vals,
        marker_color=colors,
    ))
    fig.update_layout(
        xaxis_title="号码", yaxis_title="综合权重",
        height=350, margin=dict(l=40, r=20, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_stacked_bar(details):
    """6算法贡献堆叠柱状图"""
    if not details:
        st.info("暂无算法明细数据")
        return

    nums = sorted(int(n) for n in details.keys())
    x_labels = [f"{n:02d}" for n in nums]

    fig = go.Figure()
    algo_colors = ['#e17055', '#fdcb6e', '#00b894', '#0984e3', '#6c5ce7', '#d63031']
    for i, algo in enumerate(WEIGHT_ALGOS):
        vals = []
        for n in nums:
            bd = details.get(str(n), {}).get("breakdown", {})
            vals.append(bd.get(algo, 0))
        fig.add_trace(go.Bar(
            name=algo, x=x_labels, y=vals,
            marker_color=algo_colors[i % len(algo_colors)],
        ))

    fig.update_layout(
        barmode='stack',
        xaxis_title="号码", yaxis_title="权重贡献",
        height=400, margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_radar(details, top_nums):
    """Top10 号码雷达图"""
    if not details or not top_nums:
        return

    fig = go.Figure()
    for n_str in top_nums:
        bd = details.get(n_str, {}).get("breakdown", {})
        values = [bd.get(algo, 0) for algo in WEIGHT_ALGOS]
        values.append(values[0])  # 闭合
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=WEIGHT_ALGOS + [WEIGHT_ALGOS[0]],
            name=f"号码{n_str}",
            fill='toself', opacity=0.3,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=450, margin=dict(l=60, r=60, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_weight_table(weights, details):
    """权重明细表"""
    rows = []
    for n_str in sorted(weights.keys(), key=lambda k: weights[k], reverse=True):
        row = {"号码": f"{int(n_str):02d}", "综合权重": weights[n_str]}
        bd = details.get(n_str, {}).get("breakdown", {})
        for algo in WEIGHT_ALGOS:
            row[algo] = bd.get(algo, 0)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.background_gradient(subset=["综合权重"], cmap='YlGn'),
            use_container_width=True, hide_index=True,
        )
