# -*- coding: utf-8 -*-
"""阶段1视图：排除引擎"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from .ball_display import render_ball_row


ALGO_NAMES = [
    "连续重复", "遗漏值异常", "极端组合",
    "马尔可夫", "周期性", "聚类异常",
]

ALGO_WEIGHTS = {
    "连续重复": 0.15, "遗漏值异常": 0.20, "极端组合": 0.25,
    "马尔可夫": 0.15, "周期性": 0.10, "聚类异常": 0.15,
}


def render_stage1(stage1_data, meta):
    """渲染阶段1的全部内容"""
    total = meta.get("total_numbers", 35)
    excluded = set(stage1_data.get("excluded_numbers", []))
    remaining = set(range(1, total + 1)) - excluded

    # 排除统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("排除号码", f"{len(excluded)}个")
    with col2:
        st.metric("保留号码", f"{len(remaining)}个")
    with col3:
        rate = len(excluded) / total if total else 0
        st.metric("排除率", f"{rate:.0%}")

    # 号码球全景
    st.subheader("号码排除全景")
    all_numbers = list(range(1, total + 1))
    html = render_ball_row(all_numbers, "red", excluded_set=excluded)
    st.markdown(html, unsafe_allow_html=True)
    st.caption("红色=保留，灰色删除线=排除")

    # 6算法投票热力图
    st.subheader("算法投票详情")
    details = stage1_data.get("exclusion_details", {})
    _render_vote_heatmap(details, excluded, total)

    # 排除明细表
    st.subheader("排除明细")
    _render_exclusion_table(details, excluded)

    # 算法权重饼图
    st.subheader("算法融合权重")
    _render_weight_pie()


def _render_vote_heatmap(details, excluded, total):
    """6算法投票热力图：行=被排除号码，列=算法"""
    if not details:
        st.info("暂无投票详情数据")
        return

    excluded_sorted = sorted(excluded)
    z_data = []
    for n in excluded_sorted:
        d = details.get(str(n), {})
        reasons = d.get("reasons", {})
        row = [reasons.get(algo, 0) for algo in ALGO_NAMES]
        z_data.append(row)

    if not z_data:
        return

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=ALGO_NAMES,
        y=[f"{n:02d}" for n in excluded_sorted],
        colorscale='OrRd',
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z_data],
        texttemplate="%{text}",
    ))
    fig.update_layout(
        height=max(300, len(excluded_sorted) * 28),
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis_title="排除算法",
        yaxis_title="号码",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_exclusion_table(details, excluded):
    """排除明细表"""
    rows = []
    for n in sorted(excluded):
        d = details.get(str(n), {})
        reasons = d.get("reasons", {})
        row = {"号码": f"{n:02d}", "综合置信度": d.get("confidence", 0)}
        for algo in ALGO_NAMES:
            row[algo] = reasons.get(algo, 0)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.background_gradient(subset=["综合置信度"], cmap='OrRd'),
            use_container_width=True, hide_index=True,
        )


def _render_weight_pie():
    """算法融合权重饼图"""
    fig = go.Figure(go.Pie(
        labels=list(ALGO_WEIGHTS.keys()),
        values=list(ALGO_WEIGHTS.values()),
        hole=0.4,
        marker=dict(colors=['#e17055', '#fdcb6e', '#e63946',
                            '#0984e3', '#00b894', '#6c5ce7']),
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
