# -*- coding: utf-8 -*-
"""顶部概览区：指标卡片 + 数据流漏斗图"""

import streamlit as st
import plotly.graph_objects as go


def render_overview(report_data):
    """渲染顶部概览区"""
    meta = report_data.get("metadata", {})
    stage1 = report_data.get("stage1", {})
    stage2 = report_data.get("stage2", {})
    stage3 = report_data.get("stage3", {})
    stage4 = report_data.get("stage4", {})

    total_numbers = meta.get("total_numbers", 35)
    excluded_count = len(stage1.get("excluded_numbers", []))
    remaining = total_numbers - excluded_count
    candidate_count = len(stage3.get("candidates", []))
    top_score = stage3.get("top_score", 0)
    ticket_count = len(stage4.get("tickets", []))
    total_cost = stage4.get("total_cost", 0)
    coverage_pct = stage4.get("coverage_pct", 0)
    weight_coverage = stage4.get("weight_coverage", 0)

    # 4列指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("候选号码池", f"{remaining}个", f"-{excluded_count}个(排除)")
    with col2:
        st.metric("候选组合", f"{candidate_count}组", f"Top: {top_score:.2f}")
    with col3:
        st.metric("购买方案", f"{ticket_count}张", f"花费: {total_cost}元")
    with col4:
        st.metric("号码覆盖", f"{coverage_pct:.0%}", f"权重: {weight_coverage:.0%}")


def render_funnel(report_data):
    """渲染数据流漏斗图"""
    meta = report_data.get("metadata", {})
    stage1 = report_data.get("stage1", {})
    stage2 = report_data.get("stage2", {})
    stage3 = report_data.get("stage3", {})
    stage4 = report_data.get("stage4", {})

    total = meta.get("total_numbers", 35)
    after_exclude = total - len(stage1.get("excluded_numbers", []))
    top_weight = stage2.get("top_count", after_exclude)
    candidates = len(stage3.get("candidates", []))
    final = stage4.get("total_combinations", 0)

    fig = go.Figure(go.Funnel(
        y=["全部号码 (阶段0)", "排除后 (阶段1)", "Top权重 (阶段2)",
           "候选组合 (阶段3)", "购买方案 (阶段4)"],
        x=[total, after_exclude, top_weight, candidates, final],
        textinfo="value+percent initial",
        marker=dict(color=["#e17055", "#fdcb6e", "#00b894", "#0984e3", "#6c5ce7"]),
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=10, b=10),
        font=dict(size=13),
    )
    st.plotly_chart(fig, use_container_width=True)
