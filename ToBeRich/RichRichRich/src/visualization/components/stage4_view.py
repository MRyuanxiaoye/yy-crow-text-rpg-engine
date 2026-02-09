# -*- coding: utf-8 -*-
"""阶段4视图：购买优化器"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from .ball_display import render_ball_row, show_combination


def render_stage4(stage4_data, meta):
    """渲染阶段4的全部内容"""
    tickets = stage4_data.get("tickets", [])
    budget = stage4_data.get("budget", 0)
    total_cost = stage4_data.get("total_cost", 0)
    coverage = stage4_data.get("coverage", {})

    if not tickets:
        st.info("暂无购买方案数据")
        return

    # 预算使用仪表盘 + 性价比指标
    st.subheader("预算概览")
    _render_budget_gauge(budget, total_cost, stage4_data)

    # 购买方案卡片
    st.subheader("购买方案")
    _render_tickets(tickets)

    # 覆盖分析
    st.subheader("号码覆盖分析")
    _render_coverage(coverage, meta)

    # 策略对比
    st.subheader("优化策略对比")
    _render_strategy_compare(stage4_data)


def _render_budget_gauge(budget, total_cost, data):
    """预算仪表盘 + 性价比指标"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_cost,
            title={'text': "已用预算(元)"},
            gauge={
                'axis': {'range': [0, budget]},
                'bar': {'color': '#6c5ce7'},
                'steps': [
                    {'range': [0, budget * 0.7], 'color': '#dfe6e9'},
                    {'range': [budget * 0.7, budget], 'color': '#ffeaa7'},
                ],
            },
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        combos = data.get("total_combinations", 0)
        st.metric("总注数", f"{combos}注")

    with col3:
        efficiency = data.get("efficiency", {})
        wpyuan = efficiency.get("weight_per_yuan", 0)
        st.metric("每元覆盖权重", f"{wpyuan:.3f}")

    with col4:
        coverage = data.get("coverage", {})
        top10 = coverage.get("top10_red_covered", 0)
        st.metric("Top10覆盖", f"{top10}个")


def _render_tickets(tickets):
    """购买方案卡片展示"""
    for i, ticket in enumerate(tickets):
        t_type = ticket.get("type", "单式")
        cost = ticket.get("cost", 2)
        combos = ticket.get("combinations", 1)

        with st.container():
            st.markdown(
                f'<div class="ticket-card">'
                f'<div class="ticket-card-header">'
                f'第{i+1}张 | {t_type} | {combos}注 | {cost}元'
                f'</div>',
                unsafe_allow_html=True,
            )

            if t_type == "胆拖":
                dan_red = ticket.get("dan_red", [])
                tuo_red = ticket.get("tuo_red", [])
                dan_blue = ticket.get("dan_blue", [])
                tuo_blue = ticket.get("tuo_blue", [])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**胆码（必选）**")
                    html = render_ball_row(dan_red, "red", hot_set=set(dan_red))
                    if dan_blue:
                        html += " + " + render_ball_row(dan_blue, "blue", hot_set=set(dan_blue))
                    st.markdown(html, unsafe_allow_html=True)
                with col2:
                    st.markdown("**拖码（备选）**")
                    html = render_ball_row(tuo_red, "red")
                    if tuo_blue:
                        html += " + " + render_ball_row(tuo_blue, "blue")
                    st.markdown(html, unsafe_allow_html=True)
            else:
                red = ticket.get("red_balls", [])
                blue = ticket.get("blue_balls", ticket.get("blue_ball", []))
                if isinstance(blue, int):
                    blue = [blue]
                show_combination(red, blue)

            st.markdown('</div>', unsafe_allow_html=True)


def _render_coverage(coverage, meta):
    """号码覆盖分析：全部号码球，被覆盖的高亮"""
    total = meta.get("total_numbers", 35)
    all_nums = list(range(1, total + 1))
    covered_red = set(coverage.get("all_red_numbers", []))
    covered_blue = set(coverage.get("all_blue_numbers", []))
    uncovered_red = set(all_nums) - covered_red

    st.markdown("**前区/红球覆盖**")
    html = render_ball_row(all_nums, "red", excluded_set=uncovered_red, hot_set=covered_red)
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"覆盖 {len(covered_red)}/{total} 个红球")

    blue_range = meta.get("blue_range", 12)
    all_blue = list(range(1, blue_range + 1))
    uncovered_blue = set(all_blue) - covered_blue

    st.markdown("**后区/蓝球覆盖**")
    html = render_ball_row(all_blue, "blue", excluded_set=uncovered_blue, hot_set=covered_blue)
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"覆盖 {len(covered_blue)}/{blue_range} 个蓝球")


def _render_strategy_compare(data):
    """4种优化策略对比表"""
    strategies = data.get("strategy_comparison", [])
    if not strategies:
        st.info("暂无策略对比数据")
        return

    rows = []
    for s in strategies:
        rows.append({
            "策略": s.get("name", ""),
            "综合评分": s.get("score", 0),
            "总花费": s.get("cost", 0),
            "总注数": s.get("combinations", 0),
            "权重覆盖": s.get("weight_coverage", 0),
            "Top10覆盖": s.get("top10_covered", 0),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(subset=["综合评分"], cmap='YlGn'),
        use_container_width=True, hide_index=True,
    )
