# -*- coding: utf-8 -*-
"""号码球渲染组件 - 用 HTML/CSS 渲染彩票号码球"""

import streamlit as st
from pathlib import Path


def load_custom_css():
    """加载自定义 CSS 样式"""
    css_path = Path(__file__).parent.parent / "styles" / "custom.css"
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_ball(number, ball_type="red", excluded=False, hot=False, size="normal"):
    """渲染单个号码球的 HTML"""
    css_classes = ["ball"]
    if excluded:
        css_classes.append("ball-excluded")
    elif ball_type == "red":
        css_classes.append("ball-red")
    elif ball_type == "blue":
        css_classes.append("ball-blue")
    if hot:
        css_classes.append("ball-hot")
    if size == "small":
        css_classes.append("ball-sm")
    return f'<span class="{" ".join(css_classes)}">{number:02d}</span>'


def render_ball_row(numbers, ball_type="red", excluded_set=None, hot_set=None):
    """渲染一行号码球，返回 HTML 字符串"""
    excluded_set = excluded_set or set()
    hot_set = hot_set or set()
    parts = []
    for n in numbers:
        parts.append(render_ball(
            n, ball_type,
            excluded=(n in excluded_set),
            hot=(n in hot_set),
        ))
    return " ".join(parts)


def render_combination(red_balls, blue_balls, score=None):
    """渲染一组完整号码组合（红球 + 蓝球），可选附带评分"""
    red_html = render_ball_row(red_balls, "red")
    blue_html = render_ball_row(blue_balls, "blue")
    sep = '<span class="combo-separator">+</span>'
    html = f'<div class="combo-row">{red_html}{sep}{blue_html}'
    if score is not None:
        html += f'<span style="margin-left:12px;color:#636e72;">评分: {score:.3f}</span>'
    html += '</div>'
    return html


def show_ball_row(numbers, ball_type="red", excluded_set=None, hot_set=None):
    """直接在 Streamlit 中展示一行号码球"""
    html = render_ball_row(numbers, ball_type, excluded_set, hot_set)
    st.markdown(html, unsafe_allow_html=True)


def show_combination(red_balls, blue_balls, score=None):
    """直接在 Streamlit 中展示一组号码组合"""
    html = render_combination(red_balls, blue_balls, score)
    st.markdown(html, unsafe_allow_html=True)
