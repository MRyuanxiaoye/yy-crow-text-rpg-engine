import streamlit as st
import asyncio
import time
from src.graph import app as graph_app
from src.tools.explainer import TermExplainer

st.set_page_config(page_title="Deep Search Agent", layout="wide")

st.title("🧐 Deep Search Agent (LangGraph + DeepSeek)")
st.caption("资深中国文化研究者 | 深度递归搜索 | 知识推理")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "explanations" not in st.session_state:
    st.session_state.explanations = {}

# Initialize Explainer
if "explainer" not in st.session_state:
    st.session_state.explainer = TermExplainer()

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display saved keywords/explanations for historical messages if stored
        # (This is a simplified implementation, history state management is complex in Streamlit)

query = st.chat_input("请输入您想深度研究的课题（例如：详解中国道教符箓体系）")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status_container = st.status("🧠 正在启动深度研究工作流...", expanded=True)
        logs_placeholder = status_container.empty()
        report_placeholder = st.empty()
        
        # Run Graph with Streaming
        try:
            logs = []
            final_report = ""
            keywords = []
            
            # Use LangGraph's stream method
            for event in graph_app.stream({"query": query}):
                # Event is a dict like {'node_name': {'key': 'value'}}
                for node_name, state_update in event.items():
                    
                    # Update Logs
                    if "logs" in state_update:
                        new_logs = state_update["logs"]
                        if new_logs:
                            latest_log = new_logs[-1]
                            status_container.update(label=latest_log)
                            logs.extend(new_logs)
                            logs_text = "\n\n".join([f"- {l}" for l in logs])
                            logs_placeholder.markdown(logs_text)
                    
                    # Update Report Draft (if any)
                    if "final_report" in state_update:
                        final_report = state_update["final_report"]
                        
                    if "keywords" in state_update:
                        keywords = state_update["keywords"]
            
            status_container.update(label="✅ 深度研究完成", state="complete", expanded=False)
            
            if final_report:
                report_placeholder.markdown(final_report)
                
                # Keywords Drill-down UI
                if keywords:
                    st.divider()
                    st.caption("🔍 **知识下钻** (点击关键词获取快速解释)")
                    cols = st.columns(min(len(keywords), 5))
                    for i, term in enumerate(keywords):
                        # Use a unique key for each button to avoid conflicts
                        if cols[i % 5].button(term, key=f"btn_{term}_{int(time.time())}"):
                            st.session_state.selected_term = term
                
                # Check if a term was selected (via session state trigger)
                if "selected_term" in st.session_state:
                    term = st.session_state.selected_term
                    if term not in st.session_state.explanations:
                        with st.spinner(f"正在查询 '{term}' 的定义..."):
                            exp = st.session_state.explainer.explain(term)
                            st.session_state.explanations[term] = exp
                    
                    st.info(f"**{term}**: {st.session_state.explanations[term]}")
                    # Clear selection to avoid sticky state on next run? 
                    # No, keep it visible until next interaction.
                    del st.session_state.selected_term

                # UI Optimization: Clean copy/download button
                st.divider()
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.download_button(
                        label="📥 下载 Markdown",
                        data=final_report,
                        file_name=f"deep_search_report_{int(time.time())}.md",
                        mime="text/markdown"
                    )
                
                st.session_state.messages.append({"role": "assistant", "content": final_report})
            else:
                st.error("未能生成报告")

        except Exception as e:
            st.error(f"运行出错: {e}")
