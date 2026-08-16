import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from engine.ui_theme import inject_premium_css, premium_divider

st.set_page_config(page_title="AI Project Mentor", page_icon="🎓", layout="wide")
inject_premium_css()
st.title("🎓 Offline AI Project Mentor")
premium_divider()
st.caption("Static, curated guides — no API calls, ever.")

kb_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "topics.json")
with open(kb_path, "r") as f:
    topics = json.load(f)

TOPIC_ICONS = {
    "NLP": "💬", "SLM": "🤏", "Agentic AI": "🤖", "Generative AI": "🎨",
    "RAG": "📚", "Prompt Engineering": "✍️",
}

# ---------------------------------------------------------------------------
# Prominent, clearly-visible topic selector
# ---------------------------------------------------------------------------
st.markdown('<span class="premium-badge">🎯 Choose a topic</span>', unsafe_allow_html=True)
topic_labels = [f"{TOPIC_ICONS.get(t, '📌')}  {t}" for t in topics.keys()]
label_to_topic = dict(zip(topic_labels, topics.keys()))
chosen_label = st.selectbox(
    "Choose a topic",
    topic_labels,
    label_visibility="collapsed",
    key="mentor_topic_select",
)
topic = label_to_topic[chosen_label]
data = topics[topic]

st.markdown(
    f'<div style="padding:16px 20px;border-radius:14px;'
    f'background:linear-gradient(135deg, rgba(79,70,229,0.10), rgba(219,39,119,0.08));'
    f'border:1px solid rgba(99,102,241,0.25);margin:6px 0 18px 0;">'
    f'<span style="font-size:26px;">{TOPIC_ICONS.get(topic, "📌")}</span>'
    f'&nbsp;&nbsp;<span style="font-size:22px;font-weight:800;">{topic}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.write(data["overview"])

if data.get("detailed_explanation"):
    st.markdown('<span class="premium-badge">📘 Deep dive</span>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="padding:14px 18px;border-radius:12px;background:rgba(255,255,255,0.75);'
        f'border:1px solid rgba(99,102,241,0.15);line-height:1.6;">{data["detailed_explanation"]}</div>',
        unsafe_allow_html=True,
    )

premium_divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader("📦 Required libraries")
    for lib in data["required_libraries"]:
        st.markdown(f"- `{lib}`")

    st.subheader("📁 Folder structure")
    st.code(data["folder_structure"], language="text")

with c2:
    st.subheader("🗂️ Dataset suggestions")
    for ds in data["dataset_suggestions"]:
        st.markdown(f"- {ds}")

    st.subheader("⚠️ Common errors")
    for err in data["common_errors"]:
        st.markdown(f"- {err}")

premium_divider()

# ---------------------------------------------------------------------------
# "Things to do" — phased roadmap when available (NLP, SLM, Agentic AI,
# Generative AI), falling back to the flat step list otherwise.
# ---------------------------------------------------------------------------
if data.get("roadmap"):
    st.markdown('<span class="premium-badge">🪜 Things to do — phased roadmap</span>',
                unsafe_allow_html=True)
    st.subheader("Step-by-step, phase by phase")
    for i, phase in enumerate(data["roadmap"]):
        with st.expander(f"**{phase['phase']}**", expanded=(i == 0)):
            for task in phase["tasks"]:
                st.markdown(f"- {task}")
else:
    st.subheader("🪜 Step-by-step guide")
    for i, step in enumerate(data["steps"], start=1):
        st.markdown(f"{i}. {step}")
