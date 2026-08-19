import json
from pathlib import Path

import streamlit as st

from engine.ui_theme import inject_premium_css, premium_divider
from engine.topic_viva import generate_topic_viva_questions, supported_topics

st.set_page_config(page_title="AI Project Mentor", page_icon="🎓", layout="wide")
inject_premium_css()

st.title("🎓 AI Project Mentor")
premium_divider()
st.caption("Static, offline guides + an offline viva-question generator for your own code — no external AI APIs.")

# ---------------------------------------------------------------------------
# Load the static knowledge base
# ---------------------------------------------------------------------------
KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "topics.json"


@st.cache_data
def load_topics() -> dict:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


TOPICS = load_topics()

# Order tabs so the four topics with a viva-question generator lead, matching
# what the app promises on the home page, then RAG / Prompt Engineering.
TAB_ORDER = ["NLP", "SLM", "Generative AI", "Agentic AI", "RAG", "Prompt Engineering"]
TAB_ORDER = [t for t in TAB_ORDER if t in TOPICS] + [t for t in TOPICS if t not in TAB_ORDER]

VIVA_ENABLED_TOPICS = set(supported_topics())

tabs = st.tabs([f"📌 {t}" for t in TAB_ORDER])

for tab, topic in zip(tabs, TAB_ORDER):
    with tab:
        data = TOPICS[topic]

        st.markdown(f'<span class="premium-badge">AI PROJECT MENTOR</span>', unsafe_allow_html=True)
        st.subheader(topic)
        st.write(data.get("overview", ""))

        if data.get("detailed_explanation"):
            with st.expander("📖 Detailed explanation", expanded=False):
                st.write(data["detailed_explanation"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📚 Required libraries**")
            for lib in data.get("required_libraries", []):
                st.markdown(f"- `{lib}`")

            st.markdown("**🗂️ Suggested folder structure**")
            st.code(data.get("folder_structure", ""), language="text")

        with col2:
            st.markdown("**🎯 Dataset suggestions**")
            for ds in data.get("dataset_suggestions", []):
                st.markdown(f"- {ds}")

            st.markdown("**⚠️ Common errors to avoid**")
            for err in data.get("common_errors", []):
                st.markdown(f"- {err}")

        st.markdown("**🪜 High-level steps**")
        for i, step in enumerate(data.get("steps", []), start=1):
            st.markdown(f"{i}. {step}")

        # --- Phased roadmap ---------------------------------------------------
        if data.get("roadmap"):
            st.markdown("### 🗺️ Things-to-do roadmap")
            for phase in data["roadmap"]:
                with st.expander(phase["phase"], expanded=False):
                    for task in phase.get("tasks", []):
                        st.markdown(f"- {task}")

        premium_divider()

        # --- Paste-code viva question generator --------------------------------
        if topic in VIVA_ENABLED_TOPICS:
            st.markdown(f"### 🎤 Generate {topic} Viva Questions From Your Code")
            st.caption(
                f"Paste a {topic} code snippet below (preprocessing, model, prompt, or agent code). "
                "This is analyzed entirely offline with keyword/AST rules — no AI API is called."
            )

            code_key = f"viva_code_{topic}"
            code = st.text_area(
                f"Paste your {topic} code here",
                height=220,
                key=code_key,
                placeholder="# e.g. paste your tokenizer/vectorizer, model, prompt, or agent-loop code",
            )

            if st.button(f"Generate viva questions", key=f"viva_btn_{topic}"):
                if not code.strip():
                    st.warning("Paste some code first.")
                else:
                    result = generate_topic_viva_questions(topic, code)

                    if result["note"]:
                        st.info(result["note"])

                    if result["matched_rules"]:
                        st.markdown(f"**Found {len(result['matched_rules'])} concept area(s) in your code:**")
                        for j, rule in enumerate(result["matched_rules"], start=1):
                            with st.container(border=True):
                                st.markdown(
                                    f"**{j}. Detected: `{rule['keyword_hit']}`** "
                                    f"<span style='color:#6B7280;font-size:0.85em'>({rule['category']})</span>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown("**Questions an examiner might ask:**")
                                for q in rule["questions"]:
                                    st.markdown(f"- {q}")
                                if rule["answer_points"]:
                                    with st.expander("💡 Expected answer points"):
                                        for a in rule["answer_points"]:
                                            st.markdown(f"- {a}")

                    if result["imports_detected"]:
                        st.markdown(
                            "**Libraries imported:** " +
                            ", ".join(f"`{m}`" for m in result["imports_detected"])
                        )

                    if result["conceptual_questions"]:
                        st.markdown("### 🧠 Follow-up conceptual questions")
                        st.caption(f"General {topic} questions an examiner typically asks after the code walkthrough.")
                        for q in result["conceptual_questions"]:
                            st.markdown(f"- {q}")
        else:
            st.caption(
                f"A paste-and-generate viva-question tool isn't wired up for {topic} yet — "
                "use the roadmap above, or try NLP / SLM / Generative AI / Agentic AI / RAG / "
                "Prompt Engineering for the code-based generator."
            )
