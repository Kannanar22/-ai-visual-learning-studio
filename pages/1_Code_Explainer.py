import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from engine.viva_engine import generate_viva_questions
from engine.ui_theme import inject_premium_css, premium_divider

st.set_page_config(page_title="Viva Question Generator", page_icon="📖", layout="wide")
inject_premium_css()
st.title("📖 Smart Code Explanation Engine — Viva Question Generator")
premium_divider()
st.caption("Paste your code and get the questions an examiner is likely to ask about it — "
           "generated locally with `ast`, from a rule-based question bank. No AI API involved.")

default_code = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")
X = df[["hours"]]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
"""

code = st.text_area("Paste your Python code:", value=default_code, height=280)

if st.button("Generate viva questions", type="primary"):
    result = generate_viva_questions(code)
    st.session_state["viva_result"] = result

if "viva_result" in st.session_state:
    result = st.session_state["viva_result"]

    st.subheader("🎤 Line-by-line viva questions")
    st.caption("Click each line to see the questions likely to be asked about it, "
               "and the key points an examiner would expect in your answer.")

    all_questions_text = ["VIVA PREPARATION SHEET", "=" * 40, ""]

    for item in result["line_questions"]:
        with st.expander(f"Line {item['line']}: `{item['code']}`"):
            st.markdown("**Possible questions:**")
            for q in item["questions"]:
                st.markdown(f"- {q}")
            if item["answer_points"]:
                with st.popover("💡 Expected answer points"):
                    for a in item["answer_points"]:
                        st.markdown(f"- {a}")
            st.caption(f"Category: `{item['category']}`")

        all_questions_text.append(f"Line {item['line']}: {item['code']}")
        for q in item["questions"]:
            all_questions_text.append(f"  Q: {q}")
        if item["answer_points"]:
            for a in item["answer_points"]:
                all_questions_text.append(f"     - {a}")
        all_questions_text.append("")

    if result["conceptual_questions"]:
        st.subheader("🧠 Conceptual follow-up questions")
        st.caption("These probe deeper understanding of the concepts your code touches on.")
        for q in result["conceptual_questions"]:
            st.markdown(f"- {q}")

        all_questions_text.append("CONCEPTUAL FOLLOW-UP QUESTIONS")
        all_questions_text.append("-" * 40)
        for q in result["conceptual_questions"]:
            all_questions_text.append(f"- {q}")

    st.download_button(
        "⬇️ Download viva question sheet (.txt)",
        data="\n".join(all_questions_text),
        file_name="viva_questions.txt",
        mime="text/plain",
    )

else:
    st.info("Paste your code above and click **Generate viva questions** to get started.")
