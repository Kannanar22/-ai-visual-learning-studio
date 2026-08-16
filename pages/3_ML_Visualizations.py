import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from engine.ml_predict_explainer import detect_algorithms, get_explanation, ALGO_KB
from engine.ui_theme import inject_premium_css, premium_divider

st.set_page_config(page_title="ML Code Paster", page_icon="🧪", layout="wide")
inject_premium_css()
st.title("🧪 ML Code Paster — How Does It Predict?")
st.caption("Paste your scikit-learn model code and get a step-by-step, offline explanation "
           "of exactly how that algorithm turns a new input into a prediction. Parsed locally "
           "with `ast` — no AI API involved.")
premium_divider()

SAMPLE_CODE = """from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_new)
"""

st.markdown('<span class="premium-badge">📋 Paste your ML code</span>', unsafe_allow_html=True)
code = st.text_area(
    "Paste your ML code here:",
    value=st.session_state.get("ml_code", SAMPLE_CODE),
    height=220,
    key="ml_code",
    label_visibility="collapsed",
)

with st.expander(f"📚 Supported algorithms ({len(ALGO_KB)})"):
    families = {}
    for name, info in ALGO_KB.items():
        families.setdefault(info["family"], []).append(name)
    for fam, names in families.items():
        st.markdown(f"**{fam}:** " + ", ".join(f"`{n}`" for n in names))

explain_clicked = st.button("🔍 Explain how it predicts", type="primary")

if explain_clicked:
    st.session_state["ml_detected"] = detect_algorithms(code)

if "ml_detected" in st.session_state:
    detected = st.session_state["ml_detected"]

    if not detected:
        st.warning(
            "No supported estimator constructor was found in that code. Make sure you've "
            "pasted a line like `RandomForestClassifier(...)`, `LogisticRegression(...)`, "
            "`KMeans(...)`, etc. Expand **Supported algorithms** above for the full list."
        )
    else:
        st.success(f"Found {len(detected)} model instantiation(s) in your code.")

        for idx, hit in enumerate(detected):
            info = get_explanation(hit["class_name"])
            if info is None:
                continue

            var_label = f"`{hit['var_name']}` = " if hit.get("var_name") else ""
            header = f"{idx + 1}. {var_label}**{hit['class_name']}** — {info['family']}"

            with st.expander(header, expanded=(idx == 0)):
                st.markdown(f"**Core idea:** {info['core_idea']}")

                st.markdown('<span class="premium-badge">🧮 Prediction formula</span>',
                            unsafe_allow_html=True)
                st.code(info["formula"], language="text")

                st.markdown('<span class="premium-badge">🪜 What happens when `.predict()` runs</span>',
                            unsafe_allow_html=True)
                for i, step in enumerate(info["predict_steps"], start=1):
                    st.markdown(f"{i}. {step}")

                if hit.get("params"):
                    st.markdown('<span class="premium-badge">⚙️ Hyperparameters found in your code</span>',
                                unsafe_allow_html=True)
                    for pname, pval in hit["params"].items():
                        desc = info["key_hyperparams"].get(pname)
                        if desc:
                            st.markdown(f"- `{pname} = {pval}` — {desc}")
                        else:
                            st.markdown(f"- `{pname} = {pval}`")

                relevant_defaults = {
                    k: v for k, v in info["key_hyperparams"].items()
                    if k not in hit.get("params", {})
                }
                if relevant_defaults:
                    st.markdown('<span class="premium-badge">🔧 Other tunable hyperparameters</span>',
                                unsafe_allow_html=True)
                    for pname, desc in relevant_defaults.items():
                        st.markdown(f"- `{pname}` — {desc}")

                st.info(f"💡 **Note:** {info['notes']}")
else:
    st.info("Paste your ML code above and click **Explain how it predicts** to get started.")

st.divider()
st.caption("Everything above runs entirely on your machine via static, curated explanations — "
           "no external AI service is called and no code is executed.")
