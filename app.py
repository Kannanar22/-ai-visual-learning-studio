import streamlit as st
from engine.ui_theme import inject_premium_css, premium_divider

st.set_page_config(
    page_title="AI Visual Learning Studio",
    page_icon="🧠",
    layout="wide",
)
inject_premium_css()

st.title("🧠 AI Visual Learning Studio")
premium_divider()
st.caption("A fully offline educational app — no external AI APIs, ever.")

st.markdown("""
Welcome! This app helps you visually understand AI algorithms and Python
code execution, entirely **offline**. Everything runs locally using Python's
`ast` module, rule-based logic, and static knowledge files.

### What's inside
- **📖 Viva Question Generator** — paste code and get the questions an examiner would likely ask about it, line by line, plus expected-answer points.
- **🧩 Flowchart Generator** — turn `if`/`for`/`while` logic into a Mermaid flowchart.
- **🧪 ML Code Paster** — paste your scikit-learn model code (Linear/Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, K-NN, K-Means, Naive Bayes, MLP...) and get a step-by-step, offline explanation of exactly how that algorithm turns a new input into a prediction.
- **🎓 AI Project Mentor** — static, offline guides for NLP, RAG, Agentic AI, Generative AI, Prompt Engineering, and SLMs, each with a phased "things to do" roadmap.
- **🧬 Deep Learning Studio** — pick a model demonstration (or paste your own Keras/PyTorch code — ANN, CNN, RNN, LSTM, GRU, BiLSTM) and see the input layer, hidden layers, and each layer's operation, drawn as a classic-style color-coded network diagram.

Use the sidebar to navigate between pages.
""")
