import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import streamlit.components.v1 as components
from engine.flowchart import generate_flowchart
from engine.ui_theme import inject_premium_css, premium_divider

st.set_page_config(page_title="Flowchart Generator", page_icon="🧩", layout="wide")
inject_premium_css()
st.title("🧩 Automatic Flowchart Generator")
premium_divider()
st.caption("Parses your code with `ast` and builds a Mermaid.js flowchart. No AI involved.")

default_code = """x = 5
if x > 0:
    print("Positive")
else:
    print("Negative")
"""

code = st.text_area("Python code:", value=default_code, height=200)

if st.button("Generate flowchart", type="primary"):
    mermaid_code = generate_flowchart(code)

    st.subheader("Mermaid source")
    st.code(mermaid_code, language="text")

    st.subheader("Rendered flowchart")
    html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>
    <script type="module">
        import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
        mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
    </script>
    """
    components.html(html, height=500, scrolling=True)

    st.caption("Rendering uses the mermaid.js library loaded from a CDN in your browser. "
               "The flowchart *generation logic* itself is 100% local Python (ast-based) — "
               "only the client-side drawing library is fetched, just like any web page's CSS/JS assets. "
               "For a fully air-gapped environment, download mermaid.js and serve it locally instead.")
