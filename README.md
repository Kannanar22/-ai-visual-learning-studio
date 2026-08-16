# AI Visual Learning Studio — Streamlit Edition

A fully offline educational app: AST-based viva question generator, Mermaid
flowchart generator, interactive ML visualizations, and a static
offline "AI Project Mentor". No external AI APIs are used anywhere.

## Project structure
```
ai_visual_learning_studio/
├── app.py                     # Home page + sidebar (Local AI Mode toggle)
├── pages/
│   ├── 1_Code_Explainer.py     # Viva question generator
│   ├── 2_Flowchart_Generator.py
│   ├── 3_ML_Visualizations.py
│   ├── 4_AI_Mentor.py
│   └── 5_Deep_Learning_Studio.py   # DL architecture visualizer
├── engine/
│   ├── viva_engine.py          # ast-based viva question generator
│   ├── flowchart.py            # ast -> Mermaid flowchart
│   ├── dl_parser.py            # ast -> DL layer architecture
│   └── ollama_client.py        # optional local-only LLM calls
├── knowledge_base/
│   └── topics.json             # static NLP/RAG/Agentic AI/... guides
├── .streamlit/config.toml
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy options

**1. Streamlit Community Cloud (free, easiest)**
1. Push this folder to a public/private GitHub repo.
2. Go to share.streamlit.io → "New app" → point it at `app.py`.
3. Deploy. Note: the *AI logic* is 100% local Python — only the hosting
   itself needs internet, same as any web app.

**2. Docker (fully self-hosted / air-gapped)**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t ai-visual-learning-studio .
docker run -p 8501:8501 ai-visual-learning-studio
```

**3. Any VM / on-prem server** — same `pip install` + `streamlit run` steps
work behind your own firewall with zero outbound traffic (except the
Mermaid.js CDN script used for rendering flowcharts in-browser — swap in a
locally-hosted copy of `mermaid.min.js` for a fully air-gapped setup).

## Optional local LLM (Ollama)
Toggle "Enable local Ollama model" in the sidebar. This only ever calls
`http://localhost:11434` on the machine running the app — never a hosted API.
```bash
ollama pull phi3:mini
ollama serve
```

## Extending
- Add more algorithms to `pages/3_ML_Visualizations.py` following the existing `elif algo == "..."` pattern.
- Add more question rules to `METHOD_QUESTIONS` / `CATEGORY_CONCEPTUAL_QUESTIONS` in `engine/viva_engine.py`.
- Add more topics to `knowledge_base/topics.json`.
