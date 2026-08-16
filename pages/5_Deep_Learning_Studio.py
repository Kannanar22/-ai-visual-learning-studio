import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objects as go
from engine.dl_parser import (extract_architecture, LAYER_OPERATIONS, extract_conv_blocks)
from engine.ui_theme import inject_premium_css, premium_divider, resolve_text_color

st.set_page_config(page_title="Deep Learning Studio", page_icon="🧬", layout="wide")
inject_premium_css()
st.title("🧬 Deep Learning Studio — Architecture Visualizer")
st.caption("Paste Keras or PyTorch model code (ANN, CNN, RNN, LSTM, GRU, BiLSTM...) and see "
           "the input layer, hidden layers, and each layer's operation — parsed locally with "
           "`ast`. No AI API involved.")
premium_divider()

# ---------------------------------------------------------------------------
# Model demonstrations — pick which architecture you'd like walked through.
# ---------------------------------------------------------------------------
MODEL_DEMOS = {
    "🧠 ANN — Keras MLP classifier": """model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(20,)))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
""",
    "🖼️ CNN — Keras image classifier": """model.add(Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
""",
    "🧠 ANN — PyTorch MLP": """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)
""",
    "🔁 RNN — Keras sequence model": """model.add(SimpleRNN(32, activation="tanh", input_shape=(10, 8)))
model.add(Dense(1, activation="sigmoid"))
""",
    "🔁 LSTM — Keras sequence model": """model.add(LSTM(50, activation="tanh", input_shape=(10, 8)))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
""",
    "🔁 BiLSTM — Keras text classifier": """model.add(Embedding(input_dim=5000, output_dim=64))
model.add(Bidirectional(LSTM(64, activation="tanh")))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
""",
    "🔁 BiLSTM — PyTorch": """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=100, hidden_size=64, bidirectional=True)
        self.fc = nn.Linear(128, 10)
""",
}

# Classic textbook coloring — yellow input nodes, blue hidden nodes, red
# output nodes, matching the reference "x1..x4 -> hidden -> y1..y3" style diagram.
# This is now the app's only network-diagram style.
ROLE_COLORS = {"Input": "#FDD835", "Hidden": "#42A5F5", "Output": "#EF5350"}

# Fixed, always-readable label colors — no manual "text visibility" controls needed.
LABEL_COLOR = "#1A1A2E"
HEADER_COLOR = "#1A1A2E"


def _apply_demo():
    st.session_state["dl_code"] = MODEL_DEMOS[st.session_state["demo_choice"]]


st.markdown('<span class="premium-badge">🎯 Which model would you like demonstrated?</span>',
            unsafe_allow_html=True)
demo_names = list(MODEL_DEMOS.keys())
if "demo_choice" not in st.session_state:
    st.session_state["demo_choice"] = demo_names[0]
    st.session_state["dl_code"] = MODEL_DEMOS[demo_names[0]]

st.pills(
    "Model demonstration",
    demo_names,
    key="demo_choice",
    on_change=_apply_demo,
    label_visibility="collapsed",
)
st.caption("Pick an architecture above to instantly load a ready-made demonstration below — "
           "or skip this and paste your own model code straight into the editor.")

code = st.text_area(
    "Paste your model code:",
    height=260,
    key="dl_code",
)

st.caption("Supported layers: Dense / Linear (ANN), Conv2D / Conv2d (CNN), "
           "MaxPooling2D / MaxPool2d, Flatten, Dropout, BatchNormalization, "
           "SimpleRNN / RNN, LSTM, GRU, Bidirectional(...) / bidirectional=True (BiLSTM/BiGRU/BiRNN), "
           "Embedding, and ReLU / Sigmoid / Tanh / Softmax activations.")

if st.button("Visualize architecture", type="primary"):
    result = extract_architecture(code)
    st.session_state["dl_result"] = result

if "dl_result" in st.session_state:
    result = st.session_state["dl_result"]

    if result["error"]:
        st.error(result["error"])
    else:
        layers = result["layers"]

        def layer_role(i: int) -> str:
            if i == 0:
                return "Input"
            if i == len(layers) - 1:
                return "Output"
            return "Hidden"

        # ---------------------------------------------------------------
        # 1) Neuron-level network diagram (nodes + connections) — classic
        #    textbook style only.
        # ---------------------------------------------------------------
        st.subheader("🔗 Network diagram")
        st.caption("Classic style: 🟡 Input layer · 🔵 Hidden layer(s) · 🔴 Output layer — "
                   "fully connected, like a standard neural-network textbook diagram.")

        MAX_NODES_DRAWN = 8
        fig = go.Figure()
        node_positions = []  # list of list of y-coords per layer
        max_y_seen = 0

        for x, layer in enumerate(layers):
            units = layer.get("units")
            is_dense_like = layer["type"] in ("Input", "Dense", "Linear") and isinstance(units, int)

            if is_dense_like:
                n_draw = min(units, MAX_NODES_DRAWN)
                ys = [i - (n_draw - 1) / 2 for i in range(n_draw)]
            else:
                ys = [0]
            node_positions.append(ys)
            if ys:
                max_y_seen = max(max_y_seen, max(ys))

            role = layer_role(x)
            color = ROLE_COLORS[role]
            fig.add_trace(go.Scatter(
                x=[x] * len(ys), y=ys, mode="markers",
                marker=dict(size=24 if is_dense_like else 72,
                            color=color,
                            symbol="circle" if is_dense_like else "square",
                            line=dict(width=2, color="#333333")),
                hovertext=[f"{layer['type']} ({role})<br>{layer.get('detail','')}" for _ in ys],
                hoverinfo="text",
                showlegend=False,
            ))

            fig.add_annotation(x=x, y=min(ys) - 1.2 if ys else -1.2,
                                text=f"<b>{layer['type']}</b><br>{layer.get('detail','')}"
                                     + (f"<br>act: {layer['activation']}" if layer.get('activation') else ""),
                                showarrow=False, font=dict(size=11, color=LABEL_COLOR), align="center")

            if is_dense_like and units > MAX_NODES_DRAWN:
                fig.add_annotation(x=x, y=max(ys) + 0.8, text=f"(+{units - MAX_NODES_DRAWN} more)",
                                    showarrow=False, font=dict(size=9, color=LABEL_COLOR))

        # Draw connections between consecutive layers
        line_color = "rgba(70,70,70,0.45)"
        line_width = 0.8
        for x in range(len(layers) - 1):
            for y1 in node_positions[x]:
                for y2 in node_positions[x + 1]:
                    fig.add_trace(go.Scatter(
                        x=[x, x + 1], y=[y1, y2], mode="lines",
                        line=dict(width=line_width, color=line_color),
                        hoverinfo="skip", showlegend=False,
                    ))

        # Group header labels above the diagram (Input / Hidden / Output)
        label_y = max_y_seen + 2.2
        fig.add_annotation(x=0, y=label_y, text="<b>INPUT LAYER</b>",
                            showarrow=False, font=dict(size=13, color=HEADER_COLOR))
        if len(layers) > 2:
            mid_x = (1 + (len(layers) - 2)) / 2
            fig.add_annotation(x=mid_x, y=label_y, text="<b>HIDDEN LAYER(S)</b>",
                                showarrow=False, font=dict(size=13, color=HEADER_COLOR))
        fig.add_annotation(x=len(layers) - 1, y=label_y, text="<b>OUTPUT LAYER</b>",
                            showarrow=False, font=dict(size=13, color=HEADER_COLOR))

        fig.update_layout(
            height=500, xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=40, b=80), plot_bgcolor="white",
        )
        fig.data = fig.data[::-1]  # draw lines behind nodes
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        legend_cols = st.columns(3)
        for i, (role, swatch) in enumerate(ROLE_COLORS.items()):
            with legend_cols[i]:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:6px;">'
                    f'<div style="width:14px;height:14px;border-radius:50%;background:{swatch};'
                    f'border:1.5px solid #333;"></div>'
                    f'<span style="font-size:13px;">{role} Layer</span></div>',
                    unsafe_allow_html=True,
                )

        # ---------------------------------------------------------------
        # 1b) Feature Hierarchy View — "parts converge to a decision"
        #     style diagram, for CNN architectures (Conv2D/Conv2d layers)
        # ---------------------------------------------------------------
        conv_blocks = extract_conv_blocks(layers)
        if conv_blocks:
            st.markdown('<span class="premium-badge">🐨 Part → Whole → Decision</span>',
                        unsafe_allow_html=True)
            st.subheader("Feature Hierarchy View")
            st.caption("The same idea as a classic CNN explainer diagram — small local "
                       "detectors (edges, textures, corners...) combine into whole-object "
                       "feature maps for each Conv layer, and those converge into the final "
                       "prediction. Box sizes/filter counts reflect your actual Conv2D layers; "
                       "the low-level detector labels are illustrative.")

            final_layer = layers[-1]
            final_label = f"<b>Final Prediction</b><br>({final_layer['type']}: {final_layer.get('detail','')})"

            hfig = go.Figure()

            # Depth-aware detector vocabulary: earlier conv layers "see" low-level
            # texture/edges, deeper ones "see" more abstract parts — mirrors how a
            # real CNN builds understanding, and matches the eye/nose/ear -> head,
            # hands/legs -> body style of a textbook feature-hierarchy diagram.
            DETECTOR_LEVELS = [
                ["Edge<br>detector", "Color/blob<br>detector", "Texture<br>detector"],
                ["Corner<br>detector", "Curve<br>detector"],
                ["Shape<br>detector", "Pattern<br>detector"],
                ["Part<br>detector", "Motif<br>detector"],
            ]

            MAX_BRANCHES = 4
            n_branches = min(len(conv_blocks), MAX_BRANCHES)
            BRANCH_SPACING = 3.2
            branch_y_centers = [((n_branches - 1) / 2 - b) * BRANCH_SPACING for b in range(n_branches)]

            INPUT_BG, INPUT_BORDER = "#E8ECFF", "#4338CA"
            PART_BG, PART_BORDER = "#FFF3CD", "#D69E2E"
            CONCEPT_BG, CONCEPT_BORDER = "#D6E4FF", "#3D5A80"
            FINAL_BG, FINAL_BORDER = "#FADBD8", "#C0392B"

            def box_text_color(bg_hex):
                return resolve_text_color(bg_hex, LABEL_COLOR, True)

            def add_box(x, y, text, bg, border, w=1.6, h=0.7, bold=False):
                hfig.add_shape(type="rect", x0=x - w / 2, x1=x + w / 2, y0=y - h / 2, y1=y + h / 2,
                                line=dict(color=border, width=2), fillcolor=bg, layer="below",
                                opacity=0.95)
                txt = f"<b>{text}</b>" if bold else text
                hfig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                                     font=dict(size=10.5, color=box_text_color(bg)), align="center")

            def add_arrow(x1, y1, x2, y2):
                hfig.add_annotation(x=x2, y=y2, ax=x1, ay=y1, xref="x", yref="y",
                                     axref="x", ayref="y", showarrow=True, arrowhead=2,
                                     arrowsize=1, arrowwidth=1.6, arrowcolor="#6B7C93")

            # Leftmost: a stand-in for the raw input image, like the source photo
            # in a textbook "parts -> whole -> decision" diagram.
            input_x, input_y = -3.4, sum(branch_y_centers) / len(branch_y_centers)
            add_box(input_x, input_y, "🖼️<br><b>Input<br>Image</b>", INPUT_BG, INPUT_BORDER,
                    w=1.5, h=1.6)

            concept_positions = []
            for b in range(n_branches):
                block = conv_blocks[b]
                cy = branch_y_centers[b]
                labels = DETECTOR_LEVELS[b % len(DETECTOR_LEVELS)]
                n_sub = 3 if b == 0 else 2
                sub_labels = (labels * 2)[:n_sub]
                sub_step = 0.85
                sub_ys = [cy + sub_step * (i - (n_sub - 1) / 2) for i in range(n_sub)]

                for sy, label in zip(sub_ys, sub_labels):
                    add_box(0, sy, label, PART_BG, PART_BORDER, w=1.5, h=0.62)
                    add_arrow(input_x + 0.75, input_y, -0.75, sy)
                    add_arrow(0.75, sy, 2.2 - 0.9, cy)

                concept_text = f"Conv Layer {b + 1}<br>Feature Maps<br>({block.get('detail', '')})"
                add_box(2.2, cy, concept_text, CONCEPT_BG, CONCEPT_BORDER, w=1.9, h=0.95)
                concept_positions.append((2.2, cy))

            if len(conv_blocks) > MAX_BRANCHES:
                st.caption(f"Showing the first {MAX_BRANCHES} of {len(conv_blocks)} Conv2D "
                           "layers here for readability; every layer still appears in the "
                           "network diagram and the breakdown below.")

            final_x = 4.6
            final_y = sum(y for _, y in concept_positions) / len(concept_positions)
            for cx, cy in concept_positions:
                add_arrow(cx + 0.95, cy, final_x - 1.05, final_y)
            add_box(final_x, final_y, final_label, FINAL_BG, FINAL_BORDER, w=2.3, h=1.05, bold=False)

            y_span = max(branch_y_centers) - min(branch_y_centers) if n_branches > 1 else 0
            hfig.update_layout(
                height=max(380, 220 + 90 * n_branches),
                xaxis=dict(visible=False, range=[-4.4, 6]),
                yaxis=dict(visible=False, range=[min(branch_y_centers) - 1.6 - y_span * 0.1,
                                                   max(branch_y_centers) + 1.6 + y_span * 0.1]),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(hfig, use_container_width=True)

            legend_cols = st.columns(4)
            legend_items = [
                ("Input", INPUT_BG, INPUT_BORDER), ("Local detector", PART_BG, PART_BORDER),
                ("Feature map", CONCEPT_BG, CONCEPT_BORDER), ("Prediction", FINAL_BG, FINAL_BORDER),
            ]
            for col, (label, bg, border) in zip(legend_cols, legend_items):
                with col:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:6px;">'
                        f'<div style="width:14px;height:14px;border-radius:4px;background:{bg};'
                        f'border:1.5px solid {border};"></div>'
                        f'<span style="font-size:13px;">{label}</span></div>',
                        unsafe_allow_html=True,
                    )

        # ---------------------------------------------------------------
        # 2) Per-layer operation breakdown
        # ---------------------------------------------------------------
        st.subheader("⚙️ Layer-by-layer operations")
        for i, layer in enumerate(layers):
            op = LAYER_OPERATIONS.get(layer["type"], "No description available for this layer type.")
            role = "Input Layer" if layer["type"] == "Input" else (
                "Output Layer" if i == len(layers) - 1 else "Hidden Layer")
            with st.expander(f"{i+1}. {layer['type']} — {role}  ({layer.get('detail','')})"):
                st.markdown(f"**Operation:** {op}")
                if layer.get("activation"):
                    act = layer["activation"]
                    act_op = LAYER_OPERATIONS.get(act, None)
                    st.markdown(f"**Activation ({act}):** {act_op or 'Applies a non-linear function to the layer output.'}")
                if layer["type"] in ("Dense", "Linear") and isinstance(layer.get("units"), int):
                    prev = layers[i - 1] if i > 0 else None
                    prev_units = prev.get("units") if prev else layer.get("in_features")
                    if isinstance(prev_units, int):
                        params = prev_units * layer["units"] + layer["units"]
                        st.caption(f"Approx. trainable parameters: {prev_units} × {layer['units']} "
                                   f"weights + {layer['units']} biases = **{params:,}**")

        # ---------------------------------------------------------------
        # 3) Summary table
        # ---------------------------------------------------------------
        st.subheader("📋 Summary")
        st.table([
            {"#": i + 1, "Layer": l["type"], "Detail": l.get("detail", ""),
             "Activation": l.get("activation") or "—"}
            for i, l in enumerate(layers)
        ])
else:
    st.info("Pick a model demonstration above, or paste your own code, then click "
            "**Visualize architecture** to get started.")
