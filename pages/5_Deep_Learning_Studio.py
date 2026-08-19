from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import streamlit as st

from engine.ui_theme import inject_premium_css, premium_divider, contrast_color
from engine.dl_parser import (
    extract_architecture,
    LAYER_OPERATIONS,
    LAYER_COLORS,
    DEFAULT_LAYER_COLOR,
)

st.set_page_config(page_title="Deep Learning Studio", page_icon="🧬", layout="wide")
inject_premium_css()

st.title("🧬 Deep Learning Studio")
premium_divider()
st.caption("Offline, rule-based architecture visualizer — no external AI APIs, ever.")

# ---------------------------------------------------------------------------
# Preset demo model snippets, one per requested architecture
# ---------------------------------------------------------------------------
DEMO_MODELS: Dict[str, str] = {
    "ANN": '''
model = Sequential()
model.add(Dense(64, input_shape=(20,), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
''',
    "CNN": '''
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax"))
''',
    "RNN": '''
model = Sequential()
model.add(SimpleRNN(32, input_shape=(50, 8), activation="tanh"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
''',
    "LSTM": '''
model = Sequential()
model.add(LSTM(64, input_shape=(50, 8)))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
''',
    "GRU": '''
model = Sequential()
model.add(GRU(64, input_shape=(50, 8)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
''',
    "BiLSTM": '''
model = Sequential()
model.add(Bidirectional(LSTM(64), input_shape=(50, 8)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
''',
}


# ---------------------------------------------------------------------------
# Organize the parsed layers into a correctly ordered, labeled sequence:
# Input Layer -> Hidden Layer 1 ... N -> Output Layer. This is purely
# positional (first = input if present, last = output, everything else in
# between numbered in order) so the text always matches the network's actual
# left-to-right computation order, regardless of layer types involved.
# ---------------------------------------------------------------------------
def organize_layers(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    organized = []
    n = len(layers)
    hidden_count = 0
    for i, layer in enumerate(layers):
        if layer["type"] == "Input":
            role = "Input Layer"
        elif i == n - 1:
            role = "Output Layer"
        else:
            hidden_count += 1
            role = f"Hidden Layer {hidden_count}"
        organized.append({**layer, "role": role, "position": i})
    return organized


def layer_operation_text(layer: Dict[str, Any]) -> str:
    base = LAYER_OPERATIONS.get(layer["type"], "No offline description available for this layer type.")
    parts = [base]
    if layer.get("activation"):
        act = layer["activation"]
        act_text = LAYER_OPERATIONS.get(act)
        if act_text:
            parts.append(f"**Activation ({act}):** {act_text}")
        else:
            parts.append(f"**Activation:** `{act}`")
    return "\n\n".join(parts)


def role_icon(role: str) -> str:
    if role == "Input Layer":
        return "🟢"
    if role == "Output Layer":
        return "🔴"
    return "🔵"


# ---------------------------------------------------------------------------
# Draw a classic-style, color-coded, left-to-right network diagram
# ---------------------------------------------------------------------------
def draw_network(organized: List[Dict[str, Any]]):
    max_nodes_drawn = 8  # cap per layer so wide layers (e.g. 512 units) stay readable

    def node_count(layer: Dict[str, Any]) -> int:
        units = layer.get("units")
        if isinstance(units, int) and units > 0:
            return min(units, max_nodes_drawn)
        return 3  # Flatten/Dropout/BatchNorm/etc. have no natural "unit count" — draw a placeholder block

    counts = [node_count(l) for l in organized]
    n_layers = len(organized)
    fig_width = max(6, n_layers * 2.0)
    fig_height = max(4, max(counts) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    x_positions = list(range(n_layers))
    node_positions: List[List[Tuple[float, float]]] = []

    for xi, (layer, count) in zip(x_positions, zip(organized, counts)):
        ys = [(count - 1) / 2 - k for k in range(count)]
        node_positions.append([(xi, y) for y in ys])

    # Draw connecting edges between consecutive layers first (so nodes sit on top)
    for li in range(n_layers - 1):
        for x1, y1 in node_positions[li]:
            for x2, y2 in node_positions[li + 1]:
                ax.plot([x1, x2], [y1, y2], color="#D1D5DB", linewidth=0.6, zorder=1, alpha=0.6)

    # Draw nodes
    for li, layer in enumerate(organized):
        color = LAYER_COLORS.get(layer["type"], DEFAULT_LAYER_COLOR)
        for x, y in node_positions[li]:
            ax.scatter([x], [y], s=420, color=color, zorder=3, edgecolors="white", linewidths=1.5)

        units = layer.get("units")
        label_top = layer["role"]
        label_bottom = layer["type"] if not isinstance(units, int) else f"{layer['type']} ({units})"
        top_y = max(y for _, y in node_positions[li]) + 0.9
        ax.text(x_positions[li], top_y, label_top, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#1E1B3A")
        bottom_y = min(y for _, y in node_positions[li]) - 0.6
        ax.text(x_positions[li], bottom_y, label_bottom, ha="center", va="top",
                fontsize=9, color="#4B5563")

    ax.set_xlim(-0.8, n_layers - 0.2)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
mode = st.radio(
    "Choose a source",
    ["🎬 Demo model", "✍️ Paste your own Keras/PyTorch code"],
    horizontal=True,
)

code = ""
if mode == "🎬 Demo model":
    choice = st.selectbox("Pick an architecture", list(DEMO_MODELS.keys()))
    code = DEMO_MODELS[choice].strip()
    with st.expander("View demo code", expanded=False):
        st.code(code, language="python")
else:
    code = st.text_area(
        "Paste Keras or PyTorch model code",
        height=240,
        placeholder="e.g.\nmodel = Sequential()\nmodel.add(Dense(64, activation='relu'))\n...\n\nor\n\nself.fc1 = nn.Linear(784, 128)\nself.lstm = nn.LSTM(50, 64, bidirectional=True)\n...",
    )

if st.button("🔍 Visualize architecture", type="primary") or (mode == "🎬 Demo model" and code):
    if not code.strip():
        st.warning("Paste some model code first, or pick a demo model.")
        st.stop()

    result = extract_architecture(code)

    if result["error"]:
        st.error(result["error"])
        st.stop()

    organized = organize_layers(result["layers"])

    st.markdown("### 🖼️ Network diagram")
    fig = draw_network(organized)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    premium_divider()

    st.markdown("### 📖 Layer-by-layer explanation")
    st.caption("Ordered exactly as data flows through the network: Input → Hidden layers → Output.")

    for layer in organized:
        icon = role_icon(layer["role"])
        header = f"{icon} {layer['role']} — `{layer['type']}`"
        if layer.get("detail"):
            header += f"  ({layer['detail']})"
        with st.container(border=True):
            st.markdown(f"#### {header}")
            st.markdown(layer_operation_text(layer))

    premium_divider()
    st.markdown("### 🧾 Architecture summary")
    summary_rows = []
    for layer in organized:
        summary_rows.append({
            "Position": layer["position"] + 1,
            "Role": layer["role"],
            "Layer type": layer["type"],
            "Detail": layer.get("detail", ""),
            "Activation": layer.get("activation") or "—",
        })
    st.table(summary_rows)
