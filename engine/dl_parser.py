"""
Offline, rule-based deep-learning architecture extractor.
Parses Keras-style (`Dense(64, activation="relu")`, `model.add(...)`) and
PyTorch-style (`nn.Linear(64, 32)`, `nn.Conv2d(...)`) layer definitions using
only the `ast` module — no execution, no external AI APIs.
"""
import ast
from typing import Any, Dict, List, Optional

# Layers whose "size" is a neuron/unit count worth drawing node-by-node
DENSE_LIKE = {"Dense", "Linear"}

# Human-readable operation description per layer type, used in the UI
LAYER_OPERATIONS: Dict[str, str] = {
    "Input": "Receives the raw feature vector; no computation happens here — "
             "it just defines the shape of data entering the network.",
    "Dense": "output = activation(W · x + b) — every neuron connects to every "
             "input from the previous layer (fully connected).",
    "Linear": "output = x · Wᵀ + b — a fully connected (affine) transformation, "
              "PyTorch's equivalent of a Dense layer.",
    "Conv2D": "Slides learnable filters across the input, computing a weighted "
              "sum (dot product) at each position to detect local patterns like edges/textures.",
    "Conv2d": "Slides learnable filters across the input, computing a weighted "
              "sum (dot product) at each position to detect local patterns like edges/textures.",
    "MaxPooling2D": "Downsamples the feature map by keeping only the maximum "
                     "value in each small window, reducing spatial size and computation.",
    "MaxPool2d": "Downsamples the feature map by keeping only the maximum "
                 "value in each small window, reducing spatial size and computation.",
    "Flatten": "Reshapes a multi-dimensional tensor (e.g. an image feature map) "
               "into a single 1D vector so it can feed into Dense/Linear layers.",
    "Dropout": "During training, randomly sets a fraction of activations to zero "
               "to prevent the network from over-relying on any single neuron (reduces overfitting).",
    "BatchNormalization": "Normalizes activations to have stable mean/variance "
                          "across a batch, which speeds up and stabilizes training.",
    "BatchNorm2d": "Normalizes activations to have stable mean/variance "
                   "across a batch, which speeds up and stabilizes training.",
    "LSTM": "Maintains a cell state across time steps, gated by forget/input/output "
            "gates, letting the network remember information over long sequences.",
    "GRU": "A simplified gated recurrent unit — combines LSTM's forget and input "
           "gates into a single 'update gate', using fewer parameters.",
    "SimpleRNN": "Combines the current input with the previous hidden state at "
                 "each time step; simple but prone to vanishing gradients on long sequences.",
    "RNN": "Combines the current input with the previous hidden state at "
           "each time step; simple but prone to vanishing gradients on long sequences.",
    "BiLSTM": "Runs two LSTMs over the sequence — one forward, one backward — then "
              "concatenates their hidden states, so each time step sees both past "
              "and future context (common in NLP tagging/translation tasks).",
    "BiGRU": "Runs two GRUs over the sequence — one forward, one backward — then "
             "concatenates their hidden states, giving each time step past and future context.",
    "BiRNN": "Runs two simple RNNs over the sequence — one forward, one backward — "
             "then concatenates their hidden states.",
    "BiSimpleRNN": "Runs two simple RNNs over the sequence — one forward, one backward — "
                   "then concatenates their hidden states.",
    "ANN": "A stack of fully-connected (Dense/Linear) layers — each neuron connects "
           "to every neuron in the previous layer, learning weighted combinations of features.",
    "Embedding": "Looks up a dense vector representation for each discrete input "
                 "token (e.g. a word index) from a learned embedding table.",
    "ReLU": "Activation: output = max(0, x) — zeroes out negative values, the "
            "most common activation for hidden layers.",
    "Sigmoid": "Activation: squashes values into (0, 1) — common for binary "
               "classification output layers.",
    "Tanh": "Activation: squashes values into (-1, 1) — zero-centered, often "
            "used in RNN/LSTM cells.",
    "Softmax": "Activation: converts a vector of scores into a probability "
               "distribution that sums to 1 — used for multi-class output layers.",
}

ACTIVATION_ONLY_LAYERS = {"ReLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU", "ELU"}

RECURRENT_LAYERS = {"LSTM", "GRU", "SimpleRNN", "RNN"}

# Distinct color per layer type for the network diagram — grouped by role
# (recurrent = blues, conv/vision = warm oranges, dense = violet, utility = neutral grays)
LAYER_COLORS: Dict[str, str] = {
    "Input": "#00A896",
    "Dense": "#5B5F97",
    "Linear": "#5B5F97",
    "ANN": "#5B5F97",
    "Conv2D": "#F4A259",
    "Conv2d": "#F4A259",
    "MaxPooling2D": "#F6BD60",
    "MaxPool2d": "#F6BD60",
    "Flatten": "#9E9E9E",
    "Dropout": "#E07A5F",
    "BatchNormalization": "#81B29A",
    "BatchNorm2d": "#81B29A",
    "LSTM": "#3D5A80",
    "GRU": "#4C6E8E",
    "SimpleRNN": "#5C82A3",
    "RNN": "#5C82A3",
    "BiLSTM": "#293241",
    "BiGRU": "#38455A",
    "BiRNN": "#425873",
    "BiSimpleRNN": "#425873",
    "Embedding": "#EE964B",
}
DEFAULT_LAYER_COLOR = "#6C63FF"


def _literal_or_text(node) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return None


def _get_kwarg(call: ast.Call, name: str, default=None):
    for kw in call.keywords:
        if kw.arg == name:
            return _literal_or_text(kw.value)
    return default


def _get_arg(call: ast.Call, idx: int, default=None):
    if len(call.args) > idx:
        return _literal_or_text(call.args[idx])
    return default


def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _extract_layer(call: ast.Call) -> Optional[Dict[str, Any]]:
    name = _get_call_name(call)
    if name not in LAYER_OPERATIONS:
        return None

    layer: Dict[str, Any] = {"type": name, "units": None, "activation": None, "detail": ""}

    if name == "Dense":
        layer["units"] = _get_kwarg(call, "units", _get_arg(call, 0))
        layer["activation"] = _get_kwarg(call, "activation", None)
        input_shape = _get_kwarg(call, "input_shape", None)
        if input_shape is not None:
            layer["input_shape"] = input_shape
        layer["detail"] = f"{layer['units']} units"

    elif name == "Linear":
        in_f = _get_kwarg(call, "in_features", _get_arg(call, 0))
        out_f = _get_kwarg(call, "out_features", _get_arg(call, 1))
        layer["units"] = out_f
        layer["in_features"] = in_f
        layer["detail"] = f"{in_f} → {out_f}"

    elif name in ("Conv2D", "Conv2d"):
        filters = _get_kwarg(call, "filters", _get_arg(call, 1 if name == "Conv2d" else 0))
        kernel = _get_kwarg(call, "kernel_size", _get_arg(call, 2 if name == "Conv2d" else 1))
        layer["activation"] = _get_kwarg(call, "activation", None)
        layer["units"] = filters
        layer["detail"] = f"{filters} filters, kernel {kernel}"
        input_shape = _get_kwarg(call, "input_shape", None)
        if input_shape is not None:
            layer["input_shape"] = input_shape

    elif name in ("MaxPooling2D", "MaxPool2d"):
        pool = _get_kwarg(call, "pool_size", _get_arg(call, 0, "2x2"))
        layer["detail"] = f"pool {pool}"

    elif name == "Flatten":
        layer["detail"] = "flatten to 1D"

    elif name == "Dropout":
        rate = _get_kwarg(call, "rate", _get_kwarg(call, "p", _get_arg(call, 0)))
        layer["detail"] = f"rate {rate}"

    elif name in ("BatchNormalization", "BatchNorm2d"):
        layer["detail"] = "normalize batch"

    elif name in ("LSTM", "GRU", "SimpleRNN", "RNN"):
        units = _get_kwarg(call, "units", None)
        if units is None:
            units = _get_kwarg(call, "hidden_size", None)
        if units is None:
            # Disambiguate positional args: PyTorch passes (input_size, hidden_size),
            # Keras passes just (units) as the first positional arg.
            if len(call.args) >= 2:
                units = _get_arg(call, 1)
            elif len(call.args) == 1:
                units = _get_arg(call, 0)
        layer["units"] = units
        layer["activation"] = _get_kwarg(call, "activation", None)
        input_shape = _get_kwarg(call, "input_shape", None)
        if input_shape is not None:
            layer["input_shape"] = input_shape
        bidirectional = bool(_get_kwarg(call, "bidirectional", False))
        if bidirectional:
            layer["type"] = "Bi" + name
            layer["detail"] = f"{units} hidden units/direction (bidirectional → {units * 2 if isinstance(units, int) else units} total)"
        else:
            layer["detail"] = f"{units} hidden units"

    elif name == "Embedding":
        vocab = _get_kwarg(call, "input_dim", _get_arg(call, 0))
        dim = _get_kwarg(call, "output_dim", _get_arg(call, 1))
        layer["units"] = dim
        layer["detail"] = f"vocab {vocab} → dim {dim}"

    elif name in ACTIVATION_ONLY_LAYERS:
        layer["detail"] = "activation function"
        layer["activation_only"] = True

    return layer


def extract_architecture(code: str) -> Dict[str, Any]:
    """
    Parse Keras/PyTorch-style model code and return an ordered list of layers.

    Returns:
      {"layers": [ {type, units, activation, detail, ...}, ... ], "error": str|None}
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"layers": [], "error": f"SyntaxError: {e.msg}"}

    all_calls: List[ast.Call] = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    # Detect Keras-style Bidirectional(LSTM(...)) / Bidirectional(GRU(...)) wrappers,
    # and mark their inner call so it isn't also processed as a standalone layer.
    consumed_ids = set()
    bidirectional_layers: Dict[int, Dict[str, Any]] = {}  # id(wrapper_call) -> layer dict
    for call in all_calls:
        if _get_call_name(call) == "Bidirectional" and call.args:
            inner = call.args[0]
            if isinstance(inner, ast.Call):
                inner_layer = _extract_layer(inner)
                if inner_layer and inner_layer["type"] in RECURRENT_LAYERS:
                    units = inner_layer.get("units")
                    inner_layer["type"] = "Bi" + inner_layer["type"]
                    total = units * 2 if isinstance(units, int) else units
                    inner_layer["detail"] = f"{units} hidden units/direction (bidirectional → {total} total)"
                    # Keras attaches input_shape to the Bidirectional(...) wrapper itself,
                    # not the inner recurrent layer — pull it in so the Input layer can
                    # still be inferred for a Bidirectional-first model.
                    wrapper_input_shape = _get_kwarg(call, "input_shape", None)
                    if wrapper_input_shape is not None:
                        inner_layer["input_shape"] = wrapper_input_shape
                    bidirectional_layers[id(call)] = inner_layer
                    consumed_ids.add(id(inner))

    layers: List[Dict[str, Any]] = []
    for call in all_calls:
        if id(call) in consumed_ids:
            continue
        if id(call) in bidirectional_layers:
            layers.append(bidirectional_layers[id(call)])
            continue
        layer = _extract_layer(call)
        if layer:
            layers.append(layer)

    if not layers:
        return {"layers": [], "error": "No recognizable layers found "
                 "(supported: Dense, Linear, Conv2D/Conv2d, MaxPooling2D/MaxPool2d, "
                 "Flatten, Dropout, BatchNormalization, LSTM, GRU, RNN, BiLSTM/Bidirectional, "
                 "Embedding, ReLU/Sigmoid/Tanh/Softmax)."}

    # Merge standalone activation layers (common in PyTorch) into the preceding layer
    merged: List[Dict[str, Any]] = []
    for layer in layers:
        if layer.get("activation_only") and merged and merged[-1].get("activation") is None:
            merged[-1]["activation"] = layer["type"]
            continue
        merged.append(layer)

    # Prepend an explicit Input layer if we can infer the input size
    first = merged[0]
    input_size = first.get("input_shape") or first.get("in_features")
    if input_size is not None:
        merged.insert(0, {"type": "Input", "units": input_size, "activation": None,
                           "detail": f"shape {input_size}"})

    return {"layers": merged, "error": None}


def extract_conv_blocks(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return just the Conv2D/Conv2d layers, in order, for the feature-hierarchy view."""
    return [l for l in layers if l["type"] in ("Conv2D", "Conv2d")]
