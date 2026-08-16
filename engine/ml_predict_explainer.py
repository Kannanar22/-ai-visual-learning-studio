"""
Offline, rule-based "how does it predict?" explainer for classic ML code.
Detects known scikit-learn-style estimator constructors (`LinearRegression(...)`,
`RandomForestClassifier(...)`, etc.) using only the `ast` module, pulls out any
hyperparameters the user set, and returns a static, curated explanation of the
exact prediction mechanism — no AI API involved, matching the rest of this app.
"""
import ast
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Knowledge base: for every supported estimator, how it turns a NEW input row
# into a prediction, step by step, plus its core formula and hyperparameters.
# ---------------------------------------------------------------------------
ALGO_KB: Dict[str, Dict[str, Any]] = {
    "LinearRegression": {
        "family": "Regression",
        "core_idea": "Fits a straight line (or hyperplane) that minimizes squared error "
                     "between predicted and actual values.",
        "formula": "ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b",
        "predict_steps": [
            "Take the new input's feature values x₁...xₙ.",
            "Multiply each feature by its learned weight (coefficient) wᵢ.",
            "Sum all the weighted features together.",
            "Add the learned intercept (bias) b.",
            "The result of that sum IS the prediction — no further transformation.",
        ],
        "key_hyperparams": {
            "fit_intercept": "Whether to learn a bias term b (default True).",
            "positive": "Forces all coefficients to be non-negative when True.",
        },
        "notes": "Weights are learned once during `.fit()` by minimizing mean squared "
                 "error (closed-form via the Normal Equation, or gradient-based for large data). "
                 "`.predict()` is then just one dot product — extremely fast.",
    },
    "Ridge": {
        "family": "Regression",
        "core_idea": "Same as Linear Regression, but the training objective adds an L2 "
                     "penalty on the weights to shrink them and reduce overfitting.",
        "formula": "ŷ = w₁x₁ + ... + wₙxₙ + b   (weights trained with an extra +α·Σwᵢ² penalty)",
        "predict_steps": [
            "Take the new input's feature values.",
            "Multiply each feature by its (shrunken) learned weight.",
            "Sum the weighted features and add the intercept.",
            "That sum is the prediction — prediction math is identical to plain Linear Regression.",
        ],
        "key_hyperparams": {
            "alpha": "Strength of the L2 penalty. Higher alpha = smaller, more conservative weights.",
        },
        "notes": "Only the *training* objective changes vs Linear Regression — prediction is "
                 "the same dot-product-plus-bias.",
    },
    "Lasso": {
        "family": "Regression",
        "core_idea": "Same as Linear Regression, but the training objective adds an L1 "
                     "penalty, which can shrink some weights all the way to zero (feature selection).",
        "formula": "ŷ = w₁x₁ + ... + wₙxₙ + b   (weights trained with an extra +α·Σ|wᵢ| penalty)",
        "predict_steps": [
            "Take the new input's feature values.",
            "Multiply each feature by its learned weight — some weights may be exactly 0, "
            "meaning that feature is effectively ignored.",
            "Sum the weighted features and add the intercept.",
            "That sum is the prediction.",
        ],
        "key_hyperparams": {
            "alpha": "Strength of the L1 penalty. Higher alpha = more weights pushed to exactly 0.",
        },
        "notes": "Useful when you suspect only a few features actually matter — Lasso can "
                 "zero out the rest automatically during training.",
    },
    "LogisticRegression": {
        "family": "Classification",
        "core_idea": "Computes a weighted sum of features like Linear Regression, then "
                     "squashes it through a sigmoid to output a probability between 0 and 1.",
        "formula": "P(class=1) = sigmoid(w₁x₁ + ... + wₙxₙ + b) = 1 / (1 + e^-(w·x + b))",
        "predict_steps": [
            "Take the new input's feature values.",
            "Compute the weighted sum z = w·x + b (same as linear regression).",
            "Pass z through the sigmoid function to get a probability in [0, 1].",
            "If that probability ≥ 0.5 (default threshold), predict class 1, else class 0.",
            "For multi-class, softmax is used instead and the highest-probability class wins.",
        ],
        "key_hyperparams": {
            "C": "Inverse of regularization strength — smaller C = stronger regularization.",
            "penalty": "Type of regularization applied to the weights ('l2' by default).",
            "max_iter": "Max optimizer iterations while training (doesn't affect prediction math).",
        },
        "notes": "`.predict_proba()` returns the raw probability; `.predict()` applies the "
                 "0.5 threshold (or argmax for multi-class) on top of it.",
    },
    "DecisionTreeClassifier": {
        "family": "Classification",
        "core_idea": "Learns a tree of if/else questions on features; a new sample is "
                     "routed down the tree until it reaches a leaf holding a class label.",
        "formula": "ŷ = majority class of training samples in the leaf the input lands in",
        "predict_steps": [
            "Start at the root node of the trained tree.",
            "At each node, check the learned condition (e.g. 'feature x2 <= 0.73').",
            "Go left or right depending on whether the input satisfies the condition.",
            "Repeat until reaching a leaf node (no more splits).",
            "Predict the majority class of the training samples that ended up in that leaf.",
        ],
        "key_hyperparams": {
            "max_depth": "Maximum number of splits from root to leaf — limits overfitting.",
            "criterion": "Split-quality measure used during training ('gini' or 'entropy').",
            "min_samples_split": "Minimum samples required at a node before it can be split further.",
        },
        "notes": "Splits are chosen greedily during training to best separate classes "
                 "(e.g. maximize Gini impurity reduction) — prediction itself is just tree traversal.",
    },
    "DecisionTreeRegressor": {
        "family": "Regression",
        "core_idea": "Same tree-traversal idea as a classifier, but each leaf stores the "
                     "average target value of the training samples that landed there.",
        "formula": "ŷ = mean(target values of training samples in the leaf the input lands in)",
        "predict_steps": [
            "Start at the root and evaluate each split condition on the input's features.",
            "Follow left/right branches down the tree until reaching a leaf.",
            "Predict the average target value of training samples in that leaf.",
        ],
        "key_hyperparams": {
            "max_depth": "Maximum tree depth — limits overfitting.",
            "min_samples_leaf": "Minimum samples a leaf must contain.",
        },
        "notes": "Predictions are piecewise-constant — the model can only output values "
                 "seen as leaf averages during training, never a smooth continuous curve.",
    },
    "RandomForestClassifier": {
        "family": "Classification (Ensemble)",
        "core_idea": "Trains many decision trees on random subsets of data/features "
                     "(bagging), then lets them vote on the final class.",
        "formula": "ŷ = majority vote across all trees' individual predictions",
        "predict_steps": [
            "Send the new input through EVERY tree in the forest independently.",
            "Each tree traverses its own splits and outputs a predicted class (or class probabilities).",
            "Average the probabilities (or take a majority vote) across all trees.",
            "The class with the highest combined vote/probability is the final prediction.",
        ],
        "key_hyperparams": {
            "n_estimators": "Number of trees in the forest — more trees = smoother, usually more accurate.",
            "max_depth": "Maximum depth allowed for each individual tree.",
            "max_features": "How many features each tree considers at each split (adds randomness).",
        },
        "notes": "Because each tree sees a random subset of data and features, individual "
                 "trees overfit differently — averaging them cancels out much of that noise.",
    },
    "RandomForestRegressor": {
        "family": "Regression (Ensemble)",
        "core_idea": "Trains many regression trees on random subsets of data/features, "
                     "then averages their numeric predictions.",
        "formula": "ŷ = average of all trees' individual predicted values",
        "predict_steps": [
            "Send the new input through every tree in the forest.",
            "Each tree outputs its own predicted numeric value via its leaf average.",
            "Average all trees' predictions together — this final average is the forest's output.",
        ],
        "key_hyperparams": {
            "n_estimators": "Number of trees — more trees generally reduce prediction variance.",
            "max_depth": "Maximum depth of each individual tree.",
        },
        "notes": "Averaging many slightly-different trees smooths out the 'staircase' "
                 "effect a single regression tree produces.",
    },
    "GradientBoostingClassifier": {
        "family": "Classification (Ensemble)",
        "core_idea": "Builds trees one at a time, where each new tree is trained to "
                     "correct the errors (residuals) of the trees built so far.",
        "formula": "ŷ = sigmoid/softmax( Σᵢ learning_rate × treeᵢ(x) )",
        "predict_steps": [
            "Start from an initial baseline prediction (e.g. the log-odds of the base rate).",
            "Pass the input through tree 1; scale its output by the learning rate and add it in.",
            "Pass the input through tree 2 (trained to fix tree 1's mistakes); add its scaled output.",
            "Repeat for every tree in the sequence, accumulating a running score.",
            "Convert the final accumulated score into a class probability via sigmoid/softmax.",
        ],
        "key_hyperparams": {
            "n_estimators": "Number of boosting rounds (trees) — more rounds fit training data harder.",
            "learning_rate": "How much each tree's correction contributes — lower = more conservative.",
            "max_depth": "Depth of each individual (usually shallow) tree.",
        },
        "notes": "Unlike Random Forest's independent trees, boosting trees are trained "
                 "sequentially and depend on each other — order matters.",
    },
    "AdaBoostClassifier": {
        "family": "Classification (Ensemble)",
        "core_idea": "Trains a sequence of weak learners (often shallow trees), where "
                     "each new learner focuses more on the samples the previous ones got wrong.",
        "formula": "ŷ = sign( Σᵢ αᵢ × weak_learnerᵢ(x) )",
        "predict_steps": [
            "Send the input through every weak learner in the ensemble.",
            "Each weak learner casts a weighted vote (weight αᵢ reflects how accurate it was in training).",
            "Sum up all the weighted votes.",
            "The sign/majority of that weighted sum determines the predicted class.",
        ],
        "key_hyperparams": {
            "n_estimators": "Number of weak learners to chain together.",
            "learning_rate": "Shrinks each learner's contribution/weight.",
        },
        "notes": "Misclassified training samples get higher weight for the NEXT learner "
                 "during training, forcing the ensemble to focus on hard cases.",
    },
    "SVC": {
        "family": "Classification",
        "core_idea": "Finds the boundary (possibly after a non-linear kernel transform) "
                     "that maximizes the margin between classes, defined only by the closest points (support vectors).",
        "formula": "ŷ = sign( Σᵢ αᵢ yᵢ K(xᵢ, x) + b )   — sum runs only over support vectors",
        "predict_steps": [
            "Take the new input x.",
            "For each stored support vector xᵢ, compute the kernel similarity K(xᵢ, x) "
            "(e.g. dot product for 'linear', RBF distance for 'rbf').",
            "Weight each similarity by that support vector's learned coefficient αᵢyᵢ.",
            "Sum all the weighted similarities and add the bias b.",
            "The sign of that sum gives the predicted class (or `predict_proba` if enabled).",
        ],
        "key_hyperparams": {
            "kernel": "Similarity function used ('linear', 'rbf', 'poly') — changes the "
                      "shape of decision boundaries the model can represent.",
            "C": "Trade-off between a wide margin and classifying training points correctly.",
            "gamma": "For 'rbf'/'poly' kernels — controls how far a single support vector's influence reaches.",
        },
        "notes": "Only the support vectors (points closest to the boundary) matter at "
                 "prediction time — every other training point can be discarded after training.",
    },
    "SVR": {
        "family": "Regression",
        "core_idea": "The regression counterpart of SVC — finds a function that stays "
                     "within a margin (epsilon) of as many training points as possible.",
        "formula": "ŷ = Σᵢ αᵢ K(xᵢ, x) + b   — sum runs only over support vectors",
        "predict_steps": [
            "Take the new input x.",
            "Compute the kernel similarity between x and every stored support vector.",
            "Weight each similarity by its learned coefficient αᵢ.",
            "Sum the weighted similarities and add the bias b — that sum is the predicted value.",
        ],
        "key_hyperparams": {
            "kernel": "Similarity function ('linear', 'rbf', 'poly').",
            "epsilon": "Width of the 'no-penalty' margin around the true value during training.",
            "C": "Penalty strength for points that fall outside the epsilon margin.",
        },
        "notes": "Like SVC, only a subset of training points (support vectors) actually "
                 "influence the final prediction formula.",
    },
    "LinearSVC": {
        "family": "Classification",
        "core_idea": "A linear-kernel-only, more scalable version of SVC — finds the "
                     "max-margin hyperplane directly without kernel tricks.",
        "formula": "ŷ = sign(w · x + b)",
        "predict_steps": [
            "Take the new input's feature values.",
            "Compute the weighted sum w · x + b using the learned weight vector.",
            "The sign of that value determines the predicted class side of the hyperplane.",
        ],
        "key_hyperparams": {
            "C": "Trade-off between margin width and training-point misclassification.",
        },
        "notes": "Faster than kernel SVC for large datasets since it skips the kernel "
                 "similarity computation and works directly with weights.",
    },
    "KNeighborsClassifier": {
        "family": "Classification",
        "core_idea": "Stores the entire training set; a new point is classified by a "
                     "majority vote among its K nearest neighbors — no training-time model is built.",
        "formula": "ŷ = majority class among the K closest training points to x",
        "predict_steps": [
            "Take the new input x.",
            "Compute the distance (e.g. Euclidean) from x to every stored training point.",
            "Sort and select the K closest training points.",
            "Count the class labels among those K neighbors.",
            "Predict the majority class (optionally weighted by inverse distance).",
        ],
        "key_hyperparams": {
            "n_neighbors": "The K in K-NN — how many nearby points get a vote.",
            "weights": "'uniform' (every neighbor counts equally) or 'distance' (closer neighbors count more).",
            "metric": "Distance function used to measure 'closeness' (default: Euclidean/Minkowski).",
        },
        "notes": "There's effectively no separate 'training' — all the work happens at "
                 "prediction time, which makes K-NN slow on large datasets.",
    },
    "KNeighborsRegressor": {
        "family": "Regression",
        "core_idea": "Same neighbor-lookup idea as the classifier, but predicts the "
                     "average target value of the K nearest neighbors instead of a vote.",
        "formula": "ŷ = average target value among the K closest training points to x",
        "predict_steps": [
            "Compute the distance from the new input to every stored training point.",
            "Select the K closest points.",
            "Average their target values (optionally weighted by inverse distance).",
            "That average is the prediction.",
        ],
        "key_hyperparams": {
            "n_neighbors": "The K in K-NN.",
            "weights": "'uniform' or 'distance'-weighted averaging.",
        },
        "notes": "Predictions can only ever be an average of values already seen in "
                 "training — K-NN cannot extrapolate beyond the training data's range.",
    },
    "KMeans": {
        "family": "Clustering (Unsupervised)",
        "core_idea": "Learns K cluster centroids during training; a new point is "
                     "'predicted' by assigning it to its nearest centroid.",
        "formula": "cluster(x) = argmin over centroids c of ‖x - c‖²",
        "predict_steps": [
            "Take the new input x.",
            "Compute the distance from x to each of the K learned centroids.",
            "Assign x to the cluster whose centroid is closest.",
            "Return that cluster's ID as the 'prediction'.",
        ],
        "key_hyperparams": {
            "n_clusters": "K — the number of clusters/centroids to learn.",
            "n_init": "How many random centroid initializations are tried (best result is kept).",
        },
        "notes": "There's no 'ground truth' label — clustering is unsupervised, so "
                 "`.predict()` returns a cluster index, not a class or a value with inherent meaning.",
    },
    "GaussianNB": {
        "family": "Classification",
        "core_idea": "Assumes each feature is normally (Gaussian) distributed within "
                     "each class, and uses Bayes' theorem to pick the most probable class.",
        "formula": "ŷ = argmax over classes c of  P(c) × Πᵢ P(xᵢ | c)",
        "predict_steps": [
            "For each possible class c, start with its prior probability P(c) (how common it was in training).",
            "For each feature xᵢ, compute how likely that value is under class c's learned "
            "Gaussian (mean/variance from training data).",
            "Multiply the prior by all these per-feature likelihoods (this assumes features "
            "are independent given the class — the 'naive' assumption).",
            "Repeat for every class, then predict the class with the highest resulting score.",
        ],
        "key_hyperparams": {
            "var_smoothing": "A small value added to variances for numerical stability.",
        },
        "notes": "Surprisingly effective even though the 'features are independent' "
                 "assumption is rarely exactly true — it's fast and needs little training data.",
    },
    "MultinomialNB": {
        "family": "Classification",
        "core_idea": "The text-classification variant of Naive Bayes — assumes features "
                     "are counts (e.g. word counts) drawn from a multinomial distribution per class.",
        "formula": "ŷ = argmax over classes c of  P(c) × Πᵢ P(wordᵢ | c)^countᵢ",
        "predict_steps": [
            "For each class c, start with its prior probability P(c).",
            "For each word/feature in the input, look up how likely that word is under class c "
            "(learned word frequencies from training documents of that class).",
            "Multiply these likelihoods together (raised to the power of each word's count).",
            "Repeat per class and predict the class with the highest resulting score.",
        ],
        "key_hyperparams": {
            "alpha": "Additive (Laplace) smoothing — avoids zero-probability words unseen in training.",
        },
        "notes": "This is the classic 'spam vs not spam' Naive Bayes — works directly on "
                 "TF/count vectorized text rather than continuous features.",
    },
    "MLPClassifier": {
        "family": "Classification (Neural Network)",
        "core_idea": "A small feed-forward neural network — stacks weighted sums and "
                     "non-linear activations across hidden layers to learn a decision boundary.",
        "formula": "ŷ = softmax( Wₙ · ... activation(W₂ · activation(W₁·x + b₁) + b₂) ... + bₙ )",
        "predict_steps": [
            "Feed the input features into the first hidden layer: compute weighted sum + bias, "
            "then apply the activation function (ReLU by default).",
            "Pass that layer's output as input to the next hidden layer, repeating the same "
            "weighted-sum-plus-activation step.",
            "After the last hidden layer, compute the output layer's weighted sum.",
            "Apply softmax (multi-class) or sigmoid (binary) to turn that into class probabilities.",
            "Predict the class with the highest probability.",
        ],
        "key_hyperparams": {
            "hidden_layer_sizes": "Number and width of hidden layers, e.g. (100,) = one layer of 100 neurons.",
            "activation": "Non-linearity used between layers ('relu', 'tanh', 'logistic').",
            "max_iter": "Maximum training iterations (doesn't affect prediction math, only training).",
        },
        "notes": "All the weights are frozen after `.fit()` — `.predict()` is a single "
                 "forward pass through the network, no further learning happens.",
    },
    "MLPRegressor": {
        "family": "Regression (Neural Network)",
        "core_idea": "Same feed-forward network as MLPClassifier, but the output layer "
                     "has no squashing activation — it outputs a raw numeric value.",
        "formula": "ŷ = Wₙ · ... activation(W₂ · activation(W₁·x + b₁) + b₂) ... + bₙ",
        "predict_steps": [
            "Feed the input through each hidden layer: weighted sum + bias, then activation (usually ReLU).",
            "Pass the result forward, layer by layer, to the final output layer.",
            "The output layer computes one last weighted sum — that raw number IS the prediction "
            "(no softmax/sigmoid, since this is regression, not classification).",
        ],
        "key_hyperparams": {
            "hidden_layer_sizes": "Number and width of hidden layers.",
            "activation": "Non-linearity used between layers.",
        },
        "notes": "Because there's no output activation, MLPRegressor can predict any real "
                 "number, unlike MLPClassifier which is bounded to [0, 1] probabilities.",
    },
}

# Constructor names we actively look for in pasted code (handles both
# `LinearRegression(...)` and `linear_model.LinearRegression(...)` forms).
KNOWN_CLASSES = list(ALGO_KB.keys())


def _literal_or_str(node: ast.AST) -> Any:
    """Best-effort conversion of an AST value node to a Python value/string."""
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<expr>"


def _call_name(node: ast.Call) -> Optional[str]:
    """Return the plain class name for `Foo(...)` or `mod.sub.Foo(...)` calls."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def detect_algorithms(code: str) -> List[Dict[str, Any]]:
    """
    Scan pasted code for known estimator constructors. Returns a list of
    {class_name, var_name, params, lineno} in the order they appear.
    Falls back to a plain regex scan if the code doesn't parse as valid Python
    (e.g. a fragment copy-pasted without imports/indentation).
    """
    found: List[Dict[str, Any]] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None

    if tree is not None:
        # Map "var = ClassName(...)" assignments so we can show the variable name too.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                name = _call_name(call)
                if name in ALGO_KB:
                    params = {}
                    for kw in call.keywords:
                        if kw.arg is not None:
                            params[kw.arg] = _literal_or_str(kw.value)
                    var_name = None
                    if node.targets and isinstance(node.targets[0], ast.Name):
                        var_name = node.targets[0].id
                    found.append({
                        "class_name": name,
                        "var_name": var_name,
                        "params": params,
                        "lineno": getattr(node, "lineno", 0),
                    })
        # Also catch bare calls not assigned to a variable, e.g. inside a Pipeline(...)
        assigned_linenos = {f["lineno"] for f in found}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node, "lineno", -1) not in assigned_linenos:
                name = _call_name(node)
                if name in ALGO_KB:
                    params = {}
                    for kw in node.keywords:
                        if kw.arg is not None:
                            params[kw.arg] = _literal_or_str(kw.value)
                    found.append({
                        "class_name": name,
                        "var_name": None,
                        "params": params,
                        "lineno": getattr(node, "lineno", 0),
                    })

    if not found:
        # Regex fallback for non-parseable fragments — no params extracted.
        for cls in KNOWN_CLASSES:
            for m in re.finditer(rf"\b{cls}\s*\(", code):
                found.append({"class_name": cls, "var_name": None, "params": {}, "lineno": 0})

    found.sort(key=lambda f: f["lineno"])
    # De-duplicate identical (class, lineno) hits from the two AST passes.
    seen = set()
    deduped = []
    for f in found:
        key = (f["class_name"], f["lineno"], f["var_name"])
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped


def get_explanation(class_name: str) -> Optional[Dict[str, Any]]:
    return ALGO_KB.get(class_name)
