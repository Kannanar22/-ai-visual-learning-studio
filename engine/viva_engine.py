"""
Offline, rule-based "viva question" generator.
Given a Python snippet or script, it uses the `ast` module to detect code
constructs (imports, loops, conditionals, functions, classes, ML calls like
.fit()/.predict(), etc.) and produces the kinds of questions an examiner
would likely ask about that exact line in a viva / oral exam — plus a short
set of expected-answer bullet points to help the student prepare.

No AI APIs are used — everything is templated from a local rule database.
"""
import ast
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Questions tied to a specific method/function call, e.g. model.fit(X, y)
# ---------------------------------------------------------------------------
METHOD_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "fit": {
        "questions": [
            "What does calling `.fit()` do here, in your own words?",
            "What do the arguments passed to `.fit()` represent?",
            "What is actually being learned internally when `.fit()` runs?",
            "What error would you get if you called `.predict()` before `.fit()`?",
        ],
        "answer_points": [
            "Trains the model on features (X) and labels (y)",
            "Learns internal parameters (weights/coefficients) that minimize error",
            "Calling predict() before fit() raises a NotFittedError",
        ],
        "category": "machine_learning_training",
    },
    "predict": {
        "questions": [
            "What does `.predict()` return?",
            "How is `.predict()` different from `.fit()`?",
            "Can `.predict()` be called on data the model has never seen? Why is that useful?",
        ],
        "answer_points": [
            "Returns model outputs/predictions for new input data",
            "predict() applies parameters already learned by fit(); it doesn't learn anything new",
            "Yes — that's the whole point of generalization to unseen data",
        ],
        "category": "machine_learning_inference",
    },
    "fit_transform": {
        "questions": [
            "Why is `fit_transform()` used instead of calling `fit()` and `transform()` separately?",
            "Why should you use only `.transform()` (not `fit_transform()`) on the test set?",
        ],
        "answer_points": [
            "It's a convenience method that fits and transforms in a single call",
            "Using fit_transform() on test data would leak test-set statistics into preprocessing",
        ],
        "category": "preprocessing",
    },
    "transform": {
        "questions": [
            "What transformation is being applied here, and why is it necessary?",
            "Why must the same transformer instance be reused instead of creating a new one?",
        ],
        "answer_points": [
            "Reshapes/rescales data using parameters already learned by a prior fit()",
            "Reusing the same transformer keeps train/test data on the same scale",
        ],
        "category": "preprocessing",
    },
    "train_test_split": {
        "questions": [
            "Why do we split the dataset into training and testing sets?",
            "What does the `test_size` parameter control?",
            "What could go wrong if you evaluated the model on the same data used to train it?",
            "What is the role of `random_state` here, and why might you fix it?",
        ],
        "answer_points": [
            "Lets you measure performance on data the model hasn't memorized",
            "test_size controls the fraction of data held out for testing",
            "Evaluating on training data gives an overly optimistic, misleading accuracy",
            "random_state makes the split reproducible across runs",
        ],
        "category": "data_preparation",
    },
    "compile": {
        "questions": [
            "What three things does `.compile()` configure for this neural network?",
            "What would happen if you picked a different loss function here?",
        ],
        "answer_points": [
            "Optimizer, loss function, and evaluation metric(s)",
            "The loss function shapes what 'error' means, changing how weights get updated",
        ],
        "category": "deep_learning_setup",
    },
    "array": {
        "questions": [
            "Why convert this list into a NumPy array instead of using a plain Python list?",
        ],
        "answer_points": [
            "NumPy arrays support fast, vectorized math operations that plain lists don't",
        ],
        "category": "numpy",
    },
    "DataFrame": {
        "questions": [
            "What structure does a Pandas DataFrame have, and why use it here instead of a list/array?",
        ],
        "answer_points": [
            "A 2D labeled table with rows and columns, like a spreadsheet",
            "DataFrames keep column names/types and support easy filtering, grouping, and merging",
        ],
        "category": "pandas",
    },
    "read_csv": {
        "questions": [
            "What would happen if the file path passed to `read_csv()` were wrong?",
            "How would you handle missing values that might exist in this CSV?",
        ],
        "answer_points": [
            "It would raise a FileNotFoundError",
            "e.g. df.dropna(), df.fillna(), or inspecting with df.isnull().sum()",
        ],
        "category": "pandas",
    },
}

# General "big picture" questions asked once per category detected anywhere
# in the script — the kind of follow-up an examiner asks after the line-by-line.
CATEGORY_CONCEPTUAL_QUESTIONS: Dict[str, List[str]] = {
    "machine_learning_training": [
        "How would you check if this model is overfitting?",
        "What metric would you use to evaluate this model, and why that one?",
        "What would you do if the model's accuracy on test data was much lower than on training data?",
    ],
    "data_preparation": [
        "What would happen if you forgot to split your data before training?",
        "How do you decide the right train/test split ratio?",
    ],
    "preprocessing": [
        "Why is it important to preprocess data the same way for train and test sets?",
    ],
    "deep_learning_setup": [
        "What happens during one epoch of training?",
        "How would you tell if this network is underfitting?",
    ],
    "control_flow": [
        "Can you trace through this code by hand for a specific input and state the output?",
        "What would happen if the loop/condition boundary were off by one?",
    ],
    "function_definition": [
        "What is the time complexity of this function, and why?",
        "What edge cases might break this function?",
    ],
    "class_definition": [
        "Why was this implemented as a class rather than standalone functions?",
        "What state does this class maintain between method calls?",
    ],
    "import": [
        "What would happen if this library weren't installed?",
    ],
}

DEFAULT_QUESTIONS = [
    "What is this line of code doing?",
    "Why is this line necessary for the program to work correctly?",
]


def _expr_text(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _questions_for_statement(node) -> Dict[str, Any]:
    """Return {questions, answer_points, category} for one top-level statement."""
    # Check for a recognizable method/function call first (most specific)
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _get_call_name(child)
            if name in METHOD_QUESTIONS:
                return dict(METHOD_QUESTIONS[name])

    # Fall back to statement-type-based questions
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        modules = []
        if isinstance(node, ast.Import):
            modules = [n.name for n in node.names]
        else:
            modules = [node.module or ""]
        mod_text = ", ".join(modules)
        return {
            "questions": [
                f"Why is `{mod_text}` imported in this program?",
                f"What functionality does the `{mod_text}` library provide here?",
            ],
            "answer_points": [f"{mod_text} provides functions/classes used later in the script"],
            "category": "import",
        }

    if isinstance(node, ast.Assign):
        targets = ", ".join(_expr_text(t) for t in node.targets)
        value_text = _expr_text(node.value)
        return {
            "questions": [
                f"What value does `{targets}` hold after this line runs?",
                f"Why is `{targets}` needed later in the program?",
                f"What would change if `{value_text}` were replaced with a different expression?",
            ],
            "answer_points": [f"`{targets}` stores the result of evaluating `{value_text}`"],
            "category": "assignment",
        }

    if isinstance(node, ast.For):
        return {
            "questions": [
                f"What is `{_expr_text(node.iter)}` being iterated over, and how many times will the loop run?",
                f"What role does `{_expr_text(node.target)}` play on each iteration?",
                "What would happen if the sequence being iterated over were empty?",
            ],
            "answer_points": [
                "The loop runs once per element in the iterable",
                "Empty iterable → loop body never executes",
            ],
            "category": "control_flow",
        }

    if isinstance(node, ast.While):
        return {
            "questions": [
                f"What is the stopping condition of this loop: `{_expr_text(node.test)}`?",
                "What would happen if that condition were never false?",
            ],
            "answer_points": ["An always-true condition causes an infinite loop"],
            "category": "control_flow",
        }

    if isinstance(node, ast.If):
        return {
            "questions": [
                f"What condition is being evaluated: `{_expr_text(node.test)}`?",
                "What are the two possible execution paths, and what triggers each?",
            ],
            "answer_points": ["If the condition is true the `if` branch runs, otherwise the `else` branch runs"],
            "category": "control_flow",
        }

    if isinstance(node, ast.FunctionDef):
        params = ", ".join(a.arg for a in node.args.args)
        return {
            "questions": [
                f"What is the purpose of the function `{node.name}`?",
                f"What do the parameters `{params}` represent?",
                f"What does `{node.name}` return, and under what conditions?",
                f"What edge cases could break `{node.name}`?",
            ],
            "answer_points": [f"`{node.name}` encapsulates reusable logic parameterized by ({params})"],
            "category": "function_definition",
        }

    if isinstance(node, ast.ClassDef):
        return {
            "questions": [
                f"Why is `{node.name}` implemented as a class instead of a function?",
                f"What attributes/state does `{node.name}` maintain?",
            ],
            "answer_points": ["A class bundles data (attributes) and behavior (methods) together"],
            "category": "class_definition",
        }

    return {
        "questions": list(DEFAULT_QUESTIONS),
        "answer_points": [],
        "category": "general",
    }


def generate_viva_questions(code: str) -> Dict[str, Any]:
    """
    Analyze a Python script and produce viva-style questions.

    Returns:
      {
        "line_questions": [
            {"line": int, "code": str, "questions": [...], "answer_points": [...], "category": str}
        ],
        "conceptual_questions": [str, ...]   # deduplicated, category-driven follow-ups
      }
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "line_questions": [{
                "line": 0, "code": code, "category": "syntax_error",
                "questions": [f"This code has a syntax error ({e.msg}) — can you spot and fix it?"],
                "answer_points": [],
            }],
            "conceptual_questions": [],
        }

    line_questions = []
    categories_seen = []

    for node in tree.body:
        info = _questions_for_statement(node)
        snippet = _expr_text(node)
        line_questions.append({
            "line": getattr(node, "lineno", None),
            "code": snippet,
            "questions": info["questions"],
            "answer_points": info.get("answer_points", []),
            "category": info["category"],
        })
        if info["category"] not in categories_seen:
            categories_seen.append(info["category"])

    conceptual_questions: List[str] = []
    for cat in categories_seen:
        for q in CATEGORY_CONCEPTUAL_QUESTIONS.get(cat, []):
            if q not in conceptual_questions:
                conceptual_questions.append(q)

    return {"line_questions": line_questions, "conceptual_questions": conceptual_questions}
