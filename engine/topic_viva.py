"""
Offline, rule-based "viva question" generator for AI-domain code snippets.

This is the domain-aware sibling of `viva_engine.py`. Where `viva_engine.py`
asks generic Python/ML questions line-by-line, this module looks at code
pasted under a specific AI Project Mentor topic (NLP, SLM, Generative AI,
Agentic AI, RAG, Prompt Engineering) and asks the *concept-level* questions
an examiner would ask about THAT domain — tokenization choices, quantization
tradeoffs, GAN/VAE mechanics, agent loop termination, retrieval design, etc.

Detection is keyword/substring based (domain vocabulary doesn't map cleanly
onto a fixed set of AST call names the way scikit-learn estimators do), plus
a light AST pass to detect imports and class/function definitions. Everything
is static and local — no AI API calls, matching the rest of the app.
"""
import ast
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Per-topic keyword -> {questions, answer_points, category} rule tables.
# A rule fires if ANY of its keywords appears (case-insensitive, word-ish
# boundary) in the pasted code. Rules are checked in order; each topic's
# rules are independent of the others.
# ---------------------------------------------------------------------------

TOPIC_RULES: Dict[str, List[Dict[str, Any]]] = {

    "NLP": [
        {
            "keywords": ["tokeniz"],
            "questions": [
                "What does tokenization actually split the text into here, and why is that unit chosen?",
                "What would happen downstream if a token boundary were wrong (e.g. \"don't\" split badly)?",
            ],
            "answer_points": [
                "Tokenization breaks raw text into words/sub-words/characters so the model has discrete units to work with",
                "Bad tokenization propagates errors into every later step (vectorization, embeddings, predictions)",
            ],
            "category": "nlp_preprocessing",
        },
        {
            "keywords": ["stopword"],
            "questions": [
                "Why remove stopwords here, and what's a case where removing them could hurt performance?",
            ],
            "answer_points": [
                "Stopwords (the, is, and...) usually carry little topic/sentiment signal and add noise",
                "For tasks like sentiment or authorship, function words can actually carry signal — removing them isn't always safe",
            ],
            "category": "nlp_preprocessing",
        },
        {
            "keywords": ["stem", "lemmatiz"],
            "questions": [
                "What is the difference between stemming and lemmatization, and which one is used here?",
                "What's a word this step could get wrong, and why?",
            ],
            "answer_points": [
                "Stemming chops suffixes with rules (fast, sometimes produces non-words); lemmatization uses vocabulary/grammar to return a real dictionary base form",
                "Aggressive stemming can merge unrelated words (e.g. 'university' and 'universe') into the same stem",
            ],
            "category": "nlp_preprocessing",
        },
        {
            "keywords": ["tfidf", "tf-idf", "tfidfvectorizer"],
            "questions": [
                "What does the IDF part of TF-IDF do that a plain word-count vector doesn't?",
                "Why must the same fitted TF-IDF vectorizer be reused on the test set instead of re-fitting?",
            ],
            "answer_points": [
                "IDF down-weights words that appear in almost every document (like 'the'), since they don't help distinguish documents",
                "Re-fitting on test data would use a different vocabulary/weights, leaking test statistics and breaking a fair evaluation",
            ],
            "category": "nlp_vectorization",
        },
        {
            "keywords": ["countvectorizer", "bag of words", "bag-of-words", "bow"],
            "questions": [
                "What information does a Bag-of-Words representation throw away about the original sentence?",
                "How does vocabulary size affect the size of the resulting feature vectors?",
            ],
            "answer_points": [
                "Bag-of-Words discards word order and grammar — 'dog bites man' and 'man bites dog' get the same vector",
                "Each vocabulary word becomes a feature/column, so a larger vocabulary means wider, sparser vectors",
            ],
            "category": "nlp_vectorization",
        },
        {
            "keywords": ["ngram", "n-gram"],
            "questions": [
                "What does using n-grams (instead of single words) capture that unigrams miss?",
            ],
            "answer_points": [
                "N-grams keep short local word order (e.g. 'not good' vs 'good'), which unigram Bag-of-Words loses",
            ],
            "category": "nlp_vectorization",
        },
        {
            "keywords": ["word2vec", "glove", "fasttext", "embedding"],
            "questions": [
                "How is a word embedding different from a TF-IDF vector for the same word?",
                "Why do embeddings for 'king' and 'queen' end up close together in vector space?",
            ],
            "answer_points": [
                "TF-IDF vectors are sparse and based on counts; embeddings are dense, learned vectors that capture semantic similarity",
                "Embeddings are trained so words appearing in similar contexts get similar vectors — 'king'/'queen' share a lot of context",
            ],
            "category": "nlp_embeddings",
        },
        {
            "keywords": ["bert", "transformer", "autotokenizer", "automodel", "attention"],
            "questions": [
                "How does a transformer's representation of a word differ from a static embedding like Word2Vec?",
                "What role does the attention mechanism play here?",
            ],
            "answer_points": [
                "Transformer embeddings are contextual — the same word gets a different vector depending on the surrounding sentence",
                "Attention lets each token weigh how much every other token in the sequence matters when building its representation",
            ],
            "category": "nlp_transformers",
        },
        {
            "keywords": ["ner", "named entity"],
            "questions": [
                "What is this model identifying, and what would a correct output for a sample sentence look like?",
            ],
            "answer_points": [
                "Named Entity Recognition labels spans of text as entity types (PERSON, ORG, LOCATION, etc.)",
            ],
            "category": "nlp_tasks",
        },
        {
            "keywords": ["sentiment"],
            "questions": [
                "What are the possible output classes for this sentiment task, and how is the final label chosen from raw scores?",
            ],
            "answer_points": [
                "Typically positive/negative(/neutral); the label is usually the class with the highest predicted probability",
            ],
            "category": "nlp_tasks",
        },
        {
            "keywords": ["pipeline("],
            "questions": [
                "What is a Pipeline doing here, and why chain these steps together instead of calling them separately?",
            ],
            "answer_points": [
                "A Pipeline chains preprocessing + model into one object, so fit()/predict() consistently apply the same steps in order to train and new data",
            ],
            "category": "nlp_pipeline",
        },
    ],

    "SLM": [
        {
            "keywords": ["ollama"],
            "questions": [
                "Where does this model actually run when using Ollama, and what does that mean for latency and privacy?",
                "What would happen if the Ollama server weren't running when this code executes?",
            ],
            "answer_points": [
                "Ollama runs the model fully on the local machine (CPU/GPU), so no data leaves the device and there's no network round-trip",
                "The request would fail to connect (e.g. connection refused) — the app must handle that case gracefully",
            ],
            "category": "slm_runtime",
        },
        {
            "keywords": ["quantiz", "gguf", "q4", "q5", "q8"],
            "questions": [
                "What does quantization trade off to make this model runnable locally?",
                "Why might a Q4 quantized model behave slightly worse than the same model at full precision?",
            ],
            "answer_points": [
                "Quantization reduces the precision of the model's weights (e.g. 16-bit to 4-bit), cutting RAM/disk use and speeding inference at some accuracy cost",
                "Lower-bit weights are coarser approximations of the trained values, which can lose subtle distinctions the full-precision model captured",
            ],
            "category": "slm_runtime",
        },
        {
            "keywords": ["context_length", "context window", "max_tokens", "n_ctx"],
            "questions": [
                "What happens if the input text plus expected output exceeds this model's context window?",
                "Why does an SLM's smaller context window matter more than it would for a large cloud model?",
            ],
            "answer_points": [
                "Content beyond the context limit gets truncated or the call fails, depending on the client — either way information is lost",
                "SLMs often ship with much smaller context windows, so long documents may need chunking/retrieval instead of being pasted whole",
            ],
            "category": "slm_limits",
        },
        {
            "keywords": ["temperature"],
            "questions": [
                "What does the temperature parameter control, and what would setting it to 0 do to the outputs?",
            ],
            "answer_points": [
                "Temperature controls randomness in next-token sampling — higher values produce more varied/creative output, lower values are more deterministic",
                "Temperature 0 makes the model (near-)deterministic, always picking the highest-probability token",
            ],
            "category": "slm_prompting",
        },
        {
            "keywords": ["top_p", "top_k"],
            "questions": [
                "How does this sampling parameter change which tokens the model is allowed to pick from?",
            ],
            "answer_points": [
                "top_k restricts sampling to the k most likely next tokens; top_p restricts to the smallest set of tokens whose cumulative probability exceeds p",
            ],
            "category": "slm_prompting",
        },
        {
            "keywords": ["system prompt", "role\": \"system", "role='system'"],
            "questions": [
                "What is the system prompt doing here that a user prompt alone couldn't?",
            ],
            "answer_points": [
                "A system prompt sets persistent instructions/behavior for the whole conversation, separate from the user's actual question",
            ],
            "category": "slm_prompting",
        },
        {
            "keywords": ["phi3", "phi-3", "qwen", "gemma", "mistral", "tinyllama"],
            "questions": [
                "Roughly how large is this model, and how does that size relate to the hardware it needs to run smoothly?",
            ],
            "answer_points": [
                "Small models (roughly 1B-8B parameters) need proportionally less RAM/VRAM; a rough rule of thumb is a few bytes per parameter per quant level",
            ],
            "category": "slm_selection",
        },
    ],

    "Generative AI": [
        {
            "keywords": ["generator", "discriminator", "gan"],
            "questions": [
                "What are the generator and discriminator each trying to do, and why does training them together work?",
                "What does mode collapse look like in the generator's outputs, and why does it happen?",
            ],
            "answer_points": [
                "The generator tries to produce fake samples realistic enough to fool the discriminator; the discriminator tries to tell real from fake — the adversarial pressure improves both",
                "Mode collapse is when the generator produces very similar/identical outputs regardless of input noise, because it found one output that reliably fools the current discriminator",
            ],
            "category": "genai_gan",
        },
        {
            "keywords": ["vae", "variational", "reparameter", "kl_div", "kl loss", "latent"],
            "questions": [
                "What is the latent space in this model, and what does sampling from it let you do?",
                "Why does a VAE need the reparameterization trick during training?",
            ],
            "answer_points": [
                "The latent space is a compressed representation the encoder maps inputs into; sampling a point and decoding it generates a new, plausible output",
                "Reparameterization moves the random sampling outside the network graph so gradients can still flow back through the mean/variance, letting backpropagation work",
            ],
            "category": "genai_vae",
        },
        {
            "keywords": ["encoder", "decoder"],
            "questions": [
                "What does the encoder compress the input into, and what does the decoder reconstruct from it?",
            ],
            "answer_points": [
                "The encoder maps the input down to a smaller latent representation; the decoder maps that representation back up to something in the original data space",
            ],
            "category": "genai_architecture",
        },
        {
            "keywords": ["diffusion", "denois", "noise_scheduler", "unet"],
            "questions": [
                "What is being learned at each denoising step, and how does generation start from pure noise?",
                "Why does diffusion generation typically take many steps instead of one?",
            ],
            "answer_points": [
                "The model learns to predict and remove a small amount of noise at each step; generation starts from random noise and repeatedly denoises it toward a realistic sample",
                "Removing all noise in one step is much harder to learn accurately than gradually reversing many small noising steps",
            ],
            "category": "genai_diffusion",
        },
        {
            "keywords": ["nn.sequential", "torch.nn", "keras.sequential", "build_generator", "build_discriminator"],
            "questions": [
                "What shape is the input noise vector, and what shape is the final generated output?",
            ],
            "answer_points": [
                "A generator typically maps a low-dimensional random noise vector up to the full output shape (e.g. an image's pixel dimensions)",
            ],
            "category": "genai_architecture",
        },
        {
            "keywords": ["fid", "inception score"],
            "questions": [
                "Why is a metric like FID used instead of just checking the model's training loss?",
            ],
            "answer_points": [
                "Loss doesn't directly measure how realistic/diverse generated samples look; FID compares statistics of generated vs real images to approximate perceptual quality",
            ],
            "category": "genai_evaluation",
        },
    ],

    "Agentic AI": [
        {
            "keywords": ["while true", "max_iterations", "max_steps", "for _ in range"],
            "questions": [
                "What stops this agent's loop from running forever?",
                "What happens if the stop condition is never satisfied — what's the actual worst case?",
            ],
            "answer_points": [
                "A stopping condition such as a success check, max iteration count, or timeout/budget must be checked every loop iteration",
                "Without a hard cap, the agent could call tools indefinitely, burning time/cost with no guaranteed termination",
            ],
            "category": "agent_loop",
        },
        {
            "keywords": ["tool", "function_call", "tools=", "def execute", "def run_tool"],
            "questions": [
                "How does the agent decide WHICH tool to call next?",
                "What should happen if a tool call fails or returns an error — does this code handle that?",
            ],
            "answer_points": [
                "A planning step (LLM call or rule-based logic) selects the next action based on the current state/goal and available tools",
                "Failed tool calls should be caught and fed back as an observation so the agent can react, rather than crashing the whole loop",
            ],
            "category": "agent_tools",
        },
        {
            "keywords": ["planner", "plan_step", "planning"],
            "questions": [
                "What information does the planning step use to decide the next action?",
            ],
            "answer_points": [
                "The planner typically uses the goal, the history of past actions/observations, and the current state to choose the next step",
            ],
            "category": "agent_planning",
        },
        {
            "keywords": ["memory", "scratchpad", "conversation_history", "chat_history"],
            "questions": [
                "What does this agent remember between steps, and what happens to that memory once the run ends?",
            ],
            "answer_points": [
                "Memory (a scratchpad, conversation history, or stored files) lets the agent use earlier observations when deciding later actions",
                "Unless explicitly persisted (e.g. to disk/DB), in-memory state is usually lost once the process/run ends",
            ],
            "category": "agent_memory",
        },
        {
            "keywords": ["observation", "react", "thought:", "action:"],
            "questions": [
                "What is the difference between the agent's 'thought', 'action', and 'observation' at each step?",
            ],
            "answer_points": [
                "Thought is the agent's reasoning about what to do next; Action is the tool call it chooses; Observation is the result that action returns, which feeds back into the next thought",
            ],
            "category": "agent_reasoning",
        },
        {
            "keywords": ["retry", "except", "try:"],
            "questions": [
                "How many times will this code retry a failing step, and what happens after retries are exhausted?",
            ],
            "answer_points": [
                "Retries should be capped — retrying forever on a permanently failing tool wastes budget without ever succeeding",
            ],
            "category": "agent_robustness",
        },
        {
            "keywords": ["confirm", "human_approval", "input(\"", "approve"],
            "questions": [
                "Why might this action need human confirmation before executing?",
            ],
            "answer_points": [
                "Irreversible or high-stakes actions (sending, deleting, paying, publishing) warrant a human-in-the-loop check before the agent proceeds",
            ],
            "category": "agent_safety",
        },
    ],

    "RAG": [
        {
            "keywords": ["chunk", "splitter", "chunk_size"],
            "questions": [
                "What could go wrong if the chunk size here were too large? Too small?",
            ],
            "answer_points": [
                "Too large: chunks mix unrelated content, hurting retrieval precision. Too small: chunks lose surrounding context needed to answer correctly",
            ],
            "category": "rag_chunking",
        },
        {
            "keywords": ["embed", "sentencetransformer", "encode("],
            "questions": [
                "What does embedding a chunk turn it into, and why is that useful for retrieval?",
            ],
            "answer_points": [
                "Embedding maps text to a dense vector; similarity search can then find chunks whose vectors are close to the query's vector, i.e. semantically related",
            ],
            "category": "rag_embedding",
        },
        {
            "keywords": ["faiss", "chromadb", "vector store", "vectorstore", "index"],
            "questions": [
                "What is stored in this vector index, and what operation does it optimize?",
            ],
            "answer_points": [
                "It stores embedding vectors (plus references to their source chunks) and is optimized for fast nearest-neighbor similarity search",
            ],
            "category": "rag_index",
        },
        {
            "keywords": ["top_k", "similarity_search", "retriev"],
            "questions": [
                "What happens if top_k is too large? Too small?",
            ],
            "answer_points": [
                "Too large: irrelevant chunks get fed to the LLM, diluting the answer. Too small: the correct supporting chunk might be missed entirely",
            ],
            "category": "rag_retrieval",
        },
        {
            "keywords": ["context", "prompt ="],
            "questions": [
                "How is the retrieved context combined with the user's question in the final prompt?",
            ],
            "answer_points": [
                "Retrieved chunks are typically inserted into the prompt (often with instructions to answer only from that context) before the user's actual question",
            ],
            "category": "rag_generation",
        },
    ],

    "Prompt Engineering": [
        {
            "keywords": ["few-shot", "few_shot", "examples ="],
            "questions": [
                "Why include examples in the prompt instead of just describing the task?",
            ],
            "answer_points": [
                "Few-shot examples show the model the exact input/output pattern expected, which is often more reliable than instructions alone",
            ],
            "category": "prompt_technique",
        },
        {
            "keywords": ["system prompt", "role\": \"system", "role='system'"],
            "questions": [
                "What behavior is the system prompt trying to lock in across the whole conversation?",
            ],
            "answer_points": [
                "The system prompt sets persistent constraints/persona/format rules that should apply to every response, not just one turn",
            ],
            "category": "prompt_technique",
        },
        {
            "keywords": ["json", "format", "schema"],
            "questions": [
                "What would this code do if the model returned text that wasn't valid JSON?",
            ],
            "answer_points": [
                "Well-designed prompt code should validate/parse defensively and retry or handle the error, since LLMs can't be guaranteed to follow format instructions perfectly",
            ],
            "category": "prompt_format",
        },
        {
            "keywords": ["temperature"],
            "questions": [
                "Why might this prompt use a low temperature instead of a high one (or vice versa)?",
            ],
            "answer_points": [
                "Low temperature favors consistent, deterministic output (good for factual/structured tasks); high temperature favors variety (good for brainstorming/creative tasks)",
            ],
            "category": "prompt_technique",
        },
    ],
}

# Fallback conceptual questions per topic, always appended if the topic has
# any code at all — the "why this domain, big picture" follow-ups an examiner
# asks after the specifics.
TOPIC_CONCEPTUAL_QUESTIONS: Dict[str, List[str]] = {
    "NLP": [
        "If you had 10x more labeled data, would you expect classic TF-IDF models or transformer models to improve more? Why?",
        "How would you detect that your NLP model is picking up on a spurious correlation rather than real language signal?",
    ],
    "SLM": [
        "At what point would you stop trying to make a small local model work and fall back to a larger cloud model?",
        "How would you benchmark two candidate SLMs fairly on the same task?",
    ],
    "Generative AI": [
        "How do you know when to stop training a generative model — what would you look at besides the loss curve?",
        "What's a realistic risk of deploying this generative model to real users, and how would you mitigate it?",
    ],
    "Agentic AI": [
        "What's the worst thing this agent could do if one of its tools returned malicious or unexpected data?",
        "How would you debug a run where the agent looped 20 times without completing the task?",
    ],
    "RAG": [
        "How would you tell whether a wrong answer came from bad retrieval or bad generation?",
        "What would you do if the answer needed information split across multiple non-adjacent chunks?",
    ],
    "Prompt Engineering": [
        "How would you systematically compare two prompt variants instead of eyeballing a few outputs?",
        "What's a failure mode no amount of prompt tweaking can fix, that would instead require a different model or RAG?",
    ],
}

DEFAULT_NOTE = (
    "No specific {topic} keywords were recognized in this snippet — here are "
    "the general concept questions for this topic instead. Try pasting code "
    "that uses the topic's core techniques (e.g. tokenizer/vectorizer calls "
    "for NLP, an Ollama/quantization call for SLM, a generator/discriminator "
    "or VAE for Generative AI, a tool-calling loop for Agentic AI)."
)


def _find_imports(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(n.name for n in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    return mods


def generate_topic_viva_questions(topic: str, code: str) -> Dict[str, Any]:
    """
    Analyze pasted code under a given AI Project Mentor topic and produce
    concept-level viva questions specific to that domain.

    Returns:
      {
        "matched_rules": [ {keywords_hit: str, questions: [...], answer_points: [...], category: str}, ... ],
        "conceptual_questions": [str, ...],
        "imports_detected": [str, ...],
        "note": str | None   # shown when nothing domain-specific matched
      }
    """
    rules = TOPIC_RULES.get(topic, [])
    code_lower = code.lower()

    matched_rules: List[Dict[str, Any]] = []
    for rule in rules:
        hit = None
        for kw in rule["keywords"]:
            if kw.lower() in code_lower:
                hit = kw
                break
        if hit:
            matched_rules.append({
                "keyword_hit": hit,
                "questions": rule["questions"],
                "answer_points": rule["answer_points"],
                "category": rule["category"],
            })

    conceptual_questions = list(TOPIC_CONCEPTUAL_QUESTIONS.get(topic, []))
    imports_detected = _find_imports(code)

    note = None
    if not matched_rules and code.strip():
        note = DEFAULT_NOTE.format(topic=topic)

    return {
        "matched_rules": matched_rules,
        "conceptual_questions": conceptual_questions,
        "imports_detected": imports_detected,
        "note": note,
    }


def supported_topics() -> List[str]:
    return list(TOPIC_RULES.keys())
