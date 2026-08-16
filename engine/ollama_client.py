"""
Optional integration with a LOCAL Ollama server (http://localhost:11434).
This never contacts an external service — the model runs entirely on the
user's own machine, and this module is only used if the user explicitly
enables "Local AI Mode" in the sidebar.
"""
import json
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/generate"


def is_ollama_available() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except Exception:
        return False


def ask_local_model(prompt: str, model: str = "phi3:mini", timeout: int = 30) -> str:
    """Send a prompt to a locally running Ollama model. Returns generated text
    or an error message if the local server isn't reachable."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError:
        return ("⚠️ Could not reach local Ollama server at localhost:11434. "
                "Run `ollama serve` and `ollama pull <model>` first.")
    except Exception as e:
        return f"⚠️ Local model error: {e}"
