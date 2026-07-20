"""
model_loader.py
===============
Handles OpenRouter API calls using requests.Session() and dynamic SentenceTransformer loading.
"""

import os
import logging
import requests
from dotenv import load_dotenv
import numpy as np

# Gemini / Google Generative AI SDK (used for embeddings)
try:
    import google.generativeai as genai
except Exception:
    genai = None

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Reusable session for OpenRouter API requests
_session = requests.Session()

# Cache for FAISS indexes
_faiss_cache = {}


def generate_response(prompt: str) -> str:
    """
    Generate a response using the OpenRouter API.
    Handles timeouts, rate limits, network errors, and invalid keys.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("[OpenRouter] OPENROUTER_API_KEY is not set.")
        return "Error: OpenRouter API key is missing. Please check your .env file configuration."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openrouter/free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = _session.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 401:
            logger.error("[OpenRouter] Unauthorized (401). Invalid API key.")
            return "Error: Invalid OpenRouter API key. Please check your credentials."
        
        if response.status_code == 429:
            logger.warning("[OpenRouter] Rate limit exceeded (429).")
            return "Error: Rate limit exceeded. Please try again in a moment."
            
        if response.status_code != 200:
            logger.error("[OpenRouter] API error: status %d, response: %s", response.status_code, response.text)
            return f"Error: OpenRouter API returned an error (status code {response.status_code})."
            
        data = response.json()
        if not data or "choices" not in data or not data["choices"]:
            logger.error("[OpenRouter] Empty or malformed response: %s", data)
            return "Error: Received an empty response from the OpenRouter API."
            
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            logger.warning("[OpenRouter] Generated content is empty.")
            return "Error: The model returned an empty response."
            
        return content

    except requests.exceptions.Timeout:
        logger.error("[OpenRouter] Request timed out.")
        return "Error: Request to the AI service timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error("[OpenRouter] Network error: %s", e)
        return "Error: A network error occurred while communicating with the AI service."
    except Exception as e:
        logger.error("[OpenRouter] Unexpected error: %s", e)
        return "Error: An unexpected error occurred while generating the response."


def generate_embeddings(texts):
    """Generate embeddings using the Google Gemini embedding model.

    This wrapper prefers modern SDK methods. It will attempt the following
    calls in order (if present in the installed SDK):

    1. `genai.get_embeddings(model="gemini-embedding-001", input=...)`
    2. `genai.embed(model="gemini-embedding-001", input=...)` (some SDKs)
    3. `genai.embed_text` / `genai.embed_content` (legacy)

    The function always returns an L2-normalized `numpy.ndarray` with dtype
    float32. If `texts` is a single string, a 1-D array is returned.
    """

    if genai is None:
        raise RuntimeError("google.generativeai SDK is not installed; pip install google-generativeai")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    # Configure SDK if possible (no-op for some versions)
    try:
        genai.configure(api_key=api_key)
    except Exception:
        # Older/newer SDKs may not expose configure; ignore and continue
        pass

    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    embeddings = []

    # Preferred modern method
    if hasattr(genai, "get_embeddings"):
        try:
            resp = genai.get_embeddings(model="gemini-embedding-001", input=texts)
            data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
            if not data:
                raise RuntimeError("No embedding data returned from genai.get_embeddings()")
            for item in data:
                emb = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
                embeddings.append(emb)
        except Exception as exc:
            logger.error("[EMBED] genai.get_embeddings failed: %s", exc)
            raise

    # Fallbacks for older SDKs
    elif hasattr(genai, "embed"):
        try:
            # Some SDKs accept list inputs, others per-item — try list first
            resp = genai.embed(model="gemini-embedding-001", input=texts)
            data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
            if data:
                for item in data:
                    emb = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
                    embeddings.append(emb)
            else:
                # per-item fallback
                for t in texts:
                    r = genai.embed(model="gemini-embedding-001", input=t)
                    d = getattr(r, "data", None) or (r.get("data") if isinstance(r, dict) else None)
                    if d:
                        embeddings.append(d[0].get("embedding") if isinstance(d[0], dict) else getattr(d[0], "embedding", None))
        except Exception as exc:
            logger.error("[EMBED] genai.embed failed: %s", exc)
            raise

    elif hasattr(genai, "embed_text") or hasattr(genai, "embed_content"):
        # Legacy: call per-item
        try:
            for text in texts:
                if hasattr(genai, "embed_text"):
                    r = genai.embed_text(model="gemini-embedding-001", text=text)
                else:
                    # embed_content historically used model="models/embedding-001";
                    # use gemini-embedding-001 and task_type if supported
                    r = genai.embed_content(model="gemini-embedding-001", content=text)

                d = getattr(r, "data", None) or (r.get("data") if isinstance(r, dict) else None)
                if d:
                    embeddings.append(d[0].get("embedding") if isinstance(d[0], dict) else getattr(d[0], "embedding", None))
                else:
                    # Some legacy returns the embedding directly
                    emb = r.get("embedding") if isinstance(r, dict) else getattr(r, "embedding", None)
                    embeddings.append(emb)
        except Exception as exc:
            logger.error("[EMBED] Legacy embedding call failed: %s", exc)
            raise

    else:
        raise RuntimeError(
            "Installed google.generativeai SDK does not expose a supported embedding method. "
            "Please install or upgrade to the latest `google-generativeai` package and set GEMINI_API_KEY."
        )

    # Validate and convert
    if not embeddings or any(e is None for e in embeddings):
        raise RuntimeError("Failed to obtain embeddings from Gemini API")

    arr = np.array(embeddings, dtype=np.float32)
    # L2-normalize rows
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-12)

    return arr[0] if single else arr


def cache_faiss_index(user_id, index, chunks):
    _faiss_cache[user_id] = {"index": index, "chunks": chunks}


def get_cached_faiss(user_id):
    entry = _faiss_cache.get(user_id, {})
    return entry.get("index"), entry.get("chunks")


def clear_faiss_cache(user_id=None):
    if user_id:
        _faiss_cache.pop(user_id, None)
    else:
        _faiss_cache.clear()
