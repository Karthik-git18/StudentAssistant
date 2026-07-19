"""
model_loader.py
===============
Handles OpenRouter API calls using requests.Session() and dynamic SentenceTransformer loading.
"""

import os
import logging
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Reusable session for OpenRouter API requests
_session = requests.Session()

# Cache for FAISS indexes
_faiss_cache = {}
_embedder = None


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


def get_embedder():
    """Dynamically load and return the SentenceTransformer embedder."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[EMBED] Loading SentenceTransformer...")
            _embedder = SentenceTransformer(EMBED_MODEL_NAME)
            logger.info("[EMBED] SentenceTransformer ready")
        except Exception as exc:
            logger.error("[EMBED] SentenceTransformer loading failed: %s", exc)
            _embedder = None
    return _embedder


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
