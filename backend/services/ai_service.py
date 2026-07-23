import os
import json
import hashlib
import logging
import requests
from backend.config import Config
from backend.database import (
    get_cached_embedding, save_cached_embedding,
    get_cached_prompt, save_cached_prompt, create_log
)
import numpy as np

# Gemini SDK
try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

# Reusable HTTP session for connection pooling
_http_session = requests.Session()

def _get_hash(text: str) -> str:
    """Generate MD5 hash of text for caching keys."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS (GEMINI ONLY)
# ══════════════════════════════════════════════════════════════════════════════

def generate_embeddings(texts):
    """
    Generate embeddings using Gemini API, backed by SQLite caching.
    """
    if genai is None:
        raise RuntimeError("google-generativeai SDK is not installed.")

    api_key = Config.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

    single = isinstance(texts, str)
    if single:
        texts = [texts]

    results = []
    missing_texts = []
    missing_indices = []

    # Check cache first
    for idx, text in enumerate(texts):
        text_hash = _get_hash(text)
        cached = get_cached_embedding(text_hash)
        if cached:
            try:
                results.append(np.array(json.loads(cached), dtype=np.float32))
            except Exception:
                missing_texts.append(text)
                missing_indices.append(idx)
                results.append(None)
        else:
            missing_texts.append(text)
            missing_indices.append(idx)
            results.append(None)

    # Fetch missing from Gemini
    if missing_texts:
        try:
            logger.info(f"[AI SERVICE] Fetching {len(missing_texts)} embeddings from Gemini API")
            resp = genai.embed_content(
                model=Config.EMBEDDING_MODEL,
                content=missing_texts,
                task_type="retrieval_document"
            )
            
            # Extract embedding vectors
            embeddings_data = resp.get('embedding', []) if isinstance(resp, dict) else getattr(resp, 'embeddings', [])
            if not embeddings_data:
                # Fallback for single embedding responses or differing SDK properties
                embedding_data = getattr(resp, 'embedding', None)
                if embedding_data:
                    embeddings_data = [embedding_data]
            
            for local_idx, emb in enumerate(embeddings_data):
                # Handle direct array objects or dict wrapper objects
                vector = emb if isinstance(emb, list) else getattr(emb, 'values', [])
                if not vector and isinstance(emb, dict):
                    vector = emb.get('values', [])
                
                if not vector:
                    raise RuntimeError("Received malformed embedding array from Gemini SDK")

                orig_idx = missing_indices[local_idx]
                results[orig_idx] = np.array(vector, dtype=np.float32)
                
                # Cache embedding
                text_hash = _get_hash(missing_texts[local_idx])
                save_cached_embedding(text_hash, json.dumps(vector))
                
        except Exception as e:
            logger.error(f"[AI SERVICE] Gemini embeddings call failed: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    # Stack and return
    arr = np.vstack(results)
    
    # L2-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-12)

    return arr[0] if single else arr

# Cache for FAISS indexes in-memory to prevent reading file on every Q&A
_faiss_cache = {}

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

# ══════════════════════════════════════════════════════════════════════════════
# TEXT GENERATION (OPENROUTER ONLY)
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(prompt: str, user_id=None, system_instruction=None, max_tokens=800, temperature=0.2) -> str:
    """
    Generate a text response using OpenRouter, backed by SQLite caching.
    """
    api_key = Config.OPENROUTER_API_KEY
    if not api_key:
        logger.error("[AI SERVICE] OPENROUTER_API_KEY is not set.")
        return "Error: OpenRouter API key is missing."

    # Cache lookup
    prompt_hash = _get_hash(prompt + (system_instruction or ""))
    cached = get_cached_prompt(prompt_hash)
    if cached:
        logger.info("[AI SERVICE] Prompt cache hit!")
        return cached

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://studentai.org",
        "X-Title": "Student AI Assistant"
    }
    
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": Config.OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        if user_id:
            create_log(user_id, 'ai_request', f"Model: {Config.OPENROUTER_MODEL}")
            
        logger.info(f"[AI SERVICE] Calling OpenRouter model: {Config.OPENROUTER_MODEL}")
        response = _http_session.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"[AI SERVICE] API Error {response.status_code}: {response.text}")
            return f"Error: OpenRouter returned status {response.status_code}."

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Save to cache
        if content and not content.startswith("Error:"):
            save_cached_prompt(prompt_hash, content)
            
        return content

    except Exception as e:
        logger.error(f"[AI SERVICE] OpenRouter error: {e}")
        return "Error: Unable to connect to OpenRouter."

def generate_response_stream(prompt: str, user_id=None, system_instruction=None, max_tokens=800, temperature=0.2):
    """
    Generator that calls OpenRouter with streaming enabled (SSE) and yields tokens.
    Saves the final constructed text to the prompt cache.
    """
    api_key = Config.OPENROUTER_API_KEY
    if not api_key:
        yield "data: " + json.dumps({"error": "OpenRouter API key is missing."}) + "\n\n"
        return

    # Check cache first
    prompt_hash = _get_hash(prompt + (system_instruction or ""))
    cached = get_cached_prompt(prompt_hash)
    if cached:
        logger.info("[AI SERVICE] Prompt cache hit in streaming request!")
        yield "data: " + json.dumps({"choices": [{"delta": {"content": cached}}]}) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://studentai.org",
        "X-Title": "Student AI Assistant"
    }

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": Config.OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }

    if user_id:
        create_log(user_id, 'ai_request_stream', f"Model: {Config.OPENROUTER_MODEL}")

    full_response_text = ""
    try:
        response = _http_session.post(url, headers=headers, json=payload, stream=True, timeout=30)
        
        if response.status_code != 200:
            yield "data: " + json.dumps({"error": f"OpenRouter returned status {response.status_code}."}) + "\n\n"
            return

        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8').strip()
            if line_str.startswith("data: "):
                data_content = line_str[6:]
                if data_content == "[DONE]":
                    break
                
                try:
                    data_json = json.loads(data_content)
                    delta = data_json["choices"][0]["delta"]
                    token = delta.get("content", "")
                    full_response_text += token
                except Exception:
                    pass
                
                yield line_str + "\n\n"
        
        # Stream finished, save to cache
        if full_response_text.strip():
            save_cached_prompt(prompt_hash, full_response_text)
            
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"[AI SERVICE] OpenRouter streaming error: {e}")
        yield "data: " + json.dumps({"error": f"Streaming error: {e}"}) + "\n\n"
