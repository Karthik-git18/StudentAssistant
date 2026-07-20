"""
rag.py
======
Retrieval-Augmented Generation helpers for the Student Learning Assistant.
"""

import json
import logging
import os
import re
import threading
from pathlib import Path

import faiss
import numpy as np
import fitz  # PyMuPDF

from backend.model_loader import (
    cache_faiss_index,
    clear_faiss_cache,
    get_cached_faiss,
    generate_response,
)
from backend.model_loader import generate_embeddings

logger = logging.getLogger(__name__)

# ── Directory for persisted FAISS indexes ─────────────────────────────────────
INDEX_DIR = Path(__file__).parent / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Retrieval constants ────────────────────────────────────────────────────────
FAISS_TOP_K         = 3      # Retrieve top 3 relevant chunks
CONTEXT_WORD_LIMIT  = 1200   # Limit context to 1200 words
POOR_MATCH_DISTANCE = 1.45   # L2 similarity score threshold

# ── Per-feature generation settings ───────────────────────────────────────────
_GEN_LA      = dict(max_new_tokens=180, temperature=0.2, do_sample=False, top_p=0.9, repetition_penalty=1.15)
_GEN_SUMMARY = dict(max_new_tokens=260, temperature=0.2, do_sample=False, top_p=0.9, repetition_penalty=1.2)
_GEN_TOPICS  = dict(max_new_tokens=180, temperature=0.1, do_sample=False, top_p=0.9, repetition_penalty=1.1)

# ── Static return strings ──────────────────────────────────────────────────────
_NOT_FOUND = "I couldn't find that information in the uploaded document."
_NO_INFO   = "No sufficient information found."
_NO_TOPICS = "No topics found."


# ══════════════════════════════════════════════════════════════════════════════
# PDF helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(path: str):
    """Return (cleaned_text, num_pages). Raises RuntimeError on corrupted file."""
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {exc}") from exc
    pages = len(doc)
    
    cleaned_paragraphs = []
    seen_paragraphs = set()

    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            
            # Remove blank lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)
            
            # Remove duplicate paragraphs while preserving page order
            key = cleaned_text.lower()
            if key not in seen_paragraphs:
                seen_paragraphs.add(key)
                cleaned_paragraphs.append(cleaned_text)

    return "\n\n".join(cleaned_paragraphs), pages


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """Split text into overlapping word-based chunks (300 words with 50 word overlap)."""
    tokens, chunks, i = text.split(), [], 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# FAISS helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def _encode(texts) -> np.ndarray:
    # Use remote Gemini embeddings via the model_loader wrapper
    return generate_embeddings(texts)


def _save_chunks(index_path: str, chunks: list):
    with open(index_path + ".chunks", "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.replace("\n", " ") + "\n---\n")


def _load_chunks(index_path: str) -> list:
    side = index_path + ".chunks"
    if not os.path.exists(side):
        return []
    with open(side, "r", encoding="utf-8") as fh:
        parts = fh.read().split("\n---\n")
    return [p.strip() for p in parts if p.strip()]


def build_faiss_index(chunks: list, index_path, user_id=None):
    logger.info("[RAG] Building FAISS index: %d chunks → %s", len(chunks), index_path)
    vecs  = _encode(chunks)
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(index_path))
    _save_chunks(str(index_path), chunks)
    if user_id:
        cache_faiss_index(user_id, index, chunks)
    logger.info("[RAG] FAISS index built successfully")
    return index, chunks


def build_faiss_index_async(chunks: list, index_path, user_id, callback=None):
    def _run():
        try:
            build_faiss_index(chunks, index_path, user_id)
            if callback:
                callback(True, None)
        except Exception as exc:
            logger.error("[RAG] Async build failed: %s", exc)
            if callback:
                callback(False, str(exc))
    threading.Thread(target=_run, daemon=False).start()


def append_faiss_index(chunks: list, index_path, user_id=None):
    index_path = str(index_path)
    if os.path.exists(index_path):
        index, existing = load_index(index_path, user_id)
        if index is None:
            return build_faiss_index(chunks, index_path, user_id)
        vecs = _encode(chunks)
        index.add(vecs)
        faiss.write_index(index, index_path)
        with open(index_path + ".chunks", "a", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(c.replace("\n", " ") + "\n---\n")
        all_chunks = existing + chunks
        if user_id:
            cache_faiss_index(user_id, index, all_chunks)
        return index, all_chunks
    return build_faiss_index(chunks, index_path, user_id)


def load_index(index_path, user_id=None):
    if user_id:
        idx, chunks = get_cached_faiss(user_id)
        if idx is not None:
            logger.debug("[RAG] Cache hit for user %s", user_id)
            return idx, chunks

    path_str = str(index_path)
    if not os.path.exists(path_str):
        return None, []
    if os.path.getsize(path_str) == 0:
        os.remove(path_str)
        return None, []

    try:
        idx = faiss.read_index(path_str)
    except Exception as exc:
        logger.error("[RAG] Failed to read index %s: %s", path_str, exc)
        try:
            os.remove(path_str)
        except OSError:
            pass
        return None, []

    chunks = _load_chunks(path_str)
    if user_id:
        cache_faiss_index(user_id, idx, chunks)
    logger.info("[RAG] Loaded %d chunks for user %s", len(chunks), user_id)
    return idx, chunks


def delete_document_index(index_path, cache_key=None):
    for p in (Path(index_path), Path(str(index_path) + ".chunks")):
        try:
            if p.exists():
                p.unlink()
        except OSError as exc:
            logger.warning("[RAG] Could not remove %s: %s", p, exc)
    if cache_key:
        clear_faiss_cache(cache_key)


def query_index(index, chunks, question: str, top_k: int = FAISS_TOP_K) -> list:
    """Legacy wrapper kept for backwards compatibility."""
    try:
        qvec = generate_embeddings([question])
    except Exception:
        return []
    D, I = index.search(qvec, top_k)
    return [chunks[i] for i in I[0] if 0 <= i < len(chunks)]


# ══════════════════════════════════════════════════════════════════════════════
# Context retrieval  (top-3, dedup, merge, 1200-word limit)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(
    index,
    chunks: list,
    query: str,
    top_k: int = FAISS_TOP_K,
    check_similarity: bool = False,
) -> str:
    """
    Retrieve the most relevant chunks for `query` sorted by similarity score.
    """
    if not chunks:
        return "No relevant content found."

    try:
        qvec = generate_embeddings([query])
    except Exception:
        return "No relevant content found."
    D, I = index.search(qvec, min(top_k, len(chunks)))

    if not len(D) or not len(I):
        return "No relevant content found."

    # Similarity score check
    if check_similarity and float(D[0][0]) > POOR_MATCH_DISTANCE:
        logger.info("[RAG] Best L2 %.4f > %.2f – rejecting", D[0][0], POOR_MATCH_DISTANCE)
        return "No relevant content found."

    # Deduplicate chunks & keep sorted by similarity score (as FAISS returns them sorted by L2 distance ascending)
    seen, retrieved = set(), []
    for idx in I[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        text = chunks[idx].strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            retrieved.append((int(idx), text))

    if not retrieved:
        return "No relevant content found."

    # Merge overlapping chunks if they are adjacent original indices
    retrieved_for_merging = sorted(retrieved, key=lambda x: x[0])
    groups, cur = [], [retrieved_for_merging[0]]
    for item in retrieved_for_merging[1:]:
        if item[0] - cur[-1][0] <= 1: # adjacent
            cur.append(item)
        else:
            groups.append(cur)
            cur = [item]
    groups.append(cur)

    merged = ["\n".join(t for _, t in g) for g in groups]
    full   = "\n\n".join(merged)

    words = full.split()
    if len(words) > CONTEXT_WORD_LIMIT:
        full = " ".join(words[:CONTEXT_WORD_LIMIT])

    return full


# ══════════════════════════════════════════════════════════════════════════════
# Output cleaning  (strips tokens, JSON, labels, duplicates)
# ══════════════════════════════════════════════════════════════════════════════

_RE_SPECIAL   = re.compile(r"<\|im_start\|>.*?<\|im_end\|>", re.DOTALL)
_RE_BARE_TOK  = re.compile(r"<\|im_start\|>|<\|im_end\|>|<s>|</s>")
_RE_CODE_FEN  = re.compile(r"```[a-zA-Z0-9]*\n?")
_RE_HASH      = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_RE_LABELS    = re.compile(
    r"^(human|assistant|system|user|h|a)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)
_RE_DUP_WORD  = re.compile(r"\b(\w+)(?:\s+\1)+\b", re.IGNORECASE)
_RE_SPACES    = re.compile(r" +")
_RE_NEWLINES  = re.compile(r"\n{3,}")
_RE_REF_LINE  = re.compile(
    r"^(references|bibliography|further reading|see also)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _json_to_text(text: str) -> str:
    """
    Safety net: if the LLM returned JSON, convert it to readable plain text.
    """
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return text

    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r",\s*\n", "\n", text)
        return text.strip()

    lines = []
    if isinstance(data, dict):
        for i, (key, val) in enumerate(data.items(), 1):
            if isinstance(val, dict):
                q = val.get("Question") or val.get("question") or ""
                a = val.get("Answer")   or val.get("answer")   or ""
                lines.append(f"Viva Question {i}\n\nQuestion:\n{q}\n\nAnswer:\n{a}")
            elif isinstance(val, str):
                lines.append(f"{key}: {val}")
    elif isinstance(data, list):
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                q   = item.get("Question") or item.get("question") or ""
                a   = item.get("Answer")   or item.get("answer")   or ""
                opt = {k: v for k, v in item.items() if k.upper() in ("A", "B", "C", "D")}
                if opt:
                    lines.append(f"Question {i}\n\n{q}")
                    for letter, val in sorted(opt.items()):
                        lines.append(f"{letter.upper()}) {val}")
                    correct = item.get("Correct") or item.get("correct_answer") or item.get("CorrectAnswer") or ""
                    if correct:
                        lines.append(f"Correct Answer: {correct}")
                else:
                    lines.append(f"Viva Question {i}\n\nQuestion:\n{q}\n\nAnswer:\n{a}")
            elif isinstance(item, str):
                lines.append(item)

    return "\n\n".join(lines) if lines else text


def clean_output(text: str) -> str:
    if not text:
        return ""

    text = _json_to_text(text)
    text = _RE_SPECIAL.sub("", text)
    text = _RE_BARE_TOK.sub("", text)
    text = _RE_CODE_FEN.sub("", text)
    text = text.replace("```", "")
    text = _RE_HASH.sub("", text)
    text = _RE_LABELS.sub("", text)
    text = _RE_DUP_WORD.sub(r"\1", text)
    text = text.replace("GitHub GitHub", "GitHub")

    seen, out = set(), []
    for line in text.splitlines():
        line = _RE_SPACES.sub(" ", line).strip()
        if not line:
            continue
        if _RE_REF_LINE.match(line):
            continue
        content = re.sub(r"^[\-\*•\d\.\)\s]+", "", line).strip()
        if not content:
            continue
        key = content.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)

    text = "\n".join(out)
    text = _RE_SPACES.sub(" ", text)
    text = _RE_NEWLINES.sub("\n\n", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Heading extraction directly from raw PDF text  (no LLM required)
# ══════════════════════════════════════════════════════════════════════════════

_RE_URL         = re.compile(r"https?://|www\.", re.I)
_RE_DIGIT_START = re.compile(r"^\d+[\.\)]\s+\w")
_RE_ROMAN_START = re.compile(r"^[IVXLCDM]+[\.\)]\s+\w", re.I)


def _is_heading_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if _RE_URL.search(line):
        return False
    words = line.split()
    if len(words) > 12:
        return False
    # Detect bold/large/numbered headings or short title lines
    if _RE_DIGIT_START.match(line) or _RE_ROMAN_START.match(line):
        return True
    if line.isupper() and len(words) >= 1:
        return True
    cap_count = sum(1 for w in words if w and w[0].isupper())
    if cap_count / len(words) >= 0.7 and len(words) >= 2:
        return True
    return False


def extract_headings_from_text(pdf_text: str) -> list:
    headings = []
    seen     = set()

    for line in pdf_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_heading_line(line):
            title = re.sub(r"^[\d\.\)\(IVXLCDMivxlcdm]+[\.\)]\s*", "", line).strip()
            title = re.sub(r"^[\s\-\*•]+", "", title).strip()
            if not title or len(title) < 3:
                continue
            key = title.lower()
            if key not in seen:
                seen.add(key)
                headings.append({"title": title, "description": ""})

    return headings


# ══════════════════════════════════════════════════════════════════════════════
# Intent detection
# ══════════════════════════════════════════════════════════════════════════════

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("mcq",          re.compile(r"\b(mcq|mcqs|multiple.?choice|quiz)\b",                    re.I)),
    ("viva",         re.compile(r"\b(viva|oral\s+exam|viva\s+questions?)\b",                re.I)),
    ("interview",    re.compile(r"\b(interview\s+questions?|hr\s+questions?)\b",             re.I)),
    ("summary",      re.compile(r"\b(summary|summarize|summarise|overview|brief)\b",        re.I)),
    ("topics",       re.compile(r"\b(topics?|headings?|sections?|index|table.?of.?contents?|important\s+topics?)\b", re.I)),
    ("define",       re.compile(r"\b(define|definition|meaning|what\s+is\s+meant)\b",       re.I)),
    ("explain",      re.compile(r"\b(explain|elaborate|describe|how\s+does|how\s+do)\b",    re.I)),
    ("notes",        re.compile(r"\b(notes?|study\s+notes?)\b",                             re.I)),
    ("key_points",   re.compile(r"\b(key\s+points?|bullet\s+points?|important\s+points?)\b", re.I)),
    ("long",         re.compile(r"\b(long\s+answers?|detailed\s+answers?|in\s+detail)\b",   re.I)),
    ("short",        re.compile(r"\b(short\s+answers?|brief\s+answers?|in\s+brief)\b",      re.I)),
    ("advantages",   re.compile(r"\b(advantages?|benefits?|pros?)\b",                       re.I)),
    ("disadvantages",re.compile(r"\b(disadvantages?|drawbacks?|cons?|limitations?)\b",      re.I)),
    ("difference",   re.compile(r"\b(difference|compare|comparison|vs|versus)\b",           re.I)),
]


def detect_intent(question: str) -> str:
    q = question.lower().strip()
    for label, pattern in _INTENT_PATTERNS:
        if pattern.search(q):
            return label
    return "qa"


# ══════════════════════════════════════════════════════════════════════════════
# Generation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _llm_call(prompt: str, **gen_kwargs) -> str:
    raw = generate_response(prompt)
    return clean_output(raw)


# ── Q&A ───────────────────────────────────────────────────────────────────────

def answer_question(context: str, question: str) -> str:
    if not context or context.strip() in ("", "No relevant content found."):
        return _NOT_FOUND

    prompt = (
        f"Context\n{context}\n\n"
        f"Question\n{question}\n\n"
        "Instructions\n"
        "Answer ONLY from the context.\n"
        "If the answer is unavailable say\n"
        f"\"{_NOT_FOUND}\"\n"
        "Never hallucinate.\n"
        "Never invent information."
    )
    result = _llm_call(prompt)
    return result if result.strip() else _NOT_FOUND


# ── Summary ───────────────────────────────────────────────────────────────────

def generate_summary(context: str) -> str:
    if not context or context.strip() in ("", "No relevant content found."):
        return _NO_INFO

    prompt = (
        "You are an expert document summarizer.\n"
        "Using ONLY the document context below, write a structured summary in your own words.\n"
        "Do NOT copy paragraphs directly. Remove all references, duplicate text, or GitHub citations.\n\n"
        "Your response MUST match this structure exactly:\n\n"
        "Overview\n"
        "<high-level description of the document>\n\n"
        "Objectives\n"
        "<bullet list of objectives>\n\n"
        "Key Concepts\n"
        "<bullet list of core concepts>\n\n"
        "Important Points\n"
        "<bullet list of key findings/points>\n\n"
        "Workflow\n"
        "<bullet list of processes/workflow>\n\n"
        "Technologies Used\n"
        "<bullet list of technologies used>\n\n"
        "Conclusion\n"
        "<concluding takeaway>\n\n"
        f"Document context:\n{context}\n\n"
        "Summary:"
    )
    result = _llm_call(prompt, **_GEN_SUMMARY)
    return result if result.strip() else _NO_INFO


# ── Topics Fallback ───────────────────────────────────────────────────────────

def extract_topics(context: str) -> list:
    if not context or context.strip() in ("", "No relevant content found."):
        return []

    prompt = (
        "You are an expert document analyzer.\n"
        "Extract ONLY the major topics and headings from the document below.\n"
        "Return ONLY a bullet list of topic names. Do NOT return JSON.\n\n"
        "Example output:\n"
        "• Project Introduction\n"
        "• Problem Statement\n"
        "• Objectives\n"
        "• Technology Stack\n"
        "• Architecture\n"
        "• Workflow\n"
        "• Dataset\n"
        "• EDA\n"
        "• Feature Engineering\n"
        "• Model Training\n"
        "• Results\n"
        "• Conclusion\n\n"
        f"Document context:\n{context}\n\n"
        "Topics:"
    )
    raw    = _llm_call(prompt, **_GEN_TOPICS)
    topics = []
    seen   = set()

    for line in raw.splitlines():
        line  = line.strip()
        title = re.sub(r"^[\-\*•\d\.\)\s]+", "", line).strip()
        if not title or len(title) < 3 or len(title) > 120:
            continue
        if title.endswith("?"):
            continue
        key = title.lower()
        if key not in seen:
            seen.add(key)
            topics.append({"title": title, "description": ""})
        if len(topics) >= 20:
            break

    return topics


# ── Viva Questions ────────────────────────────────────────────────────────────

def generate_viva_questions(context: str, topic: str = "") -> str:
    if not context or context.strip() in ("", "No relevant content found."):
        return _NOT_FOUND

    about = f" about {topic}" if topic.strip() else ""
    prompt = (
        "You are an intelligent Student Learning Assistant.\n"
        f"Generate 10 Viva Questions with short answers{about} using ONLY the document context below.\n"
        "Never use outside knowledge. Never output JSON.\n\n"
        "Use this format exactly:\n\n"
        "Viva Question 1\n\n"
        "Question:\n"
        "<viva question>\n\n"
        "Answer:\n"
        "<concise answer>\n\n"
        f"Document context:\n{context}\n\n"
        "Viva Questions:"
    )
    result = _llm_call(prompt, **_GEN_LA)
    return result if result.strip() else _NOT_FOUND


# ── MCQs ──────────────────────────────────────────────────────────────────────

def generate_mcqs(context: str, topic: str = "") -> str:
    if not context or context.strip() in ("", "No relevant content found."):
        return _NOT_FOUND

    about = f" about {topic}" if topic.strip() else ""
    prompt = (
        "You are an intelligent Student Learning Assistant.\n"
        f"Generate 10 MCQs{about} using ONLY the document context below.\n"
        "Never use outside knowledge. Never output JSON.\n\n"
        "Use this format exactly:\n\n"
        "Question 1\n\n"
        "<question text>\n\n"
        "A) <option>\n"
        "B) <option>\n"
        "C) <option>\n"
        "D) <option>\n\n"
        "Correct Answer: <letter>\n\n"
        f"Document context:\n{context}\n\n"
        "MCQs:"
    )
    result = _llm_call(prompt, **_GEN_LA)
    return result if result.strip() else _NOT_FOUND


# ── Notes ─────────────────────────────────────────────────────────────────────

def generate_notes(context: str, topic: str = "") -> str:
    if not context or context.strip() in ("", "No relevant content found."):
        return _NOT_FOUND

    prompt = (
        "You are an intelligent Student Learning Assistant.\n"
        f"Generate study notes for the topic '{topic}' using ONLY the document context below.\n\n"
        "Use this format exactly:\n\n"
        "Topic\n"
        "<topic title>\n\n"
        "Explanation\n"
        "<clear explanation in simple words>\n\n"
        "Key Points\n"
        "• <point 1>\n"
        "• <point 2>\n\n"
        f"Document context:\n{context}\n\n"
        "Notes:"
    )
    result = _llm_call(prompt, **_GEN_LA)
    return result if result.strip() else _NOT_FOUND


# ── Generic explanation / definition / notes ──────────────────────────────────

def _generic_qa(context: str, question: str, intent: str) -> str:
    if not context or context.strip() in ("", "No relevant content found."):
        return _NOT_FOUND

    instruction_map = {
        "define"        : "Write a clear definition under headings and bullets",
        "explain"       : "Explain step-by-step using headings and bullet points",
        "key_points"    : "List the key points as bullet points",
        "long"          : "Write a detailed long answer with headings and bullets",
        "short"         : "Write a short 3-5 sentence answer",
        "advantages"    : "List the advantages as bullet points",
        "disadvantages" : "List the disadvantages as bullet points",
        "difference"    : "Compare using clearly labelled bullet points",
        "interview"     : "Generate 5 interview questions with answers",
    }
    instruction = instruction_map.get(intent, "Answer using headings and bullet points")

    prompt = (
        "You are an intelligent Student Learning Assistant.\n"
        f"{instruction} for the request below using ONLY the document context.\n"
        "Never use outside knowledge. Never hallucinate. Never guess.\n"
        f"If the answer is not found, reply exactly with: \"{_NOT_FOUND}\"\n\n"
        "You MUST structure your response exactly as follows:\n\n"
        "Title\n"
        "<descriptive title>\n\n"
        "Explanation\n"
        "<the answer details in professional English>\n\n"
        "Important Points\n"
        "• <point 1>\n"
        "• <point 2>\n\n"
        "Conclusion\n"
        "<1-2 sentence concluding summary>\n\n"
        f"Document context:\n{context}\n\n"
        f"Request: {question}\n\n"
        "Response:"
    )
    result = _llm_call(prompt, **_GEN_LA)
    return result if result.strip() else _NOT_FOUND


# ══════════════════════════════════════════════════════════════════════════════
# Main dispatch  (single entry-point for the Learning Assistant)
# ══════════════════════════════════════════════════════════════════════════════

def route_learning_query(
    index,
    chunks: list,
    question: str,
    pdf_text: str = "",
) -> str:
    intent = detect_intent(question)
    logger.info("[RAG] Intent: %-14s | Question: %.60s", intent, question)

    # ── Topics ─────────────────────────────────────────────────────────────────
    if intent == "topics":
        if pdf_text and pdf_text.strip():
            headings = extract_headings_from_text(pdf_text)
            if headings:
                return "\n".join(f"• {h['title']}" for h in headings)

        ctx = retrieve_context(
            index, chunks,
            "table of contents sections headings introduction background methodology results conclusion",
            top_k=3, check_similarity=False,
        )
        topics = extract_topics(ctx)
        if not topics:
            return _NO_TOPICS
        return "\n".join(f"• {t['title']}" for t in topics)

    # ── Summary ────────────────────────────────────────────────────────────────
    if intent == "summary":
        ctx = retrieve_context(
            index, chunks,
            "summary overview introduction key points main concepts conclusion",
            top_k=3, check_similarity=False,
        )
        return generate_summary(ctx)

    # ── Viva ───────────────────────────────────────────────────────────────────
    if intent == "viva":
        ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=False)
        return generate_viva_questions(ctx, topic=question)

    # ── MCQ ────────────────────────────────────────────────────────────────────
    if intent == "mcq":
        ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=False)
        return generate_mcqs(ctx, topic=question)

    # ── Notes ──────────────────────────────────────────────────────────────────
    if intent == "notes":
        ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=False)
        return generate_notes(ctx, topic=question)

    # ── Intents handled by _generic_qa ────────────────────────────────────────
    if intent in ("define", "explain", "key_points", "long", "short",
                  "advantages", "disadvantages", "difference", "interview"):
        ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=True)
        if ctx == "No relevant content found.":
            return "I couldn't find that information in the uploaded PDF."
        return _generic_qa(ctx, question, intent)

    # ── Default: plain Q&A ────────────────────────────────────────────────────
    ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=True)
    if ctx == "No relevant content found.":
        return "I couldn't find that information in the uploaded PDF."
    return answer_question(ctx, question)
