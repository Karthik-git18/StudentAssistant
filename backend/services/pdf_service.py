import os
import re
import logging
from pathlib import Path
import faiss
import numpy as np
import fitz  # PyMuPDF

from backend.config import Config
from backend.services.ai_service import (
    generate_embeddings, cache_faiss_index, get_cached_faiss,
    clear_faiss_cache, generate_response
)

logger = logging.getLogger(__name__)

# Constants
INDEX_DIR = Config.DB_PATH.parent / 'indexes'
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_TOP_K = 3
CONTEXT_WORD_LIMIT = 1200
POOR_MATCH_DISTANCE = 1.45

# Prompts and response static templates
_NOT_FOUND = "I couldn't find that information in the uploaded PDF."
_NO_INFO = "No sufficient information found."
_NO_TOPICS = "No topics found."

# Intent pattern matching definitions
_INTENT_PATTERNS = [
    ("viva", re.compile(r"\b(viva|oral|examiner|test me|ask me questions)\b")),
    ("mcq", re.compile(r"\b(mcq|multiple choice|quiz|trivia)\b")),
    ("topics", re.compile(r"\b(topics|syllabus|headings|sections|index|chapters|outline)\b")),
    ("summary", re.compile(r"\b(summary|summarize|overview|abstract|outline of document)\b")),
    ("define", re.compile(r"\b(define|definition|what is|meaning of)\b")),
    ("explain", re.compile(r"\b(explain|how does|mechanism|concept of|describe)\b")),
    ("key_points", re.compile(r"\b(key points|bullet points|main points|takeaways)\b")),
    ("long", re.compile(r"\b(long answer|detailed explanation|essay|elaborate)\b")),
    ("short", re.compile(r"\b(short answer|brief|concise explanation)\b")),
    ("advantages", re.compile(r"\b(advantages|benefits|pros|strengths)\b")),
    ("disadvantages", re.compile(r"\b(disadvantages|drawbacks|cons|weaknesses|limitations)\b")),
    ("difference", re.compile(r"\b(difference|compare|contrast|distinguish|versus|vs)\b")),
    ("interview", re.compile(r"\b(interview questions|job interview|ask questions)\b")),
    ("notes", re.compile(r"\b(notes|study guide|cheat sheet|summarised notes)\b")),
]

def extract_text_from_pdf(path: str):
    """
    Open PDF using PyMuPDF and extract paragraphs.
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}") from e
    
    pages = len(doc)
    cleaned_paragraphs = []
    seen_paragraphs = set()

    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)
            
            key = cleaned_text.lower()
            if key not in seen_paragraphs:
                seen_paragraphs.add(key)
                cleaned_paragraphs.append(cleaned_text)

    return "\n\n".join(cleaned_paragraphs), pages

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Split text into word-based chunks.
    """
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks

def _save_chunks(index_path: str, chunks: list):
    with open(index_path + ".chunks", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.replace("\n", " ") + "\n---\n")

def _load_chunks(index_path: str) -> list:
    side = index_path + ".chunks"
    if not os.path.exists(side):
        return []
    with open(side, "r", encoding="utf-8") as f:
        parts = f.read().split("\n---\n")
    return [p.strip() for p in parts if p.strip()]

def build_faiss_index(chunks: list, index_path, user_id=None):
    """
    Build isolated FAISS index and cache it.
    """
    logger.info(f"[RAG] Building FAISS index: {len(chunks)} chunks -> {index_path}")
    vecs = generate_embeddings(chunks)
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(index_path))
    _save_chunks(str(index_path), chunks)
    if user_id:
        cache_faiss_index(user_id, index, chunks)
    logger.info("[RAG] FAISS index built successfully")
    return index, chunks

def load_index(index_path, user_id=None):
    """
    Load isolated FAISS index from disk or memory cache.
    """
    if user_id:
        idx, chunks = get_cached_faiss(user_id)
        if idx is not None:
            return idx, chunks

    path_str = str(index_path)
    if not os.path.exists(path_str):
        return None, []
    if os.path.getsize(path_str) == 0:
        os.remove(path_str)
        return None, []

    try:
        idx = faiss.read_index(path_str)
    except Exception as e:
        logger.error(f"[RAG] Failed to read index {path_str}: {e}")
        try:
            os.remove(path_str)
        except OSError:
            pass
        return None, []

    chunks = _load_chunks(path_str)
    if user_id:
        cache_faiss_index(user_id, idx, chunks)
    return idx, chunks

def delete_document_index(index_path, cache_key=None):
    for p in (Path(index_path), Path(str(index_path) + ".chunks")):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    if cache_key:
        clear_faiss_cache(cache_key)

def query_index(index, chunks, question: str, top_k: int = FAISS_TOP_K) -> list:
    """
    Query FAISS index and return (distance, chunk_text) list.
    """
    if index is None or not chunks:
        return []
    
    q_emb = generate_embeddings(question)
    q_emb = np.array([q_emb], dtype=np.float32)
    
    # L2-normalize
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb / np.maximum(norms, 1e-12)
    
    distances, indices = index.search(q_emb, min(top_k, len(chunks)))
    
    results = []
    for d, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(chunks):
            results.append((float(d), chunks[idx]))
    return results

def retrieve_context(index, chunks, question: str, top_k: int = FAISS_TOP_K, check_similarity: bool = True) -> str:
    """
    Retrieve contexts from FAISS, skipping poor similarity values if requested.
    """
    matches = query_index(index, chunks, question, top_k=top_k)
    if not matches:
        return "No relevant content found."
    
    # Sort matches by distance (L2 distance: lower is better)
    matches = sorted(matches, key=lambda x: x[0])
    
    valid_chunks = []
    for d, text in matches:
        if check_similarity and d > POOR_MATCH_DISTANCE:
            # Skip chunks with very large distance
            continue
        valid_chunks.append(text)
        
    if not valid_chunks:
        return "No relevant content found."
        
    # Compile text and respect limit
    compiled = []
    words_count = 0
    for chunk in valid_chunks:
        words = chunk.split()
        if words_count + len(words) > CONTEXT_WORD_LIMIT:
            break
        compiled.append(chunk)
        words_count += len(words)
        
    return "\n\n".join(compiled)

# ══════════════════════════════════════════════════════════════════════════════
# HEADING EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _is_heading_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if line.endswith(('.', ',', ';', ':')):
        return False
    
    # Common chapter/section numbering
    if re.match(r'^(chapter|section|unit|part|module)\s+\d+', line, re.IGNORECASE):
        return True
    if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
        return True
    if re.match(r'^[I|V|X|L|C|D|M]+\.\s+[A-Z]', line):
        return True
    if re.match(r'^[A-Z\s\-]{4,80}$', line) and not line.isnumeric():
        return True
    return False

def extract_headings_from_text(pdf_text: str) -> list:
    lines = pdf_text.splitlines()
    headings = []
    seen = set()
    
    for line in lines:
        line = line.strip()
        if _is_heading_line(line):
            title = re.sub(r"^[\d\.\s]+", "", line).strip()
            if not title or len(title) < 4:
                continue
            key = title.lower()
            if key not in seen:
                seen.add(key)
                headings.append({"title": title, "description": ""})
            if len(headings) >= 20:
                break
                
    return headings

def detect_intent(question: str) -> str:
    q = question.lower().strip()
    for label, pattern in _INTENT_PATTERNS:
        if pattern.search(q):
            return label
    return "qa"

# ══════════════════════════════════════════════════════════════════════════════
# CLEAN & FORMAT ANSWERS (SAAS LEVEL MARKDOWN)
# ══════════════════════════════════════════════════════════════════════════════

def clean_output(text: str) -> str:
    """Clean markdown artifacts, tags, and titles from response text."""
    text = text.strip()
    # Remove wrappers
    for prefix in ['Summary:', 'Answer:', 'Response:', 'Output:', 'Explanation:']:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text

# ══════════════════════════════════════════════════════════════════════════════
# LEARNING INTENTS GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def generate_summary_response(context: str, user_id=None) -> str:
    if not context or context == "No relevant content found.":
        return _NO_INFO
        
    prompt = (
        f"Context information is below.\n"
        f"---------------------\n"
        f"{context}\n"
        f"---------------------\n"
        f"Based on the context, write a structured and professional summary.\n"
        f"Use headings (e.g. ### Overview), bullet points, and bold keywords to highlight key concepts.\n"
        f"Keep the language extremely professional, concise, and easy to read. Do not hallucinate."
    )
    return generate_response(prompt, user_id=user_id, system_instruction="You are an expert document summarizer. Respond in clear markdown.")

def generate_viva_response(context: str, topic: str = "", user_id=None) -> str:
    if not context or context == "No relevant content found.":
        return _NOT_FOUND
        
    prompt = (
        f"Context:\n{context}\n\n"
        f"Task: Generate 5-7 viva-voce examination questions with short answers based on the topic '{topic}' and context.\n"
        f"Format as a clean markdown list:\n"
        f"### Viva Questions\n"
        f"1. **Question**: <text>?\n"
        f"   - **Answer**: <short concise explanation>\n"
        f"Only use facts directly mentioned in the context."
    )
    return generate_response(prompt, user_id=user_id)

def generate_mcq_response(context: str, topic: str = "", user_id=None) -> str:
    if not context or context == "No relevant content found.":
        return _NOT_FOUND
        
    prompt = (
        f"Context:\n{context}\n\n"
        f"Task: Generate 5 Multiple Choice Questions (MCQs) about '{topic}' based on the context.\n"
        f"Format as markdown:\n"
        f"### Multiple Choice Questions\n"
        f"1. **Question**\n"
        f"   - A) <option>\n"
        f"   - B) <option>\n"
        f"   - C) <option>\n"
        f"   - D) <option>\n"
        f"   - *Correct Answer*: A/B/C/D\n"
        f"Provide correct answers clearly."
    )
    return generate_response(prompt, user_id=user_id)

def generate_notes_response(context: str, topic: str = "", user_id=None) -> str:
    if not context or context == "No relevant content found.":
        return _NOT_FOUND
        
    prompt = (
        f"Context:\n{context}\n\n"
        f"Task: Generate detailed study notes for the topic '{topic}' using the context.\n"
        f"Organize with clear markdown headers, bold key phrases, and structured bullet lists.\n"
        f"Keep the study notes readable, educational, and clean."
    )
    return generate_response(prompt, user_id=user_id)

def extract_topics(context: str, user_id=None) -> list:
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
    raw = generate_response(prompt, user_id=user_id)
    topics = []
    seen = set()

    for line in raw.splitlines():
        line = line.strip()
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

def _generic_qa(context: str, question: str, intent: str, user_id=None) -> str:
    instruction_map = {
        "define": "Define the terms clearly using bold keywords and bulleted attributes.",
        "explain": "Explain the concept step-by-step using ordered lists and bold keywords.",
        "key_points": "Summarize the key points as clear bullet points.",
        "long": "Provide a detailed explanation with structured headings and numbered lists.",
        "short": "Provide a concise 3-4 sentence explanation.",
        "advantages": "List the advantages/pros in a bulleted list with bold headlines.",
        "disadvantages": "List the disadvantages/cons in a bulleted list with bold headlines.",
        "difference": "Compare and contrast using structured sections or bulleted lists.",
        "interview": "Generate 5 interview questions with answers based on the context.",
    }
    instruction = instruction_map.get(intent, "Explain clearly using headers and bullet points.")
    
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question/Request: {question}\n\n"
        f"Instruction: {instruction}\n"
        f"Rules:\n"
        f"- Rely ONLY on the context.\n"
        f"- Use markdown (headings, bullets, bold keywords) for readability.\n"
        f"- Keep response concise, structured, and direct.\n"
        f"- If the information is not in the context, reply exactly with '{_NOT_FOUND}'"
    )
    return generate_response(prompt, user_id=user_id)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DISPATCH ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def route_learning_query(index, chunks: list, question: str, pdf_text: str = "", user_id=None) -> str:
    intent = detect_intent(question)
    logger.info(f"[RAG] Classified Intent: {intent} for question: {question[:40]}...")

    # 1. Topics intent
    if intent == "topics":
        if pdf_text and pdf_text.strip():
            headings = extract_headings_from_text(pdf_text)
            if headings:
                return "### Topics & Headings Extracted\n" + "\n".join(f"- **{h['title']}**" for h in headings)
        
        ctx = retrieve_context(index, chunks, "table of contents sections headings introduction syllabus chapters index outline", top_k=3, check_similarity=False)
        topics = extract_headings_from_text(ctx) if ctx != "No relevant content found." else []
        if not topics:
            return _NO_TOPICS
        return "### Document Topics\n" + "\n".join(f"- **{t['title']}**" for t in topics)

    # 2. Summary intent
    if intent == "summary":
        ctx = retrieve_context(index, chunks, "summary overview introduction key points main concepts conclusion", top_k=3, check_similarity=False)
        return generate_summary_response(ctx, user_id=user_id)

    # 3. Viva intent
    if intent == "viva":
        ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=False)
        return generate_viva_response(ctx, topic=question, user_id=user_id)

    # 4. MCQ intent
    if intent == "mcq":
        ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=False)
        return generate_mcq_response(ctx, topic=question, user_id=user_id)

    # 5. Notes intent
    if intent == "notes":
        ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=False)
        return generate_notes_response(ctx, topic=question, user_id=user_id)

    # 6. Generic Q&A intents
    if intent in ("define", "explain", "key_points", "long", "short", "advantages", "disadvantages", "difference", "interview"):
        ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=True)
        if ctx == "No relevant content found.":
            return _NOT_FOUND
        return _generic_qa(ctx, question, intent, user_id=user_id)

    # Default plain Q&A
    ctx = retrieve_context(index, chunks, question, top_k=3, check_similarity=True)
    if ctx == "No relevant content found.":
        return _NOT_FOUND
    
    prompt = (
        f"Context:\n{ctx}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer the question directly and concisely from the context. Use clean markdown styling. If not found, reply with '{_NOT_FOUND}'"
    )
    return generate_response(prompt, user_id=user_id)
