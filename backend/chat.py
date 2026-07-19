"""
chat.py
=======
AI Chat blueprint – behaves like a general-purpose assistant (ChatGPT-style).
"""

import logging

from flask import Blueprint, jsonify, request, session

from database import create_chat
from model_loader import get_llm, get_llm_error
from rag import clean_output

logger = logging.getLogger(__name__)
bp = Blueprint("chat", __name__)


_SYSTEM_PROMPT = (
    "You are an AI Assistant similar to ChatGPT.\n"
    "Behave like ChatGPT. Answer in professional English.\n"
    "Never repeat sentences. Never repeat paragraphs.\n\n"
    "If the explanation is long, you MUST structure your response into these exact sections:\n\n"
    "Definition\n"
    "<definition details>\n\n"
    "Explanation\n"
    "<explanation details>\n\n"
    "Example\n"
    "<illustrative example>\n\n"
    "Applications\n"
    "<real-world applications>\n\n"
    "Conclusion\n"
    "<concluding takeaway>\n"
)

# Generation settings for chat
_GEN_CHAT = dict(
    max_new_tokens=220,
    temperature=0.3,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.15,
)


@bp.route("/api/chat", methods=["POST"])
def chat_api():
    """General-purpose AI chat endpoint (no PDF context)."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Authentication required"}), 401

    if not request.is_json:
        return jsonify({"error": "Request must use application/json."}), 415

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    logger.info("[CHAT] User %s: %.60s", user_id, msg)

    try:
        llm = get_llm()
        if llm is None:
            err = get_llm_error() or "Unknown error"
            logger.error("[CHAT] LLM unavailable: %s", err)
            return jsonify({"error": f"LLM not available: {err}"}), 503

        prompt = f"{_SYSTEM_PROMPT}\nUser: {msg}\nAssistant:"
        raw    = llm.generate(prompt, **_GEN_CHAT)
        answer = clean_output(raw)

        if not answer.strip():
            answer = "I'm not sure how to answer that. Could you rephrase?"

        # Persist (non-fatal)
        try:
            create_chat(user_id, "user",      msg)
            create_chat(user_id, "assistant", answer)
        except Exception:
            logger.exception("[CHAT] DB save failed for user %s", user_id)

        logger.info("[CHAT] Response sent to user %s", user_id)
        return jsonify({"answer": answer})

    except Exception:
        logger.exception("[CHAT] Generation failed for user %s", user_id)
        return jsonify({"error": "Unable to generate a response. Check server logs."}), 500
