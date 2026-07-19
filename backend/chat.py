"""
chat.py
=======
AI Chat blueprint – behaves like a general-purpose assistant (ChatGPT-style).
"""

import logging

from flask import Blueprint, jsonify, request, session

from database import create_chat
from model_loader import generate_response
from rag import clean_output

logger = logging.getLogger(__name__)
bp = Blueprint("chat", __name__)


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
        # Construct the requested prompt for AI Chat
        prompt = (
            "You are a helpful AI assistant.\n"
            "Give concise answers.\n"
            "Do not expose reasoning.\n\n"
            f"User: {msg}\n"
            "Assistant:"
        )
        
        raw = generate_response(prompt)
        
        if raw.startswith("Error:"):
            return jsonify({"error": raw}), 500
            
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
