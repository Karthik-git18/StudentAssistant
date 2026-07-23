import os
import json
import uuid
import logging
from pathlib import Path
from flask import Blueprint, render_template, render_template_string, request, jsonify, session, Response, send_from_directory, g
from werkzeug.utils import secure_filename
from hashlib import sha256

from backend.config import Config
from backend.middleware import login_required
from backend.database import (
    get_user_by_id, count_user_items, update_user, change_password, delete_user,
    create_upload, is_duplicate_upload, get_user_uploads, get_upload, delete_upload,
    get_recent_uploads, get_recent_chats, get_recent_plans,
    create_chat, get_chat_session_history, get_user_chat_sessions, rename_chat_session, delete_chat_session,
    create_resume, get_resume, get_recent_resume, get_user_resumes, delete_resume,
    create_log
)
from backend.services.ai_service import generate_response, generate_response_stream
from backend.services.pdf_service import (
    extract_text_from_pdf, chunk_text, build_faiss_index, load_index, INDEX_DIR,
    retrieve_context, route_learning_query, delete_document_index
)

logger = logging.getLogger(__name__)
student_bp = Blueprint('student', __name__)
from backend.database import (
    get_conn,
    get_user_by_id,
    count_user_items,
    update_user,
    change_password,
    delete_user,
    create_upload,
    is_duplicate_upload,
    get_user_uploads,
    get_upload,
    delete_upload,
    get_recent_uploads,
    get_recent_chats,
    get_recent_plans,
    create_chat,
    get_chat_session_history,
    get_user_chat_sessions,
    rename_chat_session,
    delete_chat_session,
    create_resume,
    get_resume,
    get_recent_resume,
    get_user_resumes,
    delete_resume,
    create_log
)
# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE RENDER ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@student_bp.route('/home')
@login_required
def home():
    user_id = session['user_id']
    user = get_user_by_id(user_id)
    stats = count_user_items(user_id)
    recent_uploads = get_recent_uploads(user_id, limit=3)
    recent_chats = get_recent_chats(user_id, limit=3)
    recent_resume = get_recent_resume(user_id)
    
    # Calculate a simple "Study Sessions" metric: count logins from logs
    from backend.database import get_conn
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM logs WHERE user_id = ? AND action = 'login'", (user_id,))
    study_sessions = cur.fetchone()[0]
    
    # Get last upload path/action for "continue last session"
    last_doc = recent_uploads[0] if recent_uploads else None
    
    return render_template(
        'home.html',
        user=user,
        stats=stats,
        study_sessions=study_sessions,
        recent_uploads=recent_uploads,
        recent_chats=recent_chats,
        recent_resume=recent_resume,
        last_doc=last_doc
    )

@student_bp.route('/learning')
@login_required
def learning():
    return render_template('learning.html')

@student_bp.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@student_bp.route('/resume')
@login_required
def resume():
    return render_template('resume.html')

@student_bp.route('/profile')
@login_required
def profile():
    user_id = session['user_id']
    user = get_user_by_id(user_id)
    stats = count_user_items(user_id)
    
    # Study sessions
    from backend.database import get_conn
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM logs WHERE user_id = ? AND action = 'login'", (user_id,))
    study_sessions = cur.fetchone()[0]

    return render_template('profile.html', user=user, stats=stats, study_sessions=study_sessions)

# ══════════════════════════════════════════════════════════════════════════════
# LEARNING MODULE APIs
# ══════════════════════════════════════════════════════════════════════════════

@student_bp.route('/upload_pdf', methods=['POST'])
@login_required
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file part in upload request'}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400

        if not f.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'message': 'Only PDF files allowed'}), 400

        user_id = session['user_id']
        filename = secure_filename(f.filename)
        
        file_bytes = f.read()
        file_hash = sha256(file_bytes).hexdigest()
        
        if is_duplicate_upload(user_id, file_hash):
            return jsonify({'success': False, 'message': 'Duplicate PDF detected. This file has already been uploaded.'}), 409

        upload_id = create_upload(user_id, filename, None, file_hash)
        storage_key = f"{user_id}_{upload_id}_{filename}"
        dest = Config.UPLOAD_FOLDER / storage_key
        
        with open(dest, 'wb') as out_file:
            out_file.write(file_bytes)

        # Process PDF and index FAISS
        text, pages = extract_text_from_pdf(str(dest))
        if not text or not text.strip():
            # Rollback
            delete_upload(user_id, upload_id)
            dest.unlink()
            return jsonify({'success': False, 'message': 'PDF text extraction returned empty contents.'}), 400

        # Update uploads table with pages & storage_key
        from backend.database import get_conn
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("UPDATE uploads SET pages=?, storage_key=? WHERE id=?", (pages, storage_key, upload_id))
        conn.commit()

        chunks = chunk_text(text)
        if not chunks:
            # Rollback
            delete_upload(user_id, upload_id)
            dest.unlink()
            return jsonify({'success': False, 'message': 'Could not extract chunks.'}), 400

        # Create Index
        index_path = INDEX_DIR / f"document_{user_id}_{upload_id}.index"
        build_faiss_index(chunks, index_path, user_id=f"{user_id}_{upload_id}")
        
        # Save sidecar raw text for topics extraction
        text_path = INDEX_DIR / f"document_{user_id}_{upload_id}.text"
        with open(text_path, 'w', encoding='utf-8') as tf:
            tf.write(text)

        # Set active doc
        session['current_document_id'] = upload_id

        return jsonify({'success': True, 'message': 'File uploaded and parsed successfully!', 'document_id': upload_id, 'pages': pages})
    except Exception as e:
        logger.exception("Upload PDF error")
        return jsonify({'success': False, 'message': f'Server upload error: {e}'}), 500

@student_bp.route('/api/documents', methods=['GET'])
@login_required
def get_documents():
    docs = get_user_uploads(session['user_id'])
    return jsonify({'success': True, 'documents': docs})

@student_bp.route('/api/documents/<int:doc_id>', methods=['DELETE'])
@login_required
def delete_document(doc_id):
    user_id = session['user_id']
    doc = get_upload(user_id, doc_id)
    if not doc:
        return jsonify({'success': False, 'message': 'Document not found.'}), 404
        
    # Delete database record
    delete_upload(user_id, doc_id)
    
    # Delete disk index/chunks files
    index_path = INDEX_DIR / f"document_{user_id}_{doc_id}.index"
    delete_document_index(index_path, cache_key=f"{user_id}_{doc_id}")
    
    text_path = INDEX_DIR / f"document_{user_id}_{doc_id}.text"
    if text_path.exists():
        try:
            text_path.unlink()
        except OSError:
            pass
            
    # Delete uploaded PDF file
    if doc.get('storage_key'):
        pdf_path = Config.UPLOAD_FOLDER / doc['storage_key']
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except OSError:
                pass

    if session.get('current_document_id') == doc_id:
        session.pop('current_document_id', None)
        
    return jsonify({'success': True, 'message': 'Document deleted successfully.'})

@student_bp.route('/api/documents/<int:doc_id>/view', methods=['GET'])
@login_required
def view_document(doc_id):
    user_id = session['user_id']
    doc = get_upload(user_id, doc_id)
    if not doc or not doc.get('storage_key'):
        return "Document not found", 404
    return send_from_directory(Config.UPLOAD_FOLDER, doc['storage_key'])

@student_bp.route('/api/qa', methods=['POST'])
@login_required
def qa_api():
    user_id = session['user_id']
    data = request.json or {}
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Please enter a query.'}), 400
        
    # Active document id
    document_id = session.get('current_document_id')
    if not document_id:
        docs = get_user_uploads(user_id, limit=1)
        if docs:
            document_id = docs[0]['id']
            session['current_document_id'] = document_id
            
    if not document_id:
        return jsonify({'error': 'No document found. Please upload a PDF first.'}), 400

    index_path = INDEX_DIR / f"document_{user_id}_{document_id}.index"
    index, chunks = load_index(str(index_path), user_id=f"{user_id}_{document_id}")
    if not index or not chunks:
        return jsonify({'error': 'Document FAISS index could not be loaded.'}), 400
        
    # Sidecar text
    pdf_text = ""
    text_path = INDEX_DIR / f"document_{user_id}_{document_id}.text"
    if text_path.exists():
        try:
            with open(text_path, 'r', encoding='utf-8') as tf:
                pdf_text = tf.read()
        except Exception:
            pass

    try:
        response_text = route_learning_query(index, chunks, question, pdf_text=pdf_text, user_id=user_id)
        # Log AI Request
        create_log(user_id, 'ai_learning_qa', f"Question: {question[:50]}")
        return jsonify({'answer': response_text, 'contexts': []})
    except Exception as e:
        logger.exception("QA API processing error")
        return jsonify({'error': f"Error generating answer: {e}"}), 500

@student_bp.route('/api/extract-summary', methods=['GET'])
@login_required
def extract_summary():
    user_id = session['user_id']
    document_id = session.get('current_document_id')
    if not document_id:
        docs = get_user_uploads(user_id, limit=1)
        if docs:
            document_id = docs[0]['id']
            session['current_document_id'] = document_id
            
    if not document_id:
        return jsonify({'error': 'No active document.'}), 400

    index_path = INDEX_DIR / f"document_{user_id}_{document_id}.index"
    index, chunks = load_index(str(index_path), user_id=f"{user_id}_{document_id}")
    if not index or not chunks:
        return jsonify({'error': 'Document FAISS index could not be loaded.'}), 400

    try:
        context_str = retrieve_context(index, chunks, 'summary overview introduction key points main concepts conclusion', top_k=5, check_similarity=False)
        from backend.services.pdf_service import generate_summary_response
        summary = generate_summary_response(context_str, user_id=user_id)
        create_log(user_id, 'ai_pdf_summary', f"Doc ID: {document_id}")
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@student_bp.route('/api/extract-topics', methods=['GET'])
@login_required
def extract_topics():
    user_id = session['user_id']
    document_id = session.get('current_document_id')
    if not document_id:
        docs = get_user_uploads(user_id, limit=1)
        if docs:
            document_id = docs[0]['id']
            session['current_document_id'] = document_id
            
    if not document_id:
        return jsonify({'error': 'No active document.'}), 400

    index_path = INDEX_DIR / f"document_{user_id}_{document_id}.index"
    index, chunks = load_index(str(index_path), user_id=f"{user_id}_{document_id}")
    if not index or not chunks:
        return jsonify({'error': 'Document FAISS index could not be loaded.'}), 400

    pdf_text = ""
    text_path = INDEX_DIR / f"document_{user_id}_{document_id}.text"
    if text_path.exists():
        try:
            with open(text_path, 'r', encoding='utf-8') as tf:
                pdf_text = tf.read()
        except Exception:
            pass

    try:
        from backend.services.pdf_service import extract_headings_from_text, extract_topics as extract_topics_fallback
        topics = []
        if pdf_text.strip():
            topics = extract_headings_from_text(pdf_text)
            
        if not topics:
            context_str = retrieve_context(index, chunks, 'table of contents syllabus sections headings outline chapters', top_k=3, check_similarity=False)
            topics = extract_topics_fallback(context_str)
            
        create_log(user_id, 'ai_pdf_topics', f"Doc ID: {document_id}")
        return jsonify({'success': True, 'topics': topics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@student_bp.route('/api/learning/set_active', methods=['POST'])
@login_required
def set_active_document():
    data = request.json or {}
    doc_id = data.get('document_id')
    user_id = session['user_id']
    if not doc_id or not get_upload(user_id, doc_id):
        return jsonify({'success': False, 'message': 'Invalid document ID.'}), 400
    session['current_document_id'] = doc_id
    return jsonify({'success': True, 'message': 'Active document updated.'})

# ══════════════════════════════════════════════════════════════════════════════
# AI CHAT SESSIONS & STREAMING APIs
# ══════════════════════════════════════════════════════════════════════════════

@student_bp.route('/api/chat/sessions', methods=['GET'])
@login_required
def chat_sessions():
    sessions = get_user_chat_sessions(session['user_id'])
    return jsonify({'success': True, 'sessions': sessions})

@student_bp.route('/api/chat/sessions/<session_id>', methods=['GET'])
@login_required
def chat_session_history(session_id):
    history = get_chat_session_history(session['user_id'], session_id)
    return jsonify({'success': True, 'history': history})

@student_bp.route('/api/chat/sessions/<session_id>', methods=['DELETE'])
@login_required
def delete_chat(session_id):
    user_id = session['user_id']
    delete_chat_session(user_id, session_id)
    return jsonify({'success': True, 'message': 'Chat session deleted.'})

@student_bp.route('/api/chat/sessions/<session_id>/rename', methods=['POST'])
@login_required
def rename_chat(session_id):
    user_id = session['user_id']
    data = request.json or {}
    new_title = data.get('title', '').strip()
    if not new_title:
        return jsonify({'success': False, 'message': 'Title cannot be empty.'}), 400
    rename_chat_session(user_id, session_id, new_title)
    return jsonify({'success': True, 'message': 'Chat session renamed.'})

@student_bp.route('/api/chat/stream', methods=['POST'])
@login_required
def chat_stream():
    """
    SSE stream endpoint for chat completions. Takes conversation history into context.
    """
    user_id = session['user_id']
    data = request.json or {}
    message = data.get('message', '').strip()
    session_id = data.get('session_id')
    session_title = data.get('session_title', '').strip()

    if not message:
        return jsonify({'error': 'Message cannot be empty.'}), 400

    if not session_id:
        session_id = str(uuid.uuid4())
    if not session_title:
        session_title = message[:30] + ('...' if len(message) > 30 else '')

    # Save User message
    create_chat(user_id, session_id, session_title, 'user', message)

    # Compile chat history context
    history = get_chat_session_history(user_id, session_id)
    prompt_lines = []
    
    # Limit past history context length
    history = history[-10:] # Last 10 messages
    for msg in history[:-1]: # exclude current message since we append below
        role_lbl = "User" if msg['role'] == 'user' else "Assistant"
        prompt_lines.append(f"{role_lbl}: {msg['message']}")
        
    prompt_lines.append(f"User: {message}")
    prompt_lines.append("Assistant:")
    
    system_instruction = "You are a helpful student AI assistant. Provide elegant, concise, and structured answers in markdown."
    prompt = "\n".join(prompt_lines)

    def sse_generate():
        full_assistant_text = ""
        # Yield metadata block first to let client know session ID
        yield f"data: {json.dumps({'session_id': session_id, 'session_title': session_title})}\n\n"
        
        for token_chunk in generate_response_stream(prompt, user_id=user_id, system_instruction=system_instruction, max_tokens=1000):
            if token_chunk.startswith("data: "):
                data_str = token_chunk[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data_json = json.loads(data_str)
                    if "choices" in data_json:
                        delta = data_json["choices"][0]["delta"]
                        token = delta.get("content", "")
                        full_assistant_text += token
                except Exception:
                    pass
            yield token_chunk

        # Write assistant response to DB
        if full_assistant_text.strip():
            create_chat(user_id, session_id, session_title, 'assistant', full_assistant_text)

    return Response(sse_generate(), mimetype='text/event-stream')

# ══════════════════════════════════════════════════════════════════════════════
# AI RESUME BUILDER APIs
# ══════════════════════════════════════════════════════════════════════════════

@student_bp.route('/api/resume/save', methods=['POST'])
@login_required
def api_resume_save():
    user_id = session['user_id']
    data = request.json or {}
    resume_id = data.get('resume_id')
    template = 'classic' # Removed template logic as requested
    resume_data = data.get('resume_data', {})
    ats_score = data.get('ats_score', 0)
    
    if resume_id:
        from backend.database import get_conn
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            'UPDATE resumes SET template=?, resume_json=?, ats_score=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?',
            (template, json.dumps(resume_data), ats_score, resume_id, user_id)
        )
        conn.commit()
        create_log(user_id, 'resume_update', f"Updated resume {resume_id}")
        return jsonify({'success': True, 'message': 'Resume updated successfully.', 'resume_id': resume_id})
    else:
        rid = create_resume(user_id, template, json.dumps(resume_data), ats_score)
        create_log(user_id, 'resume_create', f"Created resume {rid}")
        return jsonify({'success': True, 'message': 'Resume saved successfully.', 'resume_id': rid})

@student_bp.route('/api/resume/load', methods=['GET'])
@login_required
def api_resume_load():
    user_id = session['user_id']
    resumes = get_user_resumes(user_id)
    if resumes:
        latest_meta = resumes[0]
        full_resume = get_resume(latest_meta['id'], user_id)
        if full_resume:
            latest_dict = dict(full_resume)
            latest_dict['resume_data'] = json.loads(latest_dict['resume_json'])
            return jsonify({'success': True, 'resume': latest_dict})
    return jsonify({'success': False, 'message': 'No resumes found.'})

@student_bp.route('/api/resume/ats-score', methods=['POST'])
@login_required
def api_resume_ats_score():
    data = request.json or {}
    rd = data.get('resume_data', {})
    score = 0
    
    # Personal (20%)
    if rd.get('name'): score += 5
    if rd.get('title'): score += 5
    if rd.get('email'): score += 5
    if rd.get('phone') or rd.get('linkedin') or rd.get('github'): score += 5
    
    # Summary (15%)
    summary = rd.get('professional_summary', '')
    words = len(summary.strip().split()) if summary else 0
    if 40 <= words <= 100: score += 15
    elif words > 0: score += 8
    
    # Education (15%)
    edu = rd.get('education', [])
    if edu and all(e.get('degree') and e.get('college') for e in edu): score += 15
    
    # Skills (15%)
    skills = rd.get('skills', '')
    skills_count = len([s for s in skills.split(',') if s.strip()]) if skills else 0
    if skills_count >= 5: score += 15
    elif skills_count > 0: score += 7
    
    # Experience (20%)
    exp = rd.get('experience', [])
    if exp and all(e.get('role') and e.get('company') and len(e.get('responsibilities', '')) > 10 for e in exp): score += 20
    
    # Projects (15%)
    proj = rd.get('projects', [])
    if len(proj) >= 2 and all(p.get('name') and len(p.get('description', '')) > 10 for p in proj): score += 15
    elif len(proj) > 0: score += 8

    return jsonify({'success': True, 'score': score})

@student_bp.route('/api/resume/download', methods=['POST'])
@login_required
def api_resume_download():
    # Placeholder: The actual PDF generation remains entirely client-side using html2pdf 
    # to guarantee pixel-perfect styling fidelity from the browser.
    return jsonify({'success': True, 'message': 'Download triggered on client.'})

@student_bp.route('/api/generate-summary', methods=['POST'])
@login_required
def generate_resume_summary():
    user_id = session['user_id']
    data = request.json or {}
    rd = data.get('resume_data', {})

    prompt = f"""Write a professional resume summary (50-80 words, one paragraph) based on:
Title: {rd.get('title', '')}
Skills: {rd.get('skills', '')}
Education/Projects/Experience: {json.dumps(rd.get('projects', []))[:200]}
Do NOT write markdown, bold markings, headings, or bullet lists. Return only the summary text."""

    summary_text = generate_response(prompt, user_id=user_id, max_tokens=150)
    summary_text = summary_text.strip().strip('"\'')
    return jsonify({'success': True, 'summary': summary_text})

@student_bp.route('/api/improve-project', methods=['POST'])
@login_required
def improve_project_api():
    user_id = session['user_id']
    data = request.json or {}
    pname = data.get('name', '').strip()
    ptech = data.get('technologies', '').strip()
    pdesc = data.get('description', '').strip()

    prompt = f"""Rewrite this project description professionally for a resume.
Project: {pname}
Technologies: {ptech}
Original description: {pdesc}
Rules:
- 3 concise bullet points.
- Start each line with a dash and a space (- )
- No markdown, bold text, or headers. Return ONLY the bullet lines."""

    improved = generate_response(prompt, user_id=user_id, max_tokens=250)
    return jsonify({'success': True, 'improved_description': improved.strip()})

@student_bp.route('/api/improve-experience', methods=['POST'])
@login_required
def improve_experience_api():
    user_id = session['user_id']
    data = request.json or {}
    company = data.get('company', '').strip()
    role = data.get('role', '').strip()
    responsibilities = data.get('responsibilities', '').strip()

    prompt = f"""Rewrite this work experience into 3-4 professional resume bullet points.
Company: {company}
Role: {role}
Original: {responsibilities}
Rules:
- Professional action-verb-oriented bullet points.
- Start each line with a dash and a space (- )
- No markdown, bold text, or headers. Return ONLY the bullet lines."""

    improved = generate_response(prompt, user_id=user_id, max_tokens=250)
    return jsonify({'success': True, 'improved_responsibilities': improved.strip()})

@student_bp.route('/api/optimize-resume', methods=['POST'])
@login_required
def optimize_resume_api():
    user_id = session['user_id']
    data = request.json or {}
    rd = data.get('resume_data', {})
    
    # Let's perform sequential optimisations via OpenRouter
    results = {}
    
    # 1. Optimize summary
    summary = rd.get('professional_summary', '').strip()
    if summary:
        prompt = f"Optimize this professional summary for an ATS system, keeping facts exact: '{summary}'. Return only the summary text without headers or quotes."
        results['professional_summary'] = generate_response(prompt, user_id=user_id, max_tokens=150).strip().strip('"\'')
    else:
        results['professional_summary'] = ""
        
    # 2. Optimize projects
    opt_projects_desc = []
    for p in rd.get('projects', []):
        desc = p.get('description', '').strip()
        pname = p.get('name', '').strip()
        if desc:
            prompt = f"Optimize this project description into 2 ATS resume bullet points (each starting with '- '): Project Name: {pname}, Description: {desc}. Return only the bullet points."
            opt_projects_desc.append(generate_response(prompt, user_id=user_id, max_tokens=150).strip())
        else:
            opt_projects_desc.append("")
    results['projects'] = opt_projects_desc

    # 3. Optimize experience
    opt_exp_desc = []
    for exp in rd.get('experience', []):
        resp = exp.get('responsibilities', '').strip()
        role = exp.get('role', '').strip()
        if resp:
            prompt = f"Optimize this job experience responsibilities section into 3 ATS resume bullet points (each starting with '- '): Role: {role}, Description: {resp}. Return only the bullet points."
            opt_exp_desc.append(generate_response(prompt, user_id=user_id, max_tokens=200).strip())
        else:
            opt_exp_desc.append("")
    results['experience'] = opt_exp_desc

    return jsonify({'success': True, 'optimized': results})

@student_bp.route('/api/resume/preview', methods=['POST'])
@login_required
def api_resume_preview():
    data = request.json or {}
    resume_data = data.get('resume_data', {})
    
    html_template = """
<div class="resume-paper">
  <div class="resume-header">
    <h1>{{ rd.name }}</h1>
    <div class="resume-title">{{ rd.title }}</div>
    <div class="resume-contact">
      {% if rd.email %}{{ rd.email }}{% endif %}
      {% if rd.phone %} • {{ rd.phone }}{% endif %}
      {% if rd.linkedin %} • {{ rd.linkedin }}{% endif %}
      {% if rd.github %} • {{ rd.github }}{% endif %}
    </div>
  </div>

  {% if rd.professional_summary %}
  <div class="resume-section">
    <h2>Professional Summary</h2>
    <div class="resume-item-desc">
      {{ rd.professional_summary }}
    </div>
  </div>
  {% endif %}

  {% if rd.education and rd.education | selectattr('degree') | list | length > 0 %}
  <div class="resume-section">
    <h2>Education</h2>
    {% for edu in rd.education %}
      {% if edu.degree %}
      <div class="resume-item">
        <div class="resume-item-header">
          <div class="resume-item-title">{{ edu.degree }} {% if edu.branch %}in {{ edu.branch }}{% endif %}</div>
          <div class="resume-item-date">{{ edu.graduation_year }}</div>
        </div>
        <div class="resume-item-subtitle">{{ edu.college }}</div>
        {% if edu.cgpa %}
        <div class="resume-item-desc">CGPA / Percentage: {{ edu.cgpa }}</div>
        {% endif %}
      </div>
      {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {% if rd.skills %}
  <div class="resume-section">
    <h2>Technical Skills</h2>
    <div class="resume-item-desc">
      {{ rd.skills }}
    </div>
  </div>
  {% endif %}

  {% if rd.experience and rd.experience | selectattr('role') | list | length > 0 %}
  <div class="resume-section">
    <h2>Professional Experience</h2>
    {% for exp in rd.experience %}
      {% if exp.role %}
      <div class="resume-item">
        <div class="resume-item-header">
          <div class="resume-item-title">{{ exp.role }}</div>
          <div class="resume-item-date">{{ exp.duration }}</div>
        </div>
        <div class="resume-item-subtitle">{{ exp.company }}</div>
        {% if exp.responsibilities %}
        <div class="resume-item-desc">
          <ul>
            {% for bullet in exp.responsibilities.split('\\n') %}
              {% if bullet.strip() %}
                <li>{{ bullet.replace('-', '').replace('*', '').strip() }}</li>
              {% endif %}
            {% endfor %}
          </ul>
        </div>
        {% endif %}
      </div>
      {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {% if rd.projects and rd.projects | selectattr('name') | list | length > 0 %}
  <div class="resume-section">
    <h2>Key Projects</h2>
    {% for proj in rd.projects %}
      {% if proj.name %}
      <div class="resume-item">
        <div class="resume-item-header">
          <div class="resume-item-title">{{ proj.name }}</div>
        </div>
        {% if proj.technologies %}
        <div class="resume-item-subtitle">Technologies: {{ proj.technologies }}</div>
        {% endif %}
        {% if proj.description %}
        <div class="resume-item-desc">
          <ul>
            {% for bullet in proj.description.split('\\n') %}
              {% if bullet.strip() %}
                <li>{{ bullet.replace('-', '').replace('*', '').strip() }}</li>
              {% endif %}
            {% endfor %}
          </ul>
        </div>
        {% endif %}
      </div>
      {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {% if rd.certifications and rd.certifications | selectattr('name') | list | length > 0 %}
  <div class="resume-section">
    <h2>Certifications</h2>
    {% for cert in rd.certifications %}
      {% if cert.name %}
      <div class="resume-item">
        <div class="resume-item-header">
          <div class="resume-item-title">{{ cert.name }}</div>
          <div class="resume-item-date">{{ cert.date }}</div>
        </div>
        <div class="resume-item-subtitle">{{ cert.issuer }}</div>
      </div>
      {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {% if (rd.languages and rd.languages.strip()) or (rd.activities and rd.activities.strip()) %}
  <div class="resume-section">
    <h2>Languages & Activities</h2>
    {% if rd.languages %}
    <div class="resume-item">
      <div class="resume-item-subtitle" style="margin-bottom:4px;">Languages</div>
      <div class="resume-item-desc">{{ rd.languages }}</div>
    </div>
    {% endif %}
    {% if rd.activities %}
    <div class="resume-item">
      <div class="resume-item-subtitle" style="margin-bottom:4px;">Activities & Achievements</div>
      <div class="resume-item-desc">
        <ul>
          {% for act in rd.activities.split('\\n') %}
            {% if act.strip() %}
              <li>{{ act.replace('-', '').replace('*', '').strip() }}</li>
            {% endif %}
          {% endfor %}
        </ul>
      </div>
    </div>
    {% endif %}
  </div>
  {% endif %}
</div>
    """
    
    html_content = render_template_string(html_template, rd=resume_data)
    return jsonify({'success': True, 'resume_html': html_content})

# ══════════════════════════════════════════════════════════════════════════════
# PROFILE & ACCOUNT MANAGEMENT APIs
# ══════════════════════════════════════════════════════════════════════════════

@student_bp.route('/api/profile/update', methods=['POST'])
@login_required
def update_profile():
    user_id = session['user_id']
    data = request.json or {}
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    phone = data.get('phone', '').strip()
    department = data.get('department', '').strip()
    year = data.get('year', '').strip()
    university = data.get('university', '').strip()

    if not name or not email:
        return jsonify({'error': 'Name and Email are required.'}), 400

    try:
        # Check if email is already taken by someone else
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ? AND id != ?", (email, user_id))
        if cur.fetchone():
            return jsonify({'error': 'Email is already taken.'}), 400
            
        update_user(user_id, name, email, phone, department, year, university)
        session['user_name'] = name
        session['user_email'] = email
        return jsonify({'success': True, 'message': 'Profile updated successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@student_bp.route('/api/profile/password', methods=['POST'])
@login_required
def update_password():
    user_id = session['user_id']
    data = request.json or {}
    old_pass = data.get('old_password', '')
    new_pass = data.get('new_password', '')

    if not old_pass or not new_pass:
        return jsonify({'error': 'Both old and new passwords are required.'}), 400

    if len(new_pass) < 6:
        return jsonify({'error': 'New password must be at least 6 characters.'}), 400

    user = get_user_by_id(user_id)
    from werkzeug.security import check_password_hash
    if not check_password_hash(user['password'], old_pass):
        return jsonify({'error': 'Incorrect old password.'}), 400

    change_password(user_id, new_pass)
    return jsonify({'success': True, 'message': 'Password updated successfully.'})

@student_bp.route('/api/profile/delete', methods=['POST'])
@login_required
def delete_account():
    user_id = session['user_id']
    # Delete student uploads and indexes
    uploads = get_user_uploads(user_id, limit=100)
    for upload in uploads:
        index_path = INDEX_DIR / f"document_{user_id}_{upload['id']}.index"
        delete_document_index(index_path, cache_key=f"{user_id}_{upload['id']}")
        
        text_path = INDEX_DIR / f"document_{user_id}_{upload['id']}.text"
        if text_path.exists():
            try:
                text_path.unlink()
            except OSError:
                pass

        if upload.get('storage_key'):
            pdf_path = Config.UPLOAD_FOLDER / upload['storage_key']
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except OSError:
                    pass

    delete_user(user_id)
    session.clear()
    return jsonify({'success': True, 'message': 'Account deleted successfully.'})

@student_bp.route('/api/profile/avatar', methods=['POST'])
@login_required
def upload_avatar():
    if 'avatar' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
        
    f = request.files['avatar']
    if f.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
        
    ext = f.filename.lower().split('.')[-1]
    if ext not in ['jpg', 'jpeg', 'png']:
        return jsonify({'error': 'Invalid file type. Only JPG, JPEG, and PNG are allowed.'}), 400

    user_id = session['user_id']
    filename = f"avatar_{user_id}_{int(uuid.uuid4().time)}.{ext}"
    
    avatar_dir = Config.UPLOAD_FOLDER / 'avatars'
    avatar_dir.mkdir(parents=True, exist_ok=True)
    
    dest = avatar_dir / filename
    f.save(dest)

    # Save to user record
    from backend.database import get_conn
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET profile_pic=? WHERE id=?", (f"/static/uploads/avatars/{filename}", user_id))
    conn.commit()
    
    create_log(user_id, 'profile_pic_upload', f"Uploaded profile picture: {filename}")
    return jsonify({'success': True, 'avatar_url': f"/static/uploads/avatars/{filename}"})
