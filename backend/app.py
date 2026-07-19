from flask import Flask, render_template, session, redirect, url_for, request, flash, jsonify, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
from hashlib import sha256
import traceback
from auth import bp as auth_bp
from chat import bp as chat_bp
from rag import (
    extract_text_from_pdf, chunk_text, build_faiss_index,
    delete_document_index, INDEX_DIR, load_index, query_index,
    retrieve_context, clean_output, generate_summary, extract_topics,
    answer_question, route_learning_query,
)
from model_loader import get_llm, get_embedder
from database import get_conn, get_user_by_id, count_user_items, update_user, create_upload, is_duplicate_upload, get_user_uploads, get_upload, delete_upload, get_recent_uploads, get_recent_chats, get_recent_plans
from html import escape
import logging
import os
import json
import re

def create_app():
    logging.basicConfig(
        level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / 'frontend' / 'templates'), static_folder=str(Path(__file__).parent.parent / 'frontend' / 'static'))
    app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    # ensure database schema exists
    try:
        from database import init_db
        init_db()
    except Exception:
        pass

    # Warm-up model cache
    try:
        get_llm()
        get_embedder()
    except Exception:
        pass

    @app.context_processor
    def inject_user():
        return dict(user_name=session.get('user_name'))

    @app.route('/')
    def root():
        if session.get('user_id'):
            return redirect(url_for('home'))
        return redirect(url_for('auth.login'))

    @app.route('/home')
    def home():
        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('auth.login'))
        user = get_user_by_id(user_id)
        stats = count_user_items(user_id)
        recent_uploads = get_recent_uploads(user_id, limit=3)
        recent_chats = get_recent_chats(user_id, limit=3)
        return render_template('home.html', stats=stats, last_login=user['last_login'], recent_uploads=recent_uploads, recent_chats=recent_chats)

    @app.route('/learning')
    def learning():
        if not session.get('user_id'):
            return redirect(url_for('auth.login'))
        return render_template('learning.html')

    @app.route('/chat')
    def chat():
        if not session.get('user_id'):
            return redirect(url_for('auth.login'))
        return render_template('chat.html')

    @app.route('/resume')
    def resume():
        if not session.get('user_id'):
            return redirect(url_for('auth.login'))
        return render_template('resume.html')

    @app.route('/profile')
    def profile():
        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('auth.login'))
        user = get_user_by_id(user_id)
        stats = count_user_items(user_id)
        if not user:
            session.clear()
            return redirect(url_for('auth.login'))
        return render_template('profile.html', user=user, stats=stats)

    @app.route('/api/profile/update', methods=['POST'])
    def update_profile():
        user_id = session.get('user_id')
        if not user_id:
            return {'error': 'Authentication required'}, 401
        data = request.json or {}
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        roll = data.get('roll', '').strip()
        branch = data.get('branch', '').strip()
        semester = data.get('semester', '').strip()

        if not name or not email:
            return {'error': 'Name and email are required.'}, 400
        try:
            updated = update_user(user_id, name, email, roll, branch, semester)
            if updated == 0:
                return {'error': 'Profile update failed.'}, 400
            session['user_name'] = name
            user = get_user_by_id(user_id)
            return {
                'success': True,
                'user': {
                    'name': user['name'],
                    'email': user['email'],
                    'roll': user['roll'],
                    'branch': user['branch'],
                    'semester': user['semester']
                }
            }
        except Exception as e:
            return {'error': str(e)}, 500

    @app.route('/upload_pdf', methods=['POST'])
    def upload_pdf():
        print('[1] Upload request received')
        try:
            if 'file' not in request.files:
                print('[ERROR] No file part in upload request')
                return jsonify({'success': False, 'message': 'No file provided'}), 400

            f = request.files['file']
            if f.filename == '':
                print('[ERROR] No selected file')
                return jsonify({'success': False, 'message': 'No selected file'}), 400

            if not f.filename.lower().endswith('.pdf'):
                print('[ERROR] Invalid file extension:', f.filename)
                return jsonify({'success': False, 'message': 'Only PDF files allowed'}), 400

            user_id = session.get('user_id')
            if not user_id:
                print('[ERROR] Upload attempted without authentication')
                return jsonify({'success': False, 'message': 'Authentication required'}), 401

            filename = secure_filename(f.filename)
            if not filename:
                print('[ERROR] Invalid filename after sanitization')
                return jsonify({'success': False, 'message': 'Invalid filename'}), 400

            save_dir = Path(__file__).parent.parent / 'frontend' / 'uploads'
            save_dir.mkdir(parents=True, exist_ok=True)

            file_bytes = f.read()
            file_hash = sha256(file_bytes).hexdigest()
            
            if is_duplicate_upload(user_id, file_hash):
                print('[WARN] Duplicate upload detected for user', user_id, 'hash', file_hash)
                return jsonify({'success': False, 'message': 'Duplicate PDF detected. This file has already been uploaded.'}), 409

            # Each document owns its PDF, extracted chunks, and FAISS index.  This
            # prevents one document from resurfacing after another is deleted.
            upload_id = create_upload(user_id, filename, None, file_hash)
            storage_key = f"{user_id}_{upload_id}_{filename}"
            dest = save_dir / storage_key
            print('[2] Saving file to', dest)
            with open(dest, 'wb') as out_file:
                out_file.write(file_bytes)

            try:
                print('[3] Extracting text from PDF')
                text, pages = extract_text_from_pdf(str(dest))
                print(f'[4] Extracted text ({pages} pages)')

                if not text or not text.strip():
                    raise ValueError("The uploaded PDF is empty or contains no extractable text.")

                print('[5] Creating chunks')
                chunks = chunk_text(text)
                print(f'[6] Chunk count: {len(chunks)}')

                if not chunks:
                    raise ValueError("Could not extract any chunks from the document text.")

                index_path = INDEX_DIR / f"document_{user_id}_{upload_id}.index"
                print('[7] Creating isolated FAISS index at', index_path)
                build_faiss_index(chunks, index_path, f"document_{user_id}_{upload_id}")
                print('[8] FAISS index created')

                # Save raw text sidecar for heading extraction (used by Topics feature)
                text_path = INDEX_DIR / f"document_{user_id}_{upload_id}.text"
                try:
                    with open(text_path, 'w', encoding='utf-8') as tf:
                        tf.write(text)
                    print('[8b] PDF text sidecar saved')
                except Exception as te:
                    print('[WARN] Could not save text sidecar:', te)
                
                # A new upload starts a fresh learning session for that document.
                session['current_document_id'] = upload_id
                session['current_pdf'] = {'id': upload_id, 'filename': filename, 'pages': pages, 'hash': file_hash}
            except Exception as inner_error:
                try:
                    dest.unlink(missing_ok=True)
                    delete_document_index(INDEX_DIR / f"document_{user_id}_{upload_id}.index", f"document_{user_id}_{upload_id}")
                    delete_upload(user_id, upload_id)
                except Exception:
                    pass
                tb = traceback.format_exc()
                print('[ERROR] PDF processing failed:', inner_error)
                print(tb)
                return jsonify({'success': False, 'message': f'PDF processing failed: {str(inner_error)}'}), 400

            try:
                conn = get_conn()
                conn.execute('UPDATE uploads SET pages=?, storage_key=? WHERE id=? AND user_id=?',
                             (pages, storage_key, upload_id, user_id))
                conn.commit()
                conn.close()
                print('[9] Upload metadata saved')
            except Exception as db_error:
                print('[ERROR] Upload metadata save failed:', db_error)
                print(traceback.format_exc())

            print('[10] Upload completed successfully')
            return jsonify({'success': True, 'message': 'Upload completed', 'filename': filename, 'pages': pages})
        except Exception as e:
            tb = traceback.format_exc()
            print('[ERROR] upload_pdf exception:', e)
            print(tb)
            return jsonify({'success': False, 'message': 'Unexpected error during upload. Please check server logs.'}), 500

    @app.route('/api/documents')
    def documents_api():
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        save_dir = Path(__file__).parent.parent / 'frontend' / 'uploads'
        docs = []
        for document in get_user_uploads(user_id):
            storage_key = document.get('storage_key') or f"{user_id}_{document['filename']}"
            if (save_dir / storage_key).is_file():
                docs.append(document)
            else:
                # Never expose stale metadata in Recent Activity after a restart.
                delete_document_index(INDEX_DIR / f"document_{user_id}_{document['id']}.index", f"document_{user_id}_{document['id']}")
                delete_upload(user_id, document['id'])
        return jsonify({'documents': docs})

    @app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
    def delete_document(doc_id):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        document = get_upload(user_id, doc_id)
        if document:
            save_dir = Path(__file__).parent.parent / 'frontend' / 'uploads'
            storage_key = document.get('storage_key') or f"{user_id}_{document['filename']}"
            try:
                (save_dir / storage_key).unlink(missing_ok=True)
                # Legacy files can exist from earlier versions of the app.
                if not document.get('storage_key'):
                    (save_dir / f"{user_id}_{document['filename']}").unlink(missing_ok=True)
                delete_document_index(INDEX_DIR / f"document_{user_id}_{doc_id}.index", f"document_{user_id}_{doc_id}")
                # Also remove raw text sidecar if present
                text_path = INDEX_DIR / f"document_{user_id}_{doc_id}.text"
                try:
                    text_path.unlink(missing_ok=True)
                except OSError:
                    pass
                deleted = delete_upload(user_id, doc_id)
            except OSError as exc:
                return jsonify({'error': f'Could not delete document data: {exc}'}), 500
            if session.get('current_document_id') == doc_id:
                session.pop('current_document_id', None)
                session.pop('current_pdf', None)
            return jsonify({'success': True})
        return jsonify({'error': 'Document not found'}), 404

    @app.route('/api/documents/<int:doc_id>/view')
    def view_document(doc_id):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        document = get_upload(user_id, doc_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        storage_key = document.get('storage_key') or f"{user_id}_{document['filename']}"
        return send_from_directory(Path(__file__).parent.parent / 'frontend' / 'uploads', storage_key, mimetype='application/pdf')

    @app.route('/api/qa', methods=['POST'])
    def qa_api():
        """
        Learning Assistant – single endpoint for all PDF-related queries.

        Intent is auto-detected from the question text:
          qa | summary | topics | viva | mcq | define | explain |
          notes | long | short | advantages | disadvantages |
          difference | interview
        """
        data     = request.json or {}
        question = (data.get('question') or '').strip()
        user_id  = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        if not question:
            return jsonify({'error': 'Please enter a question.'}), 400

        # Resolve the active document
        document_id = session.get('current_document_id')
        if not document_id or not get_upload(user_id, document_id):
            docs = get_user_uploads(user_id, limit=1)
            document_id = docs[0]['id'] if docs else None
            if document_id:
                session['current_document_id'] = document_id
        if not document_id:
            return jsonify({'error': 'No document found. Upload a PDF first.'}), 400

        index_path = INDEX_DIR / f'document_{user_id}_{document_id}.index'
        index, chunks = load_index(str(index_path), f'document_{user_id}_{document_id}')
        if not index or not chunks:
            return jsonify({'error': 'No index found. Upload a PDF first.'}), 400

        # Load raw PDF text sidecar for heading extraction
        pdf_text = ""
        text_path = INDEX_DIR / f"document_{user_id}_{document_id}.text"
        try:
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as tf:
                    pdf_text = tf.read()
        except Exception:
            pass

        try:
            # route_learning_query detects intent and calls the right generator
            out = route_learning_query(index, chunks, question, pdf_text=pdf_text)
        except Exception as exc:
            logging.exception('[ERROR] Learning query failed')
            return jsonify({'error': f'Generation failed: {exc}'}), 500

        # Persist to chat history (non-fatal)
        try:
            from database import create_chat
            create_chat(user_id, 'user',      question)
            create_chat(user_id, 'assistant', out)
        except Exception:
            pass

        return jsonify({'answer': out, 'contexts': []})

    @app.route('/api/extract-summary', methods=['GET'])
    def _route_extract_summary():
        """Generate a structured PDF summary (Overview / Key Points / Important Concepts / Conclusion)."""
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        document_id = session.get('current_document_id')
        if not document_id or not get_upload(user_id, document_id):
            docs = get_user_uploads(user_id, limit=1)
            document_id = docs[0]['id'] if docs else None
        if not document_id:
            return jsonify({'error': 'No document found. Upload a PDF first.'}), 400

        index_path = INDEX_DIR / f"document_{user_id}_{document_id}.index"
        index, chunks = load_index(str(index_path), f"document_{user_id}_{document_id}")
        if not index or not chunks:
            return jsonify({'error': 'No indexed documents found. Upload a PDF first.'}), 400

        try:
            # Broad semantic query to surface the most informative chunks
            context_str = retrieve_context(
                index, chunks,
                'summary overview introduction key points main concepts conclusion',
                top_k=5, check_similarity=False
            )
            summary = generate_summary(context_str)
            return jsonify({'success': True, 'summary': summary})
        except Exception as exc:
            logging.exception('[ERROR] Summary generation failed')
            return jsonify({'error': f'Summary generation failed: {exc}'}), 500

    @app.route('/api/extract-topics', methods=['GET'])
    def _route_extract_topics():
        """Extract major topics / section headings from the uploaded PDF."""
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        document_id = session.get('current_document_id')
        if not document_id or not get_upload(user_id, document_id):
            docs = get_user_uploads(user_id, limit=1)
            document_id = docs[0]['id'] if docs else None
        if not document_id:
            return jsonify({'error': 'No document found. Upload a PDF first.'}), 400

        index_path = INDEX_DIR / f"document_{user_id}_{document_id}.index"
        index, chunks = load_index(str(index_path), f"document_{user_id}_{document_id}")
        if not index or not chunks:
            return jsonify({'error': 'No indexed documents found. Upload a PDF first.'}), 400

        # Load raw PDF text sidecar for heading extraction
        pdf_text = ""
        text_path = INDEX_DIR / f"document_{user_id}_{document_id}.text"
        try:
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as tf:
                    pdf_text = tf.read()
        except Exception:
            pass

        try:
            topics = []
            # Try raw heading extraction first
            if pdf_text.strip():
                from rag import extract_headings_from_text
                topics = extract_headings_from_text(pdf_text)

            # Fall back to LLM topics extraction if none found
            if not topics:
                context_str = retrieve_context(
                    index, chunks,
                    'table of contents sections headings introduction background methodology results conclusion',
                    top_k=3, check_similarity=False
                )
                topics = extract_topics(context_str)

            return jsonify({'success': True, 'topics': topics})
        except Exception as exc:
            logging.exception('[ERROR] Topics extraction failed')
            return jsonify({'error': f'Topics extraction failed: {exc}'}), 500

    @app.route('/api/generate-summary', methods=['POST'])
    def _route_generate_resume_summary():
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        data = request.json or {}
        rd = data.get('resume_data', {})

        name = rd.get('name', '').strip()
        title = rd.get('title', '').strip()
        skills = rd.get('skills', '').strip()
        education = rd.get('education', [])
        projects = rd.get('projects', [])
        experience = rd.get('experience', [])

        edu_str = ""
        if isinstance(education, list):
            for e in education:
                parts = [e.get('degree',''), e.get('branch',''), e.get('college','')]
                edu_str += ", ".join([p for p in parts if p.strip()]) + ". "

        proj_str = ""
        if isinstance(projects, list):
            for p in projects:
                n = p.get('name','').strip()
                t = p.get('technologies','').strip()
                if n:
                    proj_str += f"{n} ({t}). " if t else f"{n}. "

        exp_str = ""
        if isinstance(experience, list):
            for ex in experience:
                r = ex.get('role','').strip()
                c = ex.get('company','').strip()
                if c:
                    exp_str += f"{r} at {c}. " if r else f"{c}. "

        prompt = f"""<|im_start|>system
You are an expert ATS Resume Writer.

Using ONLY the candidate information provided, write a professional summary.

Rules:
- 50-80 words
- One paragraph
- Professional tone
- ATS optimized
- No markdown
- No headings
- No asterisks
- No hash symbols
- Do not repeat labels like "Name:" or "Skills:"
- Do not invent information
- Return only the summary text, nothing else
<|im_end|>
<|im_start|>user
Candidate Information:
Name: {name}
Title: {title}
Skills: {skills}
Education: {edu_str}
Projects: {proj_str}
Experience: {exp_str}
<|im_end|>
<|im_start|>assistant
"""
        try:
            llm = get_llm()
            if not llm:
                return jsonify({'error': 'LLM not available'}), 500
            raw = llm.generate(prompt, max_tokens=180).strip()
            cleaned = _clean_llm_output(raw)
            return jsonify({'success': True, 'summary': cleaned})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/improve-project', methods=['POST'])
    def improve_project():
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        data = request.json or {}
        pname = data.get('name', '').strip()
        pdesc = data.get('description', '').strip()
        ptech = data.get('technologies', '').strip()

        prompt = f"""<|im_start|>system
You are an ATS Resume Writer.

Rewrite this project description professionally.

Rules:
- 3-4 concise resume bullet points
- Professional language
- Highlight technologies
- No fake information
- No markdown
- No asterisks
- No hash symbols
- No headings
- Each bullet point starts with a dash and a space
- Return only the bullet points, nothing else
<|im_end|>
<|im_start|>user
Project Name: {pname}
Technologies: {ptech}
Description: {pdesc}
<|im_end|>
<|im_start|>assistant
"""
        try:
            llm = get_llm()
            if not llm:
                return jsonify({'error': 'LLM not available'}), 500
            raw = llm.generate(prompt, max_tokens=150).strip()
            cleaned = _clean_llm_output(raw)
            return jsonify({'success': True, 'improved_description': cleaned})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/improve-experience', methods=['POST'])
    def improve_experience():
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        data = request.json or {}
        company = data.get('company', '').strip()
        role = data.get('role', '').strip()
        responsibilities = data.get('responsibilities', '').strip()

        prompt = f"""<|im_start|>system
You are an ATS Resume Writer.

Rewrite the following work notes into 3-5 professional resume bullet points.

Rules:
- Professional language
- Strong action verbs
- No fake information
- No markdown
- No asterisks
- No hash symbols
- No headings
- Each bullet point starts with a dash and a space
- Return only the bullet points, nothing else
<|im_end|>
<|im_start|>user
Company: {company}
Role: {role}
Work Notes: {responsibilities}
<|im_end|>
<|im_start|>assistant
"""
        try:
            llm = get_llm()
            if not llm:
                return jsonify({'error': 'LLM not available'}), 500
            raw = llm.generate(prompt, max_tokens=250).strip()
            cleaned = _clean_llm_output(raw)
            # Extract bullet lines
            lines = []
            for line in cleaned.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Strip leading markers
                for marker in ['- ', '* ', '• ', '– ']:
                    if line.startswith(marker):
                        line = line[len(marker):]
                        break
                # Remove numbering like "1. " or "1) "
                import re as _re
                line = _re.sub(r'^\d+[\.\)]\s*', '', line)
                if line:
                    lines.append(line)
            lines = lines[:5]  # Max 5 bullets
            result = "\n".join(f"- {l}" for l in lines)
            return jsonify({'success': True, 'improved_responsibilities': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/optimize-resume', methods=['POST'])
    def optimize_resume():
        """Full-resume AI optimization: improve summary, projects, experience wording."""
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        data = request.json or {}
        rd = data.get('resume_data', {})
        results = {}

        llm = get_llm()
        if not llm:
            return jsonify({'error': 'LLM not available'}), 500

        # 1. Optimize summary
        summary = rd.get('professional_summary', '').strip()
        if summary:
            prompt = f"""<|im_start|>system
You are an expert ATS Resume Writer.
Improve this professional summary. Make it more professional, ATS-optimized, and polished.
Rules:
- 50-80 words
- One paragraph
- Professional tone
- Improve grammar and wording
- Add relevant ATS keywords naturally
- Do not invent any new information
- No markdown, no asterisks, no hash symbols, no headings
- Return only the improved summary text
<|im_end|>
<|im_start|>user
{summary}
<|im_end|>
<|im_start|>assistant
"""
            try:
                raw = llm.generate(prompt, max_tokens=180).strip()
                results['professional_summary'] = _clean_llm_output(raw)
            except Exception:
                results['professional_summary'] = summary

        # 2. Optimize projects
        projects = rd.get('projects', [])
        if isinstance(projects, list):
            optimized_projects = []
            for proj in projects:
                desc = proj.get('description', '').strip()
                pname = proj.get('name', '').strip()
                ptech = proj.get('technologies', '').strip()
                if desc and pname:
                    prompt = f"""<|im_start|>system
You are an ATS Resume Writer.
Improve this project description. Make it more professional, ATS-friendly, and polished.
Rules:
- 2-4 lines
- Professional language
- Improve grammar and wording
- Add ATS keywords naturally
- Do not invent any new information
- No markdown, no asterisks, no hash symbols, no headings
- Return only the improved description
<|im_end|>
<|im_start|>user
Project: {pname}
Technologies: {ptech}
Current Description: {desc}
<|im_end|>
<|im_start|>assistant
"""
                    try:
                        raw = llm.generate(prompt, max_tokens=150).strip()
                        optimized_projects.append(_clean_llm_output(raw))
                    except Exception:
                        optimized_projects.append(desc)
                else:
                    optimized_projects.append(desc)
            results['projects'] = optimized_projects

        # 3. Optimize experience
        experience = rd.get('experience', [])
        if isinstance(experience, list):
            optimized_exp = []
            for exp in experience:
                resp = exp.get('responsibilities', '').strip()
                role = exp.get('role', '').strip()
                comp = exp.get('company', '').strip()
                if resp and comp:
                    prompt = f"""<|im_start|>system
You are an ATS Resume Writer.
Improve these resume bullet points. Make them more professional and ATS-optimized.
Rules:
- 3-5 bullet points
- Strong action verbs
- Improve grammar and wording
- Add ATS keywords naturally
- Do not invent any new information
- No markdown, no asterisks, no hash symbols, no headings
- Each bullet point starts with a dash and a space
- Return only the bullet points
<|im_end|>
<|im_start|>user
Role: {role} at {comp}
Current bullets: {resp}
<|im_end|>
<|im_start|>assistant
"""
                    try:
                        raw = llm.generate(prompt, max_tokens=250).strip()
                        cleaned = _clean_llm_output(raw)
                        lines = []
                        for line in cleaned.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            for marker in ['- ', '* ', '• ', '– ']:
                                if line.startswith(marker):
                                    line = line[len(marker):]
                                    break
                            import re as _re
                            line = _re.sub(r'^\d+[\.\)]\s*', '', line)
                            if line:
                                lines.append(line)
                        lines = lines[:5]
                        optimized_exp.append("\n".join(f"- {l}" for l in lines))
                    except Exception:
                        optimized_exp.append(resp)
                else:
                    optimized_exp.append(resp)
            results['experience'] = optimized_exp

        return jsonify({'success': True, 'optimized': results})

    @app.route('/api/generate-resume', methods=['POST'])
    def generate_resume():
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        data = request.json or {}
        rd = data.get('resume_data', {})

        name = rd.get('name', '').strip()
        title = rd.get('title', '').strip()
        email = rd.get('email', '').strip()
        phone = rd.get('phone', '').strip()
        linkedin = rd.get('linkedin', '').strip()
        github = rd.get('github', '').strip()
        summary = rd.get('professional_summary', '').strip()
        skills = rd.get('skills', '').strip()
        education = rd.get('education', [])
        projects = rd.get('projects', [])
        experience = rd.get('experience', [])
        certifications = rd.get('certifications', [])
        languages = rd.get('languages', '').strip()
        activities = rd.get('activities', '').strip()

        def contact_line():
            parts = []
            if email:
                parts.append(escape(email))
            if phone:
                parts.append(escape(phone))
            if linkedin:
                disp = linkedin.replace('https://','').replace('http://','').replace('www.','')
                parts.append(escape(disp))
            if github:
                disp = github.replace('https://','').replace('http://','').replace('www.','')
                parts.append(escape(disp))
            return ' &nbsp;|&nbsp; '.join(parts)

        def section(heading, content):
            if not content:
                return ''
            return f'<div class="resume-section"><h2>{escape(heading)}</h2>{content}</div>'

        # Summary
        sum_html = f'<p>{escape(summary)}</p>' if summary else ''

        # Skills: group only supplied skills. Unknown skills remain visible under
        # Tools instead of being discarded or replaced with invented skills.
        skills_html = ''
        if skills:
            items = [s.strip() for s in skills.replace('\n', ',').split(',') if s.strip()]
            if items:
                groups = {'Languages': [], 'AI & ML': [], 'Web': [], 'Database': [], 'Frameworks': [], 'Tools': []}
                keywords = {
                    'Languages': ('python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'golang', 'ruby', 'php', 'kotlin', 'swift', 'matlab'),
                    'AI & ML': ('machine learning', 'deep learning', 'artificial intelligence', 'tensorflow', 'pytorch', 'keras', 'scikit', 'opencv', 'nlp', 'llm', 'transformer', 'pandas', 'numpy'),
                    'Web': ('html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'rest', 'api', 'bootstrap', 'tailwind'),
                    'Database': ('sql', 'mysql', 'postgres', 'mongodb', 'sqlite', 'redis', 'firebase', 'oracle'),
                    'Frameworks': ('flask', 'django', 'fastapi', 'spring', 'laravel', 'next', '.net'),
                    'Tools': ('git', 'github', 'docker', 'kubernetes', 'aws', 'azure', 'linux', 'figma', 'postman', 'jira', 'faiss')
                }
                for item in items:
                    lowered = item.lower()
                    target = next((group for group, terms in keywords.items() if any(term in lowered for term in terms)), 'Tools')
                    groups[target].append(item)
                skills_html = ''.join(
                    f'<p class="resume-skill-row"><strong>{escape(group)}:</strong> {escape(", ".join(values))}</p>'
                    for group, values in groups.items() if values
                )

        # Education
        edu_html = ''
        if isinstance(education, list):
            pieces = []
            for ed in education:
                coll = ed.get('college','').strip()
                if not coll:
                    continue
                deg = ed.get('degree','').strip()
                br = ed.get('branch','').strip()
                cgpa = ed.get('cgpa','').strip()
                yr = ed.get('graduation_year','').strip()
                line1 = f'<strong>{escape(deg)}</strong>'
                if br:
                    line1 += f' in {escape(br)}'
                line1 += f' &mdash; {escape(coll)}'
                detail_parts = []
                if cgpa:
                    detail_parts.append(f'CGPA: {escape(cgpa)}')
                if yr:
                    detail_parts.append(escape(yr))
                pieces.append(f'''<div class="resume-item">
                    <div class="resume-item-header">
                        <span class="resume-item-title">{line1}</span>
                        <span class="resume-item-date">{escape(yr)}</span>
                    </div>
                    {f'<div class="resume-item-details">{" | ".join(detail_parts)}</div>' if detail_parts else ''}
                </div>''')
            edu_html = ''.join(pieces)

        # Experience
        exp_html = ''
        if isinstance(experience, list):
            pieces = []
            for ex in experience:
                comp = ex.get('company','').strip()
                if not comp:
                    continue
                role = ex.get('role','').strip()
                dur = ex.get('duration','').strip()
                resp = ex.get('responsibilities','').strip()
                title_line = f'<strong>{escape(role)}</strong> &mdash; {escape(comp)}' if role else f'<strong>{escape(comp)}</strong>'
                bullets = ''
                if resp:
                    blines = []
                    for r in resp.splitlines():
                        r = r.strip()
                        if not r:
                            continue
                        for m in ['- ','* ','• ','– ']:
                            if r.startswith(m):
                                r = r[len(m):]
                                break
                        if r:
                            blines.append(f'<li>{escape(r)}</li>')
                    if blines:
                        bullets = '<ul class="resume-list">' + ''.join(blines) + '</ul>'
                pieces.append(f'''<div class="resume-item">
                    <div class="resume-item-header">
                        <span class="resume-item-title">{title_line}</span>
                        <span class="resume-item-date">{escape(dur)}</span>
                    </div>
                    {bullets}
                </div>''')
            exp_html = ''.join(pieces)

        # Projects
        proj_html = ''
        if isinstance(projects, list):
            pieces = []
            for p in projects:
                pn = p.get('name','').strip()
                if not pn:
                    continue
                pt = p.get('technologies','').strip()
                pd = p.get('description','').strip()
                tline = f'<strong>{escape(pn)}</strong>'
                if pt:
                    tline += f' <span class="resume-item-subtitle">| {escape(pt)}</span>'
                bullet_lines = []
                for line in pd.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    for marker in ['- ', '* ', '• ', '– ']:
                        if line.startswith(marker):
                            line = line[len(marker):]
                            break
                    bullet_lines.append(line)
                desc = ('<ul class="resume-list">' + ''.join(f'<li>{escape(line)}</li>' for line in bullet_lines) + '</ul>') if bullet_lines else ''
                pieces.append(f'''<div class="resume-item">
                    <div class="resume-item-header">
                        <span class="resume-item-title">{tline}</span>
                    </div>
                    {desc}
                </div>''')
            proj_html = ''.join(pieces)

        # Certifications
        cert_html = ''
        if isinstance(certifications, list):
            pieces = []
            for c in certifications:
                cn = c.get('name','').strip()
                if not cn:
                    continue
                ci = c.get('issuer','').strip()
                cd = c.get('date','').strip()
                tline = f'<strong>{escape(cn)}</strong>'
                if ci:
                    tline += f' &mdash; {escape(ci)}'
                date_text = f' ({escape(cd)})' if cd else ''
                pieces.append(f'<li>{tline}{date_text}</li>')
            cert_html = '<ul class="resume-list">' + ''.join(pieces) + '</ul>' if pieces else ''

        # Activities
        act_html = ''
        if activities:
            lines = [a.strip() for a in activities.splitlines() if a.strip()]
            if lines:
                act_html = '<ul class="resume-list">' + ''.join(f'<li>{escape(l)}</li>' for l in lines) + '</ul>'

        html = f'''<div class="resume-paper">
  <div class="resume-header">
    <h1>{escape(name) or "Your Name"}</h1>
    {f'<div class="resume-title">{escape(title)}</div>' if title else ''}
    <div class="resume-contact">{contact_line()}</div>
  </div>
  {section("Professional Summary", sum_html)}
  {section("Technical Skills", skills_html)}
  {section("Experience", exp_html)}
  {section("Projects", proj_html)}
  {section("Education", edu_html)}
  {section("Certifications", cert_html)}
  {section("Activities & Achievements", act_html)}
</div>'''
        return jsonify({'success': True, 'resume_html': html})

    def _clean_llm_output(text):
        """Remove markdown artifacts, headings, quotes, and labels from LLM output."""
        import re
        # Remove markdown bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        # Remove markdown headings
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Remove leading quotes
        text = text.strip('"\'')
        # Remove label-like prefixes the model might add
        for prefix in ['Summary:', 'Professional Summary:', 'Description:', 'Project Description:', 'Output:', 'Result:', 'Answer:']:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        return text.strip()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.1', port=8080, debug=True)
