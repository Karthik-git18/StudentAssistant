from flask import Blueprint, render_template, request, jsonify, session, g, abort, send_from_directory
from backend.middleware import admin_required
from backend.database import (
    admin_get_metrics, admin_get_students, admin_get_all_documents, admin_get_logs, admin_get_analytics_data,
    get_user_by_id, delete_user, set_user_status, get_upload, delete_upload, create_log,
    get_user_uploads
)
from backend.services.pdf_service import delete_document_index, INDEX_DIR
from backend.config import Config

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/admin/dashboard')
@admin_required
def dashboard():
    metrics = admin_get_metrics()
    recent_logs = admin_get_logs(limit=10)
    return render_template('admin/dashboard.html', metrics=metrics, recent_logs=recent_logs)

@admin_bp.route('/admin/students')
@admin_required
def students():
    search_q = request.args.get('search', '').strip()
    student_list = admin_get_students(search_query=search_q)
    return render_template('admin/students.html', students=student_list, search_q=search_q)

@admin_bp.route('/admin/documents')
@admin_required
def documents():
    doc_list = admin_get_all_documents()
    return render_template('admin/documents.html', documents=doc_list)

@admin_bp.route('/admin/logs')
@admin_required
def logs():
    log_list = admin_get_logs(limit=150)
    return render_template('admin/logs.html', logs=log_list)

@admin_bp.route('/admin/analytics')
@admin_required
def analytics():
    return render_template('admin/analytics.html')

# ══════════════════════════════════════════════════════════════════════════════
# ADMIN APIs
# ══════════════════════════════════════════════════════════════════════════════

@admin_bp.route('/api/admin/students/<int:student_id>', methods=['PUT', 'DELETE'])
@admin_required
def manage_student(student_id):
    student = get_user_by_id(student_id)
    if not student or student['role'] == 'admin':
        return jsonify({'success': False, 'message': 'Student not found or unauthorized.'}), 404
        
    if request.method == 'DELETE':
        # 1. Collect upload metadata BEFORE any DB writes (avoids nested cursor locking)
        from backend.database import get_user_uploads, get_conn as _get_conn
        uploads = get_user_uploads(student_id, limit=200)

        # 2. Delete index + PDF files first (no DB writes yet)
        for doc in uploads:
            delete_document_index(
                INDEX_DIR / f"document_{student_id}_{doc['id']}.index",
                cache_key=f"{student_id}_{doc['id']}"
            )
            text_path = INDEX_DIR / f"document_{student_id}_{doc['id']}.text"
            if text_path.exists():
                try:
                    text_path.unlink()
                except OSError:
                    pass
            if doc.get('storage_key'):
                pdf_path = Config.UPLOAD_FOLDER / doc['storage_key']
                if pdf_path.exists():
                    try:
                        pdf_path.unlink()
                    except OSError:
                        pass

        # 3. Delete all DB rows in ONE transaction — no nested calls
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute('DELETE FROM uploads WHERE user_id = ?', (student_id,))
        cur.execute('DELETE FROM chats WHERE user_id = ?', (student_id,))
        cur.execute('DELETE FROM resumes WHERE user_id = ?', (student_id,))
        cur.execute('DELETE FROM users WHERE id = ?', (student_id,))
        conn.commit()

        create_log(session['user_id'], 'admin_delete_student', f"Deleted student: {student['email']}")
        return jsonify({'success': True, 'message': 'Student deleted successfully.'})
        
    elif request.method == 'PUT':
        data = request.json or {}
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        phone = data.get('phone', '').strip()
        department = data.get('department', '').strip()
        year = data.get('year', '').strip()
        university = data.get('university', '').strip()
        
        if not name or not email:
            return jsonify({'success': False, 'message': 'Name and Email are required.'}), 400
            
        # Check email uniqueness
        from backend.database import get_conn
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email=? AND id!=?", (email, student_id))
        if cur.fetchone():
            return jsonify({'success': False, 'message': 'Email is already taken by another user.'}), 400
            
        from backend.database import update_user as update_user_db
        update_user_db(student_id, name, email, phone, department, year, university)
        create_log(session['user_id'], 'admin_edit_student', f"Edited details for student: {email}")
        return jsonify({'success': True, 'message': 'Student updated successfully.'})

@admin_bp.route('/api/admin/students/<int:student_id>/status', methods=['POST'])
@admin_required
def toggle_student_status(student_id):
    student = get_user_by_id(student_id)
    if not student or student['role'] == 'admin':
        return jsonify({'success': False, 'message': 'Student not found or unauthorized.'}), 404
        
    data = request.json or {}
    status = data.get('status', 'active').strip()
    if status not in ['active', 'disabled']:
        return jsonify({'success': False, 'message': 'Invalid status.'}), 400
        
    set_user_status(student_id, status)
    action_type = 'admin_enable_student' if status == 'active' else 'admin_disable_student'
    create_log(session['user_id'], action_type, f"Status updated to {status} for student: {student['email']}")
    return jsonify({'success': True, 'message': f"Student account has been {status}d."})

@admin_bp.route('/api/admin/documents/<int:doc_id>', methods=['DELETE'])
@admin_required
def admin_delete_document(doc_id):
    from backend.database import get_conn
    conn = get_conn()
    cur = conn.cursor()

    # Fetch metadata
    cur.execute("SELECT user_id, filename, storage_key FROM uploads WHERE id = ?", (doc_id,))
    row = cur.fetchone()
    if not row:
        return jsonify({'success': False, 'message': 'Document not found.'}), 404

    user_id    = row['user_id']
    filename   = row['filename']
    storage_key = row['storage_key']

    # Delete DB entry in same cursor (no nested call)
    cur.execute("DELETE FROM uploads WHERE id = ?", (doc_id,))
    conn.commit()

    # Delete FAISS index + text cache
    delete_document_index(INDEX_DIR / f"document_{user_id}_{doc_id}.index", cache_key=f"{user_id}_{doc_id}")
    text_path = INDEX_DIR / f"document_{user_id}_{doc_id}.text"
    if text_path.exists():
        try:
            text_path.unlink()
        except OSError:
            pass

    # Delete raw PDF file
    if storage_key:
        pdf_path = Config.UPLOAD_FOLDER / storage_key
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except OSError:
                pass

    create_log(session['user_id'], 'admin_delete_document', f"Deleted document: {filename} (user {user_id})")
    return jsonify({'success': True, 'message': 'Document deleted successfully.'})

@admin_bp.route('/api/admin/documents/<int:doc_id>/view', methods=['GET'])
@admin_required
def admin_view_document(doc_id):
    """Admin can view any student PDF — bypasses ownership check."""
    from backend.database import get_conn
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id, storage_key, filename FROM uploads WHERE id = ?", (doc_id,))
    row = cur.fetchone()
    if not row or not row['storage_key']:
        return "Document not found", 404
    create_log(session['user_id'], 'admin_view_document', f"Viewed document #{doc_id}: {row['filename']}")
    return send_from_directory(Config.UPLOAD_FOLDER, row['storage_key'])

@admin_bp.route('/api/admin/students/<int:student_id>/uploads', methods=['GET'])
@admin_required
def admin_student_uploads(student_id):
    """Return all PDF uploads for a specific student."""
    student = get_user_by_id(student_id)
    if not student:
        return jsonify({'success': False, 'message': 'Student not found.'}), 404
    uploads = get_user_uploads(student_id, limit=100)
    return jsonify({'success': True, 'uploads': [dict(u) for u in uploads], 'student_name': student['name']})

@admin_bp.route('/api/admin/analytics/data', methods=['GET'])
@admin_required
def admin_analytics_data():
    analytics_payload = admin_get_analytics_data()
    return jsonify({'success': True, 'analytics': analytics_payload})
