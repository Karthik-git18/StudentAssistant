import sqlite3
import logging
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from backend.config import DB_PATH

logger = logging.getLogger(__name__)

_local_conn = None

def get_conn():
    """
    Get SQLite database connection. Reuses one connection per request (via Flask g).
    busy_timeout = 10 s ensures write operations wait instead of immediately raising
    'database is locked'.
    """
    try:
        from flask import g, has_request_context
        if has_request_context():
            if 'db_conn' not in g:
                conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA busy_timeout = 10000;")
                conn.execute("PRAGMA foreign_keys = ON;")
                g.db_conn = conn
            return g.db_conn
    except Exception:
        pass

    global _local_conn
    if _local_conn is not None:
        try:
            _local_conn.execute("SELECT 1")
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            _local_conn = None

    if _local_conn is None:
        _local_conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
        _local_conn.row_factory = sqlite3.Row
        _local_conn.execute("PRAGMA journal_mode = WAL;")
        _local_conn.execute("PRAGMA busy_timeout = 10000;")
        _local_conn.execute("PRAGMA foreign_keys = ON;")

    return _local_conn

def init_db():
    """
    Initialize database, create normalized tables, add indexes, and seed default admin.
    """
    conn = get_conn()
    cur = conn.cursor()
    
    # 1. Users Table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'student',
        phone TEXT,
        department TEXT,
        year TEXT,
        university TEXT,
        profile_pic TEXT,
        status TEXT DEFAULT 'active',
        last_login DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 2. Uploads Table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        pages INTEGER,
        file_hash TEXT,
        storage_key TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')
    
    # 3. Chats Table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        session_title TEXT,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')
    
    # 4. Resumes Table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        template TEXT DEFAULT 'classic',
        resume_json TEXT NOT NULL,
        ats_score INTEGER DEFAULT 0,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')
    
    # 5. Study Plans Table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS study_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        plan_json TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')

    # 6. Audit Logs Table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT NOT NULL,
        details TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
    )
    ''')
    
    # 7. Embedding Cache
    cur.execute('''
    CREATE TABLE IF NOT EXISTS embedding_cache (
        text_hash TEXT PRIMARY KEY,
        embedding_json TEXT NOT NULL
    )
    ''')

    # 8. Prompt Cache
    cur.execute('''
    CREATE TABLE IF NOT EXISTS prompt_cache (
        prompt_hash TEXT PRIMARY KEY,
        response_text TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Dynamic schema upgrades: Ensure new columns exist if DB already exists
    cur.execute('PRAGMA table_info(users)')
    cols = [row[1] for row in cur.fetchall()]
    upgrades = {
        'role': "TEXT DEFAULT 'student'",
        'phone': "TEXT",
        'department': "TEXT",
        'year': "TEXT",
        'university': "TEXT",
        'profile_pic': "TEXT",
        'status': "TEXT DEFAULT 'active'",
        'created_at': "DATETIME"
    }
    for col, definition in upgrades.items():
        if col not in cols:
            try:
                cur.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
                conn.commit()
                logger.info(f"Database Migrations: Added column {col} to users table.")
            except Exception as e:
                logger.error(f"Migration error for column {col}: {e}")

    # Ensure chats has session_id and session_title
    cur.execute('PRAGMA table_info(chats)')
    chats_cols = [row[1] for row in cur.fetchall()]
    chats_upgrades = {
        'session_id': "TEXT NOT NULL DEFAULT 'default'",
        'session_title': "TEXT DEFAULT 'New Conversation'"
    }
    for col, definition in chats_upgrades.items():
        if col not in chats_cols:
            try:
                cur.execute(f"ALTER TABLE chats ADD COLUMN {col} {definition}")
                conn.commit()
                logger.info(f"Database Migrations: Added column {col} to chats table.")
            except Exception as e:
                logger.error(f"Migration error for column {col} in chats: {e}")

    # Ensure uploads has storage_key and file_hash
    cur.execute('PRAGMA table_info(uploads)')
    uploads_cols = [row[1] for row in cur.fetchall()]
    uploads_upgrades = {
        'file_hash': "TEXT",
        'storage_key': "TEXT"
    }
    for col, definition in uploads_upgrades.items():
        if col not in uploads_cols:
            try:
                cur.execute(f"ALTER TABLE uploads ADD COLUMN {col} {definition}")
                conn.commit()
                logger.info(f"Database Migrations: Added column {col} to uploads table.")
            except Exception as e:
                logger.error(f"Migration error for col {col} in uploads: {e}")

    # Create Indexes
    cur.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_uploads_user_id ON uploads(user_id);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_chats_session ON chats(session_id);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_resumes_user_id ON resumes(user_id);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_logs_action ON logs(action);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_logs_user_id ON logs(user_id);')

    conn.commit()

    # Seed Default Admin User
    cur.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    if cur.fetchone()[0] == 0:
        admin_pass = generate_password_hash("admin123")
        try:
            cur.execute(
                "INSERT INTO users (name, email, password, role, status) VALUES (?, ?, ?, ?, ?)",
                ("System Admin", "admin@studentai.com", admin_pass, "admin", "active")
            )
            conn.commit()
            logger.info("Database Seeding: Created default admin: admin@studentai.com / admin123")
        except Exception as e:
            logger.error(f"Failed to seed admin: {e}")

    # If connection was created locally, close it
    try:
        from flask import has_request_context
        if not has_request_context():
            conn.close()
    except Exception:
        conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_user(name, email, password, role='student', phone=None, department=None, year=None, university=None):
    conn = get_conn()
    cur = conn.cursor()
    hashed = generate_password_hash(password)
    cur.execute(
        'INSERT INTO users (name, email, password, role, phone, department, year, university) VALUES (?,?,?,?,?,?,?,?)',
        (name, email, hashed, role, phone, department, year, university)
    )
    conn.commit()
    uid = cur.lastrowid
    create_log(uid, 'register', f"Registered as {role}")
    return uid

def get_user_by_email(email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    return row

def verify_user(email, password):
    user = get_user_by_email(email)
    if not user:
        return None
    if user['status'] != 'active':
        return None
    if check_password_hash(user['password'], password):
        return user
    return None

def get_user_by_id(uid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE id = ?', (uid,))
    row = cur.fetchone()
    return row

def update_user(uid, name, email, phone=None, department=None, year=None, university=None, profile_pic=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        '''UPDATE users 
           SET name = ?, email = ?, phone = ?, department = ?, year = ?, university = ?, profile_pic = COALESCE(?, profile_pic) 
           WHERE id = ?''',
        (name, email, phone, department, year, university, profile_pic, uid)
    )
    conn.commit()
    updated = cur.rowcount
    create_log(uid, 'update_profile', "Profile details updated")
    return updated

def change_password(uid, new_password):
    conn = get_conn()
    cur = conn.cursor()
    hashed = generate_password_hash(new_password)
    cur.execute('UPDATE users SET password = ? WHERE id = ?', (hashed, uid))
    conn.commit()
    create_log(uid, 'change_password', "Password changed successfully")
    return cur.rowcount

def update_last_login(uid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (uid,))
    conn.commit()
    create_log(uid, 'login', "Logged into system")

def delete_user(uid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM users WHERE id = ?', (uid,))
    conn.commit()
    return cur.rowcount

def set_user_status(uid, status):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('UPDATE users SET status = ? WHERE id = ?', (status, uid))
    conn.commit()
    return cur.rowcount

# ══════════════════════════════════════════════════════════════════════════════
# UPLOADS & DOCUMENTS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_upload(user_id, filename, pages=None, file_hash=None, storage_key=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO uploads (user_id, filename, pages, file_hash, storage_key) VALUES (?,?,?,?,?)',
        (user_id, filename, pages, file_hash, storage_key)
    )
    conn.commit()
    uid = cur.lastrowid
    create_log(user_id, 'pdf_upload', f"Uploaded document: {filename} ({pages} pages)")
    return uid

def is_duplicate_upload(user_id, file_hash):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT id FROM uploads WHERE user_id=? AND file_hash=?', (user_id, file_hash))
    row = cur.fetchone()
    return row is not None

def get_user_uploads(user_id, limit=20):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'SELECT id, filename, pages, file_hash, storage_key, created_at FROM uploads WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT ?',
        (user_id, limit)
    )
    rows = cur.fetchall()
    return [dict(row) for row in rows]

def get_upload(user_id, upload_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'SELECT id, filename, pages, file_hash, storage_key, created_at FROM uploads WHERE id=? AND user_id=?',
        (upload_id, user_id)
    )
    row = cur.fetchone()
    return dict(row) if row else None

def delete_upload(user_id, upload_id):
    conn = get_conn()
    cur = conn.cursor()
    # Fetch filename in same cursor before deleting (avoids nested connection calls)
    cur.execute('SELECT filename FROM uploads WHERE id=? AND user_id=?', (upload_id, user_id))
    row = cur.fetchone()
    if not row:
        return 0
    filename = row['filename']
    cur.execute('DELETE FROM uploads WHERE id=? AND user_id=?', (upload_id, user_id))
    conn.commit()
    create_log(user_id, 'delete_upload', f"Deleted document: {filename}")
    return cur.rowcount

def get_recent_uploads(user_id, limit=3):
    return get_user_uploads(user_id, limit)

# ══════════════════════════════════════════════════════════════════════════════
# AI CHATS & MESSAGES
# ══════════════════════════════════════════════════════════════════════════════

def create_chat(user_id, session_id, session_title, role, message):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO chats (user_id, session_id, session_title, role, message) VALUES (?,?,?,?,?)',
        (user_id, session_id, session_title, role, message)
    )
    conn.commit()
    cid = cur.lastrowid
    return cid

def get_chat_session_history(user_id, session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'SELECT role, message, created_at FROM chats WHERE user_id=? AND session_id=? ORDER BY created_at ASC, id ASC',
        (user_id, session_id)
    )
    rows = cur.fetchall()
    return [dict(row) for row in rows]

def get_user_chat_sessions(user_id, limit=50):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        '''SELECT session_id, session_title, MAX(created_at) as last_active 
           FROM chats 
           WHERE user_id=? 
           GROUP BY session_id, session_title 
           ORDER BY last_active DESC 
           LIMIT ?''',
        (user_id, limit)
    )
    rows = cur.fetchall()
    return [dict(row) for row in rows]

def rename_chat_session(user_id, session_id, new_title):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'UPDATE chats SET session_title = ? WHERE user_id = ? AND session_id = ?',
        (new_title, user_id, session_id)
    )
    conn.commit()
    return cur.rowcount

def delete_chat_session(user_id, session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM chats WHERE user_id = ? AND session_id = ?', (user_id, session_id))
    conn.commit()
    return cur.rowcount

def get_recent_chats(user_id, limit=3):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'SELECT session_id, session_title, message, role, created_at FROM chats WHERE user_id=? ORDER BY created_at DESC LIMIT ?',
        (user_id, limit)
    )
    rows = cur.fetchall()
    return [dict(row) for row in rows]

# ══════════════════════════════════════════════════════════════════════════════
# AI RESUMES
# ══════════════════════════════════════════════════════════════════════════════

def create_resume(user_id, template, resume_json, ats_score):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO resumes (user_id, template, resume_json, ats_score, updated_at) VALUES (?,?,?,?, CURRENT_TIMESTAMP)',
        (user_id, template, resume_json, ats_score)
    )
    conn.commit()
    rid = cur.lastrowid
    create_log(user_id, 'resume_generate', f"Created resume (ATS: {ats_score}%)")
    return rid

def get_resume(resume_id, user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM resumes WHERE id=? AND user_id=?', (resume_id, user_id))
    row = cur.fetchone()
    return dict(row) if row else None

def get_recent_resume(user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM resumes WHERE user_id=? ORDER BY updated_at DESC, id DESC LIMIT 1', (user_id,))
    row = cur.fetchone()
    return dict(row) if row else None

def get_user_resumes(user_id, limit=20):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT id, template, ats_score, updated_at FROM resumes WHERE user_id=? ORDER BY updated_at DESC LIMIT ?', (user_id, limit))
    rows = cur.fetchall()
    return [dict(row) for row in rows]

def delete_resume(resume_id, user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM resumes WHERE id=? AND user_id=?', (resume_id, user_id))
    conn.commit()
    return cur.rowcount

# ══════════════════════════════════════════════════════════════════════════════
# STUDY PLANS
# ══════════════════════════════════════════════════════════════════════════════

def create_study_plan(user_id, title, plan_json):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO study_plans (user_id, title, plan_json) VALUES (?,?,?)', (user_id, title, plan_json))
    conn.commit()
    pid = cur.lastrowid
    create_log(user_id, 'study_plan', f"Created study plan: {title}")
    return pid

def get_recent_plans(user_id, limit=3):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT title, created_at FROM study_plans WHERE user_id=? ORDER BY created_at DESC LIMIT ?',
                (user_id, limit))
    rows = cur.fetchall()
    return [dict(row) for row in rows]

# ══════════════════════════════════════════════════════════════════════════════
# METRICS & STATS FOR STUDENTS
# ══════════════════════════════════════════════════════════════════════════════

def count_user_items(user_id):
    conn = get_conn()
    cur = conn.cursor()
    
    cur.execute('SELECT COUNT(*) as uploads FROM uploads WHERE user_id=?', (user_id,))
    uploads = cur.fetchone()['uploads']
    
    cur.execute("SELECT COUNT(*) as ai_requests FROM logs WHERE user_id=? AND action LIKE 'ai_%'", (user_id,))
    ai_requests = cur.fetchone()['ai_requests']
    
    cur.execute('SELECT COUNT(*) as resumes FROM resumes WHERE user_id=?', (user_id,))
    resumes = cur.fetchone()['resumes']
    
    cur.execute('SELECT COUNT(*) as plans FROM study_plans WHERE user_id=?', (user_id,))
    plans = cur.fetchone()['plans']
    
    cur.execute('SELECT COUNT(*) as questions FROM chats WHERE user_id=? AND role = ?', (user_id, 'user'))
    questions = cur.fetchone()['questions']

    return {
        'uploads': uploads,
        'ai_requests': ai_requests,
        'resumes': resumes,
        'plans': plans,
        'questions': questions
    }

# ══════════════════════════════════════════════════════════════════════════════
# CACHE IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_cached_embedding(text_hash):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT embedding_json FROM embedding_cache WHERE text_hash = ?', (text_hash,))
    row = cur.fetchone()
    return row['embedding_json'] if row else None

def save_cached_embedding(text_hash, embedding_json):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT OR REPLACE INTO embedding_cache (text_hash, embedding_json) VALUES (?, ?)',
            (text_hash, embedding_json)
        )
        conn.commit()
    except Exception:
        pass

def get_cached_prompt(prompt_hash):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT response_text FROM prompt_cache WHERE prompt_hash = ?', (prompt_hash,))
    row = cur.fetchone()
    return row['response_text'] if row else None

def save_cached_prompt(prompt_hash, response_text):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT OR REPLACE INTO prompt_cache (prompt_hash, response_text) VALUES (?, ?)',
            (prompt_hash, response_text)
        )
        conn.commit()
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING AUDIT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def create_log(user_id, action, details=None):
    # If standard connection is closed/non-writable, open a standalone connection
    # to guarantee log writes don't fail user actions.
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('INSERT INTO logs (user_id, action, details) VALUES (?, ?, ?)', (user_id, action, details))
        conn.commit()
    except Exception as e:
        logger.error(f"Audit Logging Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ADMIN PORTAL QUERIES
# ══════════════════════════════════════════════════════════════════════════════

def admin_get_metrics():
    conn = get_conn()
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM users WHERE role = 'student'")
    total_students = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM users WHERE role = 'student' AND status = 'active'")
    active_students = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM uploads")
    total_pdfs = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM logs WHERE action LIKE 'ai_%'")
    ai_requests = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT user_id) FROM logs WHERE created_at >= date('now', '-1 day')")
    dau = cur.fetchone()[0]
    
    return {
        'total_students': total_students,
        'active_students': active_students,
        'total_pdfs': total_pdfs,
        'ai_requests': ai_requests,
        'dau': dau
    }

def admin_get_students(search_query=None):
    conn = get_conn()
    cur = conn.cursor()
    if search_query:
        cur.execute(
            "SELECT id, name, email, phone, department, year, university, status, last_login, created_at FROM users WHERE role = 'student' AND (name LIKE ? OR email LIKE ?) ORDER BY created_at DESC",
            (f'%{search_query}%', f'%{search_query}%')
        )
    else:
        cur.execute(
            "SELECT id, name, email, phone, department, year, university, status, last_login, created_at FROM users WHERE role = 'student' ORDER BY created_at DESC"
        )
    return [dict(row) for row in cur.fetchall()]

def admin_get_all_documents():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        '''SELECT uploads.id, uploads.filename, uploads.pages, uploads.created_at, users.name as student_name, users.email as student_email 
           FROM uploads 
           JOIN users ON uploads.user_id = users.id 
           ORDER BY uploads.created_at DESC'''
    )
    return [dict(row) for row in cur.fetchall()]

def admin_get_logs(limit=100):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        '''SELECT logs.id, logs.action, logs.details, logs.created_at, users.name as user_name, users.email as user_email, users.role as user_role 
           FROM logs 
           LEFT JOIN users ON logs.user_id = users.id 
           ORDER BY logs.created_at DESC 
           LIMIT ?''',
        (limit,)
    )
    return [dict(row) for row in cur.fetchall()]

def admin_get_analytics_data():
    conn = get_conn()
    cur = conn.cursor()
    
    # 1. Registrations over last 7 days
    cur.execute(
        '''SELECT date(created_at) as date_val, COUNT(*) as count 
           FROM users 
           WHERE role = 'student' AND created_at >= date('now', '-7 days') 
           GROUP BY date_val 
           ORDER BY date_val ASC'''
    )
    registrations = [dict(row) for row in cur.fetchall()]
    
    # 2. AI usage by activity last 7 days
    cur.execute(
        '''SELECT date(created_at) as date_val, COUNT(*) as count 
           FROM logs 
           WHERE action LIKE 'ai_%' AND created_at >= date('now', '-7 days') 
           GROUP BY date_val 
           ORDER BY date_val ASC'''
    )
    daily_ai_requests = [dict(row) for row in cur.fetchall()]

    # 3. Most Active Students (uploads + ai requests)
    cur.execute(
        '''SELECT users.id, users.name, users.email, COUNT(logs.id) as activity_count 
           FROM users 
           JOIN logs ON logs.user_id = users.id 
           WHERE users.role = 'student' 
           GROUP BY users.id 
           ORDER BY activity_count DESC 
           LIMIT 5'''
    )
    active_students = [dict(row) for row in cur.fetchall()]

    return {
        'registrations': registrations,
        'daily_ai_requests': daily_ai_requests,
        'active_students': active_students
    }
