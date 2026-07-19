import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = Path(__file__).parent / 'database.db'


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        roll TEXT,
        branch TEXT,
        semester TEXT,
        last_login DATETIME
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        pages INTEGER,
        file_hash TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS study_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        plan_json TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        role TEXT,
        message TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')
    conn.commit()
    _ensure_upload_hash_column(conn)
    _ensure_upload_storage_column(conn)
    conn.close()


def _ensure_upload_hash_column(conn):
    cur = conn.cursor()
    cur.execute('PRAGMA table_info(uploads)')
    columns = [row[1] for row in cur.fetchall()]
    if 'file_hash' not in columns:
        cur.execute('ALTER TABLE uploads ADD COLUMN file_hash TEXT')
        conn.commit()


def _ensure_upload_storage_column(conn):
    cur = conn.cursor()
    cur.execute('PRAGMA table_info(uploads)')
    columns = [row[1] for row in cur.fetchall()]
    if 'storage_key' not in columns:
        cur.execute('ALTER TABLE uploads ADD COLUMN storage_key TEXT')
        conn.commit()


def create_user(name, email, password, roll=None, branch=None, semester=None):
    conn = get_conn()
    cur = conn.cursor()
    hashed = generate_password_hash(password)
    cur.execute('INSERT INTO users (name,email,password,roll,branch,semester) VALUES (?,?,?,?,?,?)',
                (name, email, hashed, roll, branch, semester))
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return uid


def get_user_by_email(email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    conn.close()
    return row


def verify_user(email, password):
    user = get_user_by_email(email)
    if not user:
        return None
    if check_password_hash(user['password'], password):
        return user
    return None


def get_user_by_id(uid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE id = ?', (uid,))
    row = cur.fetchone()
    conn.close()
    return row


def update_user(uid, name, email, roll=None, branch=None, semester=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'UPDATE users SET name = ?, email = ?, roll = ?, branch = ?, semester = ? WHERE id = ?',
        (name, email, roll, branch, semester, uid)
    )
    conn.commit()
    updated = cur.rowcount
    conn.close()
    return updated


def update_last_login(uid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (uid,))
    conn.commit()
    conn.close()


def create_upload(user_id, filename, pages=None, file_hash=None, storage_key=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO uploads (user_id, filename, pages, file_hash, storage_key) VALUES (?,?,?,?,?)',
                (user_id, filename, pages, file_hash, storage_key))
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return uid


def create_chat(user_id, role, message):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO chats (user_id, role, message) VALUES (?,?,?)', (user_id, role, message))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid


def create_study_plan(user_id, title, plan_json):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO study_plans (user_id, title, plan_json) VALUES (?,?,?)', (user_id, title, plan_json))
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid


def count_user_items(user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) as uploads FROM uploads WHERE user_id=?', (user_id,))
    uploads = cur.fetchone()['uploads']
    cur.execute('SELECT COUNT(*) as questions FROM chats WHERE user_id=? AND role = ?', (user_id, 'user'))
    questions = cur.fetchone()['questions']
    cur.execute('SELECT COUNT(*) as plans FROM study_plans WHERE user_id=?', (user_id,))
    plans = cur.fetchone()['plans']
    conn.close()
    return {'uploads': uploads, 'questions': questions, 'plans': plans}


def get_user_uploads(user_id, limit=20):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'SELECT id, filename, pages, file_hash, storage_key, created_at FROM uploads WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT ?',
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_upload(user_id, upload_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM uploads WHERE id=? AND user_id=?', (upload_id, user_id))
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return deleted


def get_upload(user_id, upload_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT id, filename, pages, file_hash, storage_key, created_at FROM uploads WHERE id=? AND user_id=?',
                (upload_id, user_id))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def is_duplicate_upload(user_id, file_hash):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT id FROM uploads WHERE user_id=? AND file_hash=?', (user_id, file_hash))
    row = cur.fetchone()
    conn.close()
    return row is not None


def get_recent_uploads(user_id, limit=3):
    return get_user_uploads(user_id, limit)


def get_recent_chats(user_id, limit=3):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT role, message, created_at FROM chats WHERE user_id=? ORDER BY created_at DESC LIMIT ?',
                (user_id, limit))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_recent_plans(user_id, limit=3):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT title, created_at FROM study_plans WHERE user_id=? ORDER BY created_at DESC LIMIT ?',
                (user_id, limit))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows
