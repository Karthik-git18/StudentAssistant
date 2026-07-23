from functools import wraps
from flask import session, redirect, url_for, flash, abort, g
from backend.database import get_user_by_id

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            flash('Authentication required. Please log in.', 'danger')
            return redirect(url_for('auth.login'))
        
        # Verify user still exists and is active
        user = get_user_by_id(user_id)
        if not user or user['status'] != 'active':
            session.clear()
            flash('Your account has been deactivated or does not exist.', 'danger')
            return redirect(url_for('auth.login'))
            
        g.current_user = user
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            flash('Authentication required. Please log in.', 'danger')
            return redirect(url_for('auth.login'))
            
        user = get_user_by_id(user_id)
        if not user or user['status'] != 'active':
            session.clear()
            flash('Session invalid.', 'danger')
            return redirect(url_for('auth.login'))
            
        if user['role'] != 'admin':
            # Abort with 403 Forbidden for non-admin attempts
            abort(403)
            
        g.current_user = user
        return f(*args, **kwargs)
    return decorated_function
