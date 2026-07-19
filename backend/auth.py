from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from backend.database import create_user, get_user_by_email, verify_user, init_db
from datetime import datetime

bp = Blueprint('auth', __name__)


@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        roll = request.form.get('roll')
        branch = request.form.get('branch')
        semester = request.form.get('semester')
        try:
            uid = create_user(name, email, password, roll, branch, semester)
            flash('Account created. Please log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            flash('Registration failed: ' + str(e), 'danger')
    return render_template('register.html')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = verify_user(email, password)
        if user:
            session.clear()
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            try:
                from backend.database import update_last_login
                update_last_login(user['id'])
            except Exception:
                pass
            flash('Logged in successfully', 'success')
            return redirect(url_for('home'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')


@bp.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('auth.login'))
