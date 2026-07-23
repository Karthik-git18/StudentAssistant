from flask import Blueprint, render_template, request, redirect, url_for, session, flash, g
from backend.database import create_user, get_user_by_email, verify_user, create_log
import re

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('student.home'))
        
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        phone = request.form.get('phone', '').strip()
        department = request.form.get('department', '').strip()
        year = request.form.get('year', '').strip()
        university = request.form.get('university', '').strip()
        
        # Validation
        if not name or not email or not password:
            flash('Name, Email, and Password are required fields.', 'danger')
            return render_template('register.html')
            
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('Invalid email format.', 'danger')
            return render_template('register.html')
            
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html')

        try:
            # Check if email exists
            existing = get_user_by_email(email)
            if existing:
                flash('Email is already registered. Please log in.', 'danger')
                return redirect(url_for('auth.login'))
                
            # Register user (force role = student)
            uid = create_user(
                name=name, email=email, password=password, role='student',
                phone=phone, department=department, year=year, university=university
            )
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            flash(f'Registration failed: {e}', 'danger')
            
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        # Route depending on role
        if session.get('user_role') == 'admin':
            return redirect(url_for('admin.dashboard'))
        return redirect(url_for('student.home'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Please enter email and password.', 'danger')
            return render_template('login.html')
            
        user = verify_user(email, password)
        if user:
            session.clear()
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_role'] = user['role']
            session['user_email'] = user['email']
            
            from backend.database import update_last_login
            update_last_login(user['id'])
            
            flash('Logged in successfully', 'success')
            if user['role'] == 'admin':
                return redirect(url_for('admin.dashboard'))
            return redirect(url_for('student.home'))
            
        flash('Invalid credentials or account is disabled.', 'danger')
        
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    uid = session.get('user_id')
    if uid:
        create_log(uid, 'logout', "Logged out from system")
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('auth.login'))
