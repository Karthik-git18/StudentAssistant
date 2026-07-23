import os
import logging
from pathlib import Path
from flask import Flask, render_template, session, redirect, url_for, g, jsonify, request

from backend.config import Config
from backend.database import init_db
from backend.routes.auth_routes import auth_bp
from backend.routes.student_routes import student_bp
from backend.routes.admin_routes import admin_bp

def create_app():
    # Setup logging
    logging.basicConfig(
        level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent.parent / 'frontend' / 'templates'),
        static_folder=str(Path(__file__).parent.parent / 'frontend' / 'static')
    )
    
    app.config.from_object(Config)

    # Initialize SQLite schema and seed admin user
    try:
        init_db()
    except Exception as e:
        app.logger.error(f"Error initializing database: {e}")

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(admin_bp)

    # Context processors
    @app.context_processor
    def inject_user():
        return dict(
            user_name=session.get('user_name'),
            user_role=session.get('user_role'),
            user_email=session.get('user_email')
        )

    # Root route redirection
    @app.route('/')
    def root():
        if session.get('user_id'):
            if session.get('user_role') == 'admin':
                return redirect(url_for('admin.dashboard'))
            return redirect(url_for('student.home'))
        return redirect(url_for('auth.login'))

    # Database connection teardown context handler
    @app.teardown_appcontext
    def close_db(error):
        db_conn = g.pop('db_conn', None)
        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:
                pass

    # Custom Error Handlers
    @app.errorhandler(403)
    def forbidden_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'message': 'Access forbidden.'}), 403
        return render_template('403.html'), 403

    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'message': 'Endpoint not found.'}), 404
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Server Error: {error}")
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'message': 'Internal server error.'}), 500
        return render_template('500.html'), 500

    return app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=True
    )