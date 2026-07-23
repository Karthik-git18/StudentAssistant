import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / '.env')
FRONTEND_DIR = BASE_DIR / 'frontend'
UPLOAD_FOLDER = FRONTEND_DIR / 'uploads'
DB_PATH = BASE_DIR / 'backend' / 'database.db'

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'backend' / 'indexes').mkdir(parents=True, exist_ok=True)

class Config:
    SECRET_KEY = os.environ.get('FLASK_SECRET', 'dev-secret-super-key-9988')
    DB_PATH = DB_PATH
    UPLOAD_FOLDER = UPLOAD_FOLDER
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit
    ALLOWED_EXTENSIONS = {'pdf'}
    
    # API settings
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    # Models
    # Fast OpenRouter models
    OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL', 'google/gemini-2.5-flash')
    EMBEDDING_MODEL = 'gemini-embedding-001'
