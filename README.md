# Student Assistant (Local LLM) - Premium SaaS UI

A professional, full-stack GenAI Student Assistant built with Flask and local models. Features a modern light-themed interface inspired by ChatGPT, Notion, and Linear with complete authentication, RAG (Retrieval-Augmented Generation) pipeline, AI chat, and study planning capabilities.

## ✨ Features

- **Modern SaaS UI** — Light theme with sidebar navigation, glassmorphism, smooth animations
- **User Authentication** — Secure registration and login with password hashing
- **Learning Assistant** — Upload PDFs, ask questions about content using RAG
- **AI Chat** — Chat with local Qwen LLM on any topic
- **Study Planner** — Generate personalized study schedules
- **User Profile** — View stats, manage account information
- **Fully Responsive** — Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment

### Installation

1. **Clone/extract and navigate to the project:**
```bash
cd /Users/karthikpalepu/Desktop/Student_Assistant/StudentAssistant
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download local LLM model:**
- Download `Qwen2.5-0.5B-Instruct` from Hugging Face
- Extract to: `StudentAssistant/models/Qwen2.5-0.5B-Instruct/`
- Required model structure:
  ```
  models/
  └── Qwen2.5-0.5B-Instruct/
      ├── config.json
      ├── model.safetensors (or pytorch_model.bin)
      ├── tokenizer.json
      └── ... (other model files)
  ```

5. **Run the app:**
```bash
cd backend
python3 -c "from app import create_app; a = create_app(); a.run(port=8080, debug=True)"
```

6. **Access the application:**
Open `http://127.0.0.1:8080` in your browser

## 📁 Project Structure

```
StudentAssistant/
├── backend/
│   ├── app.py              # Flask application & routes
│   ├── auth.py             # Authentication blueprint
│   ├── chat.py             # Chat API endpoints
│   ├── planner.py          # Study planner API
│   ├── rag.py              # RAG pipeline (PDF upload, Q&A)
│   ├── model_loader.py     # Local LLM & embedder caching
│   ├── database.py         # SQLite database helpers
│   ├── database.db         # SQLite database (auto-created)
│   └── indexes/            # FAISS indexes per user
├── frontend/
│   ├── templates/          # HTML templates
│   │   ├── base.html       # Layout with sidebar & navbar
│   │   ├── login.html      # Login page
│   │   ├── register.html   # Registration page
│   │   ├── home.html       # Dashboard
│   │   ├── learning.html   # Learning assistant
│   │   ├── chat.html       # AI Chat
│   │   ├── planner.html    # Study planner
│   │   └── profile.html    # User profile
│   ├── static/
│   │   ├── css/style.css   # Modern light-themed CSS
│   │   └── js/main.js      # Frontend utilities
│   └── uploads/            # Uploaded PDF files
├── models/                 # Local LLM models
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎯 Usage

### Register & Login
1. Go to register page
2. Create account with name, email, password, and optional student details
3. Login with email and password

### Upload PDF & Ask Questions
1. Go to **Learning** section
2. Drag & drop or click to upload a PDF
3. Wait for indexing to complete
4. Ask questions about the PDF content

### Chat with AI
1. Go to **AI Chat** section
2. Ask anything (programming, ML, career advice, etc.)
3. Get instant responses from local Qwen2.5 model

### Create Study Plan
1. Go to **Planner** section
2. Enter subjects, exam date, daily study hours
3. AI generates personalized study schedule

### View Profile
1. Go to **Profile** to see statistics
2. View PDFs uploaded, questions asked, study plans created

## ⚙️ Performance Optimizations

- **Cached Model Loading** — LLM and embedder loaded once on startup
- **FAISS Indexing** — In-memory caching of PDF embeddings and indexes
- **Threading** — Asynchronous PDF indexing in background
- **Lazy Loading** — Frontend pages load on-demand
- **Minimal API Calls** — Optimized AJAX requests
- **CSS/JS Compression** — Clean, minimal code

## 🎨 Design

- **Color Scheme** — Modern light theme
  - Primary: #4F46E5 (Indigo)
  - Secondary: #7C3AED (Purple)
  - Accent: #06B6D4 (Cyan)
  - Success: #10B981 (Green)
- **Typography** — Inter & Poppins fonts from Google Fonts
- **Components** — Custom HTML5/CSS3 (no Bootstrap)
- **Animations** — Smooth transitions, fade, slide effects
- **Icons** — Font Awesome 6.4
- **Responsive** — Mobile-friendly sidebar and layout

## 🔒 Security

- Passwords hashed with Werkzeug utilities
- Session management
- Input validation
- SQLite database for local storage
- No external API calls (fully local)

## ❌ Limitations & Notes

- **Model Requirements** — Qwen2.5-0.5B-Instruct must be placed locally; model not provided
- **CUDA Optional** — App runs on CPU or GPU (auto-detected)
- **First Load** — Slow first request (model + embedder loading)
- **Chat/Planner** — Require local model; will error if model missing
- **Development Only** — Flask debug server (use WSGI for production)

## 🚀 Deployment

For production, replace Flask dev server with Gunicorn:

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 "backend.app:create_app()"
```

## 📚 Tech Stack

- **Backend** — Flask, SQLite, PyMuPDF, FAISS, transformers, sentence-transformers
- **Frontend** — HTML5, CSS3, Vanilla JavaScript
- **LLM** — Qwen2.5-0.5B-Instruct (local)
- **Embeddings** — sentence-transformers/all-MiniLM-L6-v2
- **Vector DB** — FAISS (CPU)

## 🎓 Future Scope

- Profile avatar uploads
- Export study plans as PDF
- Dark mode toggle
- Advanced filtering/search
- Collaborative study groups
- Real-time progress tracking
- Multi-language support
- Custom branding options

## 📄 License

MIT License - Feel free to use and modify

---

**Built with ❤️ for students and GenAI enthusiasts**

