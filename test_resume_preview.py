from backend.app import app
import json

client = app.test_client()

with client.session_transaction() as sess:
    sess['user_id'] = 1

data = {
    "resume_data": {
        "name": "Karthik Palepu",
        "title": "Software Engineer",
        "email": "karthik@example.com",
        "education": [{"degree": "B.Tech", "graduation_year": "2026", "college": "IIT"}],
        "experience": [{"role": "Intern", "duration": "2025", "company": "Google", "responsibilities": "- Coded APIs\n- Tested features"}]
    }
}
response = client.post('/api/generate-resume', json=data)
print("Status Code:", response.status_code)
print("Response JSON:")
print(response.get_json()['resume_html'][:200])
