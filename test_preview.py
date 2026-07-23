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
    }
}
r = client.post('/api/resume/preview', json=data)
print("Preview Status:", r.status_code)
if r.status_code != 200:
    print(r.text)
