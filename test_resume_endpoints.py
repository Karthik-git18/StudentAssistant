from backend.app import app
import json

client = app.test_client()

with client.session_transaction() as sess:
    sess['user_id'] = 1

# 1. Test ATS Score
data = {
    "resume_data": {
        "name": "Karthik Palepu",
        "title": "Software Engineer",
        "email": "karthik@example.com",
        "education": [{"degree": "B.Tech", "graduation_year": "2026", "college": "IIT"}]
    }
}
r1 = client.post('/api/resume/ats-score', json=data)
print("ATS Score Status:", r1.status_code)
print(r1.get_json())

# 2. Test Save
save_data = {
    "resume_data": data["resume_data"],
    "ats_score": r1.get_json().get("score", 0)
}
r2 = client.post('/api/resume/save', json=save_data)
print("Save Status:", r2.status_code)
print(r2.get_json())

# 3. Test Load
r3 = client.get('/api/resume/load')
print("Load Status:", r3.status_code)
print("Load Success:", r3.get_json().get("success"))
