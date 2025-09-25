# tests/api/test_auth.py
def test_ingest_requires_jwt(client):
    r = client.post("/ingest", json={"urls": []})
    assert r.status_code in (401, 403)

def test_query_requires_jwt(client):
    r = client.post("/query", json={"session_id": "s1", "question": "Hi"})
    assert r.status_code in (401, 403)

def test_expired_token_rejected(client, expired_headers):
    r = client.post("/ingest", headers=expired_headers, json={"urls": []})
    assert r.status_code in (401, 403)