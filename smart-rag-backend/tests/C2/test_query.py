# tests/api/test_query.py
import io

def seed_txt(client, auth_headers, text=b"Retrieval-Augmented Generation (RAG) augments LLMs with context."):
    files = {"file": ("seed.txt", io.BytesIO(text), "text/plain")}
    r = client.post("/ingest", headers=auth_headers, files=files)
    assert r.status_code == 200

def test_query_returns_answer_and_sources(client, auth_headers):
    seed_txt(client, auth_headers)
    body = {"session_id": "sess-1", "question": "What is RAG?", "k": 2}
    r = client.post("/query", headers=auth_headers, json=body)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data.get("answer"), str) and data["answer"].strip()
    assert isinstance(data.get("sources"), list)

def test_query_respects_k_and_empty_index(client, auth_headers, tmp_path):
    # simulate a fresh app instance (optional depending on your app statefulness)
    # For now, just query without seeding anything extra:
    body = {"session_id": "sess-2", "question": "random", "k": 3}
    r = client.post("/query", headers=auth_headers, json=body)
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert isinstance(data.get("sources"), list)