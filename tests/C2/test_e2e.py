# tests/api/test_e2e.py
import io

def test_e2e_ingest_then_query(client, auth_headers):
    files = {"file": ("kb.txt", io.BytesIO(b"RAG systems improve factuality."), "text/plain")}
    r1 = client.post("/ingest", headers=auth_headers, files=files)
    assert r1.status_code == 200

    r2 = client.post("/query", headers=auth_headers, json={"session_id": "s-e2e", "question": "What improves factuality?"})
    assert r2.status_code == 200
    data = r2.json()
    assert "answer" in data and data["answer"].strip()
    assert isinstance(data.get("sources"), list)