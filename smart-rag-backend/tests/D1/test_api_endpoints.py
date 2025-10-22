# tests/D1/test_api_endpoints.py
import io
import json
import pytest
from langchain_core.documents import Document


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_requires_auth(client):
    r = client.get("/metrics")
    assert r.status_code == 401
    assert r.json()["detail"] in {"Not authenticated", "Invalid token", "Token expired"}


def test_metrics_with_auth(client, auth_headers):
    r = client.get("/metrics", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    # Ensure expected keys exist
    for key in ["requests_total", "ingest_docs_total", "ingest_chunks_total", "queries_served_total"]:
        assert key in body
        assert isinstance(body[key], int)


def test_ingest_400_when_no_input(client, auth_headers):
    r = client.post("/ingest", headers=auth_headers)
    assert r.status_code == 400
    assert "Provide a file" in r.json()["detail"] or "Provide a file upload or a JSON body with 'urls'" in r.json()["detail"]


def test_ingest_txt_upload(client, auth_headers, monkeypatch):
    # Monkeypatch the loader to avoid real disk parsing
    def fake_load_txts(paths):
        return [Document(page_content="hello text", metadata={"source": "fake.txt", "chunk_id": "A-1"})]

    monkeypatch.setattr("api.main.load_txts", fake_load_txts, raising=True)

    content = b"RAG improves factuality by grounding answers."
    files = {"file": ("policy.txt", io.BytesIO(content), "text/plain")}
    r = client.post("/ingest", headers=auth_headers, files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["ingested"] >= 1
    assert data["failures"] == []
    assert "durations_ms" in data and isinstance(data["durations_ms"], dict)


def test_ingest_pdf_upload(client, auth_headers, monkeypatch):
    # Monkeypatch PDF loader similarly
    def fake_load_pdfs(paths):
        return [Document(page_content="pdf page 1", metadata={"source": "fake.pdf", "chunk_id": "P-1"})]

    monkeypatch.setattr("api.main.load_pdfs", fake_load_pdfs, raising=True)

    files = {"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
    r = client.post("/ingest", headers=auth_headers, files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["ingested"] >= 1
    assert data["failures"] == []


def test_ingest_urls_json(client, auth_headers, monkeypatch):
    # Monkeypatch web loader to avoid network
    def fake_load_web(urls):
        return [
            Document(page_content="from web", metadata={"source": urls[0], "chunk_id": "W-1"})
        ]

    monkeypatch.setattr("api.main.load_web", fake_load_web, raising=True)

    payload = {"urls": ["https://example.com/test"]}
    r = client.post("/ingest", headers=auth_headers, json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["ingested"] >= 1
    assert data["failures"] == []


def test_query_requires_auth(client):
    r = client.post("/query", json={"session_id": "s1", "question": "hi"})
    assert r.status_code == 401


def test_query_happy_path(client, auth_headers, monkeypatch):
    # Monkeypatch the vector index search called by retriever/top-k path.
    # Our app uses a module-level singleton _INDEX; we patch `api.main._INDEX.search`.
    def fake_search(query, k, embedder=None):
        # Return plain Documents (prod signature returns List[Document])
        return [
            Document(page_content="alpha info", metadata={"source": "docA.txt", "chunk_id": "A-1"}),
            Document(page_content="beta info", metadata={"source": "docB.txt", "chunk_id": "B-1"}),
        ][:k]

    # If your app uses a Retriever that calls index.search with (query, k, embedder),
    # this signature still works because extra arg is ignored above.
    monkeypatch.setattr("api.main._INDEX.search", fake_search, raising=True)

    body = {
        "session_id": "abc123",
        "question": "What is alpha?",
        "k": 2,
        "provider": "stub",
        "use_mmr": False
    }
    r = client.post("/query", headers=auth_headers, json=body)
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert data["provider"] == "stub"
    # sources should be pulled from doc metadata
    assert data["sources"] == ["docA.txt", "docB.txt"]


def test_openapi_docs_available(client):
    # Swagger UI loads /openapi.json
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    # basic sanity checks
    assert "paths" in spec and "/ingest" in spec["paths"] and "/query" in spec["paths"]