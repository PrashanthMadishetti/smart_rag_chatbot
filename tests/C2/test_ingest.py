# tests/api/test_ingest.py
import io

def test_ingest_txt_upload(client, auth_headers, tmp_path):
    # Upload a small TXT file
    content = b"RAG improves factuality by grounding answers."
    files = {"file": ("policy.txt", io.BytesIO(content), "text/plain")}
    r = client.post("/ingest", headers=auth_headers, files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["ingested"] >= 1
    assert data["failures"] == []

def test_ingest_pdf_upload(client, auth_headers, sample_pdf_bytes):
    files = {"file": ("book.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    r = client.post("/ingest", headers=auth_headers, files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["ingested"] >= 1
    assert data["durations_ms"]["total"] >= 0

def test_ingest_urls(client, auth_headers, http_server):
    payload = {"urls": [http_server]}
    r = client.post("/ingest", headers=auth_headers, json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["ingested"] >= 1
    assert data["failures"] == []