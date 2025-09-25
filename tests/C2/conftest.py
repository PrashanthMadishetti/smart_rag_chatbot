# tests/api/conftest.py
import base64
import json
import os
import threading
import http.server
import socketserver
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

# ---- Configure a test JWT secret for the app under test
TEST_JWT_SECRET = "JWT_SECRET"
os.environ.setdefault("JWT_SECRET", TEST_JWT_SECRET)
os.environ.setdefault("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("VECTOR_BACKEND", "faiss")
os.environ.setdefault("INDEX_DIR", "./data/indexes/faiss_test")


# ---- Import the FastAPI app (supports either a factory or a module-level app)
def _get_app():
    try:
        from api.main import create_app  # app factory pattern
        return create_app()
    except Exception:
        from api.main import app  # module-level app
        return app

@pytest.fixture(scope="session")
def client():
    return TestClient(_get_app())


# ---- JWT helpers
def make_jwt(sub="tester", tenant_id="tenantA", expires_in=3600):
    payload = {
        "sub": sub,
        "tenant_id": tenant_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + int(expires_in),
        "iss": "tests",
        "aud": "tests",
    }
    
    return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")

@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {make_jwt()}"}

@pytest.fixture
def expired_headers():
    token = make_jwt(expires_in=-10)
    return {"Authorization": f"Bearer {token}"}


# ---- Local HTTP server fixture for URL ingestion
HTML_SNIPPET = b"""<!doctype html><html>
<head><title>RAG Test Page</title></head>
<body><h1>RAG Systems</h1><p>RAG improves factuality by grounding answers.</p></body></html>"""

class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # silence logs
        pass

@pytest.fixture
def http_server(tmp_path: Path):
    html = tmp_path / "index.html"
    html.write_bytes(HTML_SNIPPET)

    class CwdHandler(_Handler):
        def translate_path(self, path):
            return str(html)

    with socketserver.TCPServer(("127.0.0.1", 0), CwdHandler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}/index.html"
        finally:
            httpd.shutdown()
            thread.join()


# ---- Tiny 2-page PDF (same as A1) for upload tests
_PDF_BASE64 = b"""
JVBERi0xLjQKJcKlwrHDqwoKMSAwIG9iago8PC9UeXBlL1BhZ2VzL0NvdW50IDIvS2lkc1sgMiAw
IFIgMyAwIFJdPj4KZW5kb2JqCjIgMCBvYmoKPDwvVHlwZS9QYWdlL1BhcmVudCAxIDAgUi9NZWRp
YUJveFswIDAgNjEyIDc5Ml0vQ29udGVudHMgNCAwIFI+PgplbmRvYmoKMyAwIG9iago8PC9UeXBl
L1BhZ2UvUGFyZW50IDEgMCBSL01lZGlhQm94WzAgMCA2MTIgNzkyXS9Db250ZW50cyA1IDAgUj4+
CmVuZG9iago0IDAgb2JqCjw8L0xlbmd0aCA2Mwo+PgpzdHJlYW0KQlQKL0YxIDI0IFRmCjEwMCA3
MDAgVGQKKChIZWxsbykgVGoKRVQKZW5kc3RyZWFtCmVuZG9iago1IDAgb2JqCjw8L0xlbmd0aCA2
Nwo+PgpzdHJlYW0KQlQKL0YxIDI0IFRmCjEwMCA2MDAgVGQKKChXb3JsZCkgVGoKRVQKZW5kc3Ry
ZWFtCmVuZG9iagp4cmVmCjAgNgowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwNTYgMDAwMDAg
biAKMDAwMDAwMDEzMiAwMDAwMCBuIAowMDAwMDAwMjQ0IDAwMDAwIG4gCjAwMDAwMDAzMzUgMDAw
MDAgbiAKMDAwMDAwMDQyNiAwMDAwMCBuIAp0cmFpbGVyCjw8L1NpemUgNi9Sb290IDEgMCBSL0lu
Zm8gNiAwIFI+PgpzdGFydHhyZWYKNTQ5CiUlRU9G

"""

# @pytest.fixture
# def sample_pdf_bytes():
#     return base64.b64decode(_PDF_BASE64)
# tests/api/conftest.py
import io
from pypdf import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

@pytest.fixture
def sample_pdf_bytes():
    buf = io.BytesIO()
    # Use ReportLab to draw text into a PDF
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 750, "Hello RAG! This is a test PDF.")
    c.save()

    buf.seek(0)
    return buf.getvalue()