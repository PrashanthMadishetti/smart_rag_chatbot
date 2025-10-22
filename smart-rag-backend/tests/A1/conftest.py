# tests/conftest.py
import base64
import threading
import http.server
import socketserver
from pathlib import Path
from datetime import datetime, timezone

import pytest

# 2-page minimal PDF (hello on page 1, world on page 2)
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

HTML_SNIPPET = """<!doctype html>
<html><head><title>Test Page</title></head>
<body><h1>Local Test</h1><p>This is a local page served for loader tests.</p></body></html>
""".encode("utf-8")


@pytest.fixture
def sample_pdf() -> Path:
    return Path("/Users/jeshwanthleo/Desktop/Agentic AI Projects/Smart_RAG_Chatbot/tests/Prashanth_Madishetti.pdf")
    


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # silence test logs
        pass


@pytest.fixture
def http_server(tmp_path: Path):
    """Spin up a disposable local HTTP server serving HTML_SNIPPET.
    Useful to test the "web loader" without hitting the internet."""
    # create index.html in tmp dir
    html = tmp_path / "index.html"
    html.write_bytes(HTML_SNIPPET)

    # serve tmp_path
    class CwdHandler(_Handler):
        def translate_path(self, path):
            # force all requests to tmp_path/index.html
            return str(html)

    with socketserver.TCPServer(("127.0.0.1", 0), CwdHandler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            httpd.shutdown()
            thread.join()


@pytest.fixture
def assert_iso_timestamp():
    def _assert_iso(value: str):
        # must parse as ISO-8601
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        assert dt.tzinfo is not None
        # sane bounds (± 1 day from now)
        now = datetime.now(timezone.utc)
        delta = abs((now - dt).total_seconds())
        assert delta < 60 * 60 * 24
    return _assert_iso