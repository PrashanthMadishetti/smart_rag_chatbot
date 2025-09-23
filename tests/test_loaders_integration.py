# tests/test_loaders_integration.py
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.ingest.loaders import load_pdfs, load_txts, load_web
def test_ingest_mixed_sources_creates_documents(sample_pdf: Path, tmp_path: Path, http_server: str):
    txt = tmp_path / "policy.txt"
    txt.write_text("This is a short policy document about security and logging.")
    url = f"{http_server}/index.html"

    docs = []
    docs += load_pdfs([str(sample_pdf)])
    docs += load_txts([str(txt)])
    docs += load_web([url])

    assert len(docs) >= 4  # 2 from PDF + 1 TXT + 1 WEB
    # sanity: all have baseline metadata
    for d in docs:
        assert d.page_content.strip()
        md = d.metadata
        assert "source" in md and "type" in md and "doc_id" in md and "timestamp" in md