# tests/test_chunking_integration.py
from pathlib import Path
from langchain.schema import Document
import os
import sys

# Add project root (two levels up from tests/A2) to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from app.ingest.preprocess import clean,chunk

def test_clean_then_chunk_pipeline_counts(tmp_path: Path):
    raw = (
        "Intro paragraph.\n\n"
        + ("Body sentence repeated. " * 60) + "\n\n"
        + ("Another section. " * 50)
    )
    d = Document(page_content=raw, metadata={"doc_id": "Z", "type": "txt", "source": str(tmp_path / "doc.txt")})
    d2 = clean(d)
    assert d2.page_content.count("\n\n") >= 2, "paragraphs should be preserved after cleaning"
    parts = chunk([d2], chunk_size=800, chunk_overlap=120)
    assert len(parts) >= 2
    # Ensure metadata integrity for all parts
    for i, c in enumerate(parts):
        md = c.metadata
        assert md["doc_id"] == "Z"
        assert md["type"] == "txt"
        assert md["source"].endswith("doc.txt")
        assert md["chunk_id"] == i
        assert "chunk_uid" in md and "chunk_checksum" in md

def test_small_documents_not_overchunked():
    d = Document(page_content="Short note.\nSecond line.", metadata={"doc_id": "S", "type": "txt", "source": "/s.txt"})
    parts = chunk([d], chunk_size=800, chunk_overlap=120)
    assert len(parts) == 1, "tiny documents should remain a single chunk"