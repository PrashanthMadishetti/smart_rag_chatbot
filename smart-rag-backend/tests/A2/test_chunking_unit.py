# tests/test_chunking_unit.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from langchain.schema import Document
from app.ingest.preprocess import chunk

def _make_doc(chars: int, meta: dict | None = None) -> Document:
    text = ("A" * 200 + "\n\n") * (chars // 202)  # produce paragraphs
    return Document(page_content=text, metadata=meta or {})

def test_chunking_basic_boundaries():
    d = _make_doc(1200, {"doc_id": "d1"})
    chunks = chunk([d], chunk_size=400, chunk_overlap=50)
    # Expect at least 3 chunks for ~1200 chars with 400/50 settings
    assert len(chunks) >= 3
    # No chunk should exceed the requested size by more than a small margin
    assert all(len(c.page_content) <= 450 for c in chunks)

def test_chunking_overlap_present():
    d = _make_doc(1400, {"doc_id": "d1"})
    chunks = chunk([d], chunk_size=400, chunk_overlap=100)
    # Compute overlap between consecutive chunks
    overlaps = []
    for i in range(len(chunks) - 1):
        a, b = chunks[i].page_content, chunks[i+1].page_content
        # measure suffix/prefix overlap length (approx)
        max_overlap = min(len(a), len(b), 150)
        found = False
        for k in range(100, 0, -1):
            if a[-k:] == b[:k]:
                overlaps.append(k); found = True; break
        assert found, "expected some overlap between consecutive chunks"
    # Average overlap should be around requested value (±40 chars tolerance due to separator boundaries)
    if overlaps:
        avg = sum(overlaps) / len(overlaps)
        assert 60 <= avg <= 140

def test_chunking_preserves_metadata_and_assigns_ids():
    d = _make_doc(900, {"doc_id": "abc123", "type": "txt", "source": "/tmp/foo.txt"})
    chunks = chunk([d], chunk_size=400, chunk_overlap=100)
    for i, c in enumerate(chunks):
        md = c.metadata
        # Inherit important fields
        assert md.get("doc_id") == "abc123"
        assert md.get("type") == "txt"
        assert md.get("source") == "/tmp/foo.txt"
        # New fields for chunks
        assert isinstance(md.get("chunk_id"), int)
        assert md["chunk_id"] == i
        assert "chunk_uid" in md and isinstance(md["chunk_uid"], str)
        assert "chunk_checksum" in md and len(md["chunk_checksum"]) == 64

def test_chunking_pdf_resets_chunk_id_per_page():
    docs = [
        Document(page_content="Page1 text " * 100, metadata={"doc_id": "D", "type": "pdf", "page": 1, "source": "/x.pdf"}),
        Document(page_content="Page2 text " * 100, metadata={"doc_id": "D", "type": "pdf", "page": 2, "source": "/x.pdf"}),
    ]
    chunks = chunk(docs, chunk_size=300, chunk_overlap=60)
    # Gather chunk_ids per page
    by_page = {}
    for c in chunks:
        by_page.setdefault(c.metadata["page"], []).append(c)
    # chunk_id should start at 0 on each page
    for page, items in by_page.items():
        ids = [it.metadata["chunk_id"] for it in items]
        assert ids[0] == 0
        assert ids == list(range(len(items))), f"chunk_id must be sequential per page, failed on page {page}"
        # chunk_uid must include doc_id and the correct page number
        for it in items:
            uid = it.metadata["chunk_uid"]
            assert uid.startswith("D:")
            assert f":{page}:" in uid