# tests/test_loaders_unit.py
from pathlib import Path
import re
import time
import pytest
from langchain.schema import Document


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.ingest.loaders import load_pdfs, load_txts, load_web

def _require_meta(doc: Document, keys):
    for k in keys:
        assert k in doc.metadata, f"missing metadata key: {k}"


def test_pdf_loader_pages_have_page_field(sample_pdf: Path):
    docs = load_pdfs([str(sample_pdf)])

    assert len(docs) >= 2, "expected 2 pages from the test PDF"
    for d in docs:
        assert isinstance(d, Document)
        assert d.page_content.strip(), "empty page content"
        _require_meta(d, ["source", "type", "doc_id", "timestamp", "page", "page_id"])
        assert d.metadata["type"] == "pdf"
        assert isinstance(d.metadata["page"], int) and d.metadata["page"] >= 1
        assert re.fullmatch(r"[0-9a-f]{64}", d.metadata["doc_id"])
        assert re.fullmatch(r"[0-9a-f]{64}", d.metadata["page_id"])


def test_text_loader_sets_source_and_type(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_text("Hello text file\nThis is a line.")
    docs = load_txts([str(p)])
    assert len(docs) == 1
    d = docs[0]
    assert d.page_content.startswith("Hello")
    _require_meta(d, ["source", "type", "doc_id", "timestamp", "file_mtime"])
    assert d.metadata["type"] == "txt"
    assert str(d.metadata["source"]).endswith("note.txt")


def test_web_loader_sets_url_and_type(http_server: str):
    url = f"{http_server}/index.html"
    docs = load_web([url])
    assert len(docs) >= 1
    d = docs[0]
    _require_meta(d, ["source", "type", "url", "resolved_url", "doc_id", "timestamp"])
    assert d.metadata["type"] == "web"
    assert d.metadata["url"] == url
    # Ensure HTML was converted to readable text (no raw tags)
    assert "<html" not in d.page_content.lower()
    assert "Local Test" in d.page_content


def test_doc_id_content_first_same_path_same_content(tmp_path: Path):
    p = tmp_path / "stable.txt"
    p.write_text("Stable content")
    id_a = load_txts([str(p)])[0].metadata["doc_id"]
    id_b = load_txts([str(p)])[0].metadata["doc_id"]
    assert id_a == id_b, "doc_id must be stable for unchanged content"


def test_doc_id_content_first_different_paths_same_content(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "nested" / "b.txt"
    b.parent.mkdir(parents=True, exist_ok=True)
    text = "Identical content across paths"
    a.write_text(text)
    b.write_text(text)
    id_a = load_txts([str(a)])[0].metadata["doc_id"]
    id_b = load_txts([str(b)])[0].metadata["doc_id"]
    assert id_a == id_b, "doc_id must be path-independent (content-first)"


def test_doc_id_ignores_mtime_changes(tmp_path: Path):
    p = tmp_path / "mtime.txt"
    p.write_text("Same content, different mtime")
    id_before = load_txts([str(p)])[0].metadata["doc_id"]
    # change mtime without changing content
    time.sleep(0.01)
    p.touch()
    id_after = load_txts([str(p)])[0].metadata["doc_id"]
    assert id_before == id_after, "doc_id must not depend on mtime"


def test_doc_id_changes_when_content_changes(tmp_path: Path):
    p = tmp_path / "changing.txt"
    p.write_text("Version A")
    id_a = load_txts([str(p)])[0].metadata["doc_id"]
    p.write_text("Version B")
    id_b = load_txts([str(p)])[0].metadata["doc_id"]
    assert id_a != id_b, "doc_id must change when content changes"


def test_loader_raises_on_missing_path():
    with pytest.raises(Exception):
        load_txts(["/does/not/exist.txt"])


# # tests/test_loaders_unit.py
# from pathlib import Path
# import hashlib
# import re
# import pytest

# from langchain.schema import Document

# # These imports assume your implementation lives here (A1 target)
# from app.ingest.loaders import load_pdfs, load_texts, load_web


# def _require_meta(doc: Document, keys):
#     for k in keys:
#         assert k in doc.metadata, f"missing metadata key: {k}"


# def test_pdf_loader_pages_have_page_field(sample_pdf: Path):
#     docs = load_pdfs([str(sample_pdf)])
#     assert len(docs) >= 2, "expected 2 pages from the test PDF"
#     for d in docs:
#         assert isinstance(d, Document)
#         assert d.page_content.strip(), "empty page content"
#         _require_meta(d, ["source", "type", "doc_id", "timestamp"])
#         assert d.metadata["type"] == "pdf"
#         assert "page" in d.metadata and isinstance(d.metadata["page"], int) and d.metadata["page"] >= 1
#         # basic doc_id shape (hex)
#         assert re.fullmatch(r"[0-9a-f]{32,64}", d.metadata["doc_id"])


# def test_text_loader_sets_source_and_type(tmp_path: Path):
#     p = tmp_path / "note.txt"
#     p.write_text("Hello text file\nThis is a line.")
#     docs = load_texts([str(p)])
#     assert len(docs) == 1
#     d = docs[0]
#     assert d.page_content.startswith("Hello")
#     _require_meta(d, ["source", "type", "doc_id", "timestamp"])
#     assert d.metadata["type"] == "txt"
#     assert d.metadata["source"].endswith("note.txt")


# def test_web_loader_sets_url_and_type(http_server: str):
#     url = f"{http_server}/index.html"
#     docs = load_web([url])
#     assert len(docs) >= 1
#     d = docs[0]
#     _require_meta(d, ["source", "type", "url", "doc_id", "timestamp"])
#     assert d.metadata["type"] == "web"
#     assert d.metadata["url"] == url
#     # Ensure HTML was converted to readable text (no raw tags)
#     assert "<html" not in d.page_content.lower()
#     assert "Local Test" in d.page_content


# def test_doc_id_stability_for_same_content(tmp_path: Path):
#     p = tmp_path / "stable.txt"
#     p.write_text("Stable content")
#     docs1 = load_texts([str(p)])
#     docs2 = load_texts([str(p)])
#     id1 = docs1[0].metadata["doc_id"]
#     id2 = docs2[0].metadata["doc_id"]
#     assert id1 == id2, "doc_id must be stable for same file when unchanged"


# def test_doc_id_changes_when_content_changes(tmp_path: Path):
#     p = tmp_path / "changing.txt"
#     p.write_text("Version A")
#     id_a = load_texts([str(p)])[0].metadata["doc_id"]
#     p.write_text("Version B")  # mutate content
#     id_b = load_texts([str(p)])[0].metadata["doc_id"]
#     assert id_a != id_b, "doc_id should change when file content changes"


# def test_loader_raises_on_missing_path():
#     with pytest.raises(Exception):
#         load_texts(["/does/not/exist.txt"])