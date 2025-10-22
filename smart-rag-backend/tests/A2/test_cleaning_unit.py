# tests/test_cleaning_unit.py
from langchain.schema import Document
from app.ingest.preprocess import clean

def test_clean_removes_zero_width_and_controls():
    raw = "Hello\u200b world\u200b!\nLine\u0007 with control."
    d = Document(page_content=raw, metadata={})
    out = clean(d)
    assert "\u200b" not in out.page_content
    assert "\u0007" not in out.page_content
    assert "Hello" in out.page_content and "world" in out.page_content

def test_clean_collapses_spaces_preserves_paragraphs():
    raw = "Para A   with   extra  spaces.\n\n\n  Para B on next line.  "
    d = Document(page_content=raw, metadata={})
    out = clean(d)
    # Collapsed spaces
    print(f"output by clean function is {out}")
    assert "  " not in out.page_content
    # Preserved blank line paragraph break (exact count may differ, but must remain a paragraph gap)
    assert "\n\n" in out.page_content

def test_clean_preserves_bullets_and_newlines():
    raw = "- item one\n-   item   two\n• item three"
    d = Document(page_content=raw, metadata={})
    out = clean(d)
    lines = out.page_content.splitlines()
    assert lines[0].startswith("- ")
    assert lines[1].startswith("- ")
    assert any(l.startswith("• ") for l in lines)