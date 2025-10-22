# tests/test_embeddings_integration.py
import re
import numpy as np
import pytest
from langchain.schema import Document

from app.embeddings.encoder import Embedder
from app.vectorstore.faiss_store import FaissIndex


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def tiny_index(embedder: Embedder) -> FaissIndex:
    idx = FaissIndex(dimension=embedder.dimension, metric="cosine", model_name=embedder.model_name)
    docs = [
        Document(page_content="Retrieval-Augmented Generation (RAG) augments LLMs with context.",
                 metadata={"chunk_uid":"D:0:0","doc_id":"D","page":0,"chunk_id":0,"source":"/kb/rag.txt","type":"txt"}),
        Document(page_content="FAISS is a vector index for efficient similarity search.",
                 metadata={"chunk_uid":"D:0:1","doc_id":"D","page":0,"chunk_id":1,"source":"/kb/faiss.txt","type":"txt"}),
        Document(page_content="Chunking splits documents into overlapping windows for retrieval.",
                 metadata={"chunk_uid":"D:0:2","doc_id":"D","page":0,"chunk_id":2,"source":"/kb/chunking.txt","type":"txt"}),
    ]
    idx.add_documents(docs, embedder)
    return idx


def test_retrieval_returns_topk_and_sources(tiny_index: FaissIndex, embedder: Embedder):
    res = tiny_index.search("What is RAG?", k=2, embedder=embedder)
    assert len(res) == 2
    texts = [r.page_content.lower() for r in res]
    assert any("retrieval-augmented" in t or "rag" in t for t in texts)
    # ensure sources are present
    for r in res:
        assert "source" in r.metadata and r.metadata["source"].startswith("/kb/")


def test_query_model_mismatch_detection(tmp_path, embedder: Embedder, monkeypatch):
    """
    The index should carry the ingestion model name and warn/error on mismatched query model.
    For this test, we simulate a different model by monkeypatching Embedder.model_name.
    """
    idx = FaissIndex(dimension=embedder.dimension, metric="cosine", model_name=embedder.model_name)
    doc = Document(page_content="FAISS kNN over dense vectors", metadata={"chunk_uid":"D:0:0","doc_id":"D","page":0,"chunk_id":0,"source":"/kb/faiss.txt","type":"txt"})
    idx.add_documents([doc], embedder)

    class FakeEmbedder:
        model_name = "different-model"
        dimension = embedder.dimension
        def encode_texts(self, texts):  # reuse the real embedder to keep dims
            return embedder.encode_texts(texts)

    # Expect the index to detect mismatch and either raise or include a flag.
    with pytest.raises(Exception):
        _ = idx.search("dense vectors", k=1, embedder=FakeEmbedder())


def test_empty_index_returns_no_hits(embedder: Embedder):
    empty = FaissIndex(dimension=embedder.dimension, metric="cosine", model_name=embedder.model_name)
    res = empty.search("anything", k=3, embedder=embedder)
    assert res == [], "Empty index must return no hits gracefully"


def test_persistence_roundtrip(tmp_path, tiny_index: FaissIndex):
    path = tmp_path / "roundtrip"
    tiny_index.save_local(str(path))
    reloaded = FaissIndex.load_local(str(path))
    # Re-run the same query and expect consistent answers
    embedder = Embedder(model_name=reloaded.model_name)
    res = reloaded.search("vector index similarity", k=1, embedder=embedder)
    assert len(res) == 1
    assert "faiss" in res[0].page_content.lower()