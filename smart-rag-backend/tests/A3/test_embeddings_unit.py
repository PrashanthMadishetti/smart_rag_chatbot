# tests/test_embeddings_unit.py
import re
import numpy as np
import pytest
from langchain.schema import Document

from app.embeddings.encoder import Embedder
from app.vectorstore.faiss_store import FaissIndex


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    # Default model for Sprint 1
    return Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_embedding_dimensions_match_model(embedder: Embedder):
    vecs = embedder.encode_texts(["hello world"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.ndim == 2 and vecs.shape[0] == 1
    # MiniLM-L6-v2 is 384-dim
    assert vecs.shape[1] == 384


def test_embedder_is_deterministic(embedder: Embedder):
    v1 = embedder.encode_texts(["RAG is great"])[0]
    v2 = embedder.encode_texts(["RAG is great"])[0]
    # cosine distance close to 0
    cos = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert cos > 0.999, "Same text should embed to (nearly) identical vectors"


def test_faiss_index_add_and_count(embedder: Embedder):
    idx = FaissIndex(dimension=embedder.dimension, metric="cosine", model_name=embedder.model_name)
    docs = [
        Document(page_content="retrieval augmented generation", metadata={"chunk_uid":"D:0:0","doc_id":"D","page":0,"chunk_id":0,"source":"s.txt","type":"txt"}),
        Document(page_content="vector databases like FAISS and Pinecone", metadata={"chunk_uid":"D:0:1","doc_id":"D","page":0,"chunk_id":1,"source":"s.txt","type":"txt"}),
    ]
    idx.add_documents(docs, embedder)
    assert idx.count == 2


def test_faiss_save_and_reload(tmp_path, embedder: Embedder):
    base = tmp_path / "faiss_idx"
    idx = FaissIndex(dimension=embedder.dimension, metric="cosine", model_name=embedder.model_name)
    docs = [
        Document(page_content="Smart RAG uses embeddings", metadata={"chunk_uid":"D:0:0","doc_id":"D","page":0,"chunk_id":0,"source":"/p.txt","type":"txt"}),
        Document(page_content="FAISS provides fast kNN search", metadata={"chunk_uid":"D:0:1","doc_id":"D","page":0,"chunk_id":1,"source":"/p.txt","type":"txt"}),
    ]
    idx.add_documents(docs, embedder)
    idx.save_local(str(base))

    # Reload
    loaded = FaissIndex.load_local(str(base))
    assert loaded.count == 2
    assert loaded.model_name == embedder.model_name
    assert loaded.dimension == embedder.dimension


def test_id_format_and_metadata_persistence(tmp_path, embedder: Embedder):
    idx = FaissIndex(dimension=embedder.dimension, metric="cosine", model_name=embedder.model_name)
    d = Document(page_content="RAG systems ground answers in retrieved context.",
                 metadata={"chunk_uid":"X:2:7","doc_id":"X","page":2,"chunk_id":7,"source":"/handbook.pdf","type":"pdf"})
    idx.add_documents([d], embedder)
    res = idx.search("retrieved context", k=1, embedder=embedder)
    assert len(res) == 1
    hit = res[0]
    md = hit.metadata
    # chunk_uid format: {doc_id}:{page}:{chunk_id}
    assert re.fullmatch(r"[A-Za-z0-9]+:\d+:\d+", md["chunk_uid"])
    # critical metadata survives
    assert md["doc_id"] == "X"
    assert md["page"] == 2
    assert md["chunk_id"] == 7
    assert md["source"].endswith("handbook.pdf")