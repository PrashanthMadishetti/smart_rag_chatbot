# tests/test_embeddings_edge_cases.py
import os
import pickle
import numpy as np
import pytest
from langchain.schema import Document

from app.embeddings.encoder import Embedder
from app.vectorstore.faiss_store import FaissIndex


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_large_chunk_embedding(embedder: Embedder):
    long_text = ("RAG " * 1000) + "end."
    vec = embedder.encode_texts([long_text])
    assert vec.shape == (1, embedder.dimension)


def test_duplicate_chunks_do_not_crash(tmp_path, embedder: Embedder):
    idx = FaissIndex(dimension=embedder.dimension, metric="cosine", model_name=embedder.model_name)
    d = Document(page_content="duplicate", metadata={"chunk_uid":"D:0:0","doc_id":"D","page":0,"chunk_id":0,"source":"/d.txt","type":"txt"})
    idx.add_documents([d, d], embedder)
    assert idx.count == 2  # de-dup is optional; no crash is required


def test_corrupted_index_file_raises(tmp_path, embedder: Embedder):
    base = tmp_path / "badidx"
    base.mkdir()
    # Create bogus files where FAISS/metadata would live
    (base / "index.faiss").write_bytes(b"not a real faiss index")
    with open(base / "meta.pkl", "wb") as f:
        pickle.dump({"dimension": 384, "model_name": "fake"}, f)
    with pytest.raises(Exception):
        FaissIndex.load_local(str(base))