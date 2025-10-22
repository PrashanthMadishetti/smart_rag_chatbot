# tests/B1/test_retriever_mmr.py
import math
import numpy as np
import pytest
from langchain_core.documents import Document

# ---------- Fakes (super fast & deterministic) ----------

class FakeEmbedder:
    """Deterministic toy embedder mapping strings to 3D unit vectors."""
    dimension = 3
    model_name = "fake-embedder-3d"

    def embed(self, texts):
        # Map some keywords to axes to control similarity in tests
        vecs = []
        for t in texts:
            t = (t or "").lower()
            v = np.zeros(3, dtype=float)
            if "alpha" in t:
                v += np.array([1.0, 0.0, 0.0])
            if "beta" in t:
                v += np.array([0.0, 1.0, 0.0])
            if "gamma" in t:
                v += np.array([0.0, 0.0, 1.0])
            if np.allclose(v, 0):
                v = np.array([1.0, 1.0, 1.0])  # neutral
            v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.vstack(vecs)

    def embed_query(self, text):
        return self.embed([text])[0]

    # 🔹 add this so the retriever's MMR path can call embed_documents
    def embed_documents(self, texts):
        return self.embed(texts)


class FakeIndex:
    """Tiny in-memory vector index with cosine similarity search (prod-like signature)."""
    def __init__(self):
        self._docs = []
        self._embs = None
        self._model_name = None
        self.metric = "cosine"

    def add_documents(self, docs, embedder):
        # prefer embed_documents, fallback to embed
        if hasattr(embedder, "embed_documents"):
            embs = embedder.embed_documents([d.page_content for d in docs]).astype(np.float32)
        else:
            embs = embedder.embed([d.page_content for d in docs]).astype(np.float32)

        self._model_name = getattr(embedder, "model_name", None)
        if self._embs is None:
            self._embs = embs
            self._docs = list(docs)
        else:
            self._embs = np.vstack([self._embs, embs])
            self._docs.extend(list(docs))

    # 🔹 match production: take (query: str, k: int, embedder) and return List[Document]
    def search(self, query: str, k: int, embedder):
        if self._embs is None or len(self._docs) == 0:
            return []
        # get query vector using same embedder
        if hasattr(embedder, "embed_query"):
            qv = embedder.embed_query(query)
        else:
            qv = embedder.embed([query])[0]
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sims = self._embs @ qv
        idx = np.argsort(-sims)[:k]
        return [self._docs[i] for i in idx]


# ---------- Subject Under Test (SUT) import contract ----------

# The tests assume this interface exists in your codebase.
# Implement it in app/retrieval/retriever.py to make tests pass.
from app.retrieval.retriever import Retriever


# ---------- Fixtures ----------

@pytest.fixture
def embedder():
    return FakeEmbedder()

@pytest.fixture
def index():
    return FakeIndex()

@pytest.fixture
def docs_alpha_beta_gamma():
    # Two near-duplicate chunks from one source, plus a different source
    return [
        Document(page_content="alpha alpha alpha", metadata={"source": "docA.txt", "chunk_id": "A-1"}),
        Document(page_content="alpha very similar", metadata={"source": "docA.txt", "chunk_id": "A-2"}),
        Document(page_content="beta content", metadata={"source": "docB.txt", "chunk_id": "B-1"}),
        Document(page_content="gamma content", metadata={"source": "docC.txt", "chunk_id": "C-1"}),
    ]


# ---------- Tests ----------

def test_topk_basic_order_by_similarity(index, embedder, docs_alpha_beta_gamma):
    index.add_documents(docs_alpha_beta_gamma, embedder)
    r = Retriever(index, embedder)

    # Query is strongly along "alpha" axis
    results = r.search("alpha question", k=3, use_mmr=False, mmr_lambda=0.7)
    assert len(results) == 3

    # Expect both alpha chunks first (same source, different chunk_id), then beta or gamma
    first_sources = [d.metadata["chunk_id"] for d in results]
    assert first_sources[0].startswith("A-")
    assert first_sources[1].startswith("A-")
    # third can be B-1 or C-1 depending on tie-breaking
    assert first_sources[2] in {"B-1", "C-1"}


def test_mmr_promotes_diversity(index, embedder, docs_alpha_beta_gamma):
    index.add_documents(docs_alpha_beta_gamma, embedder)
    r = Retriever(index, embedder)

    # With MMR on, we want 1 alpha + 1 non-alpha (+ one more diverse if k=3)
    results = r.search("alpha topic", k=3, use_mmr=True, mmr_lambda=0.5)

    assert len(results) == 3

    # We should NOT get A-1 and A-2 back-to-back in top-2 if diversity works.
    top2_chunks = {results[0].metadata["chunk_id"], results[1].metadata["chunk_id"]}
    print(f"the top_chunks are {top2_chunks}")
    assert not ({"A-1", "A-2"} <= top2_chunks), "MMR failed to diversify top results"

    # Ensure at least one from a different source among top-2
    top2_sources = {results[0].metadata["source"], results[1].metadata["source"]}
    assert len(top2_sources) >= 2


def test_cap_k_and_empty_index(embedder):
    index = FakeIndex()
    r = Retriever(index, embedder)

    # Empty index → no results, no crash
    empty = r.search("anything", k=5, use_mmr=True, mmr_lambda=0.7)
    assert empty == []

    # Add only 2 docs, ask k=50 → cap & not exceed available
    docs = [
        Document(page_content="alpha", metadata={"source": "S1", "chunk_id": "1"}),
        Document(page_content="beta", metadata={"source": "S2", "chunk_id": "2"}),
    ]
    index.add_documents(docs, embedder)
    got = r.search("alpha", k=50, use_mmr=False)
    assert 1 <= len(got) <= 2