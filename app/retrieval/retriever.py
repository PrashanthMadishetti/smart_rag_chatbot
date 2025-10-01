# app/retrieval/retriever.py
from __future__ import annotations
from typing import List, Sequence, Tuple, Optional
import math
import numpy as np

from langchain_core.documents import Document
from app.embeddings.encoder import Embedder
from app.vectorstore.faiss_store import FaissIndex


# def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
#     # safe cosine for short/deterministic vectors
#     num = 0.0
#     da = 0.0
#     db = 0.0
#     for x, y in zip(a, b):
#         num += float(x) * float(y)
#         da += float(x) * float(x)
#         db += float(y) * float(y)
#     if da == 0.0 or db == 0.0:
#         return 0.0
#     return num / math.sqrt(da * db)
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    num = np.dot(a, b)
    da = np.dot(a, a)
    db = np.dot(b, b)
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(num / math.sqrt(da * db))

# def mmr(
#     query_vec: Sequence[float],
#     cand_vecs: List[Sequence[float]],
#     lambda_mult: float,
#     k: int,
# ) -> List[int]:
#     """
#     Return indices of selected candidates using Maximal Marginal Relevance.

#     lambda_mult in [0,1]:
#       - 1.0 -> pure relevance (cosine to query)
#       - 0.0 -> pure diversity (dissimilarity to selected)
#     """
#     # try:
#     #     n=len(cand_vecs)
#     # except TypeError:
#     #     n =0
#     # if n == 0 or k<=0:
#     #     return []
#     if  not len(cand_vecs)==0:
#         return []
#     # Precompute relevance scores once
#     rel = [ _cosine(query_vec, v) for v in cand_vecs ]

#     selected: List[int] = []
#     remaining = set(range(len(cand_vecs)))

#     # Pick the most relevant first
#     first = max(remaining, key=lambda i: rel[i])
#     selected.append(first)
#     remaining.remove(first)

#     while remaining and len(selected) < k:
#         best_i = None
#         best_score = -1e9
#         for i in remaining:
#             # max similarity to already selected
#             if selected:
#                 max_sim_to_selected = max(_cosine(cand_vecs[i], cand_vecs[j]) for j in selected)
#             else:
#                 max_sim_to_selected = 0.0
#             score = lambda_mult * rel[i] - (1.0 - lambda_mult) * max_sim_to_selected
#             if score > best_score:
#                 best_score = score
#                 best_i = i
#         selected.append(best_i)  # type: ignore[arg-type]
#         remaining.remove(best_i)  # type: ignore[arg-type]

#     return selected

def mmr(
    query_vec: Sequence[float],
    cand_vecs: List[Sequence[float]],
    lambda_mult: float,
    k: int,
) -> List[int]:
    """
    Select indices using Maximal Marginal Relevance (MMR).

    lambda_mult in [0,1]:
      - 1.0 -> pure relevance
      - 0.0 -> pure diversity
    """
    # robust empty/size guard
    try:
        n = len(cand_vecs)
    except TypeError:
        n = 0
    if n == 0 or k <= 0:
        return []

    # 1) precompute relevance to query
    rel = [_cosine(query_vec, v) for v in cand_vecs]

    selected: List[int] = []
    remaining = set(range(n))

    # 2) pick the most relevant first
    first = max(remaining, key=lambda i: rel[i])
    selected.append(first)
    remaining.remove(first)

    # 3) greedily add diverse-yet-relevant items
    while remaining and len(selected) < k:
        best_i = None
        best_score = -1e9
        best_max_sim = float("inf")  # for tie-break: prefer smaller similarity to selected

        for i in remaining:
            max_sim_to_selected = max(
                _cosine(cand_vecs[i], cand_vecs[j]) for j in selected
            ) if selected else 0.0

            score = lambda_mult * rel[i] - (1.0 - lambda_mult) * max_sim_to_selected

            # primary: higher score wins
            if score > best_score + 1e-12:
                best_score = score
                best_i = i
                best_max_sim = max_sim_to_selected
            # tie-break: same score → prefer lower similarity to selected (more diverse)
            elif abs(score - best_score) <= 1e-12 and max_sim_to_selected < best_max_sim:
                best_i = i
                best_max_sim = max_sim_to_selected

        selected.append(best_i)          # type: ignore[arg-type]
        remaining.remove(best_i)         # type: ignore[arg-type]

    return selected

class Retriever:
    """
    Thin retrieval layer over the vector index.
    Supports plain Top-K and MMR (diversity) re-ranking.
    """

    def __init__(self, index: FaissIndex, embedder: Embedder):
        self.index = index
        self.embedder = embedder

    def search(
        self,
        query: str,
        k: int = 4,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        fetch_k: Optional[int] = None,
    ) -> List[Document]:
        """
        - k: desired results (capped to [1, 10] by API schema, but we still clamp here)
        - use_mmr: enable diversity re-ranking
        - mmr_lambda: 0..1 (tradeoff relevance vs diversity)
        - fetch_k: initial candidate pool before MMR; defaults to max(20, k*4)
        """
        k = max(1, min(50, int(k)))
        if not use_mmr:
            # Regular Top-K from the index
            return self.index.search(query, k=k,embedder=self.embedder)

        # MMR path
        if fetch_k is None:
            fetch_k = max(20, k * 4)

        # 1) Pull a candidate pool by relevance
        candidates: List[Document] = self.index.search(query, k=fetch_k,embedder=self.embedder)
        if not candidates:
            return []

        # 2) Embed query and candidate texts (we re-embed to get vectors for MMR)
        qv = self.embedder.encode_texts([query])
        cand_texts = [d.page_content for d in candidates]
        cand_vecs = self.embedder.encode_texts(cand_texts)

        # 3) Select with MMR, then map back to docs
        chosen_idx = mmr(qv, cand_vecs, lambda_mult=mmr_lambda, k=k)
        # keep order of selection (already in list order)
        return [candidates[i] for i in chosen_idx]