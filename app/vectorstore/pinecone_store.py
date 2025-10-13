# app/vectorstore/pinecone_store.py
from __future__ import annotations
from typing import Optional, List, Any, Dict
from langchain_core.documents import Document
import os
import uuid
import numpy as np

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None
    ServerlessSpec = None


def _md_str(md: Dict[str, Any], key: str, default: str = "") -> str:
    val = md.get(key, default)
    return str(val) if val is not None else default
def _to_jsonable(v):
    # Pinecone metadata must be JSON-serializable: (str, int, float, bool, None, list, dict)
    # Anything else -> str(v)
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    # e.g. PosixPath, UUID, datetime, numpy types, etc.
    return str(v)

class PineconeIndex:
    """Pinecone backend Index"""

    def __init__(
        self,
        *,
        dimension: int,
        metric: str = "cosine",
        model_name: str = "unknown",
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        if Pinecone is None:
            raise RuntimeError("Pinecone SDK not installed, run: pip install pinecone")

        self.dimension = dimension
        self.metric = metric or "cosine"
        self.model_name = model_name

        self._namespace: str = "default"
        self._index_name = index_name or os.getenv("PINECONE_INDEX", "")
        if not self._index_name:
            raise RuntimeError("Pinecone index name not provided (env PINECONE_INDEX or index_name).")

        api_key = api_key or os.getenv("PINECONE_API_KEY", "")
        if not api_key:
            raise RuntimeError("Pinecone API key is required (env PINECONE_API_KEY).")

        self._pc = Pinecone(api_key=api_key)

        # Create an index if missing (serverless)
        existing = {ix["name"] for ix in self._pc.list_indexes()}  # SDK returns dicts
        if self._index_name not in existing:
            spec = ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", cloud),
                region=os.getenv("PINECONE_REGION", region),
            )
            self._pc.create_index(
                name=self._index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=spec,
            )

        self._index = self._pc.Index(self._index_name)

    # -------- Multi-tenant scoping -----------
    def set_namespace(self, ns: Optional[str]) -> None:
        self._namespace = (ns or "default")

    # ------ Ingestion ---------
    def add_documents(self, docs: List[Document], embedder) -> None:
        if not docs:
            return

        texts = [d.page_content or "" for d in docs]
        vecs = embedder.encode_texts(texts)  # np.ndarray (N, D)

        if not isinstance(vecs, np.ndarray):
            vecs = np.array(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding matrix shape {vecs.shape} does not match index dim {self.dimension}"
            )

        vectors = []
        for d, v in zip(docs, vecs):
            md = dict(d.metadata or {})

            # --- sanitize metadata so Pinecone accepts it ---
            md = _to_jsonable(md)

            # Normalize commonly-used fields
            doc_id   = _md_str(md, "doc_id", "")
            page     = _md_str(md, "page", "0")
            chunk_id = _md_str(md, "chunk_id", "")
            chunk_uid = _md_str(md, "chunk_uid", "") or f"{doc_id}:{page}:{chunk_id or uuid.uuid4().hex}"

            # Ensure source is a string (handles PosixPath)
            if "source" in md:
                md["source"] = _md_str(md, "source", "")

            # Keep full text for convenience (counts toward Pinecone’s per-vector metadata size limit)
            md["text"] = d.page_content or ""
            md["model_name"] = self.model_name

            vectors.append(
                {
                    "id": chunk_uid,
                    "values": [float(x) for x in v.tolist()],  # 1D numpy -> list[float]
                    "metadata": md,
                }
            )

        self._index.upsert(vectors=vectors, namespace=self._namespace)
    # def add_documents(self, docs: List[Document], embedder) -> None:
    #     if not docs:
    #         return

    #     texts = [d.page_content or "" for d in docs]
    #     vecs = embedder.encode_texts(texts)  # np.ndarray (N, D)

    #     if not isinstance(vecs, np.ndarray):
    #         vecs = np.array(vecs, dtype=np.float32)
    #     if vecs.ndim != 2 or vecs.shape[1] != self.dimension:
    #         raise ValueError(
    #             f"Embedding matrix shape {vecs.shape} does not match index dim {self.dimension}"
    #         )

    #     vectors = []
    #     for d, v in zip(docs, vecs):
    #         md = dict(d.metadata or {})
    #         doc_id = _md_str(md, "doc_id", "")
    #         page = _md_str(md, "page", "0")
    #         chunk_id = _md_str(md, "chunk_id", "")
    #         chunk_uid = _md_str(md, "chunk_uid", "") or f"{doc_id}:{page}:{chunk_id or uuid.uuid4().hex}"

    #         meta = dict(md)
    #         meta["text"] = d.page_content or ""
    #         meta["model_name"] = self.model_name

    #         # v is a 1D numpy array (D,). Convert to a plain list of floats
    #         vectors.append(
    #             {
    #                 "id": chunk_uid,
    #                 "values": [float(x) for x in v.tolist()],
    #                 "metadata": meta,
    #             }
    #         )

    #     # Batch upsert once (faster & cleaner than inside loop)
    #     self._index.upsert(vectors=vectors, namespace=self._namespace)

    # --------- Search -----------
    def search(self, query: str, k: int, embedder) -> List[Document]:
        if not query or k <= 0:
            return []

        qv = embedder.encode_texts([query])  # np.ndarray (1, D)
        if not isinstance(qv, np.ndarray):
            qv = np.array(qv, dtype=np.float32)

        # Take the first row
        if qv.ndim == 2 and qv.shape[0] == 1:
            q = qv[0]
        else:
            # Be defensive; squeeze and ensure dim D
            q = np.squeeze(qv)
        if q.ndim != 1:
            raise ValueError(f"Query embedding has unexpected shape: {qv.shape}")

        res = self._index.query(
            namespace=self._namespace,
            vector=[float(x) for x in q.tolist()],
            top_k=int(k),
            include_metadata=True,
        )

        out: List[Document] = []
        matches = getattr(res, "matches", None)
        if not matches:
            return out

        for m in matches:
            md = dict(getattr(m, "metadata", {}) or {})
            text = md.pop("text", "")
            out.append(Document(page_content=text, metadata=md))
        return out
    
    # ... existing imports / class PineconeIndex ...

    # ------ Deletions ------
    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all vectors for a specific document (by metadata filter)."""
        if not doc_id:
            return
        self._index.delete(
            namespace=self._namespace,
            filter={"doc_id": {"$eq": doc_id}},
        )

    def delete_by_filter(self, md_filter: dict) -> None:
        """Generic metadata-filtered delete."""
        if not md_filter:
            return
        self._index.delete(
            namespace=self._namespace,
            filter=md_filter,
        )


# from __future__ import annotations
# from typing import Optional,List,Any,Dict
# from langchain_core.documents import Document
# import os
# import uuid

# try:
#     from pinecone import Pinecone,ServerlessSpec
# except Exception as e:
#     Pinecone = None
#     ServerlessSpec = None

# def _md_str(md:Dict[str,Any],key:str,default:str="") -> str:
#     val = md.get(key,default)
#     return str(val) if val is not None else default

# class PineconeIndex:
#     '''Pinecone backend Index'''
#     def __init__(
#             self,
#             *,
#             dimension:int,
#             metric:str = 'cosine',
#             model_name:str = 'unknown',
#             index_name:Optional[str] = None,
#             api_key:Optional[str] = None,
#             cloud:str = "aws",
#             region:str = "us-east-1"
#     ):
#         if Pinecone is None:
#             raise RuntimeError("Pinecone SDK not installed, run: pip install pinecone")
        
#         self.dimension = dimension
#         self.metric = metric or "cosine"
#         self.model_name = model_name

#         self._namespace:str = "default"
#         self._index_name = index_name or os.getenv("PINECONE_INDEX","")
#         if not api_key:
#             raise RuntimeError("Pinecone API key is required for PineconeIndex")
#         self._pc = Pinecone(api_key=api_key)

#         #Create an index if missing (serverless)
#         have = {ix["name"] for ix in self._pc.list_indexes()}
#         if self._index_name not in have:
#             spec = ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD",cloud, region=os.getenv("PINECONE_REGION",region)))
#             self._pc.create_index(
#                 name=self._index_name,
#                 dimension=self.dimension,
#                 metric=self.metric,
#                 spec=spec
#             )

#         self._index = self._pc.Index(self._index_name)

    
#     #--------Multi-tenant scoping-----------
#     def set_namespace(self,ns:Optional[str]) -> None:
#         self._namespace = (ns or "default")
    
#     #------Ingestion---------
#     def add_documents(self,docs:List[Document],embedder):
#         if not docs:
#             return 
#         texts = [d.page_content or "" for d in docs]

#         vecs = embedder.encode_texts(texts)
#         vectors = []
#         for d,v in zip(docs,vecs):
#             md = dict(d.metadata or {})
#             doc_id = _md_str(md, "doc_id","")
#             page = _md_str(md,"page","0")
#             chunk_id = _md_str(md,"chunk_id","")
#             chunk_uid = _md_str(md, "chunk_uid","") or f"{doc_id}:{page}:{chunk_id or uuid.uuid4().hex}"

#             meta = dict(md)
#             meta["text"] = d.page_content or ""
#             meta["model_name"] = self.model_name

#             vectors.append(
#                 {
#                     "id":chunk_uid,
#                     "values":[float(x) for x in (getattr(v,"tolist",lambda:v)())],
#                     "metadata":meta
#                 }
#             )
#             self._index.upsert(vectors=vectors,namespace=self._namespace)

    
#     #---------Search-----------
#     def search(self,query:str,k:int,embedder) -> List[Document]:
#         if not query or k<=0:
#             return []
#         qv = embedder.encode_texts([query])
        
#         res = self._index.query(
#             namespace=self._namespace,
#             vector=[float(x) for x in (getattr(qv,"tolist",lambda qv:qv))],
#             top_k = int(k),
#             include_metadata=True,
#         )

#         out:List[Document] = []
#         if not res or not getattr(qv,"matches",None):
#             return out
        
#         for m in getattr(res,"matches",None):
#             md = dict(getattr(m,"metadata",{}) or {})
#             text = md.pop("text","")
#             out.append(Document(page_content=text, metadata=md))
#         return out
    




