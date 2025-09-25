# # from fastapi import FastAPI,HTTPException, status, Depends,UploadFile, File
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse
# # from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer

# # from langchain_core.documents import Document
# # from pydantic import BaseModel, Field,field_validator
# # from typing import Optional,List
# # import time
# # import jwt
# # import os
# # import tempfile

# # from app.ingest.loaders import load_pdfs,load_txts,load_web
# # from app.ingest.preprocess import clean, chunk
# # from app.embeddings.encoder import Embedder
# # from app.vectorstore.faiss_store import FaissIndex


# # #=========================
# # #       App wide Singletons
# # #=========================
# # # JWT_SECRET = os.getenv("JWT_SECRET","CHANGE_ME_FOR_PROD")
# # JWT_SECRET = "secret"
# # EMBED_MODEL = os.getenv("HF_EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2")
# # VECTOR_BACKEND = os.getenv("VECTOR_BACKEND","faiss")
# # INDEX_DIR = os.getenv("INDEX_DIR","./data/indexes/faiss")

# # #Embedder
# # _EMBEDDER = Embedder(model_name=EMBED_MODEL)

# # #Vector index
# # _INDEX = FaissIndex(dimension=_EMBEDDER.dimension, metric="cosine",model_name=_EMBEDDER.model_name)

# # #Simle In-memory metrics
# # _METRICS = {
# #     "request_total":0,
# #     "ingest_docs_total":0,
# #     "ingest_chunks_total":0,
# #     "queries_served_total":0
# # }


# # #=========================
# # #       Auth
# # #=========================
# # _bearer = HTTPBearer(auto_error=False)

# # class AuthedUser(BaseModel):
# #     sub:str
# #     tenant_id:Optional[str] = None

# # def require_bearer_auth(auth_headers:HTTPAuthorizationCredentials = Depends(_bearer)) -> AuthedUser:
# #     print(f"the authorization is {auth_headers}")
# #     print(f"The auth_headers.scheme.lower is {auth_headers.scheme.lower()}")
# #     if not auth_headers or not auth_headers.scheme.lower() != "bearer":
# #         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
# #     token = auth_headers.credentials
    
# #     try:
# #         payload = jwt.decode(token,JWT_SECRET, algorithms=["HS256"], options={"require":["exp","sub"]})
# #         return AuthedUser(sub=str(payload.get("sub")), tenant_id=payload.get("tenant_id"))
# #     except jwt.ExpiredSignatureError:
# #         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
# #     except Exception:
# #         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# # #=========================
# # #       Helpers
# # #=========================
# # def _ext_from_upload_file(file:UploadFile) -> str:
# #     name = (file.filename or "").lower()
# #     if name.endswith(".pdf") or file.content_type == "application/pdf":
# #         return "pdf"
# #     return "txt"


# # def _load_from_upload(file:UploadFile) -> List[Document]:
# #     suffix = ".pdf" if _ext_from_upload_file(file) == "pdf" else ".txt"
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
# #         tmp.write(file.file.read())
# #         tmp.flush()
# #         tmp_path = tmp.name

# #         try:
# #             if suffix == ".pdf":
# #                 docs = load_pdfs([tmp_path])
# #             else:
# #                 docs = load_txts([tmp_path])
# #         finally:
# #             try:
# #                 os.remove(tmp_path)
# #             except Exception:
# #                 pass
# # def _pipeline_ingest(docs_in: List[Document]) -> List[Document]:
# #     # Clean each doc/page, then chunk
# #     cleaned = [clean(d) for d in docs_in]
# #     # chunk defaults (800/120) from your A2 tests
# #     chunks = chunk(cleaned, chunk_size=800, chunk_overlap=120)
# #     return chunks

# # def _add_to_index(chunks: List[Document]) -> None:
# #     if not chunks:
# #         return
# #     _INDEX.add_documents(chunks, _EMBEDDER)

# # def _stub_llm_answer(question: str,ctx_docs:List[Document]) -> str:
# #     prefix = (ctx_docs[0].page_content[:100] if ctx_docs else "")
# #     return f"[ANSWER] {question}| ctx = {prefix}"




# # #=========================
# # #       Schemas
# # #=========================
# # class IngestUrlsRequest(BaseModel):
# #     urls:List[str] = Field(default_factory=list)

# # class IngestResponse(BaseModel):
# #     ingested:int
# #     failures:List[str]
# #     duration_ms:dict

# # class QueryRequest(BaseModel):
# #     session_id:str
# #     question:str
# #     k:int = 4
# #     provider: Optional[str] = Field(default="stub")

# #     @field_validator("k")
# #     def _cap_k(cls, v):
# #         return max(1, min(10, int(v)))

# # class QueryResponse(BaseModel):
# #     answer:str
# #     sources:List[str]
# #     provided:str


# # #=========================
# # #       FASTAPI app
# # #=========================

# # def create_app() -> FastAPI:
# #     app = FastAPI(title="Smart RAG Chatbot API", version="0.1.0")

# #     app.add_middleware(
# #         CORSMiddleware,
# #         allow_origins=["*"],
# #         allow_credentials=False,
# #         allow_methods=["*"],
# #         allow_headers=["*"]
# #     )

# #     @app.middleware("http")
# #     async def _count_requests(request,call_next):
# #         start = time.time()
# #         resp = await call_next(request)
# #         _METRICS["request_total"] += 1
# #         return resp

# #     @app.get('/health')
# #     def health():
# #         return {"status":"ok"}
    
# #     @app.get('/metrics')
# #     def metrics(_:AuthedUser=Depends(require_bearer_auth)):
# #         return JSONResponse(_METRICS)
    
# #     @app.post('/ingest',response_model=IngestResponse,dependencies=[Depends(require_bearer_auth)])
# #     def ingest(
# #         urls_body:Optional[IngestUrlsRequest] = None,
# #         file: Optional[UploadFile] = File(None)
# #     ):
# #         started = time.time()
# #         t0 = time.time()
# #         load_ms = clean_ms = chunk_ms = embed_ms = index_ms = 0

# #         docs:List[Document] = []
# #         failures:List[str] = []
        
# #         if file is not None:
# #             try:
# #                 docs = _load_from_upload(file)
# #             except Exception as e:
# #                 raise HTTPException(status_code=400, detail=f"Failed to load file {e}")
# #         elif urls_body is not None and urls_body.urls:
# #             try:
# #                 docs = load_web(urls_body.urls)
# #             except Exception as e:
# #                 failures.append(str(e))
# #         else:
# #             raise HTTPException(status_code=400, detail=f"Provide a file to load file or a url ")

# #         load_ms = int((time.time()-t0)*1000)

# #         #Clean
# #         t1 = time.time()
# #         cleaned =[clean(d) for d in docs]
# #         clean_ms = int((time.time()-t1)*1000)

# #         #Chunk
# #         t2 = time.time()
# #         chunks =[chunk(d) for d in docs]
# #         chunk_ms = int((time.time()-t2)*1000)

# #         #Embed + index
# #         t3 = time.time()

# #         _METRICS["ingest_docs_total"]+=len(docs)
# #         _METRICS["ingest_chunks_total"]+=len(chunks)

# #         t4 = time.time()
# #         _INDEX.add_documents(chunks, _EMBEDDER)
# #         index_ms = int((t4-time.time()*1000))

# #         total_ms = int((time.time()-started)*1000)

# #         return IngestResponse(
# #             ingested=len(chunks),
# #             failures=failures,
# #             duration_ms={
# #                 "total":total_ms,
# #                 "load":load_ms,
# #                 "clean":clean_ms,
# #                 "chunk":chunk_ms,
# #                 "embed":embed_ms,
# #                 "index":index_ms
# #             },
# #         )

# #     @app.post('/query',response_model=QueryResponse)
# #     def query(body:QueryResponse,user:AuthedUser = Depends(require_bearer_auth)):

# #         results: List[Document] = _INDEX.search(body.question, k=body.k, embedder=_EMBEDDER)
# #         _METRICS["queries_served_total"] += 1

# #         sources = []
# #         for d in results:
# #             src = d.metadata.get["source"]
# #             if isinstance(src,str):
# #                 sources.append(src)
        
# #         #LLM stub (A4 will replace with real orchestration)
# #         answer = _stub_llm_answer(body.question, results)
# #         return QueryResponse(answer=answer,sources=sources,provider=(body.provider or "stub"))

    
# #     return app

# # app = create_app()

# from __future__ import annotations

# import os
# import time
# import tempfile
# from typing import Any, List, Optional

# import jwt
# from fastapi import Body
# from fastapi import FastAPI, HTTPException, status, Depends, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# from pydantic import BaseModel, Field, field_validator

# from langchain_core.documents import Document

# from app.ingest.loaders import load_pdfs, load_txts, load_web
# from app.ingest.preprocess import clean, chunk
# from app.embeddings.encoder import Embedder
# from app.vectorstore.faiss_store import FaissIndex


# # =========================
# # App-wide singletons
# # =========================
# JWT_SECRET = os.getenv("JWT_SECRET", "JWT_SECRET")
# EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
# INDEX_DIR = os.getenv("INDEX_DIR", "./data/indexes/faiss")

# # Embedder / Index
# _EMBEDDER = Embedder(model_name=EMBED_MODEL)
# _INDEX = FaissIndex(dimension=_EMBEDDER.dimension, metric="cosine", model_name=_EMBEDDER.model_name)

# # Metrics (in-memory)
# _METRICS = {
#     "requests_total": 0,
#     "ingest_docs_total": 0,
#     "ingest_chunks_total": 0,
#     "queries_served_total": 0,
# }


# # =========================
# # Auth
# # =========================
# _bearer = HTTPBearer(auto_error=False)

# class AuthedUser(BaseModel):
#     sub: str
#     tenant_id: Optional[str] = None

# def require_bearer_auth(auth_headers: HTTPAuthorizationCredentials = Depends(_bearer)) -> AuthedUser:
#     if not auth_headers or auth_headers.scheme.lower() != "bearer":
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
#     token = auth_headers.credentials
#     try:
#         payload = jwt.decode(
#     token,
#     JWT_SECRET,
#     algorithms=["HS256"],
#     options={
#         "require": ["exp", "sub"],
#         "verify_aud": False,   # important
#         "verify_iss": False,   # important
#     },
# )
#         return AuthedUser(sub=str(payload.get("sub")), tenant_id=payload.get("tenant_id"))
#     except jwt.ExpiredSignatureError:
#         print(F"Signature expired")
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
#     except Exception:
#         print(token)
#         print(f"Invalid Token")
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# # =========================
# # Schemas
# # =========================
# class IngestUrlsRequest(BaseModel):
#     urls: List[str] = Field(default_factory=list)

# class IngestResponse(BaseModel):
#     ingested: int
#     failures: List[str]
#     durations_ms: dict

# class QueryRequest(BaseModel):
#     session_id: str
#     question: str
#     k: int = 4
#     provider: Optional[str] = Field(default="stub")

#     @field_validator("k")
#     def _cap_k(cls, v):
#         return max(1, min(10, int(v)))

# class QueryResponse(BaseModel):
#     answer: str
#     sources: List[str]
#     provider: str


# # =========================
# # Helpers
# # =========================
# def _ext_from_upload_file(file: UploadFile) -> str:
#     name = (file.filename or "").lower()
#     if name.endswith(".pdf") or file.content_type == "application/pdf":
#         return "pdf"
#     return "txt"

# def _load_from_upload(file: UploadFile) -> List[Document]:
#     suffix = ".pdf" if _ext_from_upload_file(file) == "pdf" else ".txt"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(file.file.read())
#         tmp.flush()
#         tmp_path = tmp.name
#     try:
#         if suffix == ".pdf":
#             return load_pdfs([tmp_path])
        
#         else:
#             return load_txts([tmp_path])
#     finally:
#         try:
#             os.remove(tmp_path)
#         except Exception:
#             pass

# def _stub_llm_answer(question: str, ctx_docs: List[Document]) -> str:
#     prefix = (ctx_docs[0].page_content[:100] if ctx_docs else "")
#     return f"[ANSWER] {question} | ctx={prefix}"


# # =========================
# # FastAPI app
# # =========================
# def create_app() -> FastAPI:
#     app = FastAPI(title="Smart RAG Chatbot API", version="0.1.0")

#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_credentials=False,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

#     @app.middleware("http")
#     async def _count_requests(request, call_next):
#         resp = await call_next(request)
#         _METRICS["requests_total"] += 1
#         return resp

#     @app.get("/health")
#     def health():
#         return {"status": "ok"}

#     @app.get("/metrics")
#     def metrics(_: AuthedUser = Depends(require_bearer_auth)):
#         return JSONResponse(_METRICS)

#     @app.post("/ingest", response_model=IngestResponse)
#     def ingest(
#         urls_body: Optional[IngestUrlsRequest] = None,
#         file: Optional[UploadFile] = File(None),
#         _: AuthedUser = Depends(require_bearer_auth),
#     ):
#         started = time.time()
#         load_ms = clean_ms = chunk_ms = embed_ms = index_ms = 0
#         print(f"url_body is {urls_body}")
#         docs: List[Document] = []
#         failures: List[str] = []

#         # Load
#         t0 = time.time()
#         if file is not None:
#             try:
#                 docs = _load_from_upload(file)
#             except Exception as e:
#                 raise HTTPException(status_code=400, detail=f"Failed to load file: {e}")
#         elif urls_body is not None and urls_body.urls:
#             try:
#                 docs = load_web(urls_body.urls)
#             except Exception as e:
#                 failures.append(str(e))
#         else:
#             raise HTTPException(status_code=400, detail="Provide a file upload or a JSON body with 'urls'")
#         load_ms = int((time.time() - t0) * 1000)

#         # Clean
#         t1 = time.time()
#         cleaned = [clean(d) for d in docs]
#         clean_ms = int((time.time() - t1) * 1000)

#         # Chunk
#         t2 = time.time()
#         chunks = chunk(cleaned, chunk_size=800, chunk_overlap=120)
#         chunk_ms = int((time.time() - t2) * 1000)

#         # Embed + index (embedding time occurs inside add_documents; we expose index time)
#         _METRICS["ingest_docs_total"] += len(docs)
#         _METRICS["ingest_chunks_total"] += len(chunks)

#         t3 = time.time()
#         _INDEX.add_documents(chunks, _EMBEDDER)
#         index_ms = int((time.time() - t3) * 1000)

#         total_ms = int((time.time() - started) * 1000)
#         return IngestResponse(
#             ingested=len(chunks),
#             failures=failures,
#             durations_ms={
#                 "total": total_ms,
#                 "load": load_ms,
#                 "clean": clean_ms,
#                 "chunk": chunk_ms,
#                 "embed": embed_ms,
#                 "index": index_ms,
#             },
#         )

#     @app.post("/query", response_model=QueryResponse)
#     def query(body: QueryRequest, _: AuthedUser = Depends(require_bearer_auth)):
#         results: List[Document] = _INDEX.search(body.question, k=body.k, embedder=_EMBEDDER)
#         _METRICS["queries_served_total"] += 1

#         sources: List[str] = []
#         for d in results:
#             src = d.metadata.get("source")
#             if isinstance(src, str):
#                 sources.append(src)

#         answer = _stub_llm_answer(body.question, results)
#         return QueryResponse(answer=answer, sources=sources, provider=(body.provider or "stub"))

#     return app

# app = create_app()

from __future__ import annotations

import os
import time
import tempfile
from typing import List, Optional

import jwt
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from langchain_core.documents import Document

from app.ingest.loaders import load_pdfs, load_txts, load_web
from app.ingest.preprocess import clean, chunk
from app.embeddings.encoder import Embedder
from app.vectorstore.faiss_store import FaissIndex


# =========================
# App-wide singletons
# =========================
JWT_SECRET = os.getenv("JWT_SECRET", "JWT_SECRET")
EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/indexes/faiss")

# Embedder / Index
_EMBEDDER = Embedder(model_name=EMBED_MODEL)
_INDEX = FaissIndex(dimension=_EMBEDDER.dimension, metric="cosine", model_name=_EMBEDDER.model_name)

# Metrics (in-memory)
_METRICS = {
    "requests_total": 0,
    "ingest_docs_total": 0,
    "ingest_chunks_total": 0,
    "queries_served_total": 0,
}


# =========================
# Auth
# =========================
_bearer = HTTPBearer(auto_error=False)

class AuthedUser(BaseModel):
    sub: str
    tenant_id: Optional[str] = None

def require_bearer_auth(auth_headers: HTTPAuthorizationCredentials = Depends(_bearer)) -> AuthedUser:
    if not auth_headers or auth_headers.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = auth_headers.credentials
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            options={
                "require": ["exp", "sub"],
                "verify_aud": False,
                "verify_iss": False,
            },
        )
        return AuthedUser(sub=str(payload.get("sub")), tenant_id=payload.get("tenant_id"))
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# =========================
# Schemas
# =========================
class IngestUrlsRequest(BaseModel):
    urls: List[str] = Field(default_factory=list)

class IngestResponse(BaseModel):
    ingested: int
    failures: List[str]
    durations_ms: dict

class QueryRequest(BaseModel):
    session_id: str
    question: str
    k: int = 4
    provider: Optional[str] = Field(default="stub")

    @field_validator("k")
    def _cap_k(cls, v):
        return max(1, min(10, int(v)))

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    provider: str


# =========================
# Helpers
# =========================
def _ext_from_upload_file(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    if name.endswith(".pdf") or file.content_type == "application/pdf":
        return "pdf"
    return "txt"

def _load_from_upload(file: UploadFile) -> List[Document]:
    """Persist UploadFile to a temp file and load via the right loader."""
    suffix = ".pdf" if _ext_from_upload_file(file) == "pdf" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            return load_pdfs([tmp_path])
        else:
            return load_txts([tmp_path])
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def _stub_llm_answer(question: str, ctx_docs: List[Document]) -> str:
    prefix = (ctx_docs[0].page_content[:100] if ctx_docs else "")
    return f"[ANSWER] {question} | ctx={prefix}"


# =========================
# FastAPI app
# =========================
def create_app() -> FastAPI:
    app = FastAPI(title="Smart RAG Chatbot API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _count_requests(request, call_next):
        resp = await call_next(request)
        _METRICS["requests_total"] += 1
        return resp

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics(_: AuthedUser = Depends(require_bearer_auth)):
        return JSONResponse(_METRICS)

    # ----------- INGEST (file and/or URLs; JSON and/or multipart) -----------
    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        request: Request,
        # Multipart file (optional)
        file: Optional[UploadFile] = File(None),
        # JSON body (optional)
        urls_json: Optional[IngestUrlsRequest] = Body(None),
        # Form field(s) (optional): supports repeated 'urls' or comma-separated 'urls'
        urls_form: Optional[List[str]] = Form(None),
        _: AuthedUser = Depends(require_bearer_auth),
    ):
        started = time.time()
        load_ms = clean_ms = chunk_ms = embed_ms = index_ms = 0

        # ----- Gather URLs from any supported input shape
        urls: List[str] = []

        # From explicit JSON model
        if urls_json and urls_json.urls:
            urls.extend(urls_json.urls)

        # From form fields (either repeated 'urls' or comma-separated list)
        if urls_form:
            for item in urls_form:
                if item:
                    urls.extend([u.strip() for u in item.split(",") if u.strip()])

        # Fallback: if request is application/json but FastAPI didn't bind due to File param presence
        if not urls and file is None:
            ct = request.headers.get("content-type", "")
            if ct.startswith("application/json"):
                try:
                    raw = await request.json()
                    if isinstance(raw, dict) and "urls" in raw:
                        raw_urls = raw.get("urls") or []
                        if isinstance(raw_urls, list):
                            urls.extend(list(map(str, raw_urls)))
                        elif isinstance(raw_urls, str):
                            urls.extend([u.strip() for u in raw_urls.split(",") if u.strip()])
                except Exception:
                    # ignore parse errors; we'll validate below
                    pass

        # Deduplicate URLs, keep order
        if urls:
            seen = set()
            urls = [u for u in urls if not (u in seen or seen.add(u))]

        # ----- Load docs
        t0 = time.time()
        docs: List[Document] = []
        failures: List[str] = []

        if file is not None:
            try:
                docs.extend(_load_from_upload(file))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load file: {e}")

        if urls:
            try:
                docs.extend(load_web(urls))
            except Exception as e:
                failures.append(str(e))

        if not docs and not failures:
            raise HTTPException(
                status_code=400,
                detail="Provide a file (multipart) and/or URLs (JSON {'urls': [...]} or form 'urls').",
            )
        load_ms = int((time.time() - t0) * 1000)

        # ----- Clean → Chunk → Index
        t1 = time.time()
        cleaned = [clean(d) for d in docs]
        clean_ms = int((time.time() - t1) * 1000)

        t2 = time.time()
        chunks = chunk(cleaned, chunk_size=800, chunk_overlap=120)
        chunk_ms = int((time.time() - t2) * 1000)

        _METRICS["ingest_docs_total"] += len(docs)
        _METRICS["ingest_chunks_total"] += len(chunks)

        t3 = time.time()
        _INDEX.add_documents(chunks, _EMBEDDER)
        index_ms = int((time.time() - t3) * 1000)

        total_ms = int((time.time() - started) * 1000)
        return IngestResponse(
            ingested=len(chunks),
            failures=failures,
            durations_ms={
                "total": total_ms,
                "load": load_ms,
                "clean": clean_ms,
                "chunk": chunk_ms,
                "embed": embed_ms,  # embedding time is inside add_documents; keep 0 if you prefer
                "index": index_ms,
            },
        )

    # ----------- QUERY -----------
    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, _: AuthedUser = Depends(require_bearer_auth)):
        results: List[Document] = _INDEX.search(body.question, k=body.k, embedder=_EMBEDDER)
        _METRICS["queries_served_total"] += 1

        sources: List[str] = []
        for d in results:
            src = d.metadata.get("source")
            if isinstance(src, str):
                sources.append(src)

        answer = _stub_llm_answer(body.question, results)
        return QueryResponse(answer=answer, sources=sources, provider=(body.provider or "stub"))

    return app


app = create_app()