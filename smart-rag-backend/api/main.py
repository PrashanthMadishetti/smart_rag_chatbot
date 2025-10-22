from __future__ import annotations

import os
import time
import tempfile
import uuid
from typing import List, Optional, Any

import jwt
import redis
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

import google.generativeai as genai
from groq import Groq

from pydantic import BaseModel, Field, field_validator
from langchain_core.documents import Document

from app.ingest.loaders import load_pdfs, load_txts, load_web
from app.ingest.preprocess import clean, chunk
from app.embeddings.encoder import Embedder
from app.vectorstore.faiss_store import FaissIndex
from app.vectorstore.pinecone_store import PineconeIndex
from app.retrieval.retriever import Retriever
from app.auth.auth_router import router as auth_router
from app.auth.db import Base, engine
from app.memory.store import MemoryStore
from app.prompt.builder import HistoryTurn, PromptBuilder, PromptChunk, PromptInputs
from app.docs.router import router as docs_router
from app.docs.models import DocumentRecord


# =======================
# Config & Environment
# =======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

JWT_SECRET = os.getenv("JWT_SECRET", "JWT_SECRET")
EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/indexes/faiss")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_SECONDS", "0") or 0) or None

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "tenant-documents")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")


# =========================
# Metrics (in-memory)
# =========================
_METRICS = {
    "requests_total": 0,
    "ingest_docs_total": 0,
    "ingest_chunks_total": 0,
    "queries_served_total": 0,
}


# =========================
# Lazy singletons
# =========================
_EMBEDDER: Optional[Embedder] = None
_INDEX: Optional[FaissIndex | PineconeIndex] = None
RETRIEVER: Optional[Retriever] = None
_redis_client: Optional[redis.Redis] = None
_MEMORY: Optional[Any] = None
_GEMINI: Optional[Any] = None
_GROQ: Optional[Groq] = None


# =========================
# Fallback in-process memory (if Redis fails)
# =========================
class _SimpleMemory:
    def __init__(self, max_turns: int = 6):
        self._max_turns = max_turns
        self._data: dict[tuple[str, str], list[dict]] = {}

    def append_turn(self, tenant: str, session: str, role: str, text: str):
        key = (tenant, session)
        arr = self._data.setdefault(key, [])
        arr.append({"role": role, "text": text})
        if len(arr) > self._max_turns * 2:
            self._data[key] = arr[-self._max_turns * 2 :]

    def get_recent(self, tenant: str, session: str, limit: int = 6) -> list[dict]:
        key = (tenant, session)
        arr = self._data.get(key, [])
        return arr[-limit:] if limit > 0 else []


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
    use_mmr: bool = Field(default=False, description="Enable MMR re-ranking for diversity")
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0, description="MMR tradeoff (0..1)")

    @field_validator("k")
    def _cap_k(cls, v):
        return max(1, min(10, int(v)))


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    provider: str


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
# Helpers
# =========================
def _get_gemini():
    global _GEMINI
    if _GEMINI is None and GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _GEMINI = genai.GenerativeModel(GEMINI_MODEL)
    return _GEMINI


def _get_groq():
    global _GROQ
    if _GROQ is None and GROQ_API_KEY:
        _GROQ = Groq(api_key=GROQ_API_KEY)
    return _GROQ


def _choose_free_llm(provider_hint: Optional[str]) -> tuple[str, str]:
    hint = (provider_hint or "").lower().strip()
    if hint in ("gemini", "google") and GEMINI_API_KEY:
        return ("gemini", GEMINI_MODEL)
    if hint == "groq" and GROQ_API_KEY:
        return ("groq", GROQ_MODEL)
    if GEMINI_API_KEY:
        return ("gemini", GEMINI_MODEL)
    if GROQ_API_KEY:
        return ("groq", GROQ_MODEL)
    return ("stub", "stub")


def _ext_from_upload_file(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    if name.endswith(".pdf") or file.content_type == "application/pdf":
        return "pdf"
    return "txt"


def _load_from_upload(file: UploadFile) -> List[Document]:
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


def _make_index():
    if VECTOR_BACKEND == "pinecone":
        return PineconeIndex(
            dimension=_EMBEDDER.dimension,  # type: ignore[attr-defined]
            metric="cosine",
            model_name=_EMBEDDER.model_name,  # type: ignore[attr-defined]
            index_name=PINECONE_INDEX_NAME,
            api_key=PINECONE_API_KEY,
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        )
    return FaissIndex(
        dimension=_EMBEDDER.dimension,  # type: ignore[attr-defined]
        metric="cosine",
        model_name=_EMBEDDER.model_name,  # type: ignore[attr-defined]
    )


def _guess_title_and_source(first_doc: Document) -> tuple[str, str]:
    md = first_doc.metadata or {}
    src = md.get("source") or md.get("file_path") or md.get("url") or "uploaded"
    title = md.get("title") or (os.path.basename(str(src)) if isinstance(src, str) else "Document")
    return str(title), str(src)


# =========================
# Lazy initialization
# =========================
def _init_embeddings_and_index():
    global _EMBEDDER, _INDEX, RETRIEVER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder(model_name=EMBED_MODEL)
    if _INDEX is None:
        _INDEX = _make_index()
    if RETRIEVER is None:
        RETRIEVER = Retriever(_INDEX, _EMBEDDER)


def _init_redis_and_memory():
    global _redis_client, _MEMORY
    if _MEMORY is not None:
        return
    try:
        _redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            health_check_interval=30,
        )
        _redis_client.ping()
    except Exception:
        _redis_client = None
    try:
        if _redis_client:
            _MEMORY = MemoryStore(
                redis_client=_redis_client,
                max_turns=6,
                ttl_seconds=MEMORY_TTL_SECONDS,
            )
        else:
            _MEMORY = _SimpleMemory(max_turns=6)
    except Exception:
        _MEMORY = _SimpleMemory(max_turns=6)


# =========================
# FastAPI Application
# =========================
def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart RAG Chatbot API",
        version="0.1.0",
        docs_url="/api-docs",
        redoc_url=None,
    )

    # If you later move to private SQL, consider relocating to startup
    Base.metadata.create_all(bind=engine)

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

    app.include_router(auth_router, prefix="")
    app.include_router(docs_router)

    @app.on_event("startup")
    async def _startup():
        _init_embeddings_and_index()
        _init_redis_and_memory()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics(_: AuthedUser = Depends(require_bearer_auth)):
        return JSONResponse(_METRICS)

    # ----------- INGEST -----------
    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        request: Request,
        file: Optional[UploadFile] = File(None),
        urls_json: Optional[IngestUrlsRequest] = Body(None),
        urls_form: Optional[List[str]] = Form(None),
        user: AuthedUser = Depends(require_bearer_auth),
    ):
        tenant_ns = user.tenant_id
        if not tenant_ns:
            raise HTTPException(status_code=403, detail="Tenant token required for ingestion")

        started = time.time()
        load_ms = clean_ms = chunk_ms = embed_ms = index_ms = 0
        urls: List[str] = []

        if urls_json and urls_json.urls:
            urls.extend(urls_json.urls)
        if urls_form:
            for item in urls_form:
                if item:
                    urls.extend([u.strip() for u in item.split(",") if u.strip()])

        if not urls and file is None:
            try:
                raw = await request.json()
                if isinstance(raw, dict) and "urls" in raw:
                    raw_urls = raw.get("urls") or []
                    if isinstance(raw_urls, list):
                        urls.extend(list(map(str, raw_urls)))
            except Exception:
                pass

        if urls:
            seen = set()
            urls = [u for u in urls if not (u in seen or seen.add(u))]

        t0 = time.time()
        docs: List[Document] = []
        failures: List[str] = []

        if file is not None:
            docs.extend(_load_from_upload(file))
        if urls:
            try:
                docs.extend(load_web(urls))
            except Exception as e:
                failures.append(str(e))
        if not docs and not failures:
            raise HTTPException(status_code=400, detail="No valid input provided.")
        load_ms = int((time.time() - t0) * 1000)

        cleaned = [clean(d) for d in docs]
        clean_ms = 0  # optional timing
        chunks = chunk(cleaned, chunk_size=800, chunk_overlap=120)
        chunk_ms = 0  # optional timing

        _METRICS["ingest_docs_total"] += len(docs)
        _METRICS["ingest_chunks_total"] += len(chunks)

        title, source = (
            (os.path.splitext(file.filename)[0], file.filename)
            if file
            else _guess_title_and_source(docs[0])
        )

        from app.auth.db import get_db
        db = None
        doc_rec = None
        try:
            db = next(get_db())
            doc_rec = DocumentRecord(
                tenant_id=uuid.UUID(tenant_ns),
                created_by=uuid.UUID(user.sub) if user.sub else None,
                title=title,
                source=source,
                chunk_count=0,
            )
            db.add(doc_rec)
            db.flush()
            doc_id_str = str(doc_rec.id)
        except Exception:
            doc_id_str = uuid.uuid4().hex

        for i, d in enumerate(chunks):
            md = dict(d.metadata or {})
            md["doc_id"] = doc_id_str
            md["chunk_id"] = md.get("chunk_id") or str(i)
            page = str(md.get("page") or "0")
            md["chunk_uid"] = md.get("chunk_uid") or f"{doc_id_str}:{page}:{md['chunk_id']}"
            d.metadata = md

        t3 = time.time()
        try:
            _INDEX.set_namespace(tenant_ns)  # type: ignore
        except Exception:
            pass
        _INDEX.add_documents(chunks, _EMBEDDER)  # type: ignore
        if db and doc_rec:
            doc_rec.chunk_count = len(chunks)
            db.commit()
        if db:
            db.close()

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
                "embed": embed_ms,
                "index": index_ms,
            },
        )

    # ----------- QUERY -----------
    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, user: AuthedUser = Depends(require_bearer_auth)):
        tenant = user.tenant_id or "default"
        session = body.session_id
        try:
            _INDEX.set_namespace(tenant)  # type: ignore
        except Exception:
            pass

        try:
            if _MEMORY:
                _MEMORY.append_turn(tenant, session, role="user", text=body.question)
        except Exception:
            pass

        try:
            raw_history = _MEMORY.get_recent(tenant, session, limit=6) if _MEMORY else []
        except Exception:
            raw_history = []

        history = [
            HistoryTurn(role=h["role"], text=h["text"])
            for h in raw_history
            if isinstance(h, dict) and "role" in h and "text" in h
        ]

        results: List[Document] = RETRIEVER.search(  # type: ignore
            query=body.question,
            k=body.k,
            use_mmr=body.use_mmr,
            mmr_lambda=body.mmr_lambda,
        )
        _METRICS["queries_served_total"] += 1

        sources: List[str] = []
        for d in results:
            src = d.metadata.get("source")
            if isinstance(src, str):
                sources.append(src)

        chunks = [
            PromptChunk(
                content=d.page_content,
                source=d.metadata.get("source", "unknown"),
                chunk_id=d.metadata.get("chunk_id", ""),
            )
            for d in results
        ]

        prompt_builder = PromptBuilder()
        prompt_payload = prompt_builder.build(
            inputs=PromptInputs(
                query=body.question,
                chunks=chunks,
                history=history,
                top_k=body.k,
                provider=(body.provider or "stub"),
                mmr_used=body.use_mmr,
            )
        )
        system_txt = prompt_payload["system"]
        user_msg = prompt_payload["messages"][0]["content"]

        prov, model_used = _choose_free_llm(body.provider)
        answer_text = None

        try:
            if prov == "gemini":
                gm_sys = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system_txt)
                resp = gm_sys.generate_content(user_msg)
                answer_text = (resp.text or "").strip()
            elif prov == "groq":
                gq = _get_groq()
                chat = gq.chat.completions.create(
                    model=model_used,
                    messages=[
                        {"role": "system", "content": system_txt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.2,
                )
                answer_text = (chat.choices[0].message.content or "").strip()
        except Exception:
            # Fallback chain
            if prov == "gemini" and GROQ_API_KEY:
                try:
                    gq = _get_groq()
                    chat = gq.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": system_txt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.2,
                    )
                    answer_text = (chat.choices[0].message.content or "").strip()
                except Exception:
                    pass
            elif prov == "groq" and GEMINI_API_KEY and not answer_text:
                try:
                    gm_sys = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system_txt)
                    resp = gm_sys.generate_content(user_msg)
                    answer_text = (resp.text or "").strip()
                except Exception:
                    pass

        if not answer_text:
            answer_text = _stub_llm_answer(body.question, results)

        try:
            if _MEMORY:
                _MEMORY.append_turn(tenant, session, role="assistant", text=answer_text)
        except Exception:
            pass

        return QueryResponse(answer=answer_text, sources=sources, provider=(body.provider or "stub"))

    return app


app = create_app()

# from __future__ import annotations

# import os
# import time
# import tempfile
# from typing import List, Optional,Any
# import jwt
# import redis
# from fastapi import (
#     Body,
#     Depends,
#     FastAPI,
#     File,
#     Form,
#     HTTPException,
#     Request,
#     UploadFile,
#     status,
# )
# import uuid

# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# import google.generativeai as genai
# from groq import Groq

# from pydantic import BaseModel, Field, field_validator
# from langchain_core.documents import Document

# from app.ingest.loaders import load_pdfs, load_txts, load_web
# from app.ingest.preprocess import clean, chunk
# from app.embeddings.encoder import Embedder
# from app.vectorstore.faiss_store import FaissIndex
# from app.vectorstore.pinecone_store import PineconeIndex
# from app.retrieval.retriever import Retriever
# from app.auth.auth_router import router as auth_router
# from app.auth.db import Base,engine
# from app.memory.store import MemoryStore
# from app.prompt.builder import HistoryTurn, PromptBuilder,PromptChunk,PromptInputs
# from app.docs.router import router as docs_router
# from app.docs.models import DocumentRecord

# #=======================
# # Read keys/models
# #=======================

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
# GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
# GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# # =========================
# # App-wide singletons
# # =========================
# JWT_SECRET = os.getenv("JWT_SECRET", "JWT_SECRET")
# EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
# INDEX_DIR = os.getenv("INDEX_DIR", "./data/indexes/faiss")
# REDIS_URL = os.getenv("REDIS_URL","redis://redis:6379/0")

# # Embedder / Index
# _EMBEDDER = Embedder(model_name=EMBED_MODEL)
# # _INDEX = FaissIndex(dimension=_EMBEDDER.dimension, metric="cosine", model_name=_EMBEDDER.model_name)
# # _RETRIEVER = Retriever(_INDEX, _EMBEDDER)


# #=====Pinecone========
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX","tenant-documents")
# PINECONE_CLOUD = os.getenv("PINECONE_CLOUD","aws")
# PINECONE_REGION = os.getenv("PINECONE_REGION","us-east-1")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY","")

# # Metrics (in-memory)
# _METRICS = {
#     "requests_total": 0,
#     "ingest_docs_total": 0,
#     "ingest_chunks_total": 0,
#     "queries_served_total": 0,
# }

# #=========================
# # Redis and Memory
# #=========================
# _redis = redis.from_url(REDIS_URL,decode_responses=True)
# _MEMORY = MemoryStore(redis_client=_redis, max_turns=6, ttl_seconds=int(os.getenv("MEMORY_TTL_SECONDS", "0" or 0)) or None)

# # Lazy init holders
# _GEMINI: Optional[Any] = None
# _GROQ: Optional[Groq] = None

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
#     use_mmr: bool = Field(default=False, description="Enable MMR re-ranking for diversity")
#     mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0, description="MMR tradeoff (0..1)")

#     @field_validator("k")
#     def _cap_k(cls, v):
#         return max(1, min(10, int(v)))

# class QueryResponse(BaseModel):
#     answer: str
#     sources: List[str]
#     provider: str


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
#     # print(token)
#     try:
#         payload = jwt.decode(
#             token,
#             JWT_SECRET,
#             algorithms=["HS256"],
#             options={
#                 "require": ["exp", "sub"],
#                 "verify_aud": False,
#                 "verify_iss": False,
#             },
#         )
#         return AuthedUser(sub=str(payload.get("sub")), tenant_id=payload.get("tenant_id"))
#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
#     except Exception:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token")


# # =========================
# # Helpers
# # =========================

# def _get_gemini():
#     global _GEMINI
#     if _GEMINI is None and GEMINI_API_KEY:
#         # print(GEMINI_API_KEY)
#         genai.configure(api_key=GEMINI_API_KEY)
#         # for m in genai.list_models():
#         #     print(m)
#         _GEMINI = genai.GenerativeModel(GEMINI_MODEL)
#     return _GEMINI

# def _get_groq():
#     global _GROQ
#     if _GROQ is None and GROQ_API_KEY:
#         _GROQ = Groq(api_key=GROQ_API_KEY)
#     return _GROQ

# def _choose_free_llm(provider_hint: Optional[str]) -> tuple[str, str]:
#     """
#     Returns ("gemini"|"groq", model_name). Prefers user hint if available & configured.
#     Falls back: Gemini -> Groq.
#     """
#     hint = (provider_hint or "").lower().strip()

#     if hint in ("gemini", "google") and GEMINI_API_KEY:
#         return ("gemini", GEMINI_MODEL)
#     if hint == "groq" and GROQ_API_KEY:
#         return ("groq", GROQ_MODEL)

#     # Fallback order: Gemini -> Groq
#     if GEMINI_API_KEY:
#         return ("gemini", GEMINI_MODEL)
#     if GROQ_API_KEY:
#         return ("groq", GROQ_MODEL)

#     # Nothing configured – caller should stub
#     return ("stub", "stub")

# def _ext_from_upload_file(file: UploadFile) -> str:
#     name = (file.filename or "").lower()
#     if name.endswith(".pdf") or file.content_type == "application/pdf":
#         return "pdf"
#     return "txt"

# def _load_from_upload(file: UploadFile) -> List[Document]:
#     """Persist UploadFile to a temp file and load via the right loader."""
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

# def _make_index():
#     if VECTOR_BACKEND == "pinecone":
#         return PineconeIndex(
#             dimension=_EMBEDDER.dimension,
#             metric="cosine",
#             model_name=_EMBEDDER.model_name,
#             index_name=PINECONE_INDEX_NAME,
#             api_key=PINECONE_API_KEY,
#             cloud=PINECONE_CLOUD,
#             region=PINECONE_REGION
#         )
#     return FaissIndex(dimension=_EMBEDDER.dimension,metric="cosine",model_name=_EMBEDDER.model_name)

# # Determine a title/source for record
# def _guess_title_and_source(first_doc: Document) -> tuple[str, str]:
#     md = first_doc.metadata or {}
#     src = md.get("source") or md.get("file_path") or md.get("url") or "uploaded"
#     # add parentheses so `or` doesn’t bind incorrectly
#     title = md.get("title") or (os.path.basename(str(src)) if isinstance(src, str) else "Document")
#     return str(title), str(src)


# _INDEX = _make_index()
# RETRIEVER = Retriever(_INDEX,_EMBEDDER)


# # =========================
# # FastAPI app
# # =========================

# def create_app() -> FastAPI:
#     app = FastAPI(
#         title="Smart RAG Chatbot API", 
#         version="0.1.0",
#         docs_url="/api-docs",
#         redoc_url=None)
    
#     Base.metadata.create_all(bind=engine)

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
    
#     app.include_router(auth_router,prefix="")
#     app.include_router(docs_router)  

#     @app.get("/health")
#     def health():
#         return {"status": "ok"}

#     @app.get("/metrics")
#     def metrics(_: AuthedUser = Depends(require_bearer_auth)):
#         return JSONResponse(_METRICS)

#     # ----------- INGEST (file and/or URLs; JSON and/or multipart) -----------
#     @app.post("/ingest", response_model=IngestResponse)
#     async def ingest(
#         request: Request,
#         file: Optional[UploadFile] = File(None),
#         urls_json: Optional[IngestUrlsRequest] = Body(None),
#         urls_form: Optional[List[str]] = Form(None),
#         user: AuthedUser = Depends(require_bearer_auth),
#     ):
#         # Require a tenant-scoped token (login token has no tenant_id)
#         tenant_ns = user.tenant_id
#         if not tenant_ns:
#             raise HTTPException(status_code=403, detail="Tenant token required for ingestion")

#         started = time.time()
#         load_ms = clean_ms = chunk_ms = embed_ms = index_ms = 0

#         # ----- Gather URLs (any of the supported shapes)
#         urls: List[str] = []

#         if urls_json and urls_json.urls:
#             urls.extend(urls_json.urls)

#         if urls_form:
#             for item in urls_form:
#                 if item:
#                     urls.extend([u.strip() for u in item.split(",") if u.strip()])

#         if not urls and file is None:
#             ct = request.headers.get("content-type", "")
#             if ct.startswith("application/json"):
#                 try:
#                     raw = await request.json()
#                     if isinstance(raw, dict) and "urls" in raw:
#                         raw_urls = raw.get("urls") or []
#                         if isinstance(raw_urls, list):
#                             urls.extend(list(map(str, raw_urls)))
#                         elif isinstance(raw_urls, str):
#                             urls.extend([u.strip() for u in raw_urls.split(",") if u.strip()])
#                 except Exception:
#                     pass

#         if urls:
#             seen = set()
#             urls = [u for u in urls if not (u in seen or seen.add(u))]

#         # ----- Load docs
#         t0 = time.time()
#         docs: List[Document] = []
#         failures: List[str] = []

#         if file is not None:
#             try:
#                 docs.extend(_load_from_upload(file))
#             except Exception as e:
#                 raise HTTPException(status_code=400, detail=f"Failed to load file: {e}")

#         if urls:
#             try:
#                 docs.extend(load_web(urls))
#             except Exception as e:
#                 failures.append(str(e))

#         if not docs and not failures:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Provide a file (multipart) and/or URLs (JSON {'urls': [...]} or form 'urls').",
#             )
#         load_ms = int((time.time() - t0) * 1000)

#         # ----- Clean → Chunk
#         t1 = time.time()
#         cleaned = [clean(d) for d in docs]
#         clean_ms = int((time.time() - t1) * 1000)

#         t2 = time.time()
#         chunks = chunk(cleaned, chunk_size=800, chunk_overlap=120)
#         chunk_ms = int((time.time() - t2) * 1000)

#         _METRICS["ingest_docs_total"] += len(docs)
#         _METRICS["ingest_chunks_total"] += len(chunks)

#         # ----- Create a DocumentRecord row (so /docs UI has something to list)
#         # Determine title and source — prefer the uploaded filename if available
#         if file is not None:
#             title = os.path.splitext(file.filename)[0]  # use filename (without extension)
#             source = file.filename
#         else:
#             title, source = _guess_title_and_source(docs[0])
#         # Make a DB session the same way your auth layer does
#         from app.auth.db import get_db
#         db = None
#         doc_rec = None
#         try:
#             db = next(get_db())
#             doc_rec = DocumentRecord(
#                 tenant_id=uuid.UUID(tenant_ns),  # tokens carry a UUID tenant_id
#                 created_by=uuid.UUID(user.sub) if user.sub else None,
#                 title=title,
#                 source=source,
#                 chunk_count=0,
#             )
#             db.add(doc_rec)
#             db.flush()  # assign doc_rec.id
#             doc_id_str = str(doc_rec.id)
#         except Exception:
#             # If anything goes wrong, still allow indexing (but UI list/delete will miss it)
#             doc_id_str = uuid.uuid4().hex
#         finally:
#             # keep DB open for later commit; we’ll close after indexing
#             pass

#         # Attach metadata (doc_id + stable chunk_uid) to every chunk
#         for i, d in enumerate(chunks):
#             md = dict(d.metadata or {})
#             md["doc_id"] = doc_id_str
#             md["chunk_id"] = md.get("chunk_id") or str(i)
#             page = str(md.get("page") or "0")
#             md["chunk_uid"] = md.get("chunk_uid") or f"{doc_id_str}:{page}:{md['chunk_id']}"
#             d.metadata = md

#         # ----- Index
#         t3 = time.time()
#         try:
#             _INDEX.set_namespace(tenant_ns)
#         except Exception:
#             pass

#         _INDEX.add_documents(chunks, _EMBEDDER)

#         # Update record with final chunk_count
#         try:
#             if db and doc_rec:
#                 doc_rec.chunk_count = len(chunks)
#                 db.commit()
#         except Exception:
#             pass
#         finally:
#             try:
#                 if db:
#                     db.close()
#             except Exception:
#                 pass

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
#                 "embed": embed_ms,  # if you later time encode_texts, set this properly
#                 "index": index_ms,
#             },
#         )

    
    
#     @app.post("/query", response_model=QueryResponse)
#     def query(body: QueryRequest, user: AuthedUser = Depends(require_bearer_auth)):
#         print(body)
#         tenant = user.tenant_id or "default"
#         session = body.session_id
#         try:
#             _INDEX.set_namespace(tenant)
#         except Exception:
#             pass

#         #1 Append User turn
#         try:
#             _MEMORY.append_turn(tenant,session,role="user", text=body.question)
#         except Exception:
#             pass

#         #2 Fetch recent history for prompt building
#         raw_history = _MEMORY.get_recent(tenant,session,limit=6)
#         history =[
#             HistoryTurn(role=h["role"],text=h["text"]) 
#             for h in raw_history if isinstance(h,dict) and "role" in h and "text" in h
#         ]

#         #3 Retrieve
#         results: List[Document] = RETRIEVER.search(
#         query=body.question,
#         k=body.k,
#         use_mmr=getattr(body, "use_mmr", False),
#         mmr_lambda=getattr(body, "mmr_lambda", 0.5),
#     )
#         _METRICS["queries_served_total"] += 1

#         sources: List[str] = []
#         for d in results:
#             src = d.metadata.get("source")
#             if isinstance(src, str):
#                 sources.append(src)

#         # 4) (Optional) Build prompt using B2 builder
#         chunks = [
#             PromptChunk(content=d.page_content, source=d.metadata.get("source", "unknown"), chunk_id=d.metadata.get("chunk_id", ""))
#             for d in results
#         ]

#         prompt_builder = PromptBuilder()
#         prompt_payload = prompt_builder.build(
#             inputs=PromptInputs(
#                 query=body.question,
#                 chunks=chunks,
#                 history=history,
#                 top_k=body.k,
#                 provider=(body.provider or "stub"),
#                 mmr_used=getattr(body, "use_mmr", False),
#             )
#         )
#         system_txt = prompt_payload["system"]
#         user_msg = prompt_payload["messages"][0]["content"]

#         #Choose provider
#         prov,model_used = _choose_free_llm(body.provider)
#         # print(f"The provider being used is {prov}")

#         answer_text = None
#         provider_used = prov
#         try:
#             if prov=="gemini":
#                 gm = _get_gemini()
#                 if gm is None:
#                     raise RuntimeError("Gemini Not Configured")
                
#                 gm_sys = genai.GenerativeModel(model_name=GEMINI_MODEL,system_instruction=system_txt)

#                 resp = gm_sys.generate_content(user_msg)
#                 # print(f"The response from the LLM is {resp}")
#                 answer_text = (resp.text or "").strip()
#             elif prov == "groq":
#                 gq = _get_groq()
#                 if gq is None:
#                     raise RuntimeError("Groq not configured")
#                 chat = gq.chat.completions.create(
#                     model=model_used,
#                     messages=[
#                         {"role": "system", "content": system_txt},
#                         {"role": "user", "content": user_msg},
#                     ],
#                     temperature=0.2,
#                 )
#                 answer_text = (chat.choices[0].message.content or "").strip()

#         except Exception as e:
#             # Fallback chain: try the other free provider, then stub
#             # print(f"Fell into the EXception block due to {e}")
#             if prov == "gemini" and GROQ_API_KEY:
#                 try:
#                     gq = _get_groq()
#                     chat = gq.chat.completions.create(
#                         model=GROQ_MODEL,
#                         messages=[
#                             {"role": "system", "content": system_txt},
#                             {"role": "user", "content": user_msg},
#                         ],
#                         temperature=0.2,
#                     )
#                     answer_text = (chat.choices[0].message.content or "").strip()
#                     provider_used = "groq"
#                     model_used = GROQ_MODEL
#                 except Exception:
#                     pass
#             elif prov == "groq" and GEMINI_API_KEY and answer_text is None:
#                 try:
#                     gm = _get_gemini()
#                     gm_sys = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system_txt)
#                     resp = gm_sys.generate_content(user_msg)
#                     answer_text = (resp.text or "").strip()
#                     provider_used = "gemini"
#                     model_used = GEMINI_MODEL
#                 except Exception:
#                     pass

#         # Final stub if both failed or not configured
#         if not answer_text:
#             answer_text = _stub_llm_answer(body.question, results)
#             provider_used = "stub"
#             model_used = "stub"
            
       

#         # # 5) Stub LLM
#         # answer = _stub_llm_answer(body.question, results)

#         # 6) Append assistant turn
#         try:
#             _MEMORY.append_turn(tenant, session, role="assistant", text=answer_text)
#         except Exception:
#             pass

#         return QueryResponse(answer=answer_text, sources=sources, provider=(body.provider or "stub"))


#     return app


# app = create_app()