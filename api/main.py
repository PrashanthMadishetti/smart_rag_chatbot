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
from app.retrieval.retriever import Retriever


import redis
from app.memory.store import MemoryStore
from app.prompt.builder import HistoryTurn, PromptBuilder,PromptChunk,PromptInputs




# =========================
# App-wide singletons
# =========================
JWT_SECRET = os.getenv("JWT_SECRET", "JWT_SECRET")
EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/indexes/faiss")
REDIS_URL = os.getenv("REDIS_URL","redis://redis:6379/0")

# Embedder / Index
_EMBEDDER = Embedder(model_name=EMBED_MODEL)
_INDEX = FaissIndex(dimension=_EMBEDDER.dimension, metric="cosine", model_name=_EMBEDDER.model_name)
_RETRIEVER = Retriever(_INDEX, _EMBEDDER)

# Metrics (in-memory)
_METRICS = {
    "requests_total": 0,
    "ingest_docs_total": 0,
    "ingest_chunks_total": 0,
    "queries_served_total": 0,
}

#=========================
# Redis and Memory
#=========================
_redis = redis.from_url(REDIS_URL,decode_response=True)
_MEMORY = MemoryStore(redis_client=redis, max_turns=6, ttl_seconds=int(os.getenv("MEMORY_TTL_SECONDS", "0" or 0)) or None)

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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token")


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
    # @app.post("/query", response_model=QueryResponse)
    # def query(body: QueryRequest, _: AuthedUser = Depends(require_bearer_auth)):
    #     results: List[Document] = _INDEX.search(body.question, k=body.k, embedder=_EMBEDDER)
    #     _METRICS["queries_served_total"] += 1

    #     sources: List[str] = []
    #     for d in results:
    #         src = d.metadata.get("source")
    #         if isinstance(src, str):
    #             sources.append(src)

    #     answer = _stub_llm_answer(body.question, results)
    #     return QueryResponse(answer=answer, sources=sources, provider=(body.provider or "stub"))
    
    
    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, user: AuthedUser = Depends(require_bearer_auth)):
        tenant = user.tenant_id or "default"
        session = body.session_id

        #1 Append User turn
        try:
            _MEMORY.append_turn(tenant,session,role="user", text=body.question)
        except Exception:
            pass

        #2 Fetch recent history for prompt building
        raw_history = _MEMORY.get_recent(tenant,session,limit=6)
        history =[
            HistoryTurn(role=h["role"],text=h["text"]) 
            for h in raw_history if isinstance(h,dict) and "role" in h and "text" in h
        ]

        #3 Retrieve
        results: List[Document] = _RETRIEVER.search(
        query=body.question,
        k=body.k,
        use_mmr=getattr(body, "use_mmr", False),
        mmr_lambda=getattr(body, "mmr_lambda", 0.5),
    )
        _METRICS["queries_served_total"] += 1

        sources: List[str] = []
        for d in results:
            src = d.metadata.get("source")
            if isinstance(src, str):
                sources.append(src)

        # 4) (Optional) Build prompt using B2 builder
        chunks = [
            PromptChunk(content=d.page_content, source=d.metadata.get("source", "unknown"), chunk_id=d.metadata.get("chunk_id", ""))
            for d in results
        ]
        _ = PromptBuilder.build(
            PromptInputs(
                query=body.question,
                chunks=chunks,
                history=history,
                top_k=body.k,
                provider=(body.provider or "stub"),
                mmr_used=getattr(body, "use_mmr", False),
            )
        )
        # If you want to pass the built prompt to a real LLM later, it's now ready.

        # 5) Stub LLM
        answer = _stub_llm_answer(body.question, results)

        # 6) Append assistant turn
        try:
            _MEMORY.append_turn(tenant, session, role="assistant", text=answer)
        except Exception:
            pass

        return QueryResponse(answer=answer, sources=sources, provider=(body.provider or "stub"))


    return app


app = create_app()