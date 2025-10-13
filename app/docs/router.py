# app/docs/router.py
from __future__ import annotations
import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.auth.db import get_db
from app.auth.models import Membership, Tenant
from app.docs.models import DocumentRecord
from app.auth.security import JWT_SECRET  # or re-import how you handle secrets
import jwt
import os

router = APIRouter(prefix="/documents", tags=["documents"])
_bearer = HTTPBearer(auto_error=False)
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_FOR_PROD")

def _require_tenant_token(creds: HTTPAuthorizationCredentials) -> dict:
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        decoded = jwt.decode(
            creds.credentials,
            JWT_SECRET,
            algorithms=["HS256"],
            options={"require": ["exp", "sub"], "verify_iss": False, "verify_aud": False},
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    if "tenant_id" not in decoded:
        raise HTTPException(status_code=403, detail="Tenant token required")
    return decoded

@router.get("")
def list_documents(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
):
    decoded = _require_tenant_token(creds)
    tenant_id = uuid.UUID(decoded["tenant_id"])

    rows: List[DocumentRecord] = (
        db.query(DocumentRecord)
          .filter(DocumentRecord.tenant_id == tenant_id)
          .order_by(DocumentRecord.created_at.desc())
          .all()
    )

    return {
        "items": [
            {
                "id": str(r.id),
                "title": r.title,
                "source": r.source,
                "chunk_count": r.chunk_count,
                "created_at": r.created_at.isoformat() + "Z",
            }
            for r in rows
        ]
    }

@router.delete("/{doc_id}")
def delete_document(
    doc_id: str = Path(..., description="UUID of the document"),
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
):
    from api.main import _INDEX  # reuse the global index + namespace
    decoded = _require_tenant_token(creds)
    tenant_id = uuid.UUID(decoded["tenant_id"])

    try:
        did = uuid.UUID(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    rec: DocumentRecord | None = (
        db.query(DocumentRecord)
          .filter(DocumentRecord.id == did, DocumentRecord.tenant_id == tenant_id)
          .first()
    )
    if not rec:
        raise HTTPException(status_code=404, detail="Document not found")

    # Set Pinecone namespace and delete vectors by doc_id
    try:
        _INDEX.set_namespace(str(tenant_id))
    except Exception:
        pass
    _INDEX.delete_by_doc_id(str(rec.id))

    # Remove record
    db.delete(rec)
    db.commit()

    return {"deleted": True, "doc_id": str(did)}

@router.post("/reindex/{doc_id}")
def reindex_document(
    doc_id: str = Path(...),
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
):
    # Stub: we accept the request, future work will re-run the ingestion
    decoded = _require_tenant_token(creds)
    tenant_id = uuid.UUID(decoded["tenant_id"])

    try:
        did = uuid.UUID(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    rec = (
        db.query(DocumentRecord)
          .filter(DocumentRecord.id == did, DocumentRecord.tenant_id == tenant_id)
          .first()
    )
    if not rec:
        raise HTTPException(status_code=404, detail="Document not found")

    # TODO: implement actual reindex flow (read from 'source', re-ingest)
    return {"accepted": True, "doc_id": str(did)}