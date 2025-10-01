# app/auth/auth_router.py
from __future__ import annotations

import os
import uuid
from typing import Optional, Dict, Any, List

import jwt
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth.db import get_db
from app.auth.models import User, Tenant, Membership
from app.auth.security import (
    hash_password,
    verify_password,
    issue_login_token,
    issue_tenant_token,
)
from app.auth.utils_slug import slugify

router = APIRouter(tags=["auth"])

bearer = HTTPBearer(auto_error=False)


# ---------- Helpers ----------

def _normalize_tenant_id_or_slug(db: Session, value: str) -> Optional[str]:
    """
    Accept either a tenant UUID (as string) or a slug (case-insensitive).
    Return canonical tenant_id as string, or None if not found.
    """
    if not value:
        return None

    # Try UUID first
    try:
        uid = uuid.UUID(value)
        t = db.query(Tenant).filter(Tenant.id == uid).first()
        if t:
            return str(t.id)
    except ValueError:
        pass  # not a UUID → fall through to slug

    # Slug (case-insensitive)
    t = db.query(Tenant).filter(func.lower(Tenant.slug) == value.lower()).first()
    return str(t.id) if t else None


def _require_login_token(creds: HTTPAuthorizationCredentials) -> dict:
    """
    Validate that the bearer token is a login token (scope=login).
    Returns decoded payload on success, raises HTTPException otherwise.
    """
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = creds.credentials
    try:
        decoded = jwt.decode(
            token,
            os.getenv("JWT_SECRET", "CHANGE_ME_FOR_PROD"),
            algorithms=["HS256"],
            options={"require": ["exp", "sub"], "verify_iss": False, "verify_aud": False},
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    if decoded.get("scope") != "login":
        raise HTTPException(status_code=400, detail="Not a login token")

    return decoded


# ---------- Endpoints ----------

@router.post("/auth/signup")
def signup(payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """
    payload:
      email, password,
      mode: "create_tenant" | "join_tenant"
      tenant_name (if create), tenant_id (if join)  [tenant_id or slug]
    Returns: { token, tenant_id, role }
    """
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    mode = (payload.get("mode") or "").strip()

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.flush()  # assign user.id

    if mode == "create_tenant":
        tname = (payload.get("tenant_name") or "").strip()
        if not tname:
            raise HTTPException(status_code=400, detail="tenant_name required for create_tenant")

        # generate unique slug
        slug = slugify(tname)
        base = slug
        i = 1
        while db.query(Tenant).filter(Tenant.slug == slug).first():
            i += 1
            slug = f"{base}-{i}"

        tenant = Tenant(name=tname, slug=slug)
        db.add(tenant)
        db.flush()

        ms = Membership(user_id=user.id, tenant_id=tenant.id, role="owner")
        db.add(ms)
        db.commit()

        token = issue_tenant_token(user_id=str(user.id), tenant_id=str(tenant.id), role="owner")
        return {"token": token, "tenant_id": str(tenant.id), "role": "owner"}

    elif mode == "join_tenant":
        tid_raw = (payload.get("tenant_id") or "").strip()
        if not tid_raw:
            raise HTTPException(status_code=400, detail="tenant_id (or slug) required for join_tenant")

        tenant_id_str = _normalize_tenant_id_or_slug(db, tid_raw)
        if not tenant_id_str:
            raise HTTPException(status_code=404, detail="Tenant not found")

        ms = Membership(user_id=user.id, tenant_id=tenant_id_str, role="member")
        db.add(ms)
        db.commit()

        token = issue_tenant_token(user_id=str(user.id), tenant_id=str(tenant_id_str), role="member")
        return {"token": token, "tenant_id": tenant_id_str, "role": "member"}

    else:
        db.rollback()
        raise HTTPException(status_code=400, detail="Invalid mode. Use create_tenant or join_tenant.")


@router.post("/auth/login")
def login(payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """
    payload: email, password, optional tenant_id (or slug) to directly mint a tenant token.
    Returns either:
      { token, tenant_id, role }                         # if tenant chosen & membership valid
      { login_token, memberships: [{tenant_id, tenant_name, role}] }  # otherwise
    """
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user = db.query(User).filter(User.email == email, User.is_active == True).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Fast path: login directly to a tenant if requested
    desired = (payload.get("tenant_id") or "").strip()
    if desired:
        tenant_id_str = _normalize_tenant_id_or_slug(db, desired)
        if not tenant_id_str:
            raise HTTPException(status_code=404, detail="Tenant not found")

        tenant_id = uuid.UUID(tenant_id_str)
        ms = db.query(Membership).filter(
            Membership.user_id == user.id,
            Membership.tenant_id == tenant_id
        ).first()
        if not ms:
            raise HTTPException(status_code=403, detail="Not a member of requested tenant")

        token = issue_tenant_token(user_id=str(user.id), tenant_id=str(tenant_id), role=ms.role)
        return {"token": token, "tenant_id": str(tenant_id), "role": ms.role}

    # Default: return login token and memberships list; client will call /auth/switch-tenant
    memberships = (
        db.query(Membership, Tenant)
        .join(Tenant, Tenant.id == Membership.tenant_id)
        .filter(Membership.user_id == user.id)
        .all()
    )
    out = [{"tenant_id": str(t.id), "tenant_name": t.name, "role": ms.role} for ms, t in memberships]

    return {"login_token": issue_login_token(user_id=str(user.id)), "memberships": out}


@router.get("/auth/tenants/search")
def search_tenants(q: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    """
    Simple search endpoint used by the UI during sign-up join flow.
    Returns up to 20 tenants whose name or slug contains the query (case-insensitive).
    """
    q_like = f"%{q.lower()}%"
    rows: List[Tenant] = (
        db.query(Tenant)
        .filter(
            func.lower(Tenant.name).like(q_like) |
            func.lower(Tenant.slug).like(q_like)
        )
        .order_by(Tenant.name.asc())
        .limit(20)
        .all()
    )
    return {
        "tenants": [{"id": str(t.id), "name": t.name, "slug": t.slug} for t in rows]
    }


@router.post("/auth/switch-tenant")
def switch_tenant(
    payload: Dict[str, Any] = Body(...),
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
):
    """
    Exchange a login_token (no tenant scope) for a tenant-scoped token.
    The client must send the login_token as Bearer auth.
    Body: { "tenant_id": "<uuid or slug>" }
    Returns: { token, tenant_id, role }
    """
    decoded = _require_login_token(creds)

    # Parse user id from login token
    try:
        user_id = uuid.UUID(decoded["sub"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id in token")

    tid_raw = (payload.get("tenant_id") or "").strip()
    if not tid_raw:
        raise HTTPException(status_code=400, detail="tenant_id required")

    tenant_id_str = _normalize_tenant_id_or_slug(db, tid_raw)
    if not tenant_id_str:
        raise HTTPException(status_code=404, detail="Tenant not found")

    tenant_id = uuid.UUID(tenant_id_str)

    # Verify membership
    ms = (
        db.query(Membership)
        .filter(Membership.user_id == user_id, Membership.tenant_id == tenant_id)
        .first()
    )
    if not ms:
        raise HTTPException(status_code=403, detail="Not a member of requested tenant")

    new_token = issue_tenant_token(user_id=str(user_id), tenant_id=str(tenant_id), role=ms.role)
    return {"token": new_token, "tenant_id": str(tenant_id), "role": ms.role}

# # api/auth_router.py
# from fastapi import APIRouter, Depends, HTTPException, Body, Query
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from sqlalchemy.orm import Session
# from sqlalchemy import or_, func
# from typing import Optional, List, Dict, Any
# from app.auth.db import get_db
# from app.auth.models import User, Tenant, Membership
# from app.auth.security import hash_password, verify_password, issue_login_token, issue_tenant_token
# from app.auth.utils_slug import slugify
# import jwt
# import os

# router = APIRouter(tags=["auth"])

# # ---------- Pydantic-ish request bodies (use dict for brevity) ----------
# import uuid
# from sqlalchemy import func

# def _normalize_tenant_id_or_slug(db: Session, value: str) -> Optional[str]:
#     if not value:
#         return None

#     # Try UUID first
#     try:
#         uid = uuid.UUID(value)
#         t = db.query(Tenant).filter(Tenant.id == uid).first()
#         if t:
#             return str(t.id)
#     except ValueError:
#         pass  # not a UUID, fall through to slug

#     # Fallback: slug (case-insensitive)
#     t = db.query(Tenant).filter(func.lower(Tenant.slug) == value.lower()).first()
#     return str(t.id) if t else None

# @router.post("/auth/signup")
# def signup(payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
#     """
#     payload:
#       email, password,
#       mode: "create_tenant" | "join_tenant"
#       tenant_name (if create), tenant_id (if join)  [tenant_id or slug]
#     """
#     email = (payload.get("email") or "").strip().lower()
#     password = payload.get("password") or ""
#     mode = (payload.get("mode") or "").strip()
#     if not email or not password:
#         raise HTTPException(status_code=400, detail="Email and password required")

#     existing = db.query(User).filter(User.email == email).first()
#     if existing:
#         raise HTTPException(status_code=400, detail="Email already registered")

#     user = User(email=email, password_hash=hash_password(password))
#     db.add(user)
#     db.flush()  # get user.id

#     if mode == "create_tenant":
#         tname = (payload.get("tenant_name") or "").strip()
#         if not tname:
#             raise HTTPException(status_code=400, detail="tenant_name required for create_tenant")
#         slug = slugify(tname)
#         # ensure unique slug (simple de-dupe)
#         base = slug
#         i = 1
#         while db.query(Tenant).filter(Tenant.slug == slug).first():
#             i += 1
#             slug = f"{base}-{i}"
#         tenant = Tenant(name=tname, slug=slug)
#         db.add(tenant)
#         db.flush()

#         ms = Membership(user_id=user.id, tenant_id=tenant.id, role="owner")
#         db.add(ms)
#         db.commit()
#         token = issue_tenant_token(user_id=str(user.id), tenant_id=str(tenant.id), role="owner")
#         return {"token": token, "tenant_id": str(tenant.id), "role": "owner"}

#     elif mode == "join_tenant":
#         tid_raw = (payload.get("tenant_id") or "").strip()
#         if not tid_raw:
#             raise HTTPException(status_code=400, detail="tenant_id (or slug) required for join_tenant")
#         tenant_id = _normalize_tenant_id_or_slug(db, tid_raw)
#         if not tenant_id:
#             raise HTTPException(status_code=404, detail="Tenant not found")
#         # create membership as member
#         ms = Membership(user_id=user.id, tenant_id=tenant_id, role="member")
#         db.add(ms)
#         db.commit()
#         token = issue_tenant_token(user_id=str(user.id), tenant_id=str(tenant_id), role="member")
#         return {"token": token, "tenant_id": tenant_id, "role": "member"}

#     else:
#         db.rollback()
#         raise HTTPException(status_code=400, detail="Invalid mode. Use create_tenant or join_tenant.")

# @router.post("/auth/login")
# def login(payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
#     """
#     payload: email, password, optional tenant_id (or slug) to directly mint a tenant token.
#     Returns either:
#       { token, tenant_id, role }  OR
#       { login_token, memberships: [{tenant_id, tenant_name, role}] }
#     """
#     email = (payload.get("email") or "").strip().lower()
#     password = payload.get("password") or ""
#     if not email or not password:
#         raise HTTPException(status_code=400, detail="Email and password required")

#     user = db.query(User).filter(User.email == email, User.is_active == True).first()
#     if not user or not verify_password(password, user.password_hash):
#         raise HTTPException(status_code=401, detail="Invalid credentials")

#     desired = (payload.get("tenant_id") or "").strip()
#     if desired:
#         tenant_id = _normalize_tenant_id_or_slug(db, desired)
#         if tenant_id:
#             ms = db.query(Membership).filter(Membership.user_id == user.id, Membership.tenant_id == tenant_id).first()
#             if ms:
#                 token = issue_tenant_token(user_id=str(user.id), tenant_id=str(tenant_id), role=ms.role)
#                 return {"token": token, "tenant_id": tenant_id, "role": ms.role}
#         raise HTTPException(status_code=403, detail="Not a member of requested tenant")

#     # No tenant chosen yet → return login_token and memberships so client can pick
#     memberships = (
#         db.query(Membership, Tenant)
#         .join(Tenant, Tenant.id == Membership.tenant_id)
#         .filter(Membership.user_id == user.id)
#         .all()
#     )
#     out = []
#     for ms, t in memberships:
#         out.append({"tenant_id": str(t.id), "tenant_name": t.name, "role": ms.role})
#     return {"login_token": issue_login_token(user_id=str(user.id)), "memberships": out}

# @router.post("/auth/switch-tenant")
# def switch_tenant(
#     payload: Dict[str, Any] = Body(...),
#     db: Session = Depends(get_db),
#     credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=True)),
# ):
#     """
#     Exchange a *login token* (scope=login) for a tenant-scoped JWT.
#     - Send the login token in the Authorization header: Bearer <login_token>
#     - Body: { "tenant_id": "<uuid-or-slug>" }
#     Returns: { "token": "<tenant_jwt>", "tenant_id": "...", "role": "..." }
#     """
#     raw_token = credentials.credentials
#     try:
#         decoded = jwt.decode(
#             raw_token,
#             os.getenv("JWT_SECRET", "CHANGE_ME_FOR_PROD"),
#             algorithms=["HS256"],
#             options={"require": ["exp", "sub"], "verify_iss": False, "verify_aud": False},
#         )
#     except Exception:
#         raise HTTPException(status_code=401, detail="Invalid token")

#     if decoded.get("scope") != "login":
#         raise HTTPException(status_code=400, detail="Not a login token")

#     user_id = decoded["sub"]
#     tid_raw = (payload.get("tenant_id") or "").strip()
#     if not tid_raw:
#         raise HTTPException(status_code=400, detail="tenant_id required")

#     tenant_id = _normalize_tenant_id_or_slug(db, tid_raw)
#     if not tenant_id:
#         raise HTTPException(status_code=404, detail="Tenant not found")

#     ms = (
#         db.query(Membership)
#         .filter(Membership.user_id == user_id, Membership.tenant_id == tenant_id)
#         .first()
#     )
#     if not ms:
#         raise HTTPException(status_code=403, detail="Not a member of requested tenant")

#     new_token = issue_tenant_token(user_id=str(user_id), tenant_id=str(tenant_id), role=ms.role)
#     return {"token": new_token, "tenant_id": tenant_id, "role": ms.role}