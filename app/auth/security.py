# app/auth/security.py
import os
import time
import bcrypt
import jwt
from typing import Dict, Any

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_FOR_PROD")
JWT_ISSUER = os.getenv("JWT_ISSUER", "smart-rag")
ACCESS_TTL = int(os.getenv("ACCESS_TTL_SECONDS", "3600"))   # 1 hour default
LOGIN_TTL  = int(os.getenv("LOGIN_TTL_SECONDS", "900"))     # 15 minutes default

def hash_password(plain: str) -> str:
    rounds = int(os.getenv("BCRYPT_ROUNDS", "12"))
    salt = bcrypt.gensalt(rounds)
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def _jwt(payload: Dict[str, Any], ttl: int) -> str:
    now = int(time.time())
    body = {
        "iss": JWT_ISSUER,
        "iat": now,
        "exp": now + ttl,
        **payload,
    }
    return jwt.encode(body, JWT_SECRET, algorithm="HS256")

def issue_tenant_token(*, user_id: str, tenant_id: str, role: str) -> str:
    """
    Tenant-scoped access token. Includes 'scope': 'tenant' so the client can distinguish.
    """
    return _jwt(
        {"sub": user_id, "tenant_id": tenant_id, "role": role, "scope": "tenant"},
        ACCESS_TTL,
    )

def issue_login_token(*, user_id: str) -> str:
    """
    Login-scoped token (no tenant_id). Used to enumerate/switch tenants.
    """
    return _jwt({"sub": user_id, "scope": "login"}, LOGIN_TTL)

# # api/security.py
# import os
# import time
# import bcrypt
# import jwt
# from typing import Optional, Dict, Any

# JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_FOR_PROD")
# JWT_ISSUER = os.getenv("JWT_ISSUER", "smart-rag")
# ACCESS_TTL = int(os.getenv("ACCESS_TTL_SECONDS", "3600"))
# LOGIN_TTL = int(os.getenv("LOGIN_TTL_SECONDS", "900"))

# def hash_password(plain: str) -> str:
#     rounds = int(os.getenv("PASSWORD_HASH_ROUNDS", "12"))
#     salt = bcrypt.gensalt(rounds)
#     return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")

# def verify_password(plain: str, hashed: str) -> bool:
#     try:
#         return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
#     except Exception:
#         return False

# def _jwt(payload: Dict[str, Any], ttl: int) -> str:
#     now = int(time.time())
#     body = {
#         "iss": JWT_ISSUER,
#         "iat": now,
#         "exp": now + ttl,
#         **payload,
#     }
#     return jwt.encode(body, JWT_SECRET, algorithm="HS256")

# def issue_tenant_token(*, user_id: str, tenant_id: str, role: str) -> str:
#     return _jwt({"sub": user_id, "tenant_id": tenant_id, "role": role}, ACCESS_TTL)

# def issue_login_token(*, user_id: str) -> str:
#     # no tenant_id attached
#     return _jwt({"sub": user_id, "scope": "login"}, LOGIN_TTL)