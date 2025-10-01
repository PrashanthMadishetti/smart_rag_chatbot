# tests/D1/conftest.py
import os
import time
import jwt
import pytest
from fastapi.testclient import TestClient

# IMPORTANT: your FastAPI app factory (adjust import if your path differs)
from api.main import create_app

JWT_SECRET = os.getenv("JWT_SECRET", "JWT_SECRET")  # matches app default
ALGO = "HS256"


def _make_token(sub="test-user", tenant_id="tenantA", ttl_seconds=3600):
    now = int(time.time())
    payload = {
        "sub": sub,
        "tenant_id": tenant_id,
        "iat": now,
        "exp": now + ttl_seconds,
        # iss/aud are NOT verified (verify_iss/aud disabled in app)
        "iss": "tests",
        "aud": "tests",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGO)


@pytest.fixture(scope="session")
def app():
    # FastAPI app instance
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    return TestClient(app)


@pytest.fixture
def auth_headers():
    token = _make_token()
    return {"Authorization": f"Bearer {token}"}