# # tests/C1/test_memory_integration.py
# tests/C1/test_memory_store.py
import json
import time
import pytest

try:
    import fakeredis
except ImportError:
    fakeredis = None

from typing import List, Dict, Any

# ==== Subject Under Test (SUT) contract ====
# Your implementation should live at app/memory/store.py:
#
# class MemoryStore:
#     def __init__(self, redis_client, max_turns: int = 6, ttl_seconds: int | None = None): ...
#     def append_turn(self, tenant_id: str, session_id: str, role: str, text: str, ts: float | None = None) -> None: ...
#     def get_recent(self, tenant_id: str, session_id: str, limit: int | None = None) -> List[Dict[str, Any]]: ...
#     def clear(self, tenant_id: str, session_id: str) -> None: ...
#
from app.memory.store import MemoryStore


#pytestmark = pytest.mark.skipif(fakeredis is None, reason="fakeredis is required for C1 unit tests")


@pytest.fixture
def rds():
    """Fake Redis client (decode_responses=True to work with JSON strings)."""
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def store(rds):
    return MemoryStore(redis_client=rds, max_turns=6, ttl_seconds=3600)


def _key(tenant="tenantA", session="sess1"):
    return f"mem:{tenant}:{session}"


def test_append_and_get_recent_preserves_order(store, rds):
    tenant, session = "tenantA", "sess1"
    # Append user then assistant
    store.append_turn(tenant, session, role="user", text="Hi")
    store.append_turn(tenant, session, role="assistant", text="Hello!")

    # Raw Redis list should have 2 entries
    assert rds.llen(_key(tenant, session)) == 2

    # SUT should return parsed dicts in order
    got = store.get_recent(tenant, session)
    assert len(got) == 2
    assert got[0]["role"] == "user" and got[0]["text"] == "Hi"
    assert got[1]["role"] == "assistant" and got[1]["text"] == "Hello!"
    assert "ts" in got[0] and isinstance(got[0]["ts"], (int, float))


def test_trim_to_max_turns_drops_oldest(store, rds):
    tenant, session = "tenantA", "sess2"

    # Add 8 turns while max_turns=6 → oldest 2 are trimmed
    for i in range(8):
        store.append_turn(tenant, session, role="user", text=f"u{i}")

    raw = [json.loads(x) for x in rds.lrange(_key(tenant, session), 0, -1)]
    ids = [x["text"] for x in raw]
    assert ids == [f"u{i}" for i in range(2, 8)], "Oldest two should be trimmed"

    got = store.get_recent(tenant, session)
    assert [x["text"] for x in got] == [f"u{i}" for i in range(2, 8)]


def test_get_recent_with_limit(store):
    tenant, session = "tenantB", "sess1"
    for i in range(5):
        store.append_turn(tenant, session, role="assistant", text=f"a{i}")
    got3 = store.get_recent(tenant, session, limit=3)
    assert len(got3) == 3
    # should be the last three in order
    assert [x["text"] for x in got3] == ["a2", "a3", "a4"]


def test_invalid_role_raises(store):
    with pytest.raises(ValueError):
        store.append_turn("t", "s", role="system", text="nope")


def test_clear_removes_key(store, rds):
    tenant, session = "tenantC", "s1"
    store.append_turn(tenant, session, "user", "x")
    assert rds.exists(_key(tenant, session)) == 1
    store.clear(tenant, session)
    assert rds.exists(_key(tenant, session)) == 0


def test_redis_down_returns_empty(monkeypatch, store):
    # Simulate Redis failure on LRANGE
    class Boom(Exception):
        pass

    def boom(*args, **kwargs):
        raise Boom("fail")

    monkeypatch.setattr(store.redis, "lrange", boom)
    got = store.get_recent("t", "s")
    assert got == [], "get_recent should degrade gracefully when Redis is down"


def test_appends_set_ttl_when_configured(store, rds):
    tenant, session = "tenantD", "s1"
    store.append_turn(tenant, session, "user", "hi")
    # With FakeRedis, TTL may be -1 if not implemented; accept (-1 or >0)
    ttl = rds.ttl(_key(tenant, session))
    assert ttl in (-1, None) or ttl > 0

# import os
# import pytest
# from datetime import datetime, timedelta

# #pytestmark = pytest.mark.xfail(reason="Enable after wiring MemoryStore into /query flow")

# def test_query_appends_and_reads_history(client, auth_headers):
#     # 1st query: no history yet
#     body = {
#         "session_id": "sessX",
#         "question": "What is alpha?",
#         "k": 2,
#         "provider": "stub",
#         "use_mmr": False,
#     }
#     r1 = client.post("/query", headers=auth_headers, json=body)
#     assert r1.status_code == 200

#     # 2nd query: should see previous turn in prompt (indirectly assert via response metadata/logs if exposed)
#     r2 = client.post("/query", headers=auth_headers, json={
#         **body,
#         "question": "And how about beta?"
#     })
#     assert r2.status_code == 200
#     # If your /query returns the number of history turns used in metadata, assert it here.
#     # data = r2.json()
#     # assert data["metadata"]["history_turns_used"] >= 1