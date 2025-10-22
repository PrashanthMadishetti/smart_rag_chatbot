# tests/api/test_health.py
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"