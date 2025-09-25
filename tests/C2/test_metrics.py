# tests/api/test_metrics.py
def test_metrics_requires_auth(client):
    r = client.get("/metrics")
    assert r.status_code in (401, 403)

def test_metrics_with_auth(client, auth_headers):
    r = client.get("/metrics", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    # basic shape
    assert "requests_total" in data
    assert "ingest_docs_total" in data
    assert "queries_served_total" in data