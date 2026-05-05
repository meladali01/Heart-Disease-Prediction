from src.app import app


def test_metrics_endpoint(monkeypatch):
    client = app.test_client()

    # call health and predict (predict requires model; use health which doesn't)
    r = client.get("/health")
    assert r.status_code == 200

    # metrics endpoint should be available and contain prometheus metrics
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.get_data(as_text=True)
    assert "http_requests_total" in text
    assert "http_request_latency_seconds" in text
