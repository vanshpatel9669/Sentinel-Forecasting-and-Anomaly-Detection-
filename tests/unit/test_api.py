"""API tests. `get_tf_forecaster` is overridden with the session-scoped
real (tiny, fast-trained) forecaster from conftest.py — not a mock.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sentinel.api.dependencies import get_tf_forecaster
from sentinel.api.main import app


@pytest.fixture
def client(trained_forecaster):
    app.dependency_overrides[get_tf_forecaster] = lambda: trained_forecaster
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _series(n: int = 60) -> list[float]:
    return (100 + 8 * np.sin(np.arange(n) / 4)).tolist()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_analyze_returns_all_three_forecast_methods(client):
    response = client.post("/v1/analyze", json={"values": _series(), "forecast_horizon": 5})
    assert response.status_code == 200
    body = response.json()
    assert set(body["forecasts"]) == {"moving_average", "trend", "tf_model"}
    assert "anomaly_detection" in body
    assert "drift" in body
    assert "thresholds_used" in body
    assert "incident_risk" in body


def test_analyze_with_injected_spike_flags_anomaly(client):
    values = _series(40)
    values[20] = 500.0
    response = client.post("/v1/analyze", json={"values": values, "forecast_horizon": 5})
    body = response.json()
    assert body["anomaly_detection"]["anomalies_detected"] >= 1


def test_analyze_rejects_too_short_series(client):
    response = client.post("/v1/analyze", json={"values": [1.0, 2.0], "forecast_horizon": 5})
    assert response.status_code == 422


def test_analyze_rejects_out_of_range_horizon(client):
    response = client.post("/v1/analyze", json={"values": _series(), "forecast_horizon": 100})
    assert response.status_code == 422


def test_metrics_endpoint_is_real_prometheus_not_fabricated_json(client):
    """Regression guard: the previous version of this project's /metrics
    endpoint returned hardcoded literal strings like "35%" as JSON. This
    asserts /metrics now serves real Prometheus exposition text instead.
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "35%" not in response.text
    assert "sentinel_analyses_total" in response.text


def test_analyses_counter_increments(client):
    before = client.get("/metrics").text
    client.post("/v1/analyze", json={"values": _series(), "forecast_horizon": 5})
    after = client.get("/metrics").text
    assert after != before
