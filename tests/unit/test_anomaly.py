from sentinel.anomaly.detector import (
    cusum_detection,
    detect_anomalies,
    severity_score,
    z_score_anomalies,
)


def test_z_score_detects_obvious_outlier():
    values = [100.0] * 20 + [500.0] + [100.0] * 5
    anomalies = z_score_anomalies(values, threshold=2.5)
    assert any(a["index"] == 20 for a in anomalies)


def test_z_score_no_anomalies_in_flat_series():
    assert z_score_anomalies([100.0] * 30, threshold=2.5) == []


def test_z_score_requires_at_least_two_points():
    assert z_score_anomalies([100.0], threshold=2.5) == []


def test_cusum_detects_sustained_shift():
    values = [100.0] * 20 + [130.0] * 20
    result = cusum_detection(values, threshold=8.0, drift=0.5)
    assert result["cusum_signal"] is True


def test_cusum_no_signal_on_stable_series():
    values = [100.0, 101.0, 99.0, 100.0, 100.5, 99.5] * 4
    result = cusum_detection(values, threshold=8.0, drift=0.5)
    assert result["cusum_signal"] is False


def test_severity_score_increases_with_anomaly_count():
    low = severity_score(anomaly_count=0, cusum_signal=False, values=[100.0] * 10)
    high = severity_score(anomaly_count=3, cusum_signal=True, values=[100.0] * 10)
    assert high["severity_score"] > low["severity_score"]
    assert high["risk_level"] == "high"


def test_severity_score_empty_values():
    result = severity_score(anomaly_count=0, cusum_signal=False, values=[])
    assert result == {"severity_score": 0, "risk_level": "low"}


def test_detect_anomalies_combines_z_and_cusum():
    values = [100.0] * 20 + [500.0] + [100.0] * 5
    result = detect_anomalies(values)
    assert result["anomalies_detected"] >= 1
    assert "cusum" in result
    assert "severity" in result
