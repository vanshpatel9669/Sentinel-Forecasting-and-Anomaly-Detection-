from app.anomaly import cusum_anomalies

def test_anomalies_returns_list():
    result = cusum_anomalies([100,101,99,100,150,101,99,98,97,60,100])
    assert isinstance(result, list)
