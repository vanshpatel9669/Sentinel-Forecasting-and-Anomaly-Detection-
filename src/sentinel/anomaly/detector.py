"""Statistical anomaly detection: Z-score (pointwise outliers) and CUSUM
(sustained small shifts a pointwise test would miss). Thresholds default
to `config.Settings` values but are accepted as parameters so
`drift.recalibration` can supply recalibrated ones per request.
"""

from __future__ import annotations

import numpy as np


def z_score_anomalies(values: list[float], threshold: float = 2.5) -> list[dict[str, object]]:
    values_array = np.array(values, dtype=float)
    if len(values_array) < 2:
        return []
    mean = np.mean(values_array)
    std = np.std(values_array)
    if std == 0:
        return []

    anomalies = []
    for index, value in enumerate(values_array):
        z_score = (value - mean) / std
        if abs(z_score) >= threshold:
            anomalies.append(
                {
                    "index": index,
                    "value": round(float(value), 2),
                    "z_score": round(float(z_score), 4),
                    "method": "z_score",
                }
            )
    return anomalies


def cusum_detection(
    values: list[float], threshold: float = 8.0, drift: float = 0.5
) -> dict[str, object]:
    values_array = np.array(values, dtype=float)
    if len(values_array) < 2:
        return {"cusum_signal": False, "positive_cusum": 0.0, "negative_cusum": 0.0}

    mean = np.mean(values_array)
    positive_sum = 0.0
    negative_sum = 0.0
    signal = False

    for value in values_array:
        deviation = value - mean
        positive_sum = max(0.0, positive_sum + deviation - drift)
        negative_sum = min(0.0, negative_sum + deviation + drift)
        if positive_sum > threshold or abs(negative_sum) > threshold:
            signal = True

    return {
        "cusum_signal": signal,
        "positive_cusum": round(float(positive_sum), 4),
        "negative_cusum": round(float(negative_sum), 4),
    }


def severity_score(
    anomaly_count: int, cusum_signal: bool, values: list[float]
) -> dict[str, object]:
    if not values:
        return {"severity_score": 0, "risk_level": "low"}

    volatility = float(np.std(values))
    score = anomaly_count * 20 + volatility * 0.5
    if cusum_signal:
        score += 30
    score = min(100, round(score, 2))

    if score >= 70:
        risk_level = "high"
    elif score >= 35:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {"severity_score": score, "risk_level": risk_level}


def detect_anomalies(
    values: list[float],
    z_threshold: float = 2.5,
    cusum_threshold: float = 8.0,
    cusum_drift: float = 0.5,
) -> dict[str, object]:
    z_anomalies = z_score_anomalies(values, threshold=z_threshold)
    cusum_result = cusum_detection(values, threshold=cusum_threshold, drift=cusum_drift)
    severity = severity_score(
        anomaly_count=len(z_anomalies),
        cusum_signal=bool(cusum_result["cusum_signal"]),
        values=values,
    )
    return {
        "anomalies": z_anomalies,
        "anomalies_detected": len(z_anomalies),
        "cusum": cusum_result,
        "severity": severity,
    }
