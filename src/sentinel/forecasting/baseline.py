"""Non-ML forecasting baselines: cheap, dependency-light, and the
comparison point the TensorFlow forecaster (`tf_model.py`) is actually
benchmarked against in `scripts/evaluate_system.py` — a claim that an ML
forecaster is worthwhile only means something next to a real baseline.
"""

from __future__ import annotations

import numpy as np


def moving_average_forecast(values: list[float], window: int = 5, horizon: int = 5) -> list[float]:
    if not values:
        return []
    values_array = np.array(values, dtype=float)
    window = min(window, len(values_array))
    forecast_base = float(np.mean(values_array[-window:]))
    return [round(forecast_base, 2) for _ in range(horizon)]


def trend_forecast(values: list[float], horizon: int = 5) -> list[float]:
    if len(values) < 2:
        return moving_average_forecast(values, horizon=horizon)
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(values), len(values) + horizon)
    forecast = slope * future_x + intercept
    return [round(float(v), 2) for v in forecast]


def forecast_error(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if not actual or not predicted:
        return {"mae": 0.0, "rmse": 0.0}
    n = min(len(actual), len(predicted))
    actual_array = np.array(actual[:n], dtype=float)
    predicted_array = np.array(predicted[:n], dtype=float)
    errors = actual_array - predicted_array
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}
