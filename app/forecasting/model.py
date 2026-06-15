from typing import List, Dict
import numpy as np


def moving_average_forecast(values: List[float], window: int = 5, horizon: int = 5) -> List[float]:
    if not values:
        return []

    values_array = np.array(values, dtype=float)
    window = min(window, len(values_array))

    forecast_base = float(np.mean(values_array[-window:]))
    return [round(forecast_base, 2) for _ in range(horizon)]


def trend_forecast(values: List[float], horizon: int = 5) -> List[float]:
    if len(values) < 2:
        return moving_average_forecast(values, horizon=horizon)

    x = np.arange(len(values))
    y = np.array(values, dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(values), len(values) + horizon)
    forecast = slope * future_x + intercept

    return [round(float(v), 2) for v in forecast]


def forecast_error(actual: List[float], predicted: List[float]) -> Dict[str, float]:
    if not actual or not predicted:
        return {"mae": 0.0, "rmse": 0.0}

    n = min(len(actual), len(predicted))
    actual_array = np.array(actual[:n], dtype=float)
    predicted_array = np.array(predicted[:n], dtype=float)

    errors = actual_array - predicted_array

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


def generate_forecast(values: List[float], horizon: int = 5) -> Dict[str, object]:
    moving_average = moving_average_forecast(values, horizon=horizon)
    trend = trend_forecast(values, horizon=horizon)

    return {
        "moving_average_forecast": moving_average,
        "trend_forecast": trend,
        "forecast_horizon": horizon,
    }