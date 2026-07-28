"""API request/response schemas — real Pydantic validation, replacing the
original implementation's unvalidated `List[float]` with no length or
range bounds.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SeriesRequest(BaseModel):
    values: list[float] = Field(min_length=5, max_length=5000)
    forecast_horizon: int = Field(default=5, ge=1, le=30)


class ForecastMethodResult(BaseModel):
    predicted: list[float]
    error_vs_recent_actual: dict[str, float] = Field(
        description="MAE/RMSE against the tail of the *input* series itself "
        "(an in-request sanity check, not a held-out benchmark — see "
        "scripts/evaluate_system.py for real held-out evaluation numbers)."
    )


class DriftReportResponse(BaseModel):
    drift_detected: bool
    baseline_std: float
    recent_std: float
    relative_std_change: float


class RecalibratedThresholdsResponse(BaseModel):
    cusum_threshold: float
    cusum_drift: float
    recalibrated: bool
    reason: str


class AnalyzeResponse(BaseModel):
    forecast_horizon: int
    forecasts: dict[str, ForecastMethodResult] = Field(
        description="Keyed by method: 'moving_average', 'trend', 'tf_model'."
    )
    anomaly_detection: dict[str, object]
    drift: DriftReportResponse
    thresholds_used: RecalibratedThresholdsResponse
    incident_risk: dict[str, object]
