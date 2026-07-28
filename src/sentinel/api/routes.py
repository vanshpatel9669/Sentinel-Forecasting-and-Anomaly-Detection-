"""HTTP routes — thin; forecasting/anomaly/drift/monitoring logic all live
in their own modules.

`analyses_total` / `alerts_total` are the real operational-monitoring
signal this service exposes at `GET /metrics` (Prometheus) — genuine,
incrementing counters, replacing the previous version's `/metrics`
endpoint, which returned hardcoded literal strings
(`"incident_response_time_reduction": "35%"`) on every call regardless of
what actually happened. See `monitoring/service.py` and
docs/architecture.md for the full account.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from prometheus_client import Counter

from sentinel.anomaly.detector import detect_anomalies
from sentinel.api.dependencies import get_cached_settings, get_tf_forecaster
from sentinel.api.schemas import (
    AnalyzeResponse,
    DriftReportResponse,
    ForecastMethodResult,
    RecalibratedThresholdsResponse,
    SeriesRequest,
)
from sentinel.config import Settings
from sentinel.drift.recalibration import detect_drift, recalibrate_cusum
from sentinel.forecasting.baseline import forecast_error, moving_average_forecast, trend_forecast
from sentinel.forecasting.tf_model import TFForecaster
from sentinel.logging_config import get_logger
from sentinel.monitoring.service import calculate_incident_risk

router = APIRouter()
logger = get_logger(__name__)

SettingsDep = Annotated[Settings, Depends(get_cached_settings)]
ForecasterDep = Annotated[TFForecaster, Depends(get_tf_forecaster)]

analyses_total = Counter("sentinel_analyses_total", "Total /v1/analyze calls served")
alerts_total = Counter("sentinel_alerts_total", "Total alerts issued, by priority", ["priority"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(forecaster: ForecasterDep) -> dict:
    return {"status": "ready", "window_size": forecaster.window_size}


def _method_result(predicted: list[float], recent_actual: list[float]) -> ForecastMethodResult:
    error = forecast_error(actual=recent_actual, predicted=predicted[: len(recent_actual)])
    return ForecastMethodResult(predicted=predicted, error_vs_recent_actual=error)


@router.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    payload: SeriesRequest, settings: SettingsDep, forecaster: ForecasterDep
) -> AnalyzeResponse:
    values = payload.values
    horizon = payload.forecast_horizon
    recent_actual = values[-horizon:] if len(values) >= horizon else values

    forecasts = {
        "moving_average": _method_result(
            moving_average_forecast(values, horizon=horizon), recent_actual
        ),
        "trend": _method_result(trend_forecast(values, horizon=horizon), recent_actual),
        "tf_model": _method_result(forecaster.forecast(values), recent_actual),
    }

    drift_report = detect_drift(
        values,
        baseline_window=settings.drift_baseline_window,
        recent_window=settings.drift_recent_window,
        relative_std_change_threshold=settings.drift_relative_std_change,
    )
    thresholds = recalibrate_cusum(
        values, recent_window=settings.drift_recent_window, drift_report=drift_report
    )
    cusum_threshold = (
        thresholds.cusum_threshold if thresholds.recalibrated else settings.cusum_threshold
    )
    cusum_drift = thresholds.cusum_drift if thresholds.recalibrated else settings.cusum_drift

    anomaly_report = detect_anomalies(
        values,
        z_threshold=settings.z_score_threshold,
        cusum_threshold=cusum_threshold,
        cusum_drift=cusum_drift,
    )

    severity: dict[str, object] = anomaly_report["severity"]  # type: ignore[assignment]
    incident_risk = calculate_incident_risk(
        severity_score=severity["severity_score"],  # type: ignore[arg-type]
        forecast_rmse=forecasts["tf_model"].error_vs_recent_actual["rmse"],
        anomaly_count=anomaly_report["anomalies_detected"],  # type: ignore[arg-type]
    )

    analyses_total.inc()
    alerts_total.labels(priority=incident_risk["alert_priority"]).inc()

    logger.info(
        "analysis_served",
        alert_priority=incident_risk["alert_priority"],
        anomalies_detected=anomaly_report["anomalies_detected"],
        drift_detected=drift_report.drift_detected,
    )

    return AnalyzeResponse(
        forecast_horizon=horizon,
        forecasts=forecasts,
        anomaly_detection=anomaly_report,
        drift=DriftReportResponse(
            drift_detected=drift_report.drift_detected,
            baseline_std=drift_report.baseline_std,
            recent_std=drift_report.recent_std,
            relative_std_change=drift_report.relative_std_change,
        ),
        thresholds_used=RecalibratedThresholdsResponse(
            cusum_threshold=thresholds.cusum_threshold,
            cusum_drift=thresholds.cusum_drift,
            recalibrated=thresholds.recalibrated,
            reason=thresholds.reason,
        ),
        incident_risk=incident_risk,
    )
