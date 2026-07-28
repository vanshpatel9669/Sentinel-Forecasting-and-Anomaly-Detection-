"""Environment-based configuration. Single source of truth for all settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    model_path: str = "./models/tf_forecaster.keras"
    window_size: int = 20  # trailing observations the TF model conditions on

    # Anomaly detection defaults — real starting points, not fabricated
    # targets; drift.recalibration can override these at runtime.
    z_score_threshold: float = 2.5
    cusum_threshold: float = 8.0
    cusum_drift: float = 0.5

    # Drift-triggered recalibration
    drift_baseline_window: int = 40
    drift_recent_window: int = 20
    drift_relative_std_change: float = 0.5  # trigger recalibration above this fractional change

    log_level: str = "INFO"
    service_name: str = "sentinel-monitoring"


def get_settings() -> Settings:
    return Settings()
