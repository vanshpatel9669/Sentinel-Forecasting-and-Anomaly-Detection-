"""FastAPI dependency-injection wiring."""

from __future__ import annotations

from functools import lru_cache

from sentinel.config import Settings, get_settings
from sentinel.forecasting.tf_model import TFForecaster

_FORECAST_HORIZON = 5  # the horizon the committed model was trained for


@lru_cache
def get_cached_settings() -> Settings:
    return get_settings()


@lru_cache
def get_tf_forecaster() -> TFForecaster:
    settings = get_cached_settings()
    return TFForecaster.load(
        settings.model_path, window_size=settings.window_size, horizon=_FORECAST_HORIZON
    )
