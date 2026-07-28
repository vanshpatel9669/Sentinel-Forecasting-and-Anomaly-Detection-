"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from sentinel.api.routes import router
from sentinel.config import get_settings
from sentinel.logging_config import configure_logging
from sentinel.observability import configure_tracing, instrument_app


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings.service_name)

    app = FastAPI(
        title="Sentinel Forecasting & Monitoring System",
        description=(
            "Statistical and TensorFlow forecasting, Z-score/CUSUM anomaly "
            "detection, and drift-triggered threshold recalibration."
        ),
        version="0.1.0",
    )
    app.include_router(router)
    instrument_app(app)
    return app


app = create_app()
