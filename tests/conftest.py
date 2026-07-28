"""Session-scoped fixture: a real (tiny, fast-to-train) TF forecaster used
by tests instead of a mock — mirrors the other projects in this portfolio's
"use a small real model, not a mock" convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.forecasting.tf_model import TFForecaster, train_forecaster

_WINDOW_SIZE = 20
_HORIZON = 5


@pytest.fixture(scope="session")
def trained_forecaster() -> TFForecaster:
    rng = np.random.default_rng(0)
    series_list = [
        100 + 8 * np.sin(np.arange(60) / 4) + rng.normal(0, 2, size=60) for _ in range(20)
    ]
    result = train_forecaster(
        series_list, window_size=_WINDOW_SIZE, horizon=_HORIZON, epochs=3, seed=0
    )
    return TFForecaster(model=result.model, window_size=_WINDOW_SIZE, horizon=_HORIZON)
