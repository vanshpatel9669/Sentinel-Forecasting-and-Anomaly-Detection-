import numpy as np
import pytest

from sentinel.forecasting.tf_model import TFForecaster, build_model, make_windows


def test_build_model_output_shape():
    model = build_model(window_size=20, horizon=5)
    prediction = model.predict(np.zeros((1, 20), dtype=np.float32), verbose=0)
    assert prediction.shape == (1, 5)


def test_make_windows_shapes():
    series = np.arange(30, dtype=float)
    x, y = make_windows(series, window_size=20, horizon=5)
    assert x.shape[1] == 20
    assert y.shape[1] == 5
    assert x.shape[0] == y.shape[0] == 30 - 20 - 5 + 1


def test_make_windows_too_short_returns_empty():
    x, y = make_windows(np.arange(5, dtype=float), window_size=20, horizon=5)
    assert x.shape[0] == 0


def test_forecaster_forecast_shape_and_finite(trained_forecaster):
    values = (100 + 8 * np.sin(np.arange(40) / 4)).tolist()
    prediction = trained_forecaster.forecast(values)
    assert len(prediction) == 5
    assert all(np.isfinite(v) for v in prediction)


def test_forecaster_handles_input_shorter_than_window(trained_forecaster):
    prediction = trained_forecaster.forecast([100.0, 101.0, 99.0])
    assert len(prediction) == 5


def test_forecaster_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TFForecaster.load(tmp_path / "does_not_exist.keras", window_size=20, horizon=5)
