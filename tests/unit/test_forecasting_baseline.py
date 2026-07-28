import pytest

from sentinel.forecasting.baseline import (
    forecast_error,
    moving_average_forecast,
    trend_forecast,
)


def test_moving_average_forecast_returns_horizon_length():
    result = moving_average_forecast([1.0, 2.0, 3.0, 4.0, 5.0], window=3, horizon=4)
    assert len(result) == 4
    assert all(v == result[0] for v in result)


def test_moving_average_forecast_empty_input():
    assert moving_average_forecast([], horizon=5) == []


def test_moving_average_forecast_value_is_windowed_mean():
    result = moving_average_forecast([10.0, 20.0, 30.0], window=3, horizon=1)
    assert result[0] == pytest.approx(20.0)


def test_trend_forecast_extrapolates_positive_slope():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = trend_forecast(values, horizon=2)
    assert result[0] > values[-1]
    assert result[1] > result[0]


def test_trend_forecast_falls_back_for_short_series():
    result = trend_forecast([5.0], horizon=3)
    assert len(result) == 3


def test_forecast_error_known_values():
    error = forecast_error(actual=[10.0, 10.0], predicted=[8.0, 12.0])
    assert error["mae"] == pytest.approx(2.0)
    assert error["rmse"] == pytest.approx(2.0)


def test_forecast_error_empty_input():
    assert forecast_error(actual=[], predicted=[]) == {"mae": 0.0, "rmse": 0.0}
