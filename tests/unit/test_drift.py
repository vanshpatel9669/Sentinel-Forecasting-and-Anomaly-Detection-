import numpy as np

from sentinel.drift.recalibration import detect_drift, recalibrate_cusum


def _regime_shift_series(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    calm = 100 + rng.normal(0, 2, size=40)
    volatile = 100 + rng.normal(0, 14, size=20)
    return np.concatenate([calm, volatile]).tolist()


def _stable_series(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return (100 + rng.normal(0, 2, size=60)).tolist()


def test_detect_drift_flags_volatility_increase():
    report = detect_drift(_regime_shift_series(), baseline_window=40, recent_window=20)
    assert report.drift_detected is True
    assert report.recent_std > report.baseline_std


def test_detect_drift_no_change_when_stable():
    report = detect_drift(_stable_series(), baseline_window=40, recent_window=20)
    assert report.drift_detected is False


def test_detect_drift_insufficient_data_returns_no_drift():
    report = detect_drift([1.0, 2.0, 3.0], baseline_window=40, recent_window=20)
    assert report.drift_detected is False


def test_recalibrate_cusum_only_changes_when_drift_detected():
    series = _stable_series()
    report = detect_drift(series, baseline_window=40, recent_window=20)
    thresholds = recalibrate_cusum(series, recent_window=20, drift_report=report)
    assert thresholds.recalibrated is False


def test_recalibrate_cusum_scales_with_recent_std():
    series = _regime_shift_series()
    report = detect_drift(series, baseline_window=40, recent_window=20)
    thresholds = recalibrate_cusum(series, recent_window=20, drift_report=report)
    assert thresholds.recalibrated is True
    recent_std = float(np.std(series[-20:]))
    assert thresholds.cusum_drift == round(0.5 * recent_std, 4)
    assert thresholds.cusum_threshold == round(5.0 * recent_std, 4)
