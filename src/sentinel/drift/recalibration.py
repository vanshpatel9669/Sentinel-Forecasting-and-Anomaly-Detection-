"""Drift-triggered recalibration of anomaly-detection thresholds.

The previous version of this project's `/metrics` endpoint returned a
literal, hardcoded string — `"response_time_reduction": "35%"` — on every
request, regardless of input. There was no drift detection and no
recalibration anywhere in the code; the claim existed only as a fabricated
API response. This module replaces that with a real, working feature.

**Why CUSUM parameters specifically need recalibration, and z-score
doesn't**: a z-score is already expressed in standard-deviation units, so
it's scale-invariant by construction — a z-threshold of 2.5 means the same
thing regardless of the series' volatility regime. CUSUM's `threshold`
(decision interval `h`) and `drift` (reference value `k`) are expressed in
the series' *raw* units, so they silently stop meaning what they used to
mean when volatility shifts — the textbook fix (Montgomery, *Introduction
to Statistical Quality Control*) is to express them as multiples of the
current standard deviation (commonly `k = 0.5*sigma`, `h = 5*sigma`) and
refit `sigma` from recent data. That is exactly what `recalibrate_cusum`
does here — a standard, principled tuning rule, not an invented one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriftReport:
    drift_detected: bool
    baseline_std: float
    recent_std: float
    relative_std_change: float


@dataclass(frozen=True)
class RecalibratedThresholds:
    cusum_threshold: float
    cusum_drift: float
    recalibrated: bool
    reason: str


def detect_drift(
    values: list[float],
    baseline_window: int = 40,
    recent_window: int = 20,
    relative_std_change_threshold: float = 0.5,
) -> DriftReport:
    """Compares an older baseline segment's volatility to a recent
    segment's volatility. A large relative change signals the series has
    entered a different statistical regime — the trigger condition for
    recalibration.
    """
    if len(values) < baseline_window + recent_window:
        return DriftReport(
            drift_detected=False, baseline_std=0.0, recent_std=0.0, relative_std_change=0.0
        )

    recent = np.array(values[-recent_window:], dtype=float)
    baseline = np.array(values[-(baseline_window + recent_window) : -recent_window], dtype=float)

    baseline_std = float(np.std(baseline))
    recent_std = float(np.std(recent))
    relative_change = abs(recent_std - baseline_std) / (baseline_std + 1e-6)

    return DriftReport(
        drift_detected=relative_change > relative_std_change_threshold,
        baseline_std=round(baseline_std, 4),
        recent_std=round(recent_std, 4),
        relative_std_change=round(relative_change, 4),
    )


def recalibrate_cusum(
    values: list[float],
    recent_window: int,
    drift_report: DriftReport,
    k_multiplier: float = 0.5,
    h_multiplier: float = 5.0,
) -> RecalibratedThresholds:
    """Refits CUSUM's drift (k) and threshold (h) from the recent window's
    standard deviation using the standard `k = k_multiplier*sigma`,
    `h = h_multiplier*sigma` SPC tuning rule. Only runs when drift was
    actually detected — recalibrating on every request would just be
    noisy re-fitting, not a response to an actual regime change.
    """
    if not drift_report.drift_detected:
        baseline_std = drift_report.baseline_std
        return RecalibratedThresholds(
            cusum_threshold=h_multiplier * baseline_std if baseline_std else 8.0,
            cusum_drift=k_multiplier * baseline_std if baseline_std else 0.5,
            recalibrated=False,
            reason="No drift detected; thresholds unchanged.",
        )

    recent = values[-recent_window:]
    recent_std = float(np.std(recent))
    return RecalibratedThresholds(
        cusum_threshold=round(h_multiplier * recent_std, 4),
        cusum_drift=round(k_multiplier * recent_std, 4),
        recalibrated=True,
        reason=(
            f"Recent volatility changed {drift_report.relative_std_change:.1%} vs. baseline "
            f"(std {drift_report.baseline_std} -> {drift_report.recent_std}); CUSUM k/h refit "
            f"to {k_multiplier}/{h_multiplier} x recent sigma."
        ),
    )
