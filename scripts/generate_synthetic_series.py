#!/usr/bin/env python3
"""Generates reproducible synthetic time series for training and
evaluation. Two kinds:

- Training series: seasonal + trend + noise, randomized parameters per
  series for diversity, no labeled anomalies (used to train the TF
  forecaster on the "normal" data-generating process).
- Labeled evaluation series: the same generating process, but with a
  small number of injected point anomalies (spikes/dips) at KNOWN indices
  — the ground truth `scripts/evaluate_system.py` scores the anomaly
  detector against. This is synthetic data with synthetic labels, not real
  operational data — documented as such everywhere it's used.

Usage: python scripts/generate_synthetic_series.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_series(
    rng: np.random.Generator,
    length: int = 120,
    base: float = 100.0,
    trend: float = 0.0,
    seasonal_amplitude: float = 8.0,
    seasonal_period: float = 12.0,
    noise_std: float = 2.0,
) -> np.ndarray:
    t = np.arange(length)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
    trend_component = trend * t
    noise = rng.normal(0, noise_std, size=length)
    return base + seasonal + trend_component + noise


def generate_training_series(n_series: int, length: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    series_list = []
    for _ in range(n_series):
        series_list.append(
            make_series(
                rng,
                length=length,
                base=rng.uniform(50, 500),
                trend=rng.uniform(-0.3, 0.3),
                seasonal_amplitude=rng.uniform(2, 20),
                seasonal_period=rng.uniform(8, 24),
                noise_std=rng.uniform(1, 5),
            )
        )
    return series_list


def generate_labeled_eval_series(
    n_series: int, length: int, seed: int, n_anomalies_per_series: int = 3
) -> list[dict]:
    rng = np.random.default_rng(seed)
    labeled = []
    for _ in range(n_series):
        series = make_series(
            rng,
            length=length,
            base=rng.uniform(50, 500),
            trend=rng.uniform(-0.3, 0.3),
            seasonal_amplitude=rng.uniform(2, 20),
            seasonal_period=rng.uniform(8, 24),
            noise_std=rng.uniform(1, 5),
        )
        anomaly_indices = sorted(
            rng.choice(range(20, length - 5), size=n_anomalies_per_series, replace=False).tolist()
        )
        for idx in anomaly_indices:
            direction = rng.choice([-1, 1])
            magnitude = rng.uniform(4, 8) * np.std(series)
            series[idx] += direction * magnitude
        labeled.append(
            {
                "values": [round(float(v), 3) for v in series],
                "anomaly_indices": anomaly_indices,
            }
        )
    return labeled


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    training_series = generate_training_series(n_series=200, length=120, seed=0)
    training_path = DATA_DIR / "training_series.json"
    training_path.write_text(json.dumps([[round(float(v), 3) for v in s] for s in training_series]))
    print(f"Wrote {len(training_series)} training series to {training_path}")

    eval_series = generate_labeled_eval_series(
        n_series=40, length=120, seed=1000, n_anomalies_per_series=3
    )
    eval_path = DATA_DIR / "labeled_eval_series.json"
    eval_path.write_text(json.dumps(eval_series))
    print(f"Wrote {len(eval_series)} labeled evaluation series to {eval_path}")


if __name__ == "__main__":
    main()
