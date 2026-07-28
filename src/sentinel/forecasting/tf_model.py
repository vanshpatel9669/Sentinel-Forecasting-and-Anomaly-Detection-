"""A small TensorFlow/Keras forecaster: a dense network over a sliding
window of trailing observations, predicting the next `horizon` values.

Each window is normalized by its own mean/std before being fed to the
network and de-normalized on the way out. Without this, a network trained
on one series' value range (e.g. latency in milliseconds, ~100-200) would
not generalize to a series on a different scale (e.g. request counts in
the thousands) — normalization is what lets one trained model forecast
arbitrary input series rather than only the scale it happened to train on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

_EPSILON = 1e-6


def build_model(window_size: int, horizon: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(horizon),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _normalize_window(window: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(window))
    std = float(np.std(window)) + _EPSILON
    return (window - mean) / std, mean, std


def make_windows(
    series: np.ndarray, window_size: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window (X, y) pairs from a single series, each window
    normalized independently by its own mean/std (see module docstring).
    """
    x_rows, y_rows = [], []
    for start in range(len(series) - window_size - horizon + 1):
        window = series[start : start + window_size]
        target = series[start + window_size : start + window_size + horizon]
        normed_window, mean, std = _normalize_window(window)
        x_rows.append(normed_window)
        y_rows.append((target - mean) / std)
    if not x_rows:
        return np.empty((0, window_size)), np.empty((0, horizon))
    return np.array(x_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def build_training_dataset(
    series_list: list[np.ndarray], window_size: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    x_parts, y_parts = [], []
    for series in series_list:
        x, y = make_windows(series, window_size, horizon)
        if len(x):
            x_parts.append(x)
            y_parts.append(y)
    return np.concatenate(x_parts), np.concatenate(y_parts)


@dataclass
class TrainResult:
    model: keras.Model
    final_train_loss: float
    final_val_loss: float
    epochs_run: int


def train_forecaster(
    series_list: list[np.ndarray],
    window_size: int = 20,
    horizon: int = 5,
    epochs: int = 30,
    validation_split: float = 0.2,
    seed: int = 0,
) -> TrainResult:
    tf.random.set_seed(seed)
    x, y = build_training_dataset(series_list, window_size, horizon)
    model = build_model(window_size, horizon)
    history = model.fit(
        x,
        y,
        epochs=epochs,
        validation_split=validation_split,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    return TrainResult(
        model=model,
        final_train_loss=float(history.history["loss"][-1]),
        final_val_loss=float(history.history["val_loss"][-1]),
        epochs_run=len(history.history["loss"]),
    )


class TFForecaster:
    """Loads a trained model and forecasts arbitrary-length input series by
    taking the trailing `window_size` values (padding by repeating the
    first value if the input is shorter than the window).
    """

    def __init__(self, model: keras.Model, window_size: int, horizon: int):
        self._model = model
        self.window_size = window_size
        self.horizon = horizon

    @classmethod
    def load(cls, path: str | Path, window_size: int, horizon: int) -> TFForecaster:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained forecaster at {path}. Run `python scripts/train_forecaster.py` first."
            )
        model = keras.models.load_model(path)
        return cls(model=model, window_size=window_size, horizon=horizon)

    def forecast(self, values: list[float]) -> list[float]:
        series = np.array(values, dtype=np.float32)
        if len(series) < self.window_size:
            pad = np.full(self.window_size - len(series), series[0] if len(series) else 0.0)
            series = np.concatenate([pad, series])
        window = series[-self.window_size :]
        normed_window, mean, std = _normalize_window(window)
        normed_pred = self._model.predict(normed_window.reshape(1, -1), verbose=0)[0]
        prediction = normed_pred * std + mean
        return [round(float(v), 2) for v in prediction]
