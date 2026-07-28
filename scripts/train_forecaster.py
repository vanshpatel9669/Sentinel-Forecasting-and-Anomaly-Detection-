#!/usr/bin/env python3
"""Trains the TensorFlow forecaster on the synthetic training series and
saves it to models/tf_forecaster.keras. Prints the real final train/val
loss — nothing here is a target, only a measurement of this run.

Usage: python scripts/train_forecaster.py [--epochs 30] [--window-size 20] [--horizon 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentinel.forecasting.tf_model import train_forecaster  # noqa: E402
from sentinel.logging_config import configure_logging, get_logger  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_logging()
    logger = get_logger(__name__)

    training_path = DATA_DIR / "training_series.json"
    if not training_path.exists():
        raise SystemExit("No training data. Run scripts/generate_synthetic_series.py first.")

    raw_series = json.loads(training_path.read_text())
    series_list = [np.array(s, dtype=float) for s in raw_series]

    logger.info(
        "training_started",
        series_count=len(series_list),
        window_size=args.window_size,
        horizon=args.horizon,
        epochs=args.epochs,
    )

    result = train_forecaster(
        series_list,
        window_size=args.window_size,
        horizon=args.horizon,
        epochs=args.epochs,
        seed=args.seed,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "tf_forecaster.keras"
    result.model.save(model_path)

    logger.info(
        "training_completed",
        epochs_run=result.epochs_run,
        final_train_loss=round(result.final_train_loss, 4),
        final_val_loss=round(result.final_val_loss, 4),
        model_path=str(model_path),
    )
    print(
        f"Trained {result.epochs_run} epochs. Final train loss (MSE): "
        f"{result.final_train_loss:.4f}, val loss: {result.final_val_loss:.4f}. "
        f"Saved to {model_path}"
    )


if __name__ == "__main__":
    main()
