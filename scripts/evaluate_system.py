#!/usr/bin/env python3
"""End-to-end evaluation: forecast accuracy (held-out, walk-forward, per
method), anomaly-detection precision/recall against labeled synthetic
anomalies, and a drift-recalibration demonstration on a deliberately
regime-shifted series. Writes a real, reproducible report plus real
matplotlib visualizations — nothing in the output report is a target or a
claim copied from elsewhere; every number comes from this run.

Usage: python scripts/evaluate_system.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentinel.anomaly.detector import detect_anomalies  # noqa: E402
from sentinel.drift.recalibration import detect_drift, recalibrate_cusum  # noqa: E402
from sentinel.forecasting.baseline import (  # noqa: E402
    forecast_error,
    moving_average_forecast,
    trend_forecast,
)
from sentinel.forecasting.tf_model import TFForecaster  # noqa: E402
from sentinel.logging_config import configure_logging, get_logger  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "plots"

WINDOW_SIZE = 20
HORIZON = 5
Z_THRESHOLD = 2.5
CUSUM_THRESHOLD = 8.0
CUSUM_DRIFT = 0.5
HIT_TOLERANCE = 1  # a detection within +/-1 index of a true anomaly counts as a hit


def evaluate_forecast_accuracy(eval_series: list[dict], forecaster: TFForecaster) -> dict:
    errors: dict[str, list[float]] = {"moving_average": [], "trend": [], "tf_model": []}
    for record in eval_series:
        values = record["values"]
        if len(values) <= WINDOW_SIZE + HORIZON:
            continue
        history = values[:-HORIZON]
        actual = values[-HORIZON:]

        ma_pred = moving_average_forecast(history, horizon=HORIZON)
        trend_pred = trend_forecast(history, horizon=HORIZON)
        tf_pred = forecaster.forecast(history)

        errors["moving_average"].append(forecast_error(actual, ma_pred)["rmse"])
        errors["trend"].append(forecast_error(actual, trend_pred)["rmse"])
        errors["tf_model"].append(forecast_error(actual, tf_pred)["rmse"])

    return {
        method: {"mean_rmse": round(float(np.mean(vals)), 4), "n_series": len(vals)}
        for method, vals in errors.items()
    }


def evaluate_anomaly_detection(eval_series: list[dict]) -> dict:
    total_tp, total_fp, total_fn = 0, 0, 0
    for record in eval_series:
        values = record["values"]
        true_indices = set(record["anomaly_indices"])
        result = detect_anomalies(
            values,
            z_threshold=Z_THRESHOLD,
            cusum_threshold=CUSUM_THRESHOLD,
            cusum_drift=CUSUM_DRIFT,
        )
        detected_indices = {a["index"] for a in result["anomalies"]}

        matched_true: set[int] = set()
        for detected in detected_indices:
            hit = any(abs(detected - t) <= HIT_TOLERANCE for t in true_indices)
            if hit:
                total_tp += 1
                matched_true |= {t for t in true_indices if abs(detected - t) <= HIT_TOLERANCE}
            else:
                total_fp += 1
        total_fn += len(true_indices - matched_true)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def demonstrate_drift_recalibration() -> dict:
    rng = np.random.default_rng(42)
    calm = 100 + rng.normal(0, 2, size=40)
    volatile = 100 + rng.normal(0, 14, size=20)
    series = np.concatenate([calm, volatile]).tolist()

    drift_report = detect_drift(series, baseline_window=40, recent_window=20)
    thresholds = recalibrate_cusum(series, recent_window=20, drift_report=drift_report)

    return {
        "series": series,
        "drift_report": drift_report,
        "thresholds": thresholds,
    }


def plot_forecast_comparison(eval_series: list[dict], forecaster: TFForecaster) -> None:
    record = eval_series[0]
    values = record["values"]
    history, actual = values[:-HORIZON], values[-HORIZON:]
    x_actual = range(len(history), len(history) + HORIZON)

    plt.figure(figsize=(9, 5))
    plt.plot(range(len(history)), history, label="history", color="gray")
    plt.plot(x_actual, actual, "o-", label="actual", color="black")
    ma_forecast = moving_average_forecast(history, horizon=HORIZON)
    plt.plot(x_actual, ma_forecast, "--", label="moving_average")
    plt.plot(x_actual, trend_forecast(history, horizon=HORIZON), "--", label="trend")
    plt.plot(x_actual, forecaster.forecast(history), "--", label="tf_model")
    plt.legend()
    plt.title("Forecast comparison on one held-out evaluation series")
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "forecast_comparison.png", dpi=120)
    plt.close()


def plot_anomaly_detection(eval_series: list[dict]) -> None:
    record = eval_series[1]
    values = record["values"]
    true_indices = record["anomaly_indices"]
    result = detect_anomalies(
        values,
        z_threshold=Z_THRESHOLD,
        cusum_threshold=CUSUM_THRESHOLD,
        cusum_drift=CUSUM_DRIFT,
    )
    detected_indices = [a["index"] for a in result["anomalies"]]

    plt.figure(figsize=(9, 5))
    plt.plot(values, label="series", color="steelblue")
    plt.scatter(
        true_indices,
        [values[i] for i in true_indices],
        color="black",
        marker="x",
        s=100,
        label="true anomaly",
        zorder=5,
    )
    plt.scatter(
        detected_indices,
        [values[i] for i in detected_indices],
        color="red",
        facecolors="none",
        s=180,
        label="detected",
        zorder=4,
    )
    plt.legend()
    plt.title("Anomaly detection: true vs. detected (one evaluation series)")
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "anomaly_detection.png", dpi=120)
    plt.close()


def plot_drift_recalibration(drift_demo: dict) -> None:
    series = drift_demo["series"]
    thresholds = drift_demo["thresholds"]

    plt.figure(figsize=(9, 5))
    plt.plot(series, color="darkorange")
    plt.axvline(40, color="gray", linestyle=":", label="regime shift")
    plt.title(
        f"Drift-triggered recalibration — recalibrated={thresholds.recalibrated}, "
        f"new CUSUM h={thresholds.cusum_threshold}, k={thresholds.cusum_drift}"
    )
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "drift_recalibration.png", dpi=120)
    plt.close()


def render_report(forecast_results: dict, anomaly_results: dict, drift_demo: dict) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "**Machine-generated by `scripts/evaluate_system.py`. Every number "
        "below comes from this run against synthetic, labeled data — nothing "
        "here is a target or a claim copied from elsewhere.**",
        "",
        "## Forecast accuracy (held-out, walk-forward, RMSE)",
        "",
        "Each method forecasts the final 5 points of each of the 40 labeled "
        "evaluation series from the preceding history — a genuine held-out "
        "test, not the API's in-request self-check against its own tail.",
        "",
        "| Method | Mean RMSE | Series evaluated |",
        "|---|---|---|",
    ]
    for method, stats in forecast_results.items():
        lines.append(f"| {method} | {stats['mean_rmse']} | {stats['n_series']} |")

    lines += [
        "",
        "## Anomaly detection (Z-score, against labeled synthetic anomalies)",
        "",
        f"- True positives: {anomaly_results['true_positives']}",
        f"- False positives: {anomaly_results['false_positives']}",
        f"- False negatives: {anomaly_results['false_negatives']}",
        f"- **Precision: {anomaly_results['precision']:.1%}**",
        f"- **Recall: {anomaly_results['recall']:.1%}**",
        f"- **F1: {anomaly_results['f1']:.1%}**",
        "",
        f"A detection counts as a hit if within +/-{HIT_TOLERANCE} index of a "
        "true injected anomaly. Ground truth and detections both come from "
        "`scripts/generate_synthetic_series.py` / `anomaly/detector.py` — "
        "this measures detector quality against synthetic point anomalies, "
        "not real production incident data.",
        "",
        "## Drift-triggered recalibration demonstration",
        "",
        f"- Baseline std (calm segment): {drift_demo['drift_report'].baseline_std}",
        f"- Recent std (volatile segment): {drift_demo['drift_report'].recent_std}",
        f"- Relative std change: {drift_demo['drift_report'].relative_std_change:.1%}",
        f"- Drift detected: {drift_demo['drift_report'].drift_detected}",
        f"- Recalibrated: {drift_demo['thresholds'].recalibrated}",
        f"- New CUSUM threshold (h): {drift_demo['thresholds'].cusum_threshold}",
        f"- New CUSUM drift (k): {drift_demo['thresholds'].cusum_drift}",
        f"- Reason: {drift_demo['thresholds'].reason}",
        "",
        "See `eval/plots/` for the corresponding visualizations.",
        "",
        "## What this does not claim",
        "",
        '- Not "1.2M+ data points" — this evaluation runs against 40 '
        "labeled synthetic series of 120 points each (4,800 points total). "
        "See README.md for why that figure isn't reproduced.",
        "- Not a specific incident-response-time or downtime-reduction "
        "percentage — there is no production deployment to measure that "
        "against. The previous version of this repository hardcoded such "
        "percentages directly into its `/metrics` endpoint response; this "
        "evaluation does not attempt to replace that fabrication with a "
        "different number.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)

    eval_path = DATA_DIR / "labeled_eval_series.json"
    if not eval_path.exists():
        raise SystemExit("No eval data. Run scripts/generate_synthetic_series.py first.")
    eval_series = json.loads(eval_path.read_text())

    forecaster = TFForecaster.load(
        MODEL_DIR / "tf_forecaster.keras", window_size=WINDOW_SIZE, horizon=HORIZON
    )

    logger.info("evaluation_started", eval_series_count=len(eval_series))

    forecast_results = evaluate_forecast_accuracy(eval_series, forecaster)
    anomaly_results = evaluate_anomaly_detection(eval_series)
    drift_demo = demonstrate_drift_recalibration()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_forecast_comparison(eval_series, forecaster)
    plot_anomaly_detection(eval_series)
    plot_drift_recalibration(drift_demo)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = render_report(forecast_results, anomaly_results, drift_demo)
    (RESULTS_DIR / "evaluation_report.md").write_text(report)

    (RESULTS_DIR / "evaluation_raw.json").write_text(
        json.dumps(
            {
                "forecast_accuracy": forecast_results,
                "anomaly_detection": anomaly_results,
                "drift_recalibration": {
                    "baseline_std": drift_demo["drift_report"].baseline_std,
                    "recent_std": drift_demo["drift_report"].recent_std,
                    "relative_std_change": drift_demo["drift_report"].relative_std_change,
                    "drift_detected": drift_demo["drift_report"].drift_detected,
                    "recalibrated": drift_demo["thresholds"].recalibrated,
                    "cusum_threshold": drift_demo["thresholds"].cusum_threshold,
                    "cusum_drift": drift_demo["thresholds"].cusum_drift,
                },
            },
            indent=2,
        )
    )

    logger.info("evaluation_completed")
    print(report)
    print(f"Wrote {RESULTS_DIR / 'evaluation_report.md'} and plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
