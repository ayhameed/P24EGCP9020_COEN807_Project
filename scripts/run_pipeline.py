"""End-to-end training pipeline for river discharge linear regression."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from scripts.data_loader import load_station_data
from scripts.evaluation import (
    plot_actual_vs_predicted,
    plot_residuals,
    regression_metrics,
    save_metrics,
    save_predictions,
)
from scripts.modeling import (
    get_linear_coefficients,
    train_linear_regression,
    tune_ridge_model,
)
from scripts.preprocessing import build_feature_table, temporal_train_test_split


def run_station_experiment(station_file: Path, test_fraction: float = 0.2) -> Dict[str, Any]:
    """Run the full modeling workflow for one station and return artifacts.

    Parameters
    ----------
    station_file : Path
        Path to one station CSV file.
    test_fraction : float, optional
        Fraction of records used for the chronological test split.

    Returns
    -------
    dict
        Dictionary containing metadata, models, predictions, metrics, and tables.

    Notes
    -----
    author: P24EGCP9020
    usage: result = run_station_experiment(station_file, test_fraction=0.2)
    use: Share one consistent evaluation flow across single and multi-station runs.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    station_frame, station_meta = load_station_data(station_file)

    features = build_feature_table(station_frame)
    feature_cols = [
        col
        for col in features.columns
        if col not in {"target", "discharge", "uncertainty", "satellite"}
    ]

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        features, feature_cols=feature_cols, test_fraction=test_fraction
    )

    linear_model = train_linear_regression(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_metrics = regression_metrics(y_test, linear_pred)

    ridge_model, ridge_info = tune_ridge_model(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_metrics = regression_metrics(y_test, ridge_pred)

    metrics = {
        "station": station_meta.get("station", "unknown"),
        "country": station_meta.get("country", "unknown"),
        "record_count": float(len(station_frame)),
        "linear_regression": linear_metrics,
        "ridge_regression": ridge_metrics,
        "ridge_tuning": ridge_info,
    }

    summary = pd.DataFrame(
        [
            {"model": "linear_regression", **linear_metrics},
            {"model": "ridge_regression", **ridge_metrics, **ridge_info},
        ]
    )

    return {
        "station_file": station_file,
        "station_frame": station_frame,
        "station_meta": station_meta,
        "features": features,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "linear_pred": linear_pred,
        "ridge_pred": ridge_pred,
        "linear_model": linear_model,
        "metrics": metrics,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create command line parser for pipeline arguments.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.

    Notes
    -----
    author: P24EGCP9020
    usage: parser = build_parser()
    use: Standardize reproducible command-line runs.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    parser = argparse.ArgumentParser(description="Run linear regression for one river station.")
    parser.add_argument(
        "--station-file",
        type=Path,
        default=Path(
            "neodc/esacci/river_discharge/data/RD/RD-multi/v1.2/CSV/PO_BORGOFORTE_Q_Day.Cmd.csv"
        ),
        help="Path to one station CSV file.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Test split fraction.")
    return parser


def main() -> None:
    """Run data loading, training, comparison, and artifact export.

    Notes
    -----
    author: P24EGCP9020
    usage: python -m scripts.run_pipeline --station-file <path>
    use: Produce reproducible outputs for report and presentation.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    args = build_parser().parse_args()
    result = run_station_experiment(args.station_file, test_fraction=args.test_fraction)
    station_frame = result["station_frame"]
    station_meta = result["station_meta"]
    features = result["features"]
    feature_cols = result["feature_cols"]
    y_test = result["y_test"]
    linear_pred = result["linear_pred"]
    ridge_pred = result["ridge_pred"]
    linear_model = result["linear_model"]
    metrics = result["metrics"]
    linear_metrics = metrics["linear_regression"]
    ridge_metrics = metrics["ridge_regression"]

    results_dir = Path("results")
    plots_dir = results_dir / "plots"
    processed_dir = Path("data/processed")

    save_metrics(metrics, results_dir / "metrics.json")
    save_predictions(y_test, linear_pred, results_dir / "predictions_linear.csv")
    save_predictions(y_test, ridge_pred, results_dir / "predictions_ridge.csv")

    coef_table = get_linear_coefficients(linear_model, feature_cols)
    coef_table.to_csv(results_dir / "coefficients_linear.csv", index=False)

    plot_actual_vs_predicted(
        y_test,
        linear_pred,
        plots_dir / "actual_vs_predicted_linear.png",
        "Linear Regression: Actual vs Predicted",
    )
    plot_actual_vs_predicted(
        y_test,
        ridge_pred,
        plots_dir / "actual_vs_predicted_ridge.png",
        "Ridge Regression: Actual vs Predicted",
    )
    plot_residuals(
        y_test,
        linear_pred,
        plots_dir / "residuals_linear.png",
        "Linear Regression Residuals",
    )

    full_export = features.reset_index().rename(columns={"index": "datetime"})
    full_export.to_csv(processed_dir / "feature_table.csv", index=False)

    summary = result["summary"]
    summary.to_csv(results_dir / "model_comparison.csv", index=False)

    print("Pipeline complete.")
    print(f"Station: {station_meta.get('station', 'unknown')}")
    print(f"Linear metrics: {linear_metrics}")
    print(f"Ridge metrics: {ridge_metrics}")


if __name__ == "__main__":
    main()
