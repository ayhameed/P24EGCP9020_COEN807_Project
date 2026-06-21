"""Batch runner for station-level model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from scripts.data_loader import list_station_files
from scripts.run_pipeline import run_station_experiment


def build_parser() -> argparse.ArgumentParser:
    """Create command line parser for batch station evaluation.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.

    Notes
    -----
    author: P24EGCP9020
    usage: parser = build_parser()
    use: Standardize automatic multi-station execution.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    parser = argparse.ArgumentParser(
        description="Run linear and ridge regression comparison across multiple stations."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("neodc/esacci/river_discharge/data/RD/RD-multi/v1.2/CSV"),
        help="Directory containing station CSV files.",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=5,
        help="Maximum number of station files to process.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Chronological test split fraction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/station_comparison.csv"),
        help="Output CSV path for station-level comparison table.",
    )
    return parser


def run_batch(
    csv_dir: Path,
    max_stations: int,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full pipeline on multiple stations.

    Parameters
    ----------
    csv_dir : Path
        Directory containing station CSV files.
    max_stations : int
        Maximum number of stations to evaluate.
    test_fraction : float
        Fraction used for temporal holdout split.

    Returns
    -------
    tuple of pandas.DataFrame
        First table is successful station comparisons, second table is failures.

    Notes
    -----
    author: P24EGCP9020
    usage: comparison, failures = run_batch(csv_dir, 5, 0.2)
    use: Generate station-level metrics table for report and presentation.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    station_files = list_station_files(csv_dir)[:max_stations]
    comparison_rows: List[Dict[str, float | str]] = []
    failed_rows: List[Dict[str, str]] = []

    for station_file in station_files:
        try:
            result = run_station_experiment(station_file=station_file, test_fraction=test_fraction)
            metrics = result["metrics"]
            linear = metrics["linear_regression"]
            ridge = metrics["ridge_regression"]
            tuning = metrics["ridge_tuning"]

            comparison_rows.append(
                {
                    "station_file": station_file.name,
                    "station": metrics["station"],
                    "country": metrics["country"],
                    "record_count": metrics["record_count"],
                    "linear_rmse": linear["rmse"],
                    "linear_mae": linear["mae"],
                    "linear_r2": linear["r2"],
                    "ridge_rmse": ridge["rmse"],
                    "ridge_mae": ridge["mae"],
                    "ridge_r2": ridge["r2"],
                    "ridge_best_alpha": tuning["best_alpha"],
                    "ridge_cv_rmse": tuning["cv_rmse"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed_rows.append({"station_file": station_file.name, "error": str(exc)})

    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("linear_rmse", ascending=True).reset_index(drop=True)

    failures_df = pd.DataFrame(failed_rows)
    return comparison_df, failures_df


def main() -> None:
    """Run multi-station comparison and save output tables.

    Notes
    -----
    author: P24EGCP9020
    usage: python -m scripts.run_multi_station --max-stations 5
    use: Produce one station-level comparison table from batch execution.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    args = build_parser().parse_args()
    comparison_df, failures_df = run_batch(
        csv_dir=args.csv_dir,
        max_stations=args.max_stations,
        test_fraction=args.test_fraction,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(args.output, index=False)

    failures_path = args.output.with_name("station_failures.csv")
    failures_df.to_csv(failures_path, index=False)

    print(f"Processed stations: {len(comparison_df)}")
    print(f"Failed stations: {len(failures_df)}")
    print(f"Comparison table: {args.output}")
    print(f"Failure table: {failures_path}")


if __name__ == "__main__":
    main()
