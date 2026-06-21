"""Evaluation and plotting helpers for regression experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true : pandas.Series
        Ground-truth target values.
    y_pred : numpy.ndarray
        Predicted target values.

    Returns
    -------
    dict of str to float
        RMSE, MAE, and R2 values.

    Notes
    -----
    author: P24EGCP9020
    usage: metrics = regression_metrics(y_test, y_pred)
    use: Evaluate continuous prediction quality with interpretable metrics.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    """Save metrics dictionary as JSON.

    Parameters
    ----------
    metrics : dict of str to float
        Metrics output from model evaluation.
    output_path : Path
        JSON file path.

    Notes
    -----
    author: P24EGCP9020
    usage: save_metrics(metrics, Path('results/metrics.json'))
    use: Persist reproducible evaluation results.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_predictions(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    """Save actual and predicted values to CSV.

    Parameters
    ----------
    y_true : pandas.Series
        Ground-truth target values indexed by datetime.
    y_pred : numpy.ndarray
        Predicted values from model.
    output_path : Path
        Destination CSV path.

    Notes
    -----
    author: P24EGCP9020
    usage: save_predictions(y_test, y_pred, Path('results/predictions.csv'))
    use: Enable transparent model comparison in report and slides.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"actual": y_true, "predicted": y_pred}, index=y_true.index)
    frame.to_csv(output_path, index_label="datetime")


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Create and save an actual-vs-predicted time-series plot.

    Parameters
    ----------
    y_true : pandas.Series
        Ground-truth values indexed by datetime.
    y_pred : numpy.ndarray
        Predicted values aligned to y_true.
    output_path : Path
        Image output path.
    title : str
        Plot title.

    Notes
    -----
    author: P24EGCP9020
    usage: plot_actual_vs_predicted(y_test, y_pred, output, 'Linear Regression')
    use: Visual diagnostic for forecast fit over time.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label="Actual", linewidth=1.5)
    plt.plot(y_true.index, y_pred, label="Predicted", linewidth=1.2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Discharge (m^3/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    """Create and save residual plot.

    Parameters
    ----------
    y_true : pandas.Series
        Ground-truth values.
    y_pred : numpy.ndarray
        Model predictions.
    output_path : Path
        Image output path.
    title : str
        Plot title.

    Notes
    -----
    author: P24EGCP9020
    usage: plot_residuals(y_test, y_pred, output, 'Residual Diagnostics')
    use: Check residual spread and potential model bias.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    residuals = y_true.values - y_pred
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, s=12, alpha=0.7)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Predicted discharge (m^3/s)")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
