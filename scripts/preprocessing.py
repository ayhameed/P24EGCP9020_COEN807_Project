"""Preprocessing and feature engineering for river discharge regression."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


def build_feature_table(
    frame: pd.DataFrame,
    lags: Iterable[int] = (1, 2, 7, 30),
    rolling_windows: Iterable[int] = (7, 30),
) -> pd.DataFrame:
    """Generate lag, rolling, and calendar features with regression target.

    Parameters
    ----------
    frame : pandas.DataFrame
        Time-indexed DataFrame with at least ``discharge`` column.
    lags : iterable of int, optional
        Lag offsets in days represented as previous records.
    rolling_windows : iterable of int, optional
        Rolling windows used for mean and standard deviation features.

    Returns
    -------
    pandas.DataFrame
        Feature table including ``target`` and predictor columns.

    Notes
    -----
    author: P24EGCP9020
    usage: features = build_feature_table(frame)
    use: Create supervised learning matrix for linear regression.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    working = frame.copy()
    working["target"] = working["discharge"]

    for lag in lags:
        working[f"lag_{lag}"] = working["discharge"].shift(lag)

    for window in rolling_windows:
        shifted = working["discharge"].shift(1)
        working[f"roll_mean_{window}"] = shifted.rolling(window=window).mean()
        working[f"roll_std_{window}"] = shifted.rolling(window=window).std()

    working["month"] = working.index.month
    working["day_of_year"] = working.index.dayofyear
    working["quarter"] = working.index.quarter

    required_cols = ["target"]
    required_cols.extend([f"lag_{lag}" for lag in lags])
    required_cols.extend([f"roll_mean_{window}" for window in rolling_windows])
    required_cols.extend([f"roll_std_{window}" for window in rolling_windows])
    required_cols.extend(["month", "day_of_year", "quarter"])

    return working.dropna(subset=required_cols)


def temporal_train_test_split(
    table: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str = "target",
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target using chronological order.

    Parameters
    ----------
    table : pandas.DataFrame
        Full feature table sorted by time index.
    feature_cols : iterable of str
        Feature column names used as predictors.
    target_col : str, optional
        Name of the regression target column.
    test_fraction : float, optional
        Fraction of the latest observations used for holdout testing.

    Returns
    -------
    tuple
        ``X_train, X_test, y_train, y_test`` in chronological order.

    Notes
    -----
    author: P24EGCP9020
    usage: X_train, X_test, y_train, y_test = temporal_train_test_split(...)
    use: Prevent future leakage during evaluation.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    table = table.sort_index()
    split_idx = int(len(table) * (1.0 - test_fraction))

    X = table[list(feature_cols)]
    y = table[target_col]

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    if X_train.empty or X_test.empty:
        raise ValueError(
            "Temporal split produced an empty partition. "
            "Adjust test_fraction or use a station with more observations."
        )

    return X_train, X_test, y_train, y_test
