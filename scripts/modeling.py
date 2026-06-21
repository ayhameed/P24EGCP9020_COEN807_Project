"""Model training utilities for linear regression experiments."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train a baseline linear regression model with scaling.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training target.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline containing scaler and linear regression model.

    Notes
    -----
    author: P24EGCP9020
    usage: linear_model = train_linear_regression(X_train, y_train)
    use: Build baseline supervised learning model for discharge prediction.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    return model


def tune_ridge_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_splits: int = 5,
) -> Tuple[Pipeline, Dict[str, float]]:
    """Tune Ridge regression using time-series cross-validation.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training target.
    alphas : iterable of float, optional
        Candidate regularization strengths.
    n_splits : int, optional
        Number of chronological cross-validation splits.

    Returns
    -------
    tuple
        Best fitted pipeline and summary dictionary with best alpha and CV RMSE.

    Notes
    -----
    author: P24EGCP9020
    usage: ridge_model, ridge_info = tune_ridge_model(X_train, y_train)
    use: Provide a tuned comparison model to satisfy multi-model requirement.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge()),
        ]
    )
    splitter = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"regressor__alpha": list(alphas)},
        cv=splitter,
        scoring="neg_root_mean_squared_error",
    )
    search.fit(X_train, y_train)

    best_rmse = float(abs(search.best_score_))
    info = {
        "best_alpha": float(search.best_params_["regressor__alpha"]),
        "cv_rmse": best_rmse,
    }
    return search.best_estimator_, info


def get_linear_coefficients(model: Pipeline, feature_names: Iterable[str]) -> pd.DataFrame:
    """Extract model coefficients into a sorted table.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted pipeline with a linear regressor.
    feature_names : iterable of str
        Ordered list of feature names.

    Returns
    -------
    pandas.DataFrame
        Coefficient table sorted by absolute magnitude.

    Notes
    -----
    author: P24EGCP9020
    usage: coef_table = get_linear_coefficients(model, X_train.columns)
    use: Interpret feature influence for report and presentation material.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    coefficients = model.named_steps["regressor"].coef_
    table = pd.DataFrame(
        {
            "feature": list(feature_names),
            "coefficient": np.asarray(coefficients, dtype=float),
        }
    )
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).drop(columns=["abs_coefficient"])
