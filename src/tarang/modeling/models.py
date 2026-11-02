"""Benchmark regression models and baseline estimators for TARANG Stage 2.

Implements benchmark regressors, baseline estimators (MeanValue, MedianValue, LastValue),
and model factory creating estimators in the exact established order and configurations.
"""

from __future__ import annotations

from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from tarang.config import (
    FEATURE_NAMES,
    LAG_FEATURE_DURATION,
    LAG_FEATURE_ENERGY,
    MODEL_NAMES,
    MODEL_RANDOM_STATE,
    TARGET_DURATION,
    TARGET_ENERGY,
)


class EmptyModel:
    """Base estimator compatible with scikit-learn estimator interface."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize empty model with dummy arguments."""
        pass

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        """Return parameter dictionary for scikit-learn compatibility."""
        return {}

    def score(self, X: Any, y: Any) -> float:
        """Return dummy score for pipeline compatibility."""
        return 1.0

    def fit(self, X: Any, y: Any) -> EmptyModel:
        """Fit estimator to data."""
        return self


class MeanValue(EmptyModel):
    """Baseline predicting the arithmetic mean of the training target."""

    def __init__(self) -> None:
        super().__init__()
        self.mean: float | None = None
        self.mean_: float | None = None

    def fit(self, X: Any, y: Any) -> MeanValue:
        """Fit by computing the arithmetic mean of y."""
        y_arr = np.asarray(y, dtype=np.float64)
        self.mean_ = float(np.mean(y_arr))
        self.mean = self.mean_
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict fitted mean for every sample in X."""
        if self.mean_ is None:
            raise ValueError("MeanValue model is not fitted yet.")
        n_samples = len(X)
        return np.full(n_samples, self.mean_, dtype=np.float64)


class MedianValue(EmptyModel):
    """Baseline predicting the median of the training target."""

    def __init__(self) -> None:
        super().__init__()
        self.median: float | None = None
        self.median_: float | None = None

    def fit(self, X: Any, y: Any) -> MedianValue:
        """Fit by computing the median of y."""
        y_arr = np.asarray(y, dtype=np.float64)
        self.median_ = float(np.median(y_arr))
        self.median = self.median_
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict fitted median for every sample in X."""
        if self.median_ is None:
            raise ValueError("MedianValue model is not fitted yet.")
        n_samples = len(X)
        return np.full(n_samples, self.median_, dtype=np.float64)


class LastValue(EmptyModel):
    """Baseline predicting directly from the target-specific lag feature."""

    def __init__(self, lag_feature: str = LAG_FEATURE_DURATION) -> None:
        super().__init__()
        self.lag_feature = lag_feature

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        """Return parameters."""
        return {"lag_feature": self.lag_feature}

    def fit(self, X: Any, y: Any = None) -> LastValue:
        """Fit baseline (no-op as lag values are passed during predict)."""
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict directly from the configured lag feature without interpolation."""
        if isinstance(X, pd.DataFrame):
            if self.lag_feature not in X.columns:
                raise KeyError(
                    f"Lag feature '{self.lag_feature}' not found in input columns: {list(X.columns)}"
                )
            return np.asarray(X[self.lag_feature].to_numpy(), dtype=np.float64)
        elif isinstance(X, np.ndarray):
            if self.lag_feature in FEATURE_NAMES:
                idx = FEATURE_NAMES.index(self.lag_feature)
                return np.asarray(X[:, idx], dtype=np.float64)
            raise KeyError(
                f"Lag feature '{self.lag_feature}' not found in canonical FEATURE_NAMES."
            )
        raise TypeError(f"Unsupported input type for LastValue: {type(X)}")


def get_benchmark_models(target: str = TARGET_DURATION) -> List[Tuple[str, Any]]:
    """Return the six benchmark models configured in established reference order.

    Args:
        target: Target name, either TARGET_DURATION ('Duration') or
            TARGET_ENERGY ('ConsumedkWh').

    Returns:
        List of (model_name, model_instance) tuples in the required order:
            1. LinearRegression
            2. QuantileRegressor
            3. HistGradientBoostingRegressor
            4. MedianValue
            5. MeanValue
            6. LastValue
    """
    if target == TARGET_DURATION:
        lag_feature = LAG_FEATURE_DURATION
    elif target == TARGET_ENERGY:
        lag_feature = LAG_FEATURE_ENERGY
    else:
        raise ValueError(
            f"Unknown target '{target}'. Expected '{TARGET_DURATION}' or '{TARGET_ENERGY}'."
        )

    # A fresh RandomState(0) is initialized per target run
    rng = np.random.RandomState(MODEL_RANDOM_STATE)

    models: List[Tuple[str, Any]] = [
        ("LinearRegression", LinearRegression()),
        (
            "QuantileRegressor",
            QuantileRegressor(quantile=0.5, alpha=0, solver="highs"),
        ),
        (
            "HistGradientBoostingRegressor",
            HistGradientBoostingRegressor(
                early_stopping=True,
                learning_rate=0.1,
                min_samples_leaf=100,
                max_iter=100,
                loss="absolute_error",
                random_state=rng,
            ),
        ),
        ("MedianValue", MedianValue()),
        ("MeanValue", MeanValue()),
        ("LastValue", LastValue(lag_feature=lag_feature)),
    ]

    # Verify model names match canonical MODEL_NAMES
    assert [m[0] for m in models] == MODEL_NAMES, "Model names order mismatch"
    return models

