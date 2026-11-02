"""Evaluation metrics and error-distribution statistics for TARANG Stage 2.

Implements reference-aligned metric calculations (medae, mae, medape, mape, rmse, smape)
and error distribution statistics (quantiles and established whiskers for ae and ape).
"""

from __future__ import annotations

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error


def absolute_percentage_error(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """Compute element-wise absolute percentage error (APE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        1D numpy array of absolute percentage errors: |y_true - y_pred| / y_true.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs((true - pred) / true)


def absolute_error(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """Compute element-wise absolute error (AE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        1D numpy array of absolute errors: |y_true - y_pred|.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return np.abs(true - pred)


def squared_error(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """Compute element-wise squared error (SE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        1D numpy array of squared errors: (y_true - y_pred) ** 2.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return (true - pred) ** 2


def symmetric_mean_absolute_percentage_error(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (sMAPE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        sMAPE as a ratio (fraction, not percentage).
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if len(true) == 0:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(
            np.sum(
                2.0 * np.abs(true - pred)
                / (np.abs(true) + np.abs(pred))
            ) / len(true)
        )


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """Calculate the 16 established metrics and error distribution statistics.

    Summary metrics:
        - medae: Median Absolute Error
        - mae: Mean Absolute Error
        - medape: Median Absolute Percentage Error
        - mape: Mean Absolute Percentage Error
        - rmse: Root Mean Squared Error
        - smape: Symmetric Mean Absolute Percentage Error

    Distribution statistics (for both ae and ape):
        - q1, q2, q3: 25th, 50th, 75th percentiles
        - wl, wh: Lower and upper whisker limits following the established formula:
            wl = error[error > (q1 - 1.5 * iqr)].min()
            wh = error[error < (q1 + 1.5 * iqr)].max()

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary mapping metric names to calculated floating-point values.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if len(true) == 0:
        return {}

    ae = np.abs(true - pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(ae / true)

    # Core summary metrics
    with np.errstate(divide="ignore", invalid="ignore"):
        results: Dict[str, float] = {
            "medae": float(median_absolute_error(true, pred)),
            "mae": float(mean_absolute_error(true, pred)),
            "medape": float(np.median(ape)),
            "mape": float(np.mean(ape)),
            "rmse": float(np.sqrt(mean_squared_error(true, pred))),
            "smape": float(symmetric_mean_absolute_percentage_error(true, pred)),
        }

    # Error distribution statistics for absolute error (ae) and percentage error (ape)
    for err_name, err in [("ae", ae), ("ape", ape)]:
        # Calculate quantiles on complete error vector without dropping NaNs
        quantiles = np.quantile(err, [0.25, 0.5, 0.75])
        iqr = quantiles[2] - quantiles[0]
        lower_bound = quantiles[0] - 1.5 * iqr
        upper_bound = quantiles[0] + 1.5 * iqr

        above_lower = err[err > lower_bound]
        below_upper = err[err < upper_bound]

        # In Pandas, an empty filtered Series yields NaN for min() and max().
        # In NumPy, empty arrays raise an error on min()/max(), so we return NaN.
        wl = float(above_lower.min()) if len(above_lower) > 0 else float("nan")
        wh = float(below_upper.max()) if len(below_upper) > 0 else float("nan")

        results[f"q1-{err_name}"] = float(quantiles[0])
        results[f"q2-{err_name}"] = float(quantiles[1])
        results[f"q3-{err_name}"] = float(quantiles[2])
        results[f"wl-{err_name}"] = wl
        results[f"wh-{err_name}"] = wh

    return results
