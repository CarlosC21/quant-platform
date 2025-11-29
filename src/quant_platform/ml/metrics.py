from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd

__all__ = ["sharpe_ratio", "information_coefficient", "directional_accuracy", "rmse"]


def sharpe_ratio(returns: Sequence[float], annualization: int = 252) -> float:
    """
    Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns : Sequence[float]
        Daily or periodic returns.
    annualization : int
        Factor to annualize the Sharpe ratio. 252 for daily returns, 12 for monthly.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    returns_array = np.asarray(returns, dtype=float)
    mean_ret = np.nanmean(returns_array)
    std_ret = np.nanstd(returns_array, ddof=1)

    if std_ret == 0:
        return 0.0

    return (mean_ret / std_ret) * np.sqrt(annualization)


def information_coefficient(
    predictions: Sequence[float], targets: Sequence[float]
) -> float:
    """
    Compute Spearman rank correlation (Information Coefficient) between predictions and targets.

    Parameters
    ----------
    predictions : Sequence[float]
        Model predictions.
    targets : Sequence[float]
        True values.

    Returns
    -------
    float
        Spearman rank correlation (IC). Returns np.nan if inputs are invalid.
    """
    preds = pd.Series(predictions, dtype=float)
    targs = pd.Series(targets, dtype=float)

    if len(preds) != len(targs) or len(preds) == 0:
        return np.nan

    return preds.corr(targs, method="spearman")


def directional_accuracy(
    predictions: Sequence[float], targets: Sequence[float]
) -> float:
    """
    Compute directional accuracy: fraction of times prediction sign matches target sign.

    Parameters
    ----------
    predictions : Sequence[float]
        Model predictions.
    targets : Sequence[float]
        True values.

    Returns
    -------
    float
        Fraction in [0,1] of correct directional predictions.
    """
    preds = np.asarray(predictions, dtype=float)
    targs = np.asarray(targets, dtype=float)

    if len(preds) != len(targs) or len(preds) == 0:
        return np.nan

    correct_dir = np.sign(preds) == np.sign(targs)
    return float(np.mean(correct_dir))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : Sequence[float]
        True target values.
    y_pred : Sequence[float]
        Predicted values.

    Returns
    -------
    float
        RMSE value.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
