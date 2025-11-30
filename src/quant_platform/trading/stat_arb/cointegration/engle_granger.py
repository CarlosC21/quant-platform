# src/quant_platform/trading/stat_arb/cointegration/engle_granger.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from quant_platform.trading.stat_arb.cointegration.schemas import (
    ADFResult,
    CointegrationResult,
    EngleGrangerConfig,
)


def _ols_y_on_x(
    y: np.ndarray,
    x: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    OLS regression of y on x: y = alpha + beta * x + eps.

    Returns
    -------
    alpha : float
    beta : float
    residuals : np.ndarray
    """
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape for OLS regression.")

    n = y.size
    if n < 3:
        raise ValueError("Need at least 3 observations for OLS regression.")

    X_design = np.column_stack([np.ones(n, dtype=float), x])
    coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    alpha_hat, beta_hat = float(coef[0]), float(coef[1])

    residuals = y - (alpha_hat + beta_hat * x)
    return alpha_hat, beta_hat, residuals


def _adf_residual_test(residuals: np.ndarray) -> ADFResult:
    """
    Minimal ADF(1) unit-root test on residuals.

    We estimate:
        Δe_t = gamma * e_{t-1} + ε_t

    and test H0: gamma = 0 (unit root in e_t) against H1: gamma < 0
    using the t-statistic of gamma.

    Notes
    -----
    - We do not include a constant or trend in the ADF regression, since
      the residuals are already mean-adjusted by the first-stage OLS.
    - Critical values are hard-coded for the ADF statistic with a constant
      in the cointegration context (Engle–Granger). We approximate standard
      MacKinnon critical values for large samples.
    """
    if residuals.ndim != 1:
        raise ValueError("Residual series must be 1-dimensional.")

    e = residuals.astype(float)
    # Drop any NaNs
    e = e[np.isfinite(e)]
    n = e.size
    if n < 10:
        raise ValueError("Not enough observations for ADF test on residuals.")

    # Construct Δe_t and lagged e_{t-1}
    de = np.diff(e)
    e_lag = e[:-1]
    n_reg = de.size

    # Regression: de = gamma * e_{t-1} + eps
    # Design matrix is just e_lag as a single regressor.
    # X = e_lag.reshape(-1, 1)
    y = de

    # OLS: gamma_hat = (X'X)^{-1} X'y
    xx = float(np.dot(e_lag, e_lag))
    if xx <= 0.0:
        raise ValueError("Residual variance is zero; ADF test is undefined.")

    xy = float(np.dot(e_lag, y))
    gamma_hat = xy / xx

    # Residuals and variance estimate
    eps = y - gamma_hat * e_lag
    s2 = float(np.dot(eps, eps) / (n_reg - 1))  # one parameter (gamma)
    # Var(gamma_hat) = s2 / sum(e_{t-1}^2)
    var_gamma = s2 / xx
    se_gamma = np.sqrt(var_gamma)

    test_stat = gamma_hat / se_gamma

    # Approximate critical values for ADF(1) with constant, large sample
    # These are standard "tau" critical values used in cointegration tests.
    crit_1 = -3.96
    crit_5 = -3.41
    crit_10 = -3.12

    # Very coarse p-value bucketing, only for logging/diagnostics.
    if test_stat < crit_1:
        p_approx = 0.005
    elif test_stat < crit_5:
        p_approx = 0.025
    elif test_stat < crit_10:
        p_approx = 0.075
    else:
        p_approx = 0.2

    return ADFResult(
        test_stat=float(test_stat),
        crit_1=float(crit_1),
        crit_5=float(crit_5),
        crit_10=float(crit_10),
        p_value=p_approx,
    )


class EngleGrangerTester:
    """
    Engle–Granger 2-step cointegration tester.

    Step 1:
        Regress Y_t on X_t: Y_t = alpha + beta X_t + e_t

    Step 2:
        Test residuals e_t for unit root using minimal ADF(1) test.

    Series are expected to be price levels (not returns) and aligned on a
    common time index. Alignment (dropna + intersection) is handled here.
    """

    def __init__(self, config: EngleGrangerConfig | None = None) -> None:
        self.config = config or EngleGrangerConfig()

    def test_pair(
        self,
        series_y: pd.Series,
        series_x: pd.Series,
    ) -> CointegrationResult:
        """
        Run Engle–Granger cointegration test on a pair of price series.

        Parameters
        ----------
        series_y : pd.Series
            Dependent asset (Y), price levels with datetime-like index.
        series_x : pd.Series
            Independent asset (X), price levels with same or superset index.

        Returns
        -------
        CointegrationResult
        """
        y_aligned, x_aligned = self._align_and_validate(series_y, series_x)
        n_obs = y_aligned.size

        y = y_aligned.to_numpy(dtype=float)
        x = x_aligned.to_numpy(dtype=float)

        alpha_hat, beta_hat, residuals = _ols_y_on_x(y, x)
        adf = _adf_residual_test(residuals)

        coint = adf.stationary_at_5pct
        return CointegrationResult(
            symbol_y=str(y_aligned.name),
            symbol_x=str(x_aligned.name),
            alpha=float(alpha_hat),
            beta=float(beta_hat),
            adf_result=adf,
            coint=coint,
            n_obs=n_obs,
        )

    def _align_and_validate(
        self,
        series_y: pd.Series,
        series_x: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Align two pandas Series on their intersection of indices and
        enforce minimal observation count.
        """
        joined = pd.concat([series_y, series_x], axis=1, join="inner").dropna()
        if joined.shape[0] < self.config.min_obs:
            raise ValueError(
                f"Not enough observations for Engle–Granger test: "
                f"{joined.shape[0]} < min_obs={self.config.min_obs}"
            )

        y_aligned = joined.iloc[:, 0]
        x_aligned = joined.iloc[:, 1]

        return y_aligned, x_aligned
