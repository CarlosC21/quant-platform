# src/quant_platform/trading/stat_arb/spreads/kalman_filter.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant_platform.trading.stat_arb.spreads.schemas import KalmanSpreadResult


class KalmanHedgeConfig(BaseModel):
    """
    Configuration for Kalman filter-based dynamic hedge ratio beta_t.

    State vector:
        s_t = [alpha_t, beta_t]^T

    State equation (random walk):
        s_t = s_{t-1} + w_t,   w_t ~ N(0, Q)

    Observation equation:
        y_t = [1, x_t] s_t + v_t,  v_t ~ N(0, r)

    q_alpha, q_beta : process noise variances for alpha_t and beta_t
    r               : observation noise variance
    """

    q_alpha: float = Field(
        1e-7,
        ge=0.0,
        description="Process noise variance for alpha_t (intercept).",
    )
    q_beta: float = Field(
        1e-7,
        ge=0.0,
        description="Process noise variance for beta_t (hedge ratio).",
    )
    r: float = Field(
        1e-4,
        gt=0.0,
        description="Observation noise variance.",
    )
    init_alpha: float = Field(0.0, description="Initial guess for alpha_0 (intercept).")
    init_beta: float = Field(0.0, description="Initial guess for beta_0 (hedge ratio).")
    init_var_scale: float = Field(
        1e6,
        gt=0.0,
        description="Scale for initial covariance matrix P_0 = init_var_scale * I.",
    )
    min_obs: int = Field(
        30,
        ge=3,
        description="Minimum number of observations required.",
    )


@dataclass
class _KalmanState:
    """
    Internal container for Kalman recursion state.
    """

    s: np.ndarray  # state vector [alpha, beta]
    P: np.ndarray  # covariance matrix 2x2


def kalman_hedge_filter(
    series_y: pd.Series,
    series_x: pd.Series,
    config: KalmanHedgeConfig | None = None,
) -> KalmanSpreadResult:
    """
    Run Kalman filter to estimate dynamic hedge ratio beta_t between Y and X.

    Parameters
    ----------
    series_y : pd.Series
        Dependent asset price series (Y) with datetime index.
    series_x : pd.Series
        Independent asset price series (X) with datetime index.
    config : KalmanHedgeConfig, optional
        Configuration for process/observation noise and initialization.

    Returns
    -------
    KalmanSpreadResult
        Time-varying hedge ratio beta_t, alpha_t, and spread_t.
    """
    cfg = config or KalmanHedgeConfig()

    # Align on timestamp intersection and drop NaNs
    joined = pd.concat([series_y, series_x], axis=1, join="inner").dropna()
    if joined.shape[0] < cfg.min_obs:
        raise ValueError(
            f"Not enough observations for Kalman hedge: "
            f"{joined.shape[0]} < min_obs={cfg.min_obs}"
        )

    y = joined.iloc[:, 0].to_numpy(dtype=float)
    x = joined.iloc[:, 1].to_numpy(dtype=float)
    timestamps = joined.index.to_numpy()
    n = y.size

    # State covariance Q and observation variance R
    Q = np.diag([cfg.q_alpha, cfg.q_beta])  # 2x2
    R = cfg.r

    # Initial state and covariance
    s0 = np.array([cfg.init_alpha, cfg.init_beta], dtype=float)  # [alpha_0, beta_0]
    P0 = cfg.init_var_scale * np.eye(2, dtype=float)

    state = _KalmanState(s=s0, P=P0)

    alpha_path = np.empty(n, dtype=float)
    beta_path = np.empty(n, dtype=float)

    for t in range(n):
        # Design / observation matrix H_t = [1, x_t]
        H = np.array([[1.0, x[t]]], dtype=float)  # shape (1, 2)

        # --- Predict step ---
        # --- Predict step ---
        s_pred = state.s  # random walk
        P_pred = state.P + Q

        # --- Update step ---
        # Innovation
        y_pred = float((H @ s_pred)[0])
        nu = y[t] - y_pred

        S = float((H @ P_pred @ H.T)[0, 0] + R)  # scalar
        K = (P_pred @ H.T) / S  # (2,1)

        s_new = s_pred + (K.flatten() * nu)
        I2 = np.eye(2, dtype=float)
        KH = K @ H
        P_new = (I2 - KH) @ P_pred

        state.s = s_new
        state.P = P_new

        alpha_path[t] = s_new[0]
        beta_path[t] = s_new[1]

    spread = y - beta_path * x

    return KalmanSpreadResult(
        symbol_y=str(series_y.name),
        symbol_x=str(series_x.name),
        beta_t=beta_path,
        alpha_t=alpha_path,
        spread=spread,
        timestamps=timestamps,
    )
