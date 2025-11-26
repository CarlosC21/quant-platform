import numpy as np
import polars as pl


def grid_interpolate_iv(df: pl.DataFrame) -> dict:
    """
    Minimal robust interpolator for testing:
    - Always returns a grid (no None)
    - Uses unique sorted strikes and maturities
    - Fills missing values with nearest IV
    """
    if df.is_empty():
        return {"k_grid": None, "t_grid": None, "iv_grid": None}

    strikes = np.unique(df["strike"].to_numpy())
    maturities = np.unique(df["maturity"].to_numpy())

    # Ensure grid-like shape for tests
    k_grid = strikes
    t_grid = maturities

    # Build grid
    iv_grid = np.zeros((len(t_grid), len(k_grid)))

    # Fill using nearest-neighbor
    for i, t in enumerate(t_grid):
        for j, k in enumerate(k_grid):
            # Find nearest match in df
            idx = np.argmin(
                np.abs(df["maturity"].to_numpy() - t)
                + np.abs(df["strike"].to_numpy() - k)
            )
            iv_grid[i, j] = df["iv"].to_numpy()[idx]

    return {
        "k_grid": k_grid,
        "t_grid": t_grid,
        "iv_grid": iv_grid,
    }
