from datetime import date, timedelta

import numpy as np
import polars as pl

from quant_platform.options.iv_surface.iv_surface import IVSurface
from quant_platform.options.iv_surface.surface_builder import build_iv_surface_table


# --------------------------
# Existing helper
# --------------------------
def make_sample_df():
    today = date.today()
    rows = []
    # two expiries
    for days in (30, 90):
        exp = today + timedelta(days=days)
        for k in (90, 95, 100, 105, 110):
            rows.append(
                {
                    "symbol": f"OPT_{days}_{k}",
                    "underlying_price": 100.0,
                    "trade_date": today,
                    "expiry": exp,
                    "strike": k,
                    "option_type": "call",
                    "bid": max(0.1, (100 - k) * 0.01 + 1.0),
                    "ask": max(0.1, (100 - k) * 0.01 + 1.2),
                    # leave implied_vol empty to force solver path
                    "implied_vol": None,
                }
            )
    return pl.DataFrame(rows)


# --------------------------
# Existing IV surface table test
# --------------------------
def test_build_iv_surface_table_basic():
    df = make_sample_df()
    result = build_iv_surface_table(df, r=0.01)
    assert "raw" in result and "grid" in result
    raw = result["raw"]
    grid = result["grid"]
    # raw must be a Polars DataFrame
    assert isinstance(raw, pl.DataFrame)
    # grid must contain iv_grid and numeric arrays
    assert grid is not None
    iv_grid = grid["iv_grid"]
    assert isinstance(iv_grid, np.ndarray)
    # no negative values allowed (after cleaning)
    assert np.all(np.isnan(iv_grid) | (iv_grid > 0))
    # ensure some finite values exist
    assert np.isfinite(iv_grid).sum() > 0


# --------------------------
# New: IVSurface tests
# --------------------------
def make_sample_iv_surface():
    strikes = np.array([90, 100, 110], dtype=float)
    taus = np.array([30 / 365, 90 / 365], dtype=float)
    iv_grid = np.array([[0.25, 0.20, 0.22], [0.23, 0.21, 0.23]], dtype=float)
    return IVSurface.from_grid(strikes, taus, iv_grid)


def test_ivsurface_interpolation():
    surf = make_sample_iv_surface()
    # Test exact grid points
    assert np.isclose(surf.iv(100, 30 / 365), 0.20)
    assert np.isclose(surf.iv(110, 90 / 365), 0.23)
    # Test interpolation (midpoints)
    mid_iv = surf.iv(95, 60 / 365)
    assert np.isfinite(mid_iv) and mid_iv > 0
    # Test dsigma_dK and dsigma_dT
    ds_dK = surf.dsigma_dK(100, 60 / 365)
    ds_dT = surf.dsigma_dT(100, 60 / 365)
    assert np.isfinite(ds_dK)
    assert np.isfinite(ds_dT)


def test_ivsurface_smile_and_term():
    surf = make_sample_iv_surface()
    ks, ivs = surf.smile(30 / 365)
    assert np.all(ks == np.array([90, 100, 110]))
    assert np.all(np.isfinite(ivs))
    taus, ivs_term = surf.term_structure(100)
    assert np.all(np.isfinite(ivs_term))


def test_ivsurface_compute_greeks():
    surf = make_sample_iv_surface()
    df_opts = pl.DataFrame(
        {
            "strike": [95, 100, 105],
            "ttm": [30 / 365, 60 / 365, 90 / 365],
            "option_type": ["call", "put", "call"],
            "underlying_price": [100, 100, 100],
        }
    )
    df_greeks = surf.compute_greeks(df_opts, r=0.01)
    assert isinstance(df_greeks, pl.DataFrame)
    for col in ["price", "delta", "gamma", "theta", "vega"]:
        assert col in df_greeks.columns
        # check dtype is numeric
        assert df_greeks[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
