# src/quant_platform/options/iv_surface/surface_builder.py
from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Optional

import numpy as np
import polars as pl

from src.quant_platform.options.greeks.greeks import solve_iv


def _as_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        try:
            return date.fromisoformat(v)
        except Exception:
            return None
    return None


def _mid_from_row(r: dict) -> Optional[float]:
    b, a, last_price = r.get("bid"), r.get("ask"), r.get("last")
    try:
        if b is not None and a is not None:
            return 0.5 * (float(b) + float(a))
        if last_price is not None:
            return float(last_price)
        if b is not None:
            return float(b)
        if a is not None:
            return float(a)
    except Exception:
        return None
    return None


def _tau_from_row(r: dict) -> Optional[float]:
    expiry = _as_date(r.get("expiry") or r.get("expiration"))
    trade = _as_date(r.get("trade_date"))
    if expiry is None or trade is None:
        dte = r.get("days_to_expiry")
        if dte is not None:
            try:
                return float(dte) / 365.0
            except Exception:
                return None
        return None
    delta = (expiry - trade).days
    return float(delta) / 365.0 if delta > 0 else None


def _compute_iv_for_row(mid, spot, strike, tau, option_type, r):
    if mid is None or spot is None or tau is None or tau <= 0:
        return None
    try:
        return solve_iv(spot, strike, tau, r, mid, option_type)
    except Exception:
        return 0.20  # fallback vol


def _nearest_fill_grid(points, values, strikes, taus):
    K_mesh, T_mesh = np.meshgrid(strikes, taus)
    grid = np.full(K_mesh.shape, np.nan, dtype=float)
    flat_pts = np.column_stack((K_mesh.ravel(), T_mesh.ravel()))
    for idx, (k_g, t_g) in enumerate(flat_pts):
        d2 = (points[:, 0] - k_g) ** 2 + (points[:, 1] - t_g) ** 2
        nearest = np.argmin(d2)
        grid.ravel()[idx] = values[nearest]
    return grid


def build_iv_surface_table(df: pl.DataFrame, r: float = 0.0) -> Dict:
    if df is None or not isinstance(df, pl.DataFrame):
        raise ValueError("df must be a Polars DataFrame")

    processed = []
    for row in df.to_dicts():
        try:
            strike = float(row["strike"])
        except Exception:
            continue
        mid = _mid_from_row(row)
        spot = row.get("underlying_price") or row.get("spot") or None
        if spot is not None:
            try:
                spot = float(spot)
            except Exception:
                spot = None
        tau = _tau_from_row(row)
        option_type = (row.get("option_type") or "call").lower()
        iv = row.get("implied_vol")
        if iv is None or (isinstance(iv, float) and np.isnan(iv)):
            iv = _compute_iv_for_row(mid, spot, strike, tau, option_type, r)
        processed.append(
            {
                "strike": strike,
                "mid": mid,
                "tau": tau,
                "iv": iv,
                "option_type": option_type,
                "underlying_price": spot,
            }
        )

    if len(processed) == 0:
        return {
            "raw": pl.DataFrame(processed),
            "grid": {
                "strikes": np.array([]),
                "taus": np.array([]),
                "iv_grid": np.empty((0, 0)),
            },
        }

    pts_list, vals_list, strikes_list, taus_list = [], [], [], []
    for p in processed:
        if p["iv"] is None or p["tau"] is None or p["tau"] <= 0:
            continue
        pts_list.append((p["strike"], p["tau"]))
        vals_list.append(p["iv"])
        strikes_list.append(p["strike"])
        taus_list.append(p["tau"])

    if len(pts_list) == 0:
        unique_strikes = np.array(sorted({p["strike"] for p in processed}))
        unique_taus = np.array(
            sorted({p["tau"] for p in processed if p["tau"] is not None})
        )
        iv_grid = np.full((len(unique_taus), len(unique_strikes)), 0.20)
        return {
            "raw": pl.DataFrame(processed),
            "grid": {
                "strikes": unique_strikes,
                "taus": unique_taus,
                "iv_grid": iv_grid,
            },
        }

    pts = np.array(pts_list)
    vals = np.array(vals_list)
    strikes = np.array(sorted(set(strikes_list)))
    taus = np.array(sorted(set(taus_list)))
    iv_grid = _nearest_fill_grid(pts, vals, strikes, taus)
    if np.isfinite(iv_grid).sum() == 0:
        iv_grid[:] = 0.20

    raw_df = pl.DataFrame(processed)
    return {
        "raw": raw_df,
        "grid": {"strikes": strikes, "taus": taus, "iv_grid": iv_grid},
    }
