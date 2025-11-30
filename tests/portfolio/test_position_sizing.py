# tests/portfolio/test_position_sizing.py
import numpy as np

from quant_platform.portfolio.position_sizing import (
    classical_kelly,
    fractional_kelly,
    vol_target_scaler,
    size_stat_arb_position,
)


def test_classical_kelly_basic():
    mu = 0.10
    sigma = 0.20
    f = classical_kelly(mu, sigma)
    # f = mu / sigma^2 = 0.10 / 0.04 = 2.5
    assert np.isclose(f, 2.5, atol=1e-12)


def test_fractional_kelly_half():
    mu = 0.10
    sigma = 0.20

    full = classical_kelly(mu, sigma)
    frac = fractional_kelly(mu, sigma, fraction=0.5)

    assert np.isclose(frac, 0.5 * full, atol=1e-12)
    assert frac < full


def test_vol_target_scaler_basic():
    realized = 0.20
    target = 0.10

    scale = vol_target_scaler(realized, target)

    # target_vol / realized_vol = 0.1 / 0.2 = 0.5
    assert np.isclose(scale, 0.5, atol=1e-12)


def test_stat_arb_position_sizing_long():
    # Setup: reasonably strong edge, z-score beyond entry, current vol > target
    side = "long"
    zscore = 3.0
    z_entry = 2.0
    mu = 0.05
    sigma = 0.10
    target_vol = 0.10
    realized_vol = 0.20
    kelly_fraction = 0.5
    leverage_limit = 1.0

    pos = size_stat_arb_position(
        side=side,
        zscore=zscore,
        z_entry=z_entry,
        mu=mu,
        sigma=sigma,
        target_vol=target_vol,
        realized_vol=realized_vol,
        kelly_fraction=kelly_fraction,
        leverage_limit=leverage_limit,
    )

    # With these parameters, raw Kelly exposure without cap would be > 1,
    # so we expect to hit the leverage limit on a strong signal.
    assert pos > 0.0
    assert pos <= leverage_limit
    assert np.isclose(pos, leverage_limit, atol=1e-12)


def test_stat_arb_position_sizing_short():
    side = "short"
    zscore = -3.0
    z_entry = 2.0
    mu = 0.05
    sigma = 0.10
    target_vol = 0.10
    realized_vol = 0.20
    kelly_fraction = 0.5
    leverage_limit = 1.0

    pos = size_stat_arb_position(
        side=side,
        zscore=zscore,
        z_entry=z_entry,
        mu=mu,
        sigma=sigma,
        target_vol=target_vol,
        realized_vol=realized_vol,
        kelly_fraction=kelly_fraction,
        leverage_limit=leverage_limit,
    )

    assert pos < 0.0
    assert abs(pos) <= leverage_limit
    assert np.isclose(abs(pos), leverage_limit, atol=1e-12)


def test_stat_arb_position_sizing_flat():
    pos = size_stat_arb_position(
        side="flat",
        zscore=3.0,
        z_entry=2.0,
        mu=0.05,
        sigma=0.10,
        target_vol=0.10,
        realized_vol=0.20,
        kelly_fraction=0.5,
        leverage_limit=1.0,
    )
    assert pos == 0.0


def test_stat_arb_leverage_clipping():
    # Extreme parameters to force raw position >> leverage_limit
    side = "long"
    zscore = 10.0
    z_entry = 1.0
    mu = 0.20
    sigma = 0.10
    target_vol = 0.10
    realized_vol = 0.10
    kelly_fraction = 1.0
    leverage_limit = 0.5

    pos = size_stat_arb_position(
        side=side,
        zscore=zscore,
        z_entry=z_entry,
        mu=mu,
        sigma=sigma,
        target_vol=target_vol,
        realized_vol=realized_vol,
        kelly_fraction=kelly_fraction,
        leverage_limit=leverage_limit,
    )

    assert pos > 0.0
    assert pos <= leverage_limit
    assert np.isclose(pos, leverage_limit, atol=1e-12)
