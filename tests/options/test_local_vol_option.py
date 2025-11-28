# tests/options/test_local_vol_option.py
import numpy as np
import pytest

from quant_platform.options.models.local_vol import LocalVolSurface
from quant_platform.options.models.local_vol import LocalVolOption
from quant_platform.options.models.delta_hedge import delta_hedge_simulator


@pytest.fixture
def lv_surface():
    strikes = np.array([80, 100, 120])
    maturities = np.array([0.5, 1.0, 1.5])
    implied_vols = np.array([[0.25, 0.2, 0.22], [0.23, 0.2, 0.21], [0.22, 0.19, 0.2]])
    return LocalVolSurface(strikes, maturities, implied_vols)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_local_vol_option_price_positive(lv_surface, option_type):
    opt = LocalVolOption(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        local_vol_surface=lv_surface,
        option_type=option_type,
    )
    price = opt.price()
    assert price > 0, f"Price should be positive, got {price}"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_local_vol_option_delta_bounds(lv_surface, option_type):
    opt = LocalVolOption(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        local_vol_surface=lv_surface,
        option_type=option_type,
    )
    delta = opt.delta()
    if option_type == "call":
        assert 0 <= delta <= 1
    else:
        assert -1 <= delta <= 0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_hedge_pnl_local_vol(lv_surface, option_type):
    np.random.seed(42)
    S_path = 100 * np.exp(np.cumsum(0.01 * np.random.randn(50)))  # simulate path

    for t, S in enumerate(S_path):
        sigma = lv_surface.vol(100, 1.0)
        print(f"t={t}, S={S}, sigma={sigma}")

    pnl = delta_hedge_simulator(
        LocalVolOption,
        S_path,
        r=0.05,
        K=100,
        option_type=option_type,
        local_vol_surface=lv_surface,
    )

    print("P&L:", pnl)
    assert abs(pnl) < 5, f"Delta-hedge P&L too large: {pnl}"
