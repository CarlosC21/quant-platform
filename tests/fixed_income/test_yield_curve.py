# tests/fixed_income/test_yield_curve.py
import math

import numpy as np
import pytest

from src.quant_platform.fixed_income.yield_curve import NSParams, fit_ns, ns_yield


def test_ns_fit_recovery():
    # Synthetic NS parameters
    true_p = NSParams(beta0=2.5, beta1=-1.0, beta2=1.2, tau=1.5)
    # maturities in years
    t = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    y = ns_yield(t, true_p)
    # add tiny noise
    y_noisy = y + np.random.normal(scale=1e-3, size=y.shape)
    fitted, res = fit_ns(t, y_noisy)
    # Assert beta0 close to true long rate within small tolerance
    assert pytest.approx(true_p.beta0, rel=1e-2) == fitted.beta0
    # tau should be within a reasonable factor
    assert math.isfinite(fitted.tau) and fitted.tau > 0
