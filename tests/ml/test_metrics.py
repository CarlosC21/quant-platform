import numpy as np
from quant_platform.ml.metrics import (
    sharpe_ratio,
    information_coefficient,
    directional_accuracy,
)


def test_sharpe_ratio_basic():
    returns = [0.01, 0.02, -0.01, 0.03, 0.0]
    sr = sharpe_ratio(returns, annualization=252)
    assert sr > 0


def test_sharpe_ratio_zero_std():
    returns = [0.0, 0.0, 0.0]
    sr = sharpe_ratio(returns)
    assert sr == 0.0


def test_ic_basic():
    preds = [0.1, 0.2, 0.3, 0.4]
    targs = [1, 2, 3, 4]
    ic = information_coefficient(preds, targs)
    assert np.isclose(ic, 1.0)


def test_ic_mismatched_length():
    preds = [0.1, 0.2]
    targs = [1]
    ic = information_coefficient(preds, targs)
    assert np.isnan(ic)


def test_directional_accuracy_basic():
    preds = [0.1, -0.2, 0.3, -0.4]
    targs = [0.2, -0.1, -0.3, -0.4]
    acc = directional_accuracy(preds, targs)
    assert np.isclose(acc, 0.75)  # updated expected value


def test_directional_accuracy_empty():
    acc = directional_accuracy([], [])
    assert np.isnan(acc)
