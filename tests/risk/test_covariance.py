# tests/risk/test_covariance.py
import numpy as np

from quant_platform.risk.covariance import (
    compute_sample_covariance,
    compute_ewma_covariance,
    compute_ledoit_wolf_covariance,
)


def _is_psd(a: np.ndarray, tol: float = 1e-8) -> bool:
    eigvals = np.linalg.eigvalsh(a)
    return np.all(eigvals >= -tol)


def test_sample_covariance_basic():
    rng = np.random.default_rng(42)
    r = rng.normal(size=(100, 3))

    cov = compute_sample_covariance(r)

    assert cov.shape == (3, 3)
    assert _is_psd(cov)


def test_ewma_covariance_shapes_and_psd():
    rng = np.random.default_rng(123)
    r = rng.normal(size=(200, 4))

    cov = compute_ewma_covariance(r, lambda_decay=0.94)

    assert cov.shape == (4, 4)
    assert _is_psd(cov)


def test_ledoit_wolf_covariance_basic():
    rng = np.random.default_rng(7)
    r = rng.normal(size=(250, 5))

    cov = compute_ledoit_wolf_covariance(r)

    assert cov.shape == (5, 5)
    assert _is_psd(cov)
