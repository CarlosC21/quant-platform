# tests/risk/test_factor.py
import numpy as np

from quant_platform.risk.factor import (
    compute_pca_factor_model,
    compute_factor_exposures_regression,
)


def test_pca_factor_model_shapes_and_variance():
    rng = np.random.default_rng(123)

    n_obs = 500
    n_assets = 5
    n_factors = 2

    # Simulate simple factor structure: R = F B^T + eps
    F_true = rng.normal(size=(n_obs, n_factors))
    B_true = rng.normal(size=(n_assets, n_factors))
    eps = 0.1 * rng.normal(size=(n_obs, n_assets))
    R = F_true @ B_true.T + eps

    result = compute_pca_factor_model(R, n_factors=n_factors)

    # Check shapes
    assert len(result.factor_returns) == n_obs
    assert len(result.factor_returns[0]) == n_factors

    assert len(result.loadings) == n_assets
    assert len(result.loadings[0]) == n_factors

    assert len(result.specific_var) == n_assets
    assert len(result.explained_variance_ratio) == n_factors

    # PCA should explain a decent chunk of variance in this synthetic setup
    total_explained = sum(result.explained_variance_ratio)
    assert total_explained > 0.5  # at least 50%


def test_factor_exposures_regression_basic():
    rng = np.random.default_rng(42)

    n_obs = 300
    n_assets = 3
    n_factors = 2

    F = rng.normal(size=(n_obs, n_factors))
    # True betas for three assets
    betas_true = np.array(
        [
            [1.0, 0.5],
            [-0.5, 1.5],
            [0.2, -0.3],
        ],
    )
    intercepts_true = np.array([0.01, -0.02, 0.0])

    eps = 0.05 * rng.normal(size=(n_obs, n_assets))
    R = intercepts_true + F @ betas_true.T + eps

    symbols = ["A", "B", "C"]

    exposures = compute_factor_exposures_regression(R, F, symbols)

    assert len(exposures) == n_assets
    for j, exp in enumerate(exposures):
        assert exp.symbol == symbols[j]
        est_betas = np.array(exp.betas)
        assert est_betas.shape == (n_factors,)
        # Allow for noise but should be close-ish
        assert np.allclose(est_betas, betas_true[j], atol=0.15)
