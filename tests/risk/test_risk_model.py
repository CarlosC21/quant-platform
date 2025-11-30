# tests/risk/test_risk_model.py

import numpy as np

from quant_platform.risk.model import CovarianceRiskModel, RiskModelConfig


def _is_psd(a: np.ndarray, tol: float = 1e-8) -> bool:
    return np.all(np.linalg.eigvalsh(a) >= -tol)


def test_risk_model_sample():
    rng = np.random.default_rng(1)
    r = rng.normal(size=(100, 3))

    rm = CovarianceRiskModel(RiskModelConfig(method="sample"))

    cov = rm.compute_covariance(r)

    assert cov.shape == (3, 3)
    assert _is_psd(cov)


def test_risk_model_ewma():
    rng = np.random.default_rng(2)
    r = rng.normal(size=(200, 4))

    rm = CovarianceRiskModel(RiskModelConfig(method="ewma", lambda_decay=0.94))

    cov = rm.compute_covariance(r)

    assert cov.shape == (4, 4)
    assert _is_psd(cov)


def test_risk_model_ledoit_wolf():
    rng = np.random.default_rng(3)
    r = rng.normal(size=(300, 5))

    rm = CovarianceRiskModel(RiskModelConfig(method="ledoit_wolf"))

    cov = rm.compute_covariance(r)

    assert cov.shape == (5, 5)
    assert _is_psd(cov)
