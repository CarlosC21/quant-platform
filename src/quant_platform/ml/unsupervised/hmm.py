# src/quant_platform/ml/unsupervised/hmm.py
from __future__ import annotations

from typing import Optional, Any
import numpy as np
import pandas as pd

from scipy.special import logsumexp

from quant_platform.ml.unsupervised.base import UnsupervisedModel


class RegimeHMM(UnsupervisedModel):
    """
    Hidden Markov Model-based regime detector.

    Wraps hmmlearn's GaussianHMM with the UnsupervisedModel protocol.

    Provides:
      - fit(X): estimate HMM parameters
      - predict(X): most likely regime sequence (Viterbi path)
      - transform(X): smoothed state probabilities (N x n_states)
      - score_samples(X): per-sample log-likelihoods
    """

    def __init__(
        self,
        n_states: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: Optional[int] = 42,
        verbose: bool = False,
        **kwargs: Any,
    ):
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore[attr-defined]
        except Exception as exc:
            raise ImportError(
                "hmmlearn is required to use RegimeHMM. "
                "Install via `pip install hmmlearn`."
            ) from exc

        # store config (needed by pipeline/tests)
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

        # underlying hmmlearn model
        self._GaussianHMM = GaussianHMM
        self.model = self._GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )

        self.fitted_: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_array(self, X) -> np.ndarray:
        """
        Ensure X is a 2D array shaped (n_samples, n_features).

        This is crucial so that hmmlearn interprets rows as observations.
        """
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy()
        else:
            arr = np.asarray(X)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        return arr

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------
    def fit(self, X) -> "RegimeHMM":
        """Fit the HMM on full historical data."""
        X_arr = self._to_array(X)
        self.model.fit(X_arr)
        self.fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        """Return most likely regime sequence (Viterbi path)."""
        if not self.fitted_:
            raise RuntimeError("RegimeHMM must be fitted before calling predict().")
        X_arr = self._to_array(X)
        return self.model.predict(X_arr)

    def transform(self, X) -> np.ndarray:
        """
        Return smoothed state probabilities for each observation.

        Shape: (n_samples, n_states)
        """
        if not self.fitted_:
            raise RuntimeError("RegimeHMM must be fitted before calling transform().")
        X_arr = self._to_array(X)
        return self.model.predict_proba(X_arr)

    def score_samples(self, X) -> np.ndarray:
        """
        Return per-sample log-likelihoods (vector length T).

        hmmlearn's GaussianHMM.score_samples returns:
            - total logprob (scalar)
            - posteriors (T x n_states)

        That is not per-observation logprob, so we derive it from
        frame-wise log-likelihoods via logsumexp over states.
        """
        if not self.fitted_:
            raise RuntimeError(
                "RegimeHMM must be fitted before calling score_samples()."
            )

        X_arr = self._to_array(X)

        # frame_logprob: shape (T, n_states)
        frame_logprob = self.model._compute_log_likelihood(X_arr)

        # per-sample log p(x_t) = logsumexp_s log p(x_t | state=s)
        per_sample = logsumexp(frame_logprob, axis=1)

        return per_sample
