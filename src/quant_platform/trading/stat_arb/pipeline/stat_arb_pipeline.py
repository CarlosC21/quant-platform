# src/quant_platform/trading/stat_arb/pipeline/stat_arb_pipeline.py
from __future__ import annotations

from typing import Optional

import pandas as pd

from quant_platform.trading.stat_arb.schemas import (
    StatArbPairConfig,
    StatArbPipelineResult,
    StatArbSignal,
)
from quant_platform.trading.stat_arb.cointegration.engle_granger import (
    EngleGrangerTester,
)
from quant_platform.trading.stat_arb.cointegration.schemas import EngleGrangerConfig
from quant_platform.trading.stat_arb.spreads.spread_builder import build_static_spread
from quant_platform.trading.stat_arb.spreads.kalman_filter import (
    KalmanHedgeConfig,
    kalman_hedge_filter,
)
from quant_platform.trading.stat_arb.spreads.ou_model import (
    fit_ou_to_spread,
)
from quant_platform.trading.stat_arb.spreads.zscore import zscore_from_ou
from quant_platform.trading.stat_arb.spreads.regime_filter import (
    RegimeFilterConfig,
)
from quant_platform.trading.stat_arb.signals import build_signals_from_zscores


class StatArbPairPipeline:
    """
    End-to-end stat-arb pipeline for a single pair (Y, X).

    Workflow:
      1. Align Y and X price series.
      2. Engle–Granger test for cointegration, get static beta.
      3. Construct spread:
            - static beta: build_static_spread()
            - or dynamic beta_t: kalman_hedge_filter()
      4. Fit OU to spread path.
      5. Compute OU-based z-scores.
      6. Apply regime filter using RegimeFilterConfig + regime DataFrame.
      7. Build StatArbSignal list.

    This pipeline operates on in-memory pandas Series + regime DataFrame.
    Integration with FeatureStore / RegimeFeatureStore can be done by
    higher-level orchestration that loads these inputs.
    """

    def __init__(
        self,
        pair_config: StatArbPairConfig,
        regime_config: RegimeFilterConfig,
        engle_granger_config: Optional[EngleGrangerConfig] = None,
        kalman_config: Optional[KalmanHedgeConfig] = None,
    ) -> None:
        self.pair_config = pair_config
        self.regime_config = regime_config
        self.eg_config = engle_granger_config or EngleGrangerConfig(min_obs=50)
        self.kalman_config = kalman_config or KalmanHedgeConfig(min_obs=50)

        if self.pair_config.z_exit >= self.pair_config.z_entry:
            raise ValueError(
                "StatArbPairConfig requires z_exit < z_entry for sensible hysteresis."
            )

    def run(
        self,
        series_y: pd.Series,
        series_x: pd.Series,
        regime_df: pd.DataFrame,
    ) -> StatArbPipelineResult:
        """
        Run the full stat-arb pipeline on price series and regime info.

        Parameters
        ----------
        series_y : pd.Series
            Dependent asset Y price history (levels).
        series_x : pd.Series
            Independent asset X price history (levels).
        regime_df : pd.DataFrame
            Regime table with columns:
              - date
              - regime_<name>
              - prob_<name>_<state> (optional)
            Typically from RegimeFeatureStore.load_regime(..., as_pandas=True).

        Returns
        -------
        StatArbPipelineResult
        """
        # Ensure correct names for consistency
        series_y = series_y.copy()
        series_y.name = self.pair_config.symbol_y
        series_x = series_x.copy()
        series_x.name = self.pair_config.symbol_x

        # 1) Engle–Granger cointegration test
        eg_tester = EngleGrangerTester(config=self.eg_config)
        eg_result = eg_tester.test_pair(series_y, series_x)

        if not eg_result.coint:
            if self.pair_config.fail_if_not_coint:
                raise ValueError(
                    f"Pair {self.pair_config.symbol_y}/{self.pair_config.symbol_x} "
                    "is not cointegrated (Engle–Granger)."
                )

            # Build flat, non-tradable signals that still preserve timestamps.
            return self._build_non_coint_result(series_y, regime_df, eg_result)

        # 2) Build spread (static beta or Kalman)
        if self.pair_config.use_kalman:
            spread_result = kalman_hedge_filter(
                series_y=series_y,
                series_x=series_x,
                config=self.kalman_config,
            )
        else:
            spread_result = build_static_spread(
                series_y=series_y,
                series_x=series_x,
                beta=eg_result.beta,
            )

        # 3) Fit OU to spread
        ou_params = fit_ou_to_spread(
            spread=spread_result.spread,
            dt=self.pair_config.dt,
        )

        # 4) Compute OU-based z-scores
        zscore_result = zscore_from_ou(spread_result, ou_params)

        # 5) Build signals using regimes + z-score logic
        signals = build_signals_from_zscores(
            zscore_result=zscore_result,
            z_entry=self.pair_config.z_entry,
            z_exit=self.pair_config.z_exit,
            regime_df=regime_df,
            regime_config=self.regime_config,
        )

        # 6) Prepare meta diagnostics
        meta = {
            "engle_granger": {
                "alpha": eg_result.alpha,
                "beta": eg_result.beta,
                "adf_stat": eg_result.adf_result.test_stat,
                "adf_crit_5": eg_result.adf_result.crit_5,
                "coint": eg_result.coint,
                "n_obs": eg_result.n_obs,
            },
            "ou_params": {
                "kappa": ou_params.kappa,
                "theta": ou_params.theta,
                "sigma": ou_params.sigma,
                "dt": ou_params.dt,
                "half_life": ou_params.half_life,
            },
            "config": {
                "use_kalman": self.pair_config.use_kalman,
            },
        }

        return StatArbPipelineResult(
            pair_config=self.pair_config,
            signals=signals,
            cointegrated=True,
            meta=meta,
        )

    def _build_non_coint_result(
        self,
        series_y: pd.Series,
        regime_df: pd.DataFrame,
        eg_result,
    ) -> StatArbPipelineResult:
        """
        Build flat, non-tradable StatArbSignal list for a non-cointegrated pair.
        """
        # Align regime_df to Y timestamps so that the signal timestamps are valid.
        y_idx = series_y.index
        # Just ensure there is a 'date' column to resemble RegimeFeatureStore output
        if "date" in regime_df.columns:
            reg_dates = pd.to_datetime(regime_df["date"])
            regime_df = regime_df.copy()
            regime_df["date"] = reg_dates
        else:
            # fabricate dates from Y index for this path
            regime_df = regime_df.copy()
            regime_df["date"] = y_idx

        # For non-cointegrated pairs, we skip spread/OU/z-score and just
        # return 'flat', non-tradable signals.
        signals: list[StatArbSignal] = []
        for ts in y_idx:
            ts_py = pd.Timestamp(ts).to_pydatetime()
            signals.append(
                StatArbSignal(
                    timestamp=ts_py,
                    symbol_y=self.pair_config.symbol_y,
                    symbol_x=self.pair_config.symbol_x,
                    zscore=0.0,
                    spread=0.0,
                    side="flat",
                    z_entry=self.pair_config.z_entry,
                    z_exit=self.pair_config.z_exit,
                    regime=None,
                    tradable=False,
                    reason="not_cointegrated",
                    meta={},
                )
            )

        meta = {
            "engle_granger": {
                "alpha": eg_result.alpha,
                "beta": eg_result.beta,
                "adf_stat": eg_result.adf_result.test_stat,
                "adf_crit_5": eg_result.adf_result.crit_5,
                "coint": eg_result.coint,
                "n_obs": eg_result.n_obs,
            },
            "ou_params": None,
            "config": {
                "use_kalman": self.pair_config.use_kalman,
            },
        }

        return StatArbPipelineResult(
            pair_config=self.pair_config,
            signals=signals,
            cointegrated=False,
            meta=meta,
        )
