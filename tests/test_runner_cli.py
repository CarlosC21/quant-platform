from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from quant_platform.runner.run import run_from_config
from quant_platform.runner.strategy_factory import register_strategy
from quant_platform.runner.core import Strategy, RunContext


class AlwaysFlatStrategy(Strategy):
    def on_start(self, context: RunContext):
        return

    def on_bar(self, context: RunContext, bar_data: pd.DataFrame):
        return

    def on_end(self, context: RunContext):
        return


def test_run_from_config(tmp_path: Path):
    # Register simple strategy
    register_strategy("always_flat", AlwaysFlatStrategy)

    # Create simple market data
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "symbol": ["AAA", "AAA", "AAA"],
            "close": [100, 102, 101],
        }
    )

    data_path = tmp_path / "md.csv"
    df.to_csv(data_path, index=False)

    cfg = {
        "name": "test_run",
        "strategy": {"params": {"name": "always_flat"}},
        "seeds": {"global_seed": 1},
        "data_source": str(data_path),
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    result = run_from_config(cfg_path)
    assert result.equity_curve.shape[0] == 3
