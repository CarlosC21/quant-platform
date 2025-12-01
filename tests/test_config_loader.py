from __future__ import annotations

from pathlib import Path
import json

from quant_platform.runner.config.loader import load_config, seed_everything
from quant_platform.runner.config.models import BacktestConfig


def test_load_config_yaml(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
name: test_run
strategy:
  params:
    lookback: 20
execution:
  latency_seconds: 0.5
seeds:
  global_seed: 123
data_source: "/tmp/data.parquet"
"""
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, BacktestConfig)
    assert cfg.name == "test_run"
    assert cfg.strategy.params["lookback"] == 20
    assert cfg.execution.latency_seconds == 0.5
    assert cfg.seeds.global_seed == 123
    assert cfg.data_source == "/tmp/data.parquet"


def test_load_config_json(tmp_path: Path):
    cfg_path = tmp_path / "cfg.json"
    data = {
        "name": "abc",
        "execution": {"latency_seconds": 1.0},
        "strategy": {"params": {"x": 1}},
        "seeds": {"global_seed": 111},
    }
    cfg_path.write_text(json.dumps(data))
    cfg = load_config(cfg_path)
    assert cfg.name == "abc"
    assert cfg.execution.latency_seconds == 1.0
    assert cfg.strategy.params["x"] == 1
    assert cfg.seeds.global_seed == 111


def test_seed_everything_global():
    cfg = BacktestConfig(seeds={"global_seed": 999})
    seed_everything(cfg)
    # Validate deterministic random numbers
    a = [__import__("random").random(), __import__("numpy").random.rand()]
    seed_everything(cfg)
    b = [__import__("random").random(), __import__("numpy").random.rand()]
    assert a == b
