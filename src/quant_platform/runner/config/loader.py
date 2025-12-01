from __future__ import annotations

import json
import yaml
import random
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from quant_platform.runner.config.models import BacktestConfig


def load_config(path: str | Path) -> BacktestConfig:
    """
    Load a BacktestConfig from YAML or JSON.

    Automatically validates using Pydantic v2.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    text = path.read_text()

    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(text)
        elif path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raise ValueError("Config path must be YAML or JSON.")
    except Exception as e:
        raise ValueError(f"Failed to parse config: {e}") from e

    try:
        return BacktestConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid BacktestConfig: {e}") from e


def seed_everything(cfg: BacktestConfig) -> None:
    """
    Set Python & NumPy random seeds for reproducibility.

    Priority:
        1. global_seed
        2. python + numpy
    """
    seed = None

    if cfg.seeds.global_seed is not None:
        seed = cfg.seeds.global_seed
        random.seed(seed)
        np.random.seed(seed)
        return

    if cfg.seeds.python is not None:
        random.seed(cfg.seeds.python)

    if cfg.seeds.numpy is not None:
        np.random.seed(cfg.seeds.numpy)
