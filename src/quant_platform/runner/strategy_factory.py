from __future__ import annotations

from typing import Type, Dict, Any
import warnings

# All imports MUST be at the top (fixes E402)
from quant_platform.runner.core import Strategy
from quant_platform.runner.strategy_meta import (
    get_strategy_param_schema,
    StrategyParamInfo,
)

# Import strategy modules **at top**, so registration happens automatically
# This fixes E402 — cannot import inside lower part of module
from quant_platform.examples.stat_arb_backtest_with_execution import (
    StatArbExecutionStrategy,
)


# ===============================================================
# Strategy Registry
# ===============================================================

STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {}


def register_strategy(name: str, cls: Type[Strategy]) -> None:
    """Register a Strategy class by its name."""
    STRATEGY_REGISTRY[name] = cls


def get_strategy_class(name: str) -> Type[Strategy]:
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy '{name}' not registered. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name]


# ===============================================================
# Schema Retrieval
# ===============================================================


def get_strategy_schema(name: str) -> Dict[str, StrategyParamInfo]:
    cls = get_strategy_class(name)
    return get_strategy_param_schema(cls)


# ===============================================================
# Param Validation
# ===============================================================


def _validate_and_filter_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_strategy_schema(name)
    if not schema:
        return params

    missing = [p for p, info in schema.items() if info.required and p not in params]
    if missing:
        raise ValueError(
            f"Missing required parameters for strategy '{name}': {missing}"
        )

    filtered: Dict[str, Any] = {}

    for p, v in params.items():
        info = schema.get(p)
        if info is None:
            warnings.warn(
                f"Unknown parameter '{p}' for strategy '{name}'. Ignoring.",
                stacklevel=2,
            )
            continue

        expected_type = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }.get(info.type)

        if expected_type is not None and not isinstance(v, expected_type):
            raise TypeError(
                f"Parameter '{p}' expected {info.type}, got {type(v).__name__}"
            )

        filtered[p] = v

    return filtered


# ===============================================================
# Strategy Creation
# ===============================================================


def create_strategy(
    strategy_name: str,
    params: Dict[str, Any] | None = None,
) -> Strategy:
    if strategy_name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy '{strategy_name}' not registered. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    cls = STRATEGY_REGISTRY[strategy_name]

    if params is None:
        return cls()  # type: ignore

    params = dict(params)
    params.pop("name", None)

    params = _validate_and_filter_params(strategy_name, params)

    if not params:
        return cls()  # type: ignore

    return cls(**params)  # type: ignore


# ===============================================================
# REGISTER STRATEGIES (importing at top ensures this runs)
# ===============================================================

register_strategy("stat_arb_exec", StatArbExecutionStrategy)

print(">>> strategy_factory loaded. Registered strategies:", STRATEGY_REGISTRY.keys())
