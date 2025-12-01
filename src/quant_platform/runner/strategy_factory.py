from __future__ import annotations

from typing import Type, Dict, Any

from quant_platform.runner.core import Strategy

# Registry: maps name -> Strategy class
STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {}


def register_strategy(name: str, cls: Type[Strategy]) -> None:
    """
    Register a Strategy class by string name so config-driven backtests
    can instantiate it.
    """
    STRATEGY_REGISTRY[name] = cls


def create_strategy(
    strategy_name: str,
    params: Dict[str, Any] | None = None,
) -> Strategy:
    """
    Instantiate a registered Strategy class.

    Rules:
    - Remove the "name" field from params
    - If params empty, call cls() with no arguments
    - Strategy classes do NOT need to implement __init__(**kwargs)
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy '{strategy_name}' not registered. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    cls = STRATEGY_REGISTRY[strategy_name]

    if params is None:
        return cls()  # type: ignore[call-arg]

    params = dict(params)
    params.pop("name", None)

    if not params:
        return cls()  # type: ignore[call-arg]

    return cls(**params)  # type: ignore[call-arg]


# ======================================================================
# Built-in Strategy Registration (Week 12)
# ======================================================================

try:
    # Import the example stat-arb strategy for CLI availability
    from quant_platform.examples.stat_arb_backtest_with_execution import (
        StatArbExecutionStrategy,
    )

    # Register it under name used in config files
    register_strategy("stat_arb_exec", StatArbExecutionStrategy)

except Exception:
    # Do nothing if example module missing (safe for packaging/tests)
    pass
