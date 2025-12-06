# src/quant_platform/examples/__init__.py

"""
Example strategies registration
"""

from quant_platform.runner.strategy_factory import register_strategy

# Import your fail-safe stat-arb strategy
from quant_platform.examples.stat_arb_backtest_with_execution import (
    StatArbExecutionStrategy,
)

# Register it under the correct name
register_strategy("stat_arb_exec", StatArbExecutionStrategy)
