from quant_platform.runner.strategy_factory import register_strategy
from quant_platform.examples.stat_arb_backtest_with_execution import (
    StatArbExecutionStrategy,
)

# Register with the global registry
register_strategy("stat_arb_exec", StatArbExecutionStrategy)
