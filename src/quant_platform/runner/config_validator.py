from __future__ import annotations

from typing import Dict, Any, List

from quant_platform.runner.strategy_factory import STRATEGY_REGISTRY
from quant_platform.runner.strategy_meta import get_strategy_param_schema


class ConfigValidationError(Exception):
    """Raised when a config file does not meet strategy requirements."""

    pass


def validate_strategy_params(strategy_name: str, params: Dict[str, Any]) -> List[str]:
    """
    Validate user-provided strategy parameters against PARAM_SCHEMA.
    Returns a list of warnings (non-fatal). Raises on fatal errors.
    """
    errors = []
    warnings = []

    # Does strategy exist?
    if strategy_name not in STRATEGY_REGISTRY:
        raise ConfigValidationError(
            f"Strategy '{strategy_name}' not registered. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    cls = STRATEGY_REGISTRY[strategy_name]
    schema = get_strategy_param_schema(cls)

    # If no schema → nothing to validate
    if not schema:
        return warnings

    supplied = dict(params)
    supplied.pop("name", None)

    # Required parameters missing
    for p, info in schema.items():
        if info.required and p not in supplied:
            errors.append(f"Missing required strategy parameter: '{p}'")

    # Unknown parameters
    for p in supplied:
        if p not in schema:
            warnings.append(f"Unknown parameter '{p}' for strategy '{strategy_name}'")

    # Type checks (lightweight)
    for p, info in schema.items():
        if p in supplied:
            val = supplied[p]
            t = info.type
            # simple type enforcement
            if t == "int" and not isinstance(val, int):
                errors.append(f"Param '{p}' must be int (got {type(val).__name__})")
            if t == "float" and not isinstance(val, (int, float)):
                errors.append(f"Param '{p}' must be float (got {type(val).__name__})")
            if t == "bool" and not isinstance(val, bool):
                errors.append(f"Param '{p}' must be bool (got {type(val).__name__})")
            if t == "choice" and info.choices and val not in info.choices:
                errors.append(
                    f"Param '{p}' must be one of {info.choices} (got '{val}')"
                )

    if errors:
        raise ConfigValidationError("\n".join(errors))

    return warnings


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """
    Validate an entire config dict.
    Returns warnings. Raises ConfigValidationError for fatal issues.
    """
    if "strategy" not in cfg:
        raise ConfigValidationError("Config missing 'strategy' section")

    strat_params = cfg["strategy"].get("params", {})
    strategy_name = strat_params.get("name")

    if strategy_name is None:
        raise ConfigValidationError("Config must include strategy.params.name")

    # Validate strategy parameters
    warnings = validate_strategy_params(strategy_name, strat_params)

    return warnings
