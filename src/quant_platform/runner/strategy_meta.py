"""
Strategy metadata & parameter introspection utilities.

Goal:
- Allow strategies to optionally declare a PARAM_SCHEMA so that
  UIs/CLIs can dynamically render configuration forms.
- This is strictly additive and non-breaking: if a strategy does
  not declare PARAM_SCHEMA, everything still works as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

from quant_platform.runner.core import Strategy


@dataclass
class StrategyParamInfo:
    """
    Lightweight description of a single strategy parameter.

    Fields are intentionally simple so that:
      - Streamlit / CLI / other frontends can easily map them
        to widgets / arguments.
      - Strategy authors can declare PARAM_SCHEMA without pulling
        in heavy dependencies (no Pydantic required).
    """

    name: str
    type: str | None = None
    default: Any | None = None
    required: bool = False
    description: str | None = None

    # Optional UI hints
    choices: List[Any] | None = None
    min: float | None = None
    max: float | None = None


def get_strategy_param_schema(cls: Type[Strategy]) -> Dict[str, StrategyParamInfo]:
    """
    Introspect a Strategy class to obtain a parameter schema.
    """

    raw = getattr(cls, "PARAM_SCHEMA", None)
    if raw is None or not isinstance(raw, dict):
        return {}

    out: Dict[str, StrategyParamInfo] = {}

    for name, spec in raw.items():
        if not isinstance(spec, dict):
            spec = {}

        out[name] = StrategyParamInfo(
            name=name,
            type=spec.get("type"),
            default=spec.get("default"),
            required=bool(spec.get("required", False)),
            description=spec.get("description"),
            choices=spec.get("choices"),
            min=spec.get("min"),
            max=spec.get("max"),
        )

    return out
