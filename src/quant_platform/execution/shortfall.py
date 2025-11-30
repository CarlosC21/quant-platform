from __future__ import annotations

from quant_platform.execution.models import Fill
from quant_platform.execution.enums import Side


def implementation_shortfall(
    fills: list[Fill],
    ideal_price: float,
    side: Side,
) -> float:
    """
    IS = (actual_price - ideal_price) * signed_qty + sum(costs)

    Positive IS = worse execution.
    """

    if not fills:
        return 0.0

    signed = side.sign * sum(f.quantity for f in fills)
    vwap = sum(f.quantity * f.price for f in fills) / sum(f.quantity for f in fills)
    fees = sum(f.cost for f in fills)

    return (vwap - ideal_price) * signed + fees
