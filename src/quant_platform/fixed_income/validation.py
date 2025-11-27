# src/quant_platform/fixed_income/validation.py

from typing import List

from quant_platform.fixed_income.schemas import YieldPoint


def validate_yield_points(points: List[YieldPoint]) -> None:
    """
    Domain-specific validation for fixed income yields.
    Ensures:
        - No negative yields
        - Sorted maturities per date
        - No duplicate maturities per date

    NOTE:
        We do NOT enforce continuous daily coverage because
        FRED and many fixed-income data sources naturally omit
        weekends, holidays, or missing observations.
    """

    if not points:
        raise ValueError("No yield points provided.")

    # ---- 1. Non-negative yields ----
    for p in points:
        if p.yield_pct < 0:
            raise ValueError(f"Negative yield encountered: {p}")

    # ---- 2. Group by date ----
    by_date = {}
    for p in points:
        by_date.setdefault(p.date, []).append(p)

    # ---- 3. Check per-date maturities ----
    for d, rows in by_date.items():
        maturities = [r.maturity_months for r in rows]

        # Duplicate maturity check
        if len(maturities) != len(set(maturities)):
            raise ValueError(f"Duplicate maturity points on {d}: {maturities}")

        # Ascending maturity check
        if sorted(maturities) != maturities:
            raise ValueError(f"Maturities must be sorted for {d}: {maturities}")

    # All good
    return None
