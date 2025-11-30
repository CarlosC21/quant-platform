from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ExecutionEvent:
    timestamp: datetime
    type: str
    order_id: str
    payload: dict[str, Any]
