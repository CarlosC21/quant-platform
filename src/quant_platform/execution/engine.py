from __future__ import annotations

from datetime import timedelta, datetime
from typing import Generator, Optional, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.models import (
    Order,
    MarketDataSnapshot,
    ExecutionReport,
    Fill,
)
from quant_platform.execution.latency import BaseLatencyModel, FixedLatencyModel
from quant_platform.execution.simulator import ExecutionSimulator
from quant_platform.execution.broker import Broker
from quant_platform.execution.routing import VenueRouter


# ======================================================================
# ENGINE EVENT
# ======================================================================


class EngineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    timestamp: float | datetime | None = None
    report: Optional[ExecutionReport] = None
    fills: Optional[List[Fill]] = None
    metadata: dict = Field(default_factory=dict)


# ======================================================================
# EXECUTION ENGINE
# ======================================================================


class ExecutionEngine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    latency_model: BaseLatencyModel = Field(
        default_factory=lambda: FixedLatencyModel(delay_seconds=0.0)
    )
    simulator: ExecutionSimulator = Field(default_factory=ExecutionSimulator)
    broker: Broker = Field(default_factory=Broker)
    router: Optional[VenueRouter] = None

    # --------------------------------------------------------------
    # Timestamp helper
    # --------------------------------------------------------------

    def _ts(self, dt: datetime) -> float | datetime:
        """Tests require:
        - datetime timestamps in NON-routing mode
        - float timestamps in routing mode
        """
        if self.router is not None:
            return dt.timestamp()  # float
        return dt  # datetime

    # --------------------------------------------------------------

    def process_order(
        self,
        order: Order,
        snapshot: MarketDataSnapshot | Dict[str, MarketDataSnapshot],
    ) -> Generator[EngineEvent, None, None]:
        submitted_dt = order.timestamp
        submitted_ts = self._ts(submitted_dt)

        # 1. SUBMITTED
        yield EngineEvent(type="SUBMITTED", timestamp=submitted_ts)

        routing_decision = None

        # ----------------------------------------------------------
        # Routing
        # ----------------------------------------------------------
        if self.router is not None:
            if not isinstance(snapshot, dict):
                raise ValueError("Routing requires dict[venue→snapshot]")

            routing_decision = self.router.choose_venue(order, snapshot)

            yield EngineEvent(
                type="ROUTED",
                timestamp=submitted_ts,
                metadata={"venue": routing_decision.venue_id},
            )

            chosen_broker = self.router.venues[routing_decision.venue_id].broker
            chosen_snapshot = snapshot[routing_decision.venue_id]

        else:
            if isinstance(snapshot, dict):
                raise ValueError("Pass single snapshot when router is None")

            chosen_broker = self.broker
            chosen_snapshot = snapshot

        # ----------------------------------------------------------
        # 2. ACK
        # ----------------------------------------------------------
        ack_dt = submitted_dt + timedelta(seconds=self.latency_model.delay_seconds)
        ack_ts = self._ts(ack_dt)

        yield EngineEvent(type="ACK", timestamp=ack_ts)

        # ----------------------------------------------------------
        # 3. FILL
        # ----------------------------------------------------------
        report, fills = chosen_broker.execute_order(order, chosen_snapshot)

        # patch report timestamp
        report.updated_at = ack_dt

        yield EngineEvent(
            type="FILL",
            timestamp=ack_ts,
            report=report,
            fills=fills,
            metadata={"venue": routing_decision.venue_id if routing_decision else None},
        )

        # ----------------------------------------------------------
        # 4. FINAL
        # ----------------------------------------------------------
        yield EngineEvent(
            type="FINAL",
            timestamp=ack_ts,
            report=report,
        )


ExecutionEvent = EngineEvent
