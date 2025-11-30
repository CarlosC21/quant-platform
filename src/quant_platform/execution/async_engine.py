from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Optional

from quant_platform.execution.events import ExecutionEvent
from quant_platform.execution.latency import BaseLatencyModel, FixedLatencyModel
from quant_platform.execution.simulator import ExecutionSimulator
from quant_platform.execution.models import Order, MarketDataSnapshot


class AsyncExecutionEngine:
    """
    Fully asynchronous event-driven execution engine.

    Architecture:
        - inbound orders go into order_queue
        - engine processes each order applying latency + simulation
        - execution events go into event_queue
    """

    def __init__(
        self,
        simulator: Optional[ExecutionSimulator] = None,
        latency_model: Optional[BaseLatencyModel] = None,
    ):
        self.simulator = simulator or ExecutionSimulator()
        self.latency_model = latency_model or FixedLatencyModel(delay_seconds=0.0)

        self.order_queue: asyncio.Queue[Order] = asyncio.Queue()
        self.event_queue: asyncio.Queue[ExecutionEvent] = asyncio.Queue()

    async def submit_order(self, order: Order) -> None:
        """User-facing API to submit an order asynchronously."""
        await self.order_queue.put(order)

    async def run(self, snapshot_provider) -> None:
        """
        Main engine loop.

        snapshot_provider: callable(order) -> MarketDataSnapshot
        """

        while True:
            order = await self.order_queue.get()

            # 1 — SUBMITTED
            submitted_ts = order.timestamp
            await self.event_queue.put(
                ExecutionEvent(
                    timestamp=submitted_ts,
                    type="SUBMITTED",
                    order_id=order.order_id,
                    payload={"order": order},
                )
            )

            # 2 — latency
            delay = self.latency_model.sample_delay(order, None)
            exec_ts = submitted_ts + timedelta(seconds=delay)

            await self.event_queue.put(
                ExecutionEvent(
                    timestamp=exec_ts,
                    type="ACK",
                    order_id=order.order_id,
                    payload={"effective_time": exec_ts},
                )
            )

            # 3 — snapshot for this specific order
            snapshot: MarketDataSnapshot = snapshot_provider(order)

            # 4 — run simulation
            report, fills = self.simulator.simulate_order(order, snapshot)

            for f in fills:
                await self.event_queue.put(
                    ExecutionEvent(
                        timestamp=exec_ts,
                        type="FILL" if f.quantity == order.quantity else "PARTIAL_FILL",
                        order_id=order.order_id,
                        payload={"fill": f},
                    )
                )

            # FINAL
            await self.event_queue.put(
                ExecutionEvent(
                    timestamp=exec_ts,
                    type="FINAL",
                    order_id=order.order_id,
                    payload={"report": report},
                )
            )
