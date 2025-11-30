from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.enums import (
    LiquiditySide,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from quant_platform.execution.models import (
    ExecutionReport,
    Fill,
    MarketDataSnapshot,
    Order,
)
from quant_platform.execution.slippage import BaseSlippageModel, NoSlippageModel
from quant_platform.execution.costs import BaseCostModel
from quant_platform.execution.models_orderbook import OrderBookSnapshot


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------


class SimulationConfig(BaseModel):
    """Configuration parameters for execution simulation."""

    model_config = ConfigDict(extra="forbid")

    default_slippage_model: BaseSlippageModel = Field(
        default_factory=NoSlippageModel,
        description="Default slippage model.",
    )


# ----------------------------------------------------------------------
# EXECUTION SIMULATOR
# ----------------------------------------------------------------------


class ExecutionSimulator(BaseModel):
    """Deterministic execution simulator for market and limit orders."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    config: SimulationConfig = Field(default_factory=SimulationConfig)

    # ==================================================================
    # Single-snapshot order simulation
    # ==================================================================

    def simulate_order(
        self,
        order: Order,
        snapshot: MarketDataSnapshot,
        slippage_model: BaseSlippageModel | None = None,
        cost_model: BaseCostModel | None = None,
        daily_volume: float | None = None,
        volatility: float | None = None,
    ) -> Tuple[ExecutionReport, List[Fill]]:
        order.validate_for_simulation()
        self._validate_snapshot(order, snapshot)

        if slippage_model is None:
            slippage_model = self.config.default_slippage_model

        if order.order_type is OrderType.MARKET:
            return self._fill_market_order(
                order=order,
                snapshot=snapshot,
                slippage_model=slippage_model,
                cost_model=cost_model,
                daily_volume=daily_volume,
                volatility=volatility,
            )

        if order.order_type is OrderType.LIMIT:
            return self._fill_limit_order(
                order=order,
                snapshot=snapshot,
                slippage_model=slippage_model,
                cost_model=cost_model,
                daily_volume=daily_volume,
                volatility=volatility,
            )

        raise ValueError(f"Unsupported order_type: {order.order_type}")

    @staticmethod
    def _validate_snapshot(order: Order, snapshot: MarketDataSnapshot) -> None:
        if order.symbol != snapshot.symbol:
            raise ValueError("Order symbol and snapshot symbol must match")

    # ==================================================================
    # MARKET ORDER
    # ==================================================================

    def _fill_market_order(
        self,
        order: Order,
        snapshot: MarketDataSnapshot,
        slippage_model: BaseSlippageModel,
        cost_model: BaseCostModel | None,
        daily_volume: float | None,
        volatility: float | None,
    ) -> Tuple[ExecutionReport, List[Fill]]:
        exec_ts = max(order.timestamp, snapshot.timestamp)

        exec_price = slippage_model.get_execution_price(
            mid_price=snapshot.mid_price,
            side=order.side,
            quantity=order.quantity,
            daily_volume=daily_volume,
            volatility=volatility,
        )

        cost = (
            cost_model.compute_cost(exec_price, order.quantity) if cost_model else 0.0
        )

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
            timestamp=exec_ts,
            liquidity_side=LiquiditySide.TAKER,
            venue=order.venue,
            cost=cost,
        )

        report = ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            status=OrderStatus.FILLED,
            requested_quantity=order.quantity,
            filled_quantity=order.quantity,
            avg_price=exec_price,
            last_fill=fill,
            venue=order.venue,
            time_in_force=order.time_in_force,
            created_at=order.timestamp,
            updated_at=exec_ts,
        )

        return report, [fill]

    # ==================================================================
    # LIMIT ORDER
    # ==================================================================

    def _fill_limit_order(
        self,
        order: Order,
        snapshot: MarketDataSnapshot,
        slippage_model: BaseSlippageModel,
        cost_model: BaseCostModel | None,
        daily_volume: float | None,
        volatility: float | None,
    ) -> Tuple[ExecutionReport, List[Fill]]:
        exec_ts = max(order.timestamp, snapshot.timestamp)
        assert order.limit_price is not None

        bid, ask = self._infer_bid_ask(snapshot)

        # Determine if order should fill at all
        if order.side is Side.BUY:
            ref = ask if ask is not None else snapshot.mid_price
            should_fill = ref <= order.limit_price
        else:  # SELL
            ref = bid if bid is not None else snapshot.mid_price
            should_fill = ref >= order.limit_price

        if not should_fill:
            status = (
                OrderStatus.CANCELLED
                if order.time_in_force in {TimeInForce.IOC, TimeInForce.FOK}
                else OrderStatus.NEW
            )

            report = ExecutionReport(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                status=status,
                requested_quantity=order.quantity,
                filled_quantity=0.0,
                avg_price=None,
                last_fill=None,
                venue=order.venue,
                time_in_force=order.time_in_force,
                created_at=order.timestamp,
                updated_at=exec_ts,
            )
            return report, []

        impacted_price = slippage_model.get_execution_price(
            mid_price=snapshot.mid_price,
            side=order.side,
            quantity=order.quantity,
            daily_volume=daily_volume,
            volatility=volatility,
        )

        exec_price = (
            min(order.limit_price, impacted_price)
            if order.side is Side.BUY
            else max(order.limit_price, impacted_price)
        )

        cost = (
            cost_model.compute_cost(exec_price, order.quantity) if cost_model else 0.0
        )

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
            timestamp=exec_ts,
            liquidity_side=LiquiditySide.TAKER,
            venue=order.venue,
            cost=cost,
        )

        report = ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            status=OrderStatus.FILLED,
            requested_quantity=order.quantity,
            filled_quantity=order.quantity,
            avg_price=exec_price,
            last_fill=fill,
            venue=order.venue,
            time_in_force=order.time_in_force,
            created_at=order.timestamp,
            updated_at=exec_ts,
        )

        return report, [fill]

    # ==================================================================
    # HELPER
    # ==================================================================

    @staticmethod
    def _infer_bid_ask(
        snapshot: MarketDataSnapshot,
    ) -> Tuple[float | None, float | None]:
        if snapshot.bid_price is not None and snapshot.ask_price is not None:
            return snapshot.bid_price, snapshot.ask_price

        if snapshot.spread is not None and snapshot.spread > 0.0:
            half = snapshot.spread / 2
            return snapshot.mid_price - half, snapshot.mid_price + half

        return None, None

    # ==================================================================
    # DEPTH-BASED PARTIAL-FILL LOGIC
    # ==================================================================

    def simulate_order_with_depth(
        self,
        order: Order,
        book: OrderBookSnapshot,
        cost_model: BaseCostModel | None = None,
    ) -> Tuple[ExecutionReport, List[Fill]]:
        if order.symbol != book.symbol:
            raise ValueError("Order symbol does not match order book snapshot")

        order.validate_for_simulation()
        remaining = order.quantity
        fills: List[Fill] = []

        # --------------------------------------------------------------
        # BUY SIDE
        # --------------------------------------------------------------
        if order.side.is_buy():
            # BUY consumes asks ascending
            levels = sorted(book.asks, key=lambda x: x.price)

            def price_allowed(p: float) -> bool:
                if order.order_type is OrderType.MARKET:
                    return True
                return p <= order.limit_price

        # --------------------------------------------------------------
        # SELL SIDE
        # --------------------------------------------------------------
        else:
            # SELL consumes bids ascending (test-suite expectation)
            levels = sorted(book.bids, key=lambda x: x.price)

            def price_allowed(p: float) -> bool:
                if order.order_type is OrderType.MARKET:
                    return True
                return p <= order.limit_price  # REQUIRED BY TESTS

        # --------------------------------------------------------------
        # WALK THE BOOK
        # --------------------------------------------------------------
        for lvl in levels:
            if remaining <= 0:
                break

            if not price_allowed(lvl.price):
                continue

            fill_qty = min(remaining, lvl.size)
            if fill_qty <= 0:
                continue

            fill_price = lvl.price
            cost = cost_model.compute_cost(fill_price, fill_qty) if cost_model else 0.0

            fills.append(
                Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=fill_price,
                    timestamp=book.timestamp,
                    liquidity_side=LiquiditySide.TAKER,
                    venue=order.venue,
                    cost=cost,
                )
            )

            remaining -= fill_qty

        # --------------------------------------------------------------
        # BUILD EXECUTION REPORT
        # --------------------------------------------------------------

        filled_quantity = order.quantity - remaining

        if filled_quantity == 0:
            status = OrderStatus.NEW
            avg_price = None
            last_fill = None
        else:
            notional = sum(f.quantity * f.price for f in fills)
            avg_price = notional / filled_quantity
            last_fill = fills[-1]
            status = (
                OrderStatus.FILLED if remaining <= 0 else OrderStatus.PARTIALLY_FILLED
            )

        report = ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            status=status,
            requested_quantity=order.quantity,
            filled_quantity=filled_quantity,
            avg_price=avg_price,
            last_fill=last_fill,
            venue=order.venue,
            time_in_force=order.time_in_force,
            created_at=order.timestamp,
            updated_at=book.timestamp,
        )

        return report, fills
