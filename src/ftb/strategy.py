"""Strategy API.

A strategy is a plain object with callbacks. Everything it can know arrives
through those callbacks or through the :class:`Context`, which only exposes
state built from events that have *already* been delivered:

* market data up to and including the current event;
* order states and positions as of the execution reports received so far
  (reports are delayed by the latency model).

The strategy never sees the data iterator, the simulated exchange, or the
ground-truth ledger, so it cannot peek ahead.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ftb.contracts import ContractSpec
from ftb.events import QuoteEvent, Side, TradeEvent, ns_to_datetime
from ftb.orders import Fill, Order, OrderType

if TYPE_CHECKING:
    from ftb.engine import Backtester


class Strategy:
    """Base class: override the callbacks you need."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def on_start(self, ctx: Context) -> None:
        """Called once before the first event."""

    def on_quote(self, ctx: Context, quote: QuoteEvent) -> None:
        """Top-of-book update."""

    def on_trade(self, ctx: Context, trade: TradeEvent) -> None:
        """A print on the tape."""

    def on_fill(self, ctx: Context, fill: Fill) -> None:
        """One of our orders was (partially) filled. ``ctx.now`` >= ``fill.ts``."""

    def on_order_update(self, ctx: Context, order: Order) -> None:
        """Order status changed (accepted, filled, cancelled, rejected, cancel rejected)."""

    def on_halt(self, ctx: Context, reason: str) -> None:
        """A risk rule halted trading: working orders are cancelled, positions flattened."""

    def on_end(self, ctx: Context) -> None:
        """Called once after the data is exhausted. New orders are rejected."""


class Context:
    """The strategy's window into the simulation (past and present only)."""

    __slots__ = ("_engine",)

    def __init__(self, engine: Backtester) -> None:
        self._engine = engine

    # ------------------------------------------------------------------ time & data
    @property
    def now(self) -> int:
        """Current simulation time (epoch ns)."""
        return self._engine._now

    def now_datetime(self, tz=timezone.utc) -> datetime:
        return ns_to_datetime(self._engine._now, tz)

    def contract(self, symbol: str) -> ContractSpec:
        return self._engine.contracts.resolve(symbol)

    def last_quote(self, symbol: str) -> QuoteEvent | None:
        return self._engine._last_quote.get(symbol)

    def last_trade(self, symbol: str) -> TradeEvent | None:
        return self._engine._last_trade.get(symbol)

    # ------------------------------------------------------------------ positions & orders
    def position(self, symbol: str) -> int:
        """Signed position as known from the fills reported so far."""
        return self._engine._known_pos.get(symbol, 0)

    def order(self, order_id: int) -> Order | None:
        return self._engine._orders.get(order_id)

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        active = self._engine._active.values()
        if symbol is None:
            return list(active)
        return [o for o in active if o.symbol == symbol]

    @property
    def halted(self) -> str | None:
        """Reason trading was halted by a risk rule, if any."""
        return self._engine._halted

    # ------------------------------------------------------------------ order entry
    def submit_market(self, symbol: str, side: Side, qty: int, tag: str = "") -> Order:
        return self._engine._submit(symbol, side, qty, OrderType.MARKET, tag=tag)

    def submit_limit(self, symbol: str, side: Side, qty: int, price: float, tag: str = "") -> Order:
        return self._engine._submit(symbol, side, qty, OrderType.LIMIT, limit_price=price, tag=tag)

    def submit_stop(
        self, symbol: str, side: Side, qty: int, stop_price: float, tag: str = ""
    ) -> Order:
        return self._engine._submit(
            symbol, side, qty, OrderType.STOP, stop_price=stop_price, tag=tag
        )

    def buy(self, symbol: str, qty: int = 1, tag: str = "") -> Order:
        return self.submit_market(symbol, Side.BUY, qty, tag)

    def sell(self, symbol: str, qty: int = 1, tag: str = "") -> Order:
        return self.submit_market(symbol, Side.SELL, qty, tag)

    def cancel(self, order: Order | int) -> bool:
        """Request a cancel (processed after cancel latency). False if nothing to cancel."""
        order_id = order if isinstance(order, int) else order.id
        return self._engine._cancel(order_id)

    def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel working limit/stop orders (in-flight market orders are left alone)."""
        return sum(
            self.cancel(o) for o in self.open_orders(symbol) if o.type is not OrderType.MARKET
        )

    def flatten(
        self, symbol: str, tag: str = "flatten", cancel_orders: bool = True
    ) -> Order | None:
        """Cancel working orders (optional) and send a market order closing the known position.

        In-flight market orders are taken into account so repeated calls do not
        over-flatten.
        """
        if cancel_orders:
            self.cancel_all(symbol)
        return self._engine._flatten(symbol, tag=tag)
