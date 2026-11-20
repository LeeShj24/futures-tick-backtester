"""Shared test helpers (imported directly by the test modules)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ftb import (
    ZERO_LATENCY,
    Backtester,
    BacktestResult,
    Context,
    QuoteEvent,
    Side,
    Strategy,
    TradeEvent,
    default_contracts,
)
from ftb.events import MarketEvent
from ftb.execution import TopOfBookExecution
from ftb.orders import Fill, Order, OrderRequest, OrderType, ReportType

MS = 1_000_000
# 2024-01-02 09:30:00 America/New_York == 14:30:00 UTC
T0 = 1_704_205_800 * 1_000_000_000

REGISTRY = default_contracts()


def q(ts: int, bid: float, ask: float, bid_size: int = 10, ask_size: int = 10, symbol: str = "NQ"):
    return QuoteEvent(ts, symbol, bid, bid_size, ask, ask_size)


def t(ts: int, price: float, size: int = 1, aggressor: Side = Side.NONE, symbol: str = "NQ"):
    return TradeEvent(ts, symbol, price, size, aggressor)


_next_id = iter(range(1, 10_000_000))


def req(
    side: Side,
    qty: int,
    otype: OrderType,
    *,
    limit: float | None = None,
    stop: float | None = None,
    symbol: str = "NQ",
    order_id: int | None = None,
) -> OrderRequest:
    return OrderRequest(
        id=order_id if order_id is not None else next(_next_id),
        symbol=symbol,
        side=side,
        qty=qty,
        type=otype,
        limit_price=limit,
        stop_price=stop,
    )


def exchange(**config) -> TopOfBookExecution:
    model = TopOfBookExecution(**config)
    model.bind(REGISTRY)
    return model


def fills(reports) -> list[tuple[float, int, str]]:
    return [
        (r.fill_price, r.fill_qty, r.liquidity.value) for r in reports if r.type is ReportType.FILL
    ]


def types(reports) -> list[ReportType]:
    return [r.type for r in reports]


class Scripted(Strategy):
    """Runs ``actions[i](ctx)`` on the i-th market event and records everything it sees."""

    def __init__(self, actions: dict[int, Callable[[Context], Any]] | None = None) -> None:
        self.actions = actions or {}
        self.n_events = 0
        self.log: list[tuple[str, int, Any]] = []
        self.fills: list[tuple[int, Fill]] = []
        self.updates: list[tuple[int, int, str]] = []
        self.placed: list[Order] = []
        self.positions_seen: list[tuple[int, int]] = []

    def _event(self, ctx: Context, kind: str, event: MarketEvent) -> None:
        index = self.n_events
        self.n_events += 1
        self.log.append((kind, ctx.now, event))
        self.positions_seen.append((ctx.now, ctx.position(event.symbol)))
        action = self.actions.get(index)
        if action is not None:
            result = action(ctx)
            if isinstance(result, Order):
                self.placed.append(result)

    def on_quote(self, ctx: Context, quote: QuoteEvent) -> None:
        self._event(ctx, "quote", quote)

    def on_trade(self, ctx: Context, trade: TradeEvent) -> None:
        self._event(ctx, "trade", trade)

    def on_fill(self, ctx: Context, fill: Fill) -> None:
        self.fills.append((ctx.now, fill))
        self.log.append(("fill", ctx.now, fill))

    def on_order_update(self, ctx: Context, order: Order) -> None:
        self.updates.append((ctx.now, order.id, order.status.value))
        self.log.append(("order", ctx.now, (order.id, order.status.value)))


def run(events: Iterable[MarketEvent], strategy: Strategy, **kwargs) -> BacktestResult:
    kwargs.setdefault("latency", ZERO_LATENCY)
    return Backtester(strategy, events, **kwargs).run()
