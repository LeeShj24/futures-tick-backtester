"""No-look-ahead guarantees.

The strongest check is *prefix invariance*: if two datasets are identical up to
time T and differ afterwards, everything the strategy saw and did before T must
be identical in both runs. Any leak of future data breaks it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ftb import (
    ConstantLatency,
    Context,
    DataOrderError,
    QuoteEvent,
    Side,
    Strategy,
    TradeEvent,
    generate_events,
)
from ftb.orders import OrderStatus
from helpers import MS, T0, Scripted, q, run, t


class ActiveTrader(Strategy):
    """Trades often, using only what it has observed; checks invariants on every callback."""

    def __init__(self) -> None:
        self.decisions: list[tuple[int, str, float]] = []
        self.seen: list[tuple[str, int]] = []
        self.n_trades = 0

    def _check(self, ctx: Context, event_ts: int | None = None) -> None:
        if event_ts is not None:
            assert ctx.now == event_ts
        lq, lt = ctx.last_quote("NQ"), ctx.last_trade("NQ")
        assert lq is None or lq.ts <= ctx.now
        assert lt is None or lt.ts <= ctx.now
        for o in ctx.open_orders():
            assert o.submit_ts <= ctx.now
            for f in o.fills:
                assert f.ts <= ctx.now

    def on_quote(self, ctx: Context, quote: QuoteEvent) -> None:
        self._check(ctx, quote.ts)
        self.seen.append(("q", quote.ts))

    def on_trade(self, ctx: Context, trade: TradeEvent) -> None:
        self._check(ctx, trade.ts)
        self.seen.append(("t", trade.ts))
        self.n_trades += 1
        if self.n_trades % 25:
            return
        pos = ctx.position("NQ")
        quote = ctx.last_quote("NQ")
        if quote is None or not quote.has_bid or not quote.has_ask:
            return
        if pos == 0 and not ctx.open_orders("NQ"):
            ctx.buy("NQ", 1)
            ctx.submit_limit("NQ", Side.BUY, 1, quote.bid_price - 0.5)
            self.decisions.append((ctx.now, "buy+bid", quote.ask_price))
        elif pos > 0:
            ctx.cancel_all("NQ")
            ctx.submit_limit("NQ", Side.SELL, pos, quote.ask_price)
            self.decisions.append((ctx.now, "offer", quote.ask_price))

    def on_fill(self, ctx, fill) -> None:
        self._check(ctx)
        assert fill.ts <= ctx.now
        self.decisions.append((ctx.now, f"fill:{fill.side.name}", fill.price))


def _strict_cut(events, fraction=0.5):
    """Index i such that events[i].ts > events[i-1].ts (a clean time boundary)."""
    i = int(len(events) * fraction)
    while events[i].ts == events[i - 1].ts:
        i += 1
    return i


def _perturb(event):
    if isinstance(event, QuoteEvent):
        return replace(event, bid_price=event.bid_price + 50.0, ask_price=event.ask_price + 50.0)
    return replace(event, price=event.price + 50.0, size=event.size + 3)


def test_prefix_invariance():
    events = generate_events(days=1, events_per_second=1.0, seed=3)
    cut = _strict_cut(events)
    boundary = events[cut].ts
    altered = events[:cut] + [_perturb(e) for e in events[cut:]]

    latency = ConstantLatency(order_ms=5, report_ms=5)
    a, b = ActiveTrader(), ActiveTrader()
    res_a = run(events, a, latency=latency)
    res_b = run(altered, b, latency=latency)

    def before(items, key):
        return [x for x in items if key(x) < boundary]

    assert len(before(a.decisions, lambda d: d[0])) > 10  # the test has teeth
    assert before(a.decisions, lambda d: d[0]) == before(b.decisions, lambda d: d[0])
    assert before(a.seen, lambda s: s[1]) == before(b.seen, lambda s: s[1])
    assert before(res_a.fills, lambda f: f.ts) == before(res_b.fills, lambda f: f.ts)
    # ...and the futures really were different.
    assert a.decisions != b.decisions


def test_engine_pulls_one_event_at_a_time():
    events = generate_events(days=1, events_per_second=0.2, seed=5)[:2_000]

    class CountingIterator:
        def __init__(self, items):
            self._it = iter(items)
            self.pulled = 0

        def __iter__(self):
            return self

        def __next__(self):
            item = next(self._it)
            self.pulled += 1
            return item

    data = CountingIterator(events)

    class Probe(Strategy):
        delivered = 0

        def _on(self, ctx):
            Probe.delivered += 1
            assert data.pulled == Probe.delivered  # nothing fetched beyond "now"

        def on_quote(self, ctx, quote):
            self._on(ctx)

        def on_trade(self, ctx, trade):
            self._on(ctx)

    run(data, Probe())
    assert Probe.delivered == len(events)


def test_fills_never_precede_order_latency():
    events = generate_events(days=1, events_per_second=1.0, seed=8)[:20_000]
    strat = ActiveTrader()
    result = run(events, strat, latency=ConstantLatency(order_ms=3, report_ms=2))
    assert result.fills
    orders = {o.id: o for o in result.orders}
    for f in result.fills:
        assert f.ts >= orders[f.order_id].submit_ts + 3 * MS


def test_reaction_to_an_event_cannot_trade_at_that_events_snapshot_when_book_moves():
    # The strategy reacts to a trade print at T0+1ms. Even with zero latency it
    # cannot fill against quote updates that share that timestamp and come after.
    events = [
        q(T0, 100.0, 100.25),
        t(T0 + MS, 100.25, 5, Side.BUY),
        q(T0 + MS, 100.25, 100.5),
        q(T0 + 2 * MS, 100.25, 100.5),
    ]
    strat = Scripted({1: lambda ctx: ctx.buy("NQ", 1)})
    result = run(events, strat)
    assert [f.price for f in result.fills] == [100.5]


def test_unsorted_data_raises():
    events = [q(T0 + MS, 100.0, 100.25), q(T0, 100.0, 100.25)]
    with pytest.raises(DataOrderError):
        run(events, Scripted())


def test_strategy_order_view_lags_exchange():
    events = [q(T0 + i * MS, 100.0, 100.25) for i in range(10)]
    statuses: list[tuple[int, OrderStatus]] = []

    class Watcher(Scripted):
        def on_quote(self, ctx, quote):
            super().on_quote(ctx, quote)
            if self.placed:
                statuses.append((ctx.now, self.placed[0].status))

    strat = Watcher({0: lambda ctx: ctx.buy("NQ", 1)})
    result = run(events, strat, latency=ConstantLatency(order_ms=1, report_ms=4))
    assert result.fills[0].ts == T0 + 1 * MS
    # Filled at the exchange at +1ms, but the strategy still sees PENDING until +5ms.
    for now, status in statuses:
        if now <= T0 + 5 * MS:
            assert status is OrderStatus.PENDING
        else:
            assert status is OrderStatus.FILLED
