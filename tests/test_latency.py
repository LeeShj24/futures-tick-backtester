"""Latency: orders see the book at arrival, reports reach the strategy late."""

from ftb import ConstantLatency, Side
from ftb.orders import OrderStatus
from helpers import MS, T0, Scripted, q, run, t


def test_market_order_fills_against_book_at_arrival_time():
    events = [
        q(T0, 100.0, 100.25),
        q(T0 + 3 * MS, 100.25, 100.5),  # before arrival: visible to the order
        q(T0 + 8 * MS, 100.5, 100.75),  # after arrival: irrelevant
    ]
    strat = Scripted({0: lambda ctx: ctx.buy("NQ", 1)})
    result = run(events, strat, latency=ConstantLatency(order_ms=5, report_ms=0))
    (fill,) = result.fills
    assert fill.ts == T0 + 5 * MS
    assert fill.price == 100.5


def test_fill_report_is_delayed_and_position_unknown_until_then():
    events = [q(T0 + i * MS, 100.0, 100.25) for i in range(20)]
    strat = Scripted({0: lambda ctx: ctx.buy("NQ", 1)})
    run(events, strat, latency=ConstantLatency(order_ms=2, report_ms=7))
    ((seen_at, fill),) = strat.fills
    assert fill.ts == T0 + 2 * MS
    assert seen_at == T0 + 9 * MS
    # The market event stamped exactly T0+9ms is processed before the report
    # (ties go to the market), so the position becomes visible only afterwards.
    for now, pos in strat.positions_seen:
        assert pos == (1 if now > T0 + 9 * MS else 0)


def test_cancel_racing_a_fill_loses():
    events = [
        q(T0, 100.0, 100.25),
        q(T0 + 10 * MS, 100.0, 100.25),  # strategy cancels here; arrives at +15ms
        t(T0 + 12 * MS, 99.75, 5, Side.SELL),  # trade-through before the cancel lands
        q(T0 + 30 * MS, 99.75, 100.0),
    ]
    strat = Scripted(
        {
            0: lambda ctx: ctx.submit_limit("NQ", Side.BUY, 1, 100.0),
            1: lambda ctx: ctx.cancel(ctx.open_orders()[0]),
        }
    )
    result = run(events, strat, latency=ConstantLatency(order_ms=5, report_ms=1))
    order = strat.placed[0]
    assert order.status is OrderStatus.FILLED
    assert [f.price for f in result.fills] == [100.0]
    assert any(status == "working" for _, _, status in strat.updates)
    assert not order.cancel_requested  # cancel was rejected by the exchange


def test_cancel_that_arrives_in_time_prevents_fill():
    events = [
        q(T0, 100.0, 100.25),
        q(T0 + 10 * MS, 100.0, 100.25),
        t(T0 + 20 * MS, 99.75, 5, Side.SELL),
    ]
    strat = Scripted(
        {
            0: lambda ctx: ctx.submit_limit("NQ", Side.BUY, 1, 100.0),
            1: lambda ctx: ctx.cancel(ctx.open_orders()[0]),
        }
    )
    result = run(events, strat, latency=ConstantLatency(order_ms=5, report_ms=1))
    assert result.fills == []
    assert strat.placed[0].status is OrderStatus.CANCELLED


def test_simultaneous_market_events_are_processed_before_the_order():
    # Zero latency: the order still arrives *after* every event stamped T0.
    events = [q(T0, 100.0, 100.25), q(T0, 100.25, 100.5), q(T0 + MS, 100.5, 100.75)]
    strat = Scripted({0: lambda ctx: ctx.buy("NQ", 1)})
    result = run(events, strat)
    assert [f.price for f in result.fills] == [100.5]
    assert result.fills[0].ts == T0


def test_order_arriving_exactly_at_an_event_sees_that_event():
    events = [q(T0, 100.0, 100.25), q(T0 + 5 * MS, 100.25, 100.5), q(T0 + 6 * MS, 101.0, 101.25)]
    strat = Scripted({0: lambda ctx: ctx.buy("NQ", 1)})
    result = run(events, strat, latency=ConstantLatency(order_ms=5, report_ms=0))
    assert [f.price for f in result.fills] == [100.5]


def test_latency_validation():
    import pytest

    with pytest.raises(ValueError):
        ConstantLatency(order_ms=-1)
    lat = ConstantLatency(order_ms=1.5, report_ms=2, cancel_ms=0.5)
    assert lat.order_delay(0) == 1_500_000
    assert lat.cancel_delay(0) == 500_000
    assert lat.report_delay(0) == 2_000_000
