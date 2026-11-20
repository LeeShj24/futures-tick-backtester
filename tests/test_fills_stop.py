"""Stop-market orders."""

from ftb import Side
from ftb.orders import OrderType, ReportType
from helpers import T0, exchange, fills, q, req, t, types

BUY, SELL, STP = Side.BUY, Side.SELL, OrderType.STOP


def armed(ex, side, stop, qty=1, last=100.0):
    ex.on_quote(q(T0, 100.0, 100.25, ask_size=5, bid_size=5))
    ex.on_trade(t(T0 + 1, last, 1))
    reports = ex.submit(req(side, qty, STP, stop=stop), T0 + 2)
    assert types(reports) == [ReportType.ACCEPTED]


def test_buy_stop_triggers_on_trade_not_on_quote():
    ex = exchange()
    armed(ex, BUY, 100.5)
    assert ex.on_quote(q(T0 + 3, 100.25, 100.5)) == []
    assert ex.on_quote(q(T0 + 4, 100.5, 100.75)) == []  # quotes alone never trigger
    reports = ex.on_trade(t(T0 + 5, 100.5, 3, Side.BUY))
    # Triggered by the print at 100.5 -> market buy against the current ask (100.75).
    assert fills(reports) == [(100.75, 1, "taker")]
    assert ex.working_orders() == []


def test_stop_fill_never_better_than_triggering_print_on_gap():
    ex = exchange()
    armed(ex, BUY, 100.5)
    # Book still shows ask 100.25 (stale), print gaps to 101.0.
    assert fills(ex.on_trade(t(T0 + 3, 101.0, 1, Side.BUY))) == [(101.0, 1, "taker")]


def test_sell_stop_symmetric():
    ex = exchange()
    armed(ex, SELL, 99.5)
    assert ex.on_trade(t(T0 + 3, 99.75, 1, Side.SELL)) == []
    ex.on_quote(q(T0 + 4, 99.25, 99.5))
    assert fills(ex.on_trade(t(T0 + 5, 99.5, 1, Side.SELL))) == [(99.25, 1, "taker")]


def test_triggered_stop_walks_the_book():
    ex = exchange()
    armed(ex, BUY, 100.25, qty=3)
    ex.on_quote(q(T0 + 3, 100.0, 100.25, ask_size=1))
    assert fills(ex.on_trade(t(T0 + 4, 100.25, 1, Side.BUY))) == [
        (100.25, 1, "taker"),
        (100.5, 1, "taker"),
        (100.75, 1, "taker"),
    ]


def test_stop_extra_slippage():
    ex = exchange(extra_slippage_ticks=2)
    armed(ex, BUY, 100.25)
    assert fills(ex.on_trade(t(T0 + 3, 100.25, 1, Side.BUY))) == [(100.75, 1, "taker")]


def test_stop_through_market_rejected_by_default():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25))
    ex.on_trade(t(T0 + 1, 100.0, 1))
    reports = ex.submit(req(BUY, 1, STP, stop=100.0), T0 + 2)
    assert types(reports) == [ReportType.REJECTED]
    assert "through" in reports[0].reason


def test_stop_through_market_can_trigger_immediately():
    ex = exchange(stop_through_market="trigger")
    ex.on_quote(q(T0, 100.0, 100.25))
    ex.on_trade(t(T0 + 1, 100.0, 1))
    reports = ex.submit(req(BUY, 1, STP, stop=99.75), T0 + 2)
    assert types(reports)[0] is ReportType.ACCEPTED
    assert fills(reports) == [(100.25, 1, "taker")]


def test_stop_without_prior_trade_is_accepted():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25))
    assert types(ex.submit(req(SELL, 1, STP, stop=99.0), T0)) == [ReportType.ACCEPTED]
