"""Market-order fills against top of book."""

import math

from ftb import Side
from ftb.orders import OrderType, ReportType
from helpers import T0, exchange, fills, q, req, types

BUY, SELL, MKT = Side.BUY, Side.SELL, OrderType.MARKET


def test_buy_fills_at_ask_within_displayed_size():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=5, ask_size=5))
    reports = ex.submit(req(BUY, 3, MKT), T0 + 1)
    assert types(reports)[0] is ReportType.ACCEPTED
    assert fills(reports) == [(100.25, 3, "taker")]
    assert all(r.ts == T0 + 1 for r in reports)


def test_sell_hits_bid():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=5, ask_size=5))
    assert fills(ex.submit(req(SELL, 2, MKT), T0)) == [(100.0, 2, "taker")]


def test_size_beyond_displayed_walks_levels_with_displayed_depth():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, ask_size=2))
    # 2 at the ask, then assumed 2 per level (= displayed size) one tick apart.
    assert fills(ex.submit(req(BUY, 5, MKT), T0)) == [
        (100.25, 2, "taker"),
        (100.5, 2, "taker"),
        (100.75, 1, "taker"),
    ]


def test_configured_depth_per_level():
    ex = exchange(depth_per_level=10)
    ex.on_quote(q(T0, 100.0, 100.25, ask_size=2))
    assert fills(ex.submit(req(BUY, 5, MKT), T0)) == [(100.25, 2, "taker"), (100.5, 3, "taker")]


def test_sell_slippage_goes_down():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=1))
    assert fills(ex.submit(req(SELL, 3, MKT), T0)) == [
        (100.0, 1, "taker"),
        (99.75, 1, "taker"),
        (99.5, 1, "taker"),
    ]


def test_consumed_liquidity_is_not_reused_until_next_quote():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, ask_size=2))
    assert fills(ex.submit(req(BUY, 2, MKT), T0)) == [(100.25, 2, "taker")]
    # Same book snapshot: the displayed 2 lots are gone.
    assert fills(ex.submit(req(BUY, 1, MKT), T0 + 1)) == [(100.5, 1, "taker")]
    # A new quote is new information: liquidity is replenished.
    ex.on_quote(q(T0 + 2, 100.0, 100.25, ask_size=2))
    assert fills(ex.submit(req(BUY, 1, MKT), T0 + 3)) == [(100.25, 1, "taker")]


def test_extra_slippage_ticks():
    ex = exchange(extra_slippage_ticks=1)
    ex.on_quote(q(T0, 100.0, 100.25))
    assert fills(ex.submit(req(BUY, 1, MKT), T0)) == [(100.5, 1, "taker")]
    assert fills(ex.submit(req(SELL, 1, MKT), T0)) == [(99.75, 1, "taker")]


def test_rejected_without_quote():
    ex = exchange()
    reports = ex.submit(req(BUY, 1, MKT), T0)
    assert types(reports) == [ReportType.REJECTED]


def test_rejected_when_opposite_side_missing():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, math.nan, ask_size=0))
    assert types(ex.submit(req(BUY, 1, MKT), T0)) == [ReportType.REJECTED]
    assert fills(ex.submit(req(SELL, 1, MKT), T0)) == [(100.0, 1, "taker")]


def test_zero_displayed_size_starts_one_tick_worse():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, ask_size=0))
    assert fills(ex.submit(req(BUY, 1, MKT), T0)) == [(100.5, 1, "taker")]
