"""Limit-order fills: trade-through, queue approximation, quote cross, marketable limits."""

from ftb import FillOnTouch, Side, TradeThroughOnly, VolumeQueue
from ftb.orders import OrderType, ReportType
from helpers import T0, exchange, fills, q, req, t, types

BUY, SELL, LMT = Side.BUY, Side.SELL, OrderType.LIMIT


def resting_buy(ex, price, qty=1, order_id=None):
    reports = ex.submit(req(BUY, qty, LMT, limit=price, order_id=order_id), T0 + 1)
    assert types(reports) == [ReportType.ACCEPTED]
    return ex.working_orders("NQ")[-1]


def test_resting_buy_not_filled_while_market_stays_above():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25))
    resting_buy(ex, 99.75)
    assert ex.on_trade(t(T0 + 2, 100.0, 50, Side.SELL)) == []
    assert ex.on_quote(q(T0 + 3, 100.0, 100.25, bid_size=1)) == []
    assert ex.on_trade(t(T0 + 4, 100.25, 50, Side.BUY)) == []


def test_trade_through_fills_at_limit_up_to_print_size():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25))
    resting_buy(ex, 99.75, qty=3)
    assert fills(ex.on_trade(t(T0 + 2, 99.5, 2, Side.SELL))) == [(99.75, 2, "maker")]
    assert fills(ex.on_trade(t(T0 + 3, 99.5, 5, Side.SELL))) == [(99.75, 1, "maker")]
    assert ex.working_orders() == []


def test_volume_queue_fills_only_after_queue_ahead_trades():
    ex = exchange(queue_model=VolumeQueue(ahead_fraction=1.0))
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=5))
    order = resting_buy(ex, 100.0, qty=2)
    assert order.queue_ahead == 5
    assert fills(ex.on_trade(t(T0 + 2, 100.0, 3, Side.SELL))) == []
    assert order.queue_ahead == 2
    assert fills(ex.on_trade(t(T0 + 3, 100.0, 3, Side.SELL))) == [(100.0, 1, "maker")]
    assert fills(ex.on_trade(t(T0 + 4, 100.0, 1, Side.SELL))) == [(100.0, 1, "maker")]


def test_unknown_aggressor_prints_count_buy_aggressor_prints_do_not():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=2))
    order = resting_buy(ex, 100.0)
    # A buyer lifting an offer at our price is not a seller hitting our bid.
    assert fills(ex.on_trade(t(T0 + 2, 100.0, 10, Side.BUY))) == []
    assert order.queue_ahead == 2
    assert fills(ex.on_trade(t(T0 + 3, 100.0, 3, Side.NONE))) == [(100.0, 1, "maker")]


def test_queue_estimate_is_clamped_by_displayed_size():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=5))
    order = resting_buy(ex, 100.0)
    ex.on_quote(q(T0 + 2, 100.0, 100.25, bid_size=2))  # cancellations ahead of us
    assert order.queue_ahead == 2
    ex.on_quote(q(T0 + 3, 100.0, 100.25, bid_size=9))  # joiners behind us don't count
    assert order.queue_ahead == 2
    assert fills(ex.on_trade(t(T0 + 4, 100.0, 3, Side.SELL))) == [(100.0, 1, "maker")]


def test_joining_behind_best_bid_has_unknown_queue_until_level_is_best():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=5))
    order = resting_buy(ex, 99.75)
    assert order.queue_ahead is None
    ex.on_quote(q(T0 + 2, 99.75, 100.0, bid_size=4))
    assert order.queue_ahead == 4


def test_joining_inside_the_spread_has_nobody_ahead():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.5))
    order = resting_buy(ex, 100.25)
    assert order.queue_ahead == 0
    assert fills(ex.on_trade(t(T0 + 2, 100.25, 1, Side.SELL))) == [(100.25, 1, "maker")]


def test_ahead_fraction():
    ex = exchange(queue_model=VolumeQueue(ahead_fraction=0.5))
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=9))
    assert resting_buy(ex, 100.0).queue_ahead == 5  # ceil(4.5)


def test_trade_through_only_never_fills_at_the_touch():
    ex = exchange(queue_model=TradeThroughOnly())
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=1))
    resting_buy(ex, 100.0)
    for i in range(5):
        assert ex.on_trade(t(T0 + 2 + i, 100.0, 100, Side.SELL)) == []
    assert fills(ex.on_trade(t(T0 + 10, 99.75, 1, Side.SELL))) == [(100.0, 1, "maker")]


def test_fill_on_touch_is_optimistic():
    ex = exchange(queue_model=FillOnTouch())
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=500))
    resting_buy(ex, 100.0)
    assert fills(ex.on_trade(t(T0 + 2, 100.0, 1, Side.SELL))) == [(100.0, 1, "maker")]


def test_quote_cross_fills_up_to_displayed_size():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=5))
    resting_buy(ex, 100.0, qty=3)
    # Offer drops to our bid: those sellers would have matched our order.
    assert fills(ex.on_quote(q(T0 + 2, 99.75, 100.0, ask_size=2))) == [(100.0, 2, "maker")]
    assert ex.working_orders()[0].remaining == 1


def test_quote_cross_can_be_disabled():
    ex = exchange(fill_on_quote_cross=False)
    ex.on_quote(q(T0, 100.0, 100.25, bid_size=5))
    resting_buy(ex, 100.0)
    assert ex.on_quote(q(T0 + 2, 99.75, 100.0)) == []


def test_marketable_limit_takes_with_price_improvement_then_rests():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, ask_size=2))
    reports = ex.submit(req(BUY, 5, LMT, limit=100.5), T0 + 1)
    assert types(reports)[0] is ReportType.ACCEPTED
    assert fills(reports) == [(100.25, 2, "taker"), (100.5, 2, "taker")]
    rest = ex.working_orders("NQ")
    assert len(rest) == 1
    assert rest[0].remaining == 1
    assert rest[0].queue_ahead == 0


def test_sell_limit_symmetric():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25, ask_size=3))
    ex.submit(req(SELL, 2, LMT, limit=100.25), T0 + 1)
    assert fills(ex.on_trade(t(T0 + 2, 100.25, 4, Side.BUY))) == [(100.25, 1, "maker")]
    assert fills(ex.on_trade(t(T0 + 3, 100.5, 7, Side.BUY))) == [(100.25, 1, "maker")]


def test_print_volume_is_shared_by_price_time_priority():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25))
    low = req(BUY, 1, LMT, limit=99.75)
    high = req(BUY, 1, LMT, limit=100.0)
    ex.submit(low, T0 + 1)
    ex.submit(high, T0 + 2)
    reports = ex.on_trade(t(T0 + 3, 99.5, 1, Side.SELL))
    assert [(r.order_id, r.fill_price) for r in reports] == [(high.id, 100.0)]


def test_off_grid_limit_rejected_by_exchange():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25))
    reports = ex.submit(req(BUY, 1, LMT, limit=100.1), T0)
    assert types(reports) == [ReportType.REJECTED]
    assert "tick" in reports[0].reason


def test_cancel_and_cancel_reject():
    ex = exchange()
    ex.on_quote(q(T0, 100.0, 100.25))
    r = req(BUY, 1, LMT, limit=99.0)
    ex.submit(r, T0)
    assert types(ex.cancel(r.id, T0 + 1)) == [ReportType.CANCELLED]
    assert types(ex.cancel(r.id, T0 + 2)) == [ReportType.CANCEL_REJECTED]
    assert ex.on_trade(t(T0 + 3, 98.0, 10, Side.SELL)) == []
