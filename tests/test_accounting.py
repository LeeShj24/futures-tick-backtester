"""Positions, realized/unrealized PnL, commissions, round trips, drawdown."""

import pytest

from ftb import Portfolio, Side
from ftb.orders import Fill, Liquidity
from helpers import REGISTRY, T0, Scripted, q, run, t

NQ_FEE = 1.90  # per side, default NQ schedule
ES_FEE = 1.90


def fill(side, qty, price, ts=T0, symbol="NQ"):
    return Fill(
        order_id=0,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        ts=ts,
        liquidity=Liquidity.TAKER,
        commission=REGISTRY[symbol].commission(qty),
    )


def test_long_round_trip_dollars_and_commissions():
    p = Portfolio(REGISTRY)
    p.on_fill(fill(Side.BUY, 1, 18000.0))
    p.on_fill(fill(Side.SELL, 1, 18001.0, ts=T0 + 1))
    assert p.realized_pnl == pytest.approx(20.0)  # 4 ticks * $5
    assert p.commissions == pytest.approx(2 * NQ_FEE)
    assert p.equity == pytest.approx(20.0 - 2 * NQ_FEE)
    (trade,) = p.trades
    assert trade.direction is Side.BUY
    assert trade.gross_pnl == pytest.approx(20.0)
    assert trade.net_pnl == pytest.approx(20.0 - 2 * NQ_FEE)
    assert trade.pnl_ticks == pytest.approx(4.0)
    assert trade.entry_ts == T0 and trade.exit_ts == T0 + 1
    assert p.positions["NQ"].qty == 0


def test_short_es():
    p = Portfolio(REGISTRY)
    p.on_fill(fill(Side.SELL, 2, 5000.0, symbol="ES"))
    p.on_fill(fill(Side.BUY, 2, 4999.5, symbol="ES"))
    assert p.realized_pnl == pytest.approx(2 * 0.5 * 50)
    assert p.commissions == pytest.approx(4 * ES_FEE)
    assert p.trades[0].direction is Side.SELL
    assert p.trades[0].pnl_ticks == pytest.approx(2.0)


def test_micro_contract_fees_differ():
    p = Portfolio(REGISTRY)
    p.on_fill(fill(Side.BUY, 1, 18000.0, symbol="MNQ"))
    p.on_fill(fill(Side.SELL, 1, 18010.0, symbol="MNQ"))
    assert p.realized_pnl == pytest.approx(10 * 2.0)
    assert p.commissions == pytest.approx(2 * 0.62)


def test_scaling_in_uses_average_price():
    p = Portfolio(REGISTRY)
    p.on_fill(fill(Side.BUY, 1, 100.0))
    p.on_fill(fill(Side.BUY, 1, 101.0))
    assert p.positions["NQ"].avg_price == pytest.approx(100.5)
    p.on_fill(fill(Side.SELL, 2, 102.0))
    assert p.realized_pnl == pytest.approx(1.5 * 2 * 20)
    (trade,) = p.trades
    assert trade.qty == 2
    assert trade.entry_price == pytest.approx(100.5)
    assert trade.commissions == pytest.approx(4 * NQ_FEE)


def test_partial_close_keeps_average_and_trade_open():
    p = Portfolio(REGISTRY)
    p.on_fill(fill(Side.BUY, 2, 100.0))
    p.on_fill(fill(Side.SELL, 1, 101.0))
    assert p.realized_pnl == pytest.approx(20.0)
    assert p.positions["NQ"].qty == 1
    assert p.positions["NQ"].avg_price == pytest.approx(100.0)
    assert p.trades == []


def test_reversal_closes_one_trade_and_opens_another():
    p = Portfolio(REGISTRY)
    p.on_fill(fill(Side.BUY, 1, 100.0))
    p.on_fill(fill(Side.SELL, 3, 101.0))  # close 1 long, open 2 short
    assert p.positions["NQ"].qty == -2
    assert p.positions["NQ"].avg_price == pytest.approx(101.0)
    p.on_fill(fill(Side.BUY, 2, 100.0))
    long_trade, short_trade = p.trades
    assert long_trade.gross_pnl == pytest.approx(20.0)
    assert short_trade.gross_pnl == pytest.approx(40.0)
    # The 3-lot fill's commission is split 1:2 between the two trades.
    assert long_trade.commissions == pytest.approx(2 * NQ_FEE)
    assert short_trade.commissions == pytest.approx(4 * NQ_FEE)
    assert p.commissions == pytest.approx(6 * NQ_FEE)
    assert sum(tr.net_pnl for tr in p.trades) == pytest.approx(p.equity)


@pytest.mark.parametrize(
    ("method", "expected"),
    [("liquidation", -5.0), ("mid", 0.0), ("last", 5.0)],
)
def test_unrealized_by_mark_method(method, expected):
    p = Portfolio(REGISTRY, mark_method=method)
    p.on_quote(q(T0, 100.0, 100.5))
    p.on_trade(t(T0, 100.5, 1))
    p.on_fill(fill(Side.BUY, 1, 100.25))
    assert p.unrealized_pnl == pytest.approx(expected)
    assert p.equity == pytest.approx(expected - NQ_FEE)


def test_short_liquidation_marks_at_ask():
    p = Portfolio(REGISTRY)
    p.on_quote(q(T0, 100.0, 100.5))
    p.on_fill(fill(Side.SELL, 1, 100.25))
    assert p.unrealized_pnl == pytest.approx(-5.0)


def test_drawdown_is_tracked_between_samples():
    p = Portfolio(REGISTRY, sample_interval_ns=10**18, record_on_fill=False)
    p.on_quote(q(T0, 100.0, 100.25))
    p.on_fill(fill(Side.BUY, 1, 100.25))
    p.maybe_sample(T0)
    p.on_quote(q(T0 + 1, 102.0, 102.25))  # +$35
    p.on_quote(q(T0 + 2, 99.0, 99.25))  # -$25 => drawdown $60 from peak
    p.on_quote(q(T0 + 3, 101.0, 101.25))
    p.maybe_sample(T0 + 4)
    assert len(p.equity_curve) == 1  # only one sample taken...
    assert p.max_drawdown == pytest.approx(60.0)  # ...but the dip was seen
    assert p.max_drawdown_ts == T0 + 2


def test_mae_mfe():
    p = Portfolio(REGISTRY)
    p.on_quote(q(T0, 100.0, 100.25))
    p.on_fill(fill(Side.BUY, 1, 100.25))
    p.on_quote(q(T0 + 1, 99.0, 99.25))  # -25
    p.on_quote(q(T0 + 2, 101.25, 101.5))  # +20
    p.on_fill(fill(Side.SELL, 1, 101.0, ts=T0 + 3))
    trade = p.trades[0]
    assert trade.mae == pytest.approx(-25.0)
    assert trade.mfe == pytest.approx(20.0)
    assert trade.gross_pnl == pytest.approx(15.0)


def test_equity_curve_sampling_interval():
    p = Portfolio(REGISTRY, sample_interval_ns=1_000, record_on_fill=False)
    for ts in range(0, 5_000, 100):
        p.maybe_sample(ts)
    assert [pt.ts for pt in p.equity_curve] == [0, 1_000, 2_000, 3_000, 4_000]


def test_engine_end_to_end_equity_matches_trades():
    events = [q(T0 + i, 100.0 + 0.25 * (i % 5), 100.25 + 0.25 * (i % 5)) for i in range(30)]
    strat = Scripted(
        {
            0: lambda ctx: ctx.buy("NQ", 2),
            10: lambda ctx: ctx.sell("NQ", 2),
            12: lambda ctx: ctx.sell("NQ", 1),
            20: lambda ctx: ctx.buy("NQ", 1),
        }
    )
    result = run(events, strat, initial_capital=10_000.0)
    assert len(result.trades) == 2
    assert result.open_positions == {}
    assert result.final_equity == pytest.approx(10_000.0 + sum(tr.net_pnl for tr in result.trades))
    assert sum(f.commission for f in result.fills) == pytest.approx(6 * NQ_FEE)
