"""Engine behaviour: order lifecycle, risk hooks, end of data."""

import pytest

from ftb import Backtester, MaxContracts, RiskRule, Side
from ftb.orders import OrderStatus
from helpers import MS, T0, Scripted, q, run, t


def flat_quotes(n=10, step=MS):
    return [q(T0 + i * step, 100.0, 100.25) for i in range(n)]


def test_order_lifecycle_callbacks():
    strat = Scripted({0: lambda ctx: ctx.submit_limit("NQ", Side.BUY, 1, 99.0)})
    events = [*flat_quotes(3), t(T0 + 5 * MS, 98.75, 1, Side.SELL), q(T0 + 6 * MS, 98.5, 98.75)]
    run(events, strat)
    assert [s for _, _, s in strat.updates] == ["working", "filled"]
    order = strat.placed[0]
    assert order.filled_qty == 1 and order.avg_fill_price == 99.0
    assert order.fills[0].commission == pytest.approx(1.90)


def test_partial_fill_status():
    strat = Scripted({0: lambda ctx: ctx.submit_limit("NQ", Side.BUY, 3, 99.0)})
    events = [*flat_quotes(2), t(T0 + 5 * MS, 98.75, 1, Side.SELL), q(T0 + 6 * MS, 100.0, 100.25)]
    run(events, strat)
    assert strat.placed[0].status is OrderStatus.PARTIALLY_FILLED
    assert strat.placed[0].remaining == 2


def test_input_validation_raises_immediately():
    def bad_price(ctx):
        with pytest.raises(ValueError):
            ctx.submit_limit("NQ", Side.BUY, 1, 100.1)
        with pytest.raises(ValueError):
            ctx.buy("NQ", 0)
        with pytest.raises(KeyError):
            ctx.buy("CL", 1)

    run(flat_quotes(2), Scripted({0: bad_price}))


def test_max_contracts_rejects_worst_case_exposure():
    strat = Scripted(
        {
            0: lambda ctx: ctx.submit_limit("NQ", Side.BUY, 1, 99.0),
            1: lambda ctx: ctx.submit_limit("NQ", Side.BUY, 1, 98.75),
            2: lambda ctx: ctx.submit_limit("NQ", Side.SELL, 2, 101.0),
        }
    )
    run(flat_quotes(5), strat, risk_rules=[MaxContracts(1)])
    first, second, third = strat.placed
    assert first.status is OrderStatus.WORKING
    assert second.status is OrderStatus.REJECTED
    assert "max_contracts" in second.reject_reason
    assert third.status is OrderStatus.REJECTED  # -2 would exceed the limit too


class StopOut(RiskRule):
    name = "stop_out"

    def __init__(self, floor):
        self.floor = floor

    def on_update(self, ts, portfolio):
        if portfolio.equity < self.floor:
            return f"equity {portfolio.equity:.2f} below {self.floor}"
        return None


def test_halt_rule_cancels_flattens_and_blocks_new_risk():
    falling = [q(T0 + i * MS, 100.0 - 0.25 * i, 100.25 - 0.25 * i) for i in range(30)]
    halts = []

    class Halting(Scripted):
        def on_halt(self, ctx, reason):
            halts.append((ctx.now, reason))

    strat = Halting(
        {
            0: lambda ctx: ctx.buy("NQ", 1),
            1: lambda ctx: ctx.submit_limit("NQ", Side.SELL, 1, 105.0),
            25: lambda ctx: ctx.buy("NQ", 1),  # after the halt: must be rejected
        }
    )
    result = run(falling, strat, initial_capital=1_000.0, risk_rules=[StopOut(1_000.0 - 30.0)])
    assert result.halted is not None and "stop_out" in result.halted
    assert len(halts) == 1
    assert result.open_positions == {}
    target, late_buy = strat.placed[1], strat.placed[2]
    assert target.status is OrderStatus.CANCELLED
    assert late_buy.status is OrderStatus.REJECTED and "halted" in late_buy.reject_reason
    assert [f.tag for f in result.fills] == ["", "halt_flatten"]


def test_orders_after_data_end_are_rejected_and_positions_reported():
    class EndTrader(Scripted):
        def on_end(self, ctx):
            self.placed.append(ctx.sell("NQ", 1))

    strat = EndTrader({0: lambda ctx: ctx.buy("NQ", 1)})
    result = run(flat_quotes(3), strat)
    assert strat.placed[-1].status is OrderStatus.REJECTED
    assert result.open_positions["NQ"].qty == 1
    # Open position is marked to market (liquidation: long at the bid).
    assert result.final_equity == pytest.approx((100.0 - 100.25) * 20 - 1.90)


def test_in_flight_orders_at_data_end_never_fill():
    strat = Scripted({2: lambda ctx: ctx.buy("NQ", 1)})
    result = Backtester(strat, flat_quotes(3)).run()  # default 5 ms latency
    assert result.fills == []
    assert strat.placed[0].status is OrderStatus.REJECTED


def test_backtester_runs_once():
    bt = Backtester(Scripted(), flat_quotes(2))
    bt.run()
    with pytest.raises(RuntimeError):
        bt.run()


def test_flatten_accounts_for_in_flight_market_orders():
    def double_flatten(ctx):
        ctx.flatten("NQ")
        ctx.flatten("NQ")

    strat = Scripted({0: lambda ctx: ctx.buy("NQ", 2), 5: double_flatten})
    result = run(flat_quotes(10), strat)
    assert [f.signed_qty for f in result.fills] == [2, -2]
    assert result.open_positions == {}
