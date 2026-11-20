"""The event loop.

Time model
----------
There is one clock. Two kinds of things happen on it:

1. **Market events** from the data iterator (already sorted by timestamp).
2. **Scheduled actions** kept in a priority queue: an order reaching the
   exchange, a cancel reaching the exchange, or an execution report reaching
   the strategy. Each is scheduled ``now + latency``.

Before a market event at time ``T`` is processed, every scheduled action with
time ``< T`` runs. An action scheduled at exactly ``T`` runs *after* all market
events stamped ``T`` - ties go to the market, which is the conservative
choice (an order never trades against a book state it was sent in reaction to
*and* the simultaneous updates it could not have seen).

For each market event the order is: simulated exchange matches resting orders
-> ledger is marked to market -> risk rules -> strategy callback.

The loop pulls exactly one event at a time from the iterator and never peeks
ahead, so neither the engine nor the strategy can use future data.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections.abc import Iterable
from enum import Enum
from typing import Any

from ftb.accounting import MarkMethod, Portfolio
from ftb.contracts import ContractRegistry, load_contracts
from ftb.data.stream import DataOrderError
from ftb.events import NS_PER_SECOND, MarketEvent, QuoteEvent, Side, TradeEvent
from ftb.execution.base import ExecutionModel
from ftb.execution.latency import ConstantLatency, LatencyModel
from ftb.execution.top_of_book import TopOfBookExecution
from ftb.orders import ExecReport, Fill, Order, OrderStatus, OrderType, ReportType
from ftb.results import BacktestResult
from ftb.risk import RiskRule
from ftb.strategy import Context, Strategy

_ARRIVE = 0
_CANCEL = 1
_REPORT = 2


class _State(Enum):
    NEW = "new"
    RUNNING = "running"
    DRAINING = "draining"
    DONE = "done"


class Backtester:
    """Runs one strategy over one market-data stream.

    Parameters
    ----------
    strategy:
        A :class:`~ftb.strategy.Strategy` instance.
    data:
        Iterable of :class:`~ftb.events.MarketEvent` in non-decreasing ``ts``.
        Consumed lazily, once.
    contracts:
        Contract specs (defaults to the bundled ES/NQ/MES/MNQ).
    execution:
        Simulated exchange (defaults to :class:`TopOfBookExecution`).
    latency:
        Latency model (defaults to ``ConstantLatency()``: 5 ms each way).
    risk_rules:
        Pre-trade checks and halt rules, see :mod:`ftb.risk`.
    initial_capital:
        Starting account equity in USD.
    mark_method:
        How open positions are valued: ``"liquidation"`` (long at bid, short
        at ask; default), ``"mid"`` or ``"last"``.
    equity_interval_s:
        Sampling interval of the stored equity curve (drawdown is tracked on
        every event regardless).
    """

    def __init__(
        self,
        strategy: Strategy,
        data: Iterable[MarketEvent],
        *,
        contracts: ContractRegistry | None = None,
        execution: ExecutionModel | None = None,
        latency: LatencyModel | None = None,
        risk_rules: Iterable[RiskRule] = (),
        initial_capital: float = 0.0,
        mark_method: MarkMethod = "liquidation",
        equity_interval_s: float = 60.0,
    ) -> None:
        self.strategy = strategy
        self.contracts = contracts or load_contracts()
        self.execution = execution or TopOfBookExecution()
        self.execution.bind(self.contracts)
        self.latency = latency or ConstantLatency()
        self.risk_rules = list(risk_rules)
        self._update_rules = [r for r in self.risk_rules if r.monitors_updates]
        self.portfolio = Portfolio(
            self.contracts,
            initial_capital=initial_capital,
            mark_method=mark_method,
            sample_interval_ns=int(equity_interval_s * NS_PER_SECOND),
        )
        self.ctx = Context(self)
        self._data = data

        self._state = _State.NEW
        self._now = 0
        self._heap: list[tuple[int, int, int, Any]] = []
        self._seq = itertools.count()
        self._ids = itertools.count(1)
        self._orders: dict[int, Order] = {}
        self._active: dict[int, Order] = {}
        self._known_pos: dict[str, int] = {}
        self._last_quote: dict[str, QuoteEvent] = {}
        self._last_trade: dict[str, TradeEvent] = {}
        self._halted: str | None = None
        self._halt_ts: int | None = None
        self._n_events = 0
        self._first_ts: int | None = None

    # ================================================================== run
    def run(self) -> BacktestResult:
        if self._state is not _State.NEW:
            raise RuntimeError("a Backtester instance can only be run once")
        wall_start = time.perf_counter()
        self._state = _State.RUNNING
        strategy, ctx, execution, portfolio = (
            self.strategy,
            self.ctx,
            self.execution,
            self.portfolio,
        )
        heap = self._heap
        last_ts: int | None = None

        strategy.on_start(ctx)

        for event in self._data:
            ts = event.ts
            if last_ts is not None and ts < last_ts:
                raise DataOrderError(
                    f"market data out of order: ts={ts} after ts={last_ts} ({event!r})"
                )
            if last_ts is None:
                self._first_ts = ts
            last_ts = ts
            if heap and heap[0][0] < ts:
                self._process_until(ts)
            self._now = ts
            self._n_events += 1

            if isinstance(event, QuoteEvent):
                reports = execution.on_quote(event)
                if reports:
                    self._handle_reports(reports)
                portfolio.on_quote(event)
                self._last_quote[event.symbol] = event
                self._check_risk_updates()
                portfolio.maybe_sample(ts)
                strategy.on_quote(ctx, event)
            elif isinstance(event, TradeEvent):
                reports = execution.on_trade(event)
                if reports:
                    self._handle_reports(reports)
                portfolio.on_trade(event)
                self._last_trade[event.symbol] = event
                self._check_risk_updates()
                portfolio.maybe_sample(ts)
                strategy.on_trade(ctx, event)
            else:
                raise TypeError(f"unsupported market event type: {type(event).__name__}")

        # Data exhausted: orders still in flight cannot reach a market anymore,
        # but reports already generated are delivered.
        self._state = _State.DRAINING
        self._process_until(None)
        strategy.on_end(ctx)
        self._process_until(None)
        if last_ts is not None:
            portfolio.sample(max(self._now, last_ts))
        self._state = _State.DONE

        return BacktestResult(
            strategy_name=strategy.name,
            initial_capital=portfolio.initial_capital,
            final_equity=portfolio.equity,
            max_drawdown=portfolio.max_drawdown,
            fills=list(portfolio.fills),
            trades=list(portfolio.trades),
            equity_curve=list(portfolio.equity_curve),
            orders=list(self._orders.values()),
            positions={s: p for s, p in portfolio.positions.items()},
            contracts=self.contracts,
            halted=self._halted,
            halt_ts=self._halt_ts,
            events_processed=self._n_events,
            start_ts=self._first_ts,
            end_ts=last_ts,
            elapsed_s=time.perf_counter() - wall_start,
        )

    # ================================================================== scheduling
    def _push(self, ts: int, kind: int, payload: Any) -> None:
        heapq.heappush(self._heap, (ts, next(self._seq), kind, payload))

    def _process_until(self, ts_exclusive: int | None) -> None:
        heap = self._heap
        while heap and (ts_exclusive is None or heap[0][0] < ts_exclusive):
            ts, _, kind, payload = heapq.heappop(heap)
            self._now = ts
            if kind == _REPORT:
                self._deliver(*payload)
            elif kind == _ARRIVE:
                if self._state is _State.DRAINING:
                    reports = [
                        ExecReport(
                            ReportType.REJECTED,
                            payload.id,
                            payload.symbol,
                            ts,
                            reason="end of data",
                        )
                    ]
                else:
                    reports = self.execution.submit(payload, ts)
                self._handle_reports(reports)
            elif kind == _CANCEL:
                self._handle_reports(self.execution.cancel(payload, ts))

    def _handle_reports(self, reports: list[ExecReport]) -> None:
        filled = False
        for report in reports:
            fill = None
            if report.type is ReportType.FILL:
                order = self._orders[report.order_id]
                spec = self.contracts.resolve(report.symbol)
                assert report.liquidity is not None
                fill = Fill(
                    order_id=report.order_id,
                    symbol=report.symbol,
                    side=order.side,
                    qty=report.fill_qty,
                    price=report.fill_price,
                    ts=report.ts,
                    liquidity=report.liquidity,
                    commission=spec.commission(report.fill_qty),
                    tag=order.tag,
                )
                self.portfolio.on_fill(fill)
                filled = True
            self._push(report.ts + self.latency.report_delay(report.ts), _REPORT, (report, fill))
        if filled:
            self._check_risk_updates()

    # ================================================================== strategy side
    def _deliver(self, report: ExecReport, fill: Fill | None) -> None:
        order = self._orders.get(report.order_id)
        if order is None:  # pragma: no cover - defensive
            return
        order.last_update_ts = self._now
        strategy, ctx = self.strategy, self.ctx
        rtype = report.type

        if rtype is ReportType.ACCEPTED:
            if order.status is OrderStatus.PENDING:
                order.status = OrderStatus.WORKING
                strategy.on_order_update(ctx, order)
        elif rtype is ReportType.FILL:
            assert fill is not None
            prev = order.filled_qty
            order.filled_qty = prev + fill.qty
            order.avg_fill_price = (
                fill.price
                if prev == 0
                else (order.avg_fill_price * prev + fill.price * fill.qty) / order.filled_qty
            )
            order.fills.append(fill)
            self._known_pos[fill.symbol] = self._known_pos.get(fill.symbol, 0) + fill.signed_qty
            old_status = order.status
            if order.filled_qty >= order.qty:
                order.status = OrderStatus.FILLED
                self._active.pop(order.id, None)
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            strategy.on_fill(ctx, fill)
            if order.status is not old_status:
                strategy.on_order_update(ctx, order)
            if self._halted is not None and self._state is _State.RUNNING:
                self._flatten(fill.symbol, tag="halt_flatten", internal=True)
        elif rtype is ReportType.CANCELLED:
            order.status = OrderStatus.CANCELLED
            self._active.pop(order.id, None)
            strategy.on_order_update(ctx, order)
        elif rtype is ReportType.REJECTED:
            order.status = OrderStatus.REJECTED
            order.reject_reason = report.reason
            self._active.pop(order.id, None)
            strategy.on_order_update(ctx, order)
        elif rtype is ReportType.CANCEL_REJECTED:
            order.cancel_requested = False
            strategy.on_order_update(ctx, order)

    def _submit(
        self,
        symbol: str,
        side: Side,
        qty: int,
        otype: OrderType,
        *,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tag: str = "",
        internal: bool = False,
    ) -> Order:
        if self._state is _State.NEW or self._state is _State.DONE:
            raise RuntimeError("orders can only be submitted while the backtest is running")
        if side not in (Side.BUY, Side.SELL):
            raise ValueError("side must be Side.BUY or Side.SELL")
        if int(qty) != qty or qty <= 0:
            raise ValueError(f"qty must be a positive integer, got {qty!r}")
        qty = int(qty)
        spec = self.contracts.resolve(symbol)  # KeyError for unknown symbols
        if otype is OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("limit orders need a price")
            spec.to_ticks(limit_price)  # ValueError if off the tick grid
        if otype is OrderType.STOP:
            if stop_price is None:
                raise ValueError("stop orders need a stop_price")
            spec.to_ticks(stop_price)

        order = Order(
            id=next(self._ids),
            symbol=symbol,
            side=side,
            qty=qty,
            type=otype,
            limit_price=limit_price,
            stop_price=stop_price,
            tag=tag,
            submit_ts=self._now,
            last_update_ts=self._now,
        )
        self._orders[order.id] = order

        if self._state is _State.DRAINING:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "end of data"
            return order

        reason = None
        if self._halted is not None and not self._is_reducing(order):
            reason = f"trading halted ({self._halted})"
        elif not internal:
            for rule in self.risk_rules:
                why = rule.check_order(order, self.ctx)
                if why:
                    reason = f"{rule.name}: {why}"
                    break
        if reason is not None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = reason
            self._push(
                self._now,
                _REPORT,
                (ExecReport(ReportType.REJECTED, order.id, symbol, self._now, reason=reason), None),
            )
            return order

        self._active[order.id] = order
        self._push(self._now + self.latency.order_delay(self._now), _ARRIVE, order.to_request())
        return order

    def _cancel(self, order_id: int) -> bool:
        order = self._orders.get(order_id)
        if order is None or not order.is_active or order.cancel_requested:
            return False
        if self._state is not _State.RUNNING and self._state is not _State.DRAINING:
            return False
        order.cancel_requested = True
        self._push(self._now + self.latency.cancel_delay(self._now), _CANCEL, order_id)
        return True

    def _flatten(self, symbol: str, tag: str = "flatten", internal: bool = False) -> Order | None:
        exposure = self._known_pos.get(symbol, 0)
        for o in self._active.values():
            if o.symbol == symbol and o.type is OrderType.MARKET:
                exposure += o.side.sign * o.remaining
        if exposure == 0:
            return None
        side = Side.SELL if exposure > 0 else Side.BUY
        return self._submit(
            symbol, side, abs(exposure), OrderType.MARKET, tag=tag, internal=internal
        )

    def _is_reducing(self, order: Order) -> bool:
        pos = self._known_pos.get(order.symbol, 0)
        return pos != 0 and (pos > 0) != (order.side is Side.BUY) and order.qty <= abs(pos)

    # ================================================================== risk
    def _check_risk_updates(self) -> None:
        if self._halted is not None or not self._update_rules:
            return
        for rule in self._update_rules:
            reason = rule.on_update(self._now, self.portfolio)
            if reason:
                self._halt(f"{rule.name}: {reason}")
                return

    def _halt(self, reason: str) -> None:
        self._halted = reason
        self._halt_ts = self._now
        for order in list(self._active.values()):
            if order.type is not OrderType.MARKET:
                self._cancel(order.id)
        for symbol in list(self._known_pos):
            self._flatten(symbol, tag="halt_flatten", internal=True)
        self.strategy.on_halt(self.ctx, reason)
