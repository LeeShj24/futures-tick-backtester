"""Execution model v1: fills against top-of-book (MBP-1) data.

Rules (all comparisons in integer ticks):

Market orders
    Fill against the opposite best price. Quantity beyond the displayed size
    walks a *virtual ladder*: each further level is one tick worse and holds
    ``depth_per_level`` contracts (default: the displayed top size). Liquidity
    our own orders took is remembered until the next quote update.

Limit orders
    * Marketable on arrival -> fill as taker up to the limit price (with price
      improvement), the rest rests.
    * Resting buy at L fills (as maker, at L) when
        - a print happens below L (trade-through), up to the print's size;
        - a sell-aggressor print happens at L and the queue model says the
          volume reached us;
        - the ask drops to <= L (quote cross), up to the displayed ask size.
      Sells are symmetric.

Stop orders (stop-market)
    Trigger on a *trade* at or through the stop price, then execute as a market
    order starting from the worse of the opposite best price and the triggering
    print. A stop that is already through the last trade on arrival is rejected
    (``stop_through_market="reject"``) or executed immediately (``"trigger"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ftb.contracts import ContractSpec
from ftb.events import QuoteEvent, Side, TradeEvent
from ftb.execution.base import ExecutionModel
from ftb.execution.book import TopOfBook, WorkingOrder
from ftb.execution.queue import QueueModel, VolumeQueue
from ftb.orders import ExecReport, Liquidity, OrderRequest, OrderType, ReportType


@dataclass
class ExecutionConfig:
    queue_model: QueueModel = field(default_factory=VolumeQueue)
    #: Contracts assumed at each level beyond the top of book. None = use the
    #: displayed top-of-book size (at least 1).
    depth_per_level: int | None = None
    #: Extra ticks of slippage added to every market/stop fill (not to limits).
    extra_slippage_ticks: int = 0
    #: Fill resting limits when the opposite quote crosses their price.
    fill_on_quote_cross: bool = True
    #: What to do with a stop that is already through the last trade on arrival.
    stop_through_market: Literal["reject", "trigger"] = "reject"

    def __post_init__(self) -> None:
        if self.depth_per_level is not None and self.depth_per_level < 1:
            raise ValueError("depth_per_level must be >= 1 or None")
        if self.extra_slippage_ticks < 0:
            raise ValueError("extra_slippage_ticks must be >= 0")
        if self.stop_through_market not in ("reject", "trigger"):
            raise ValueError("stop_through_market must be 'reject' or 'trigger'")


class TopOfBookExecution(ExecutionModel):
    def __init__(self, config: ExecutionConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is not None and kwargs:
            raise TypeError("pass either an ExecutionConfig or keyword arguments, not both")
        self.config = config or ExecutionConfig(**kwargs)
        self._books: dict[str, TopOfBook] = {}
        self._working: dict[str, dict[int, WorkingOrder]] = {}
        self._symbol_of: dict[int, str] = {}
        self._seq = 0

    # ------------------------------------------------------------------ introspection
    def book(self, symbol: str) -> TopOfBook:
        book = self._books.get(symbol)
        if book is None:
            book = self._books[symbol] = TopOfBook()
        return book

    def working_orders(self, symbol: str | None = None) -> list[WorkingOrder]:
        if symbol is not None:
            return list(self._working.get(symbol, {}).values())
        return [o for orders in self._working.values() for o in orders.values()]

    # ------------------------------------------------------------------ market data
    def on_quote(self, quote: QuoteEvent) -> list[ExecReport]:
        spec = self.spec(quote.symbol)
        book = self.book(quote.symbol)
        bp, ap = quote.bid_price, quote.ask_price
        book.bid = None if bp != bp else spec.nearest_ticks(bp)
        book.ask = None if ap != ap else spec.nearest_ticks(ap)
        book.bid_size = quote.bid_size if book.bid is not None else 0
        book.ask_size = quote.ask_size if book.ask is not None else 0
        book.consumed_bid = book.consumed_ask = 0
        book.ts = quote.ts

        orders = self._working.get(quote.symbol)
        if not orders:
            return []
        queue = self.config.queue_model
        cross = self.config.fill_on_quote_cross
        crossing: dict[Side, list[WorkingOrder]] = {Side.BUY: [], Side.SELL: []}
        for o in orders.values():
            if o.limit_ticks is None:
                continue
            queue.on_quote(o, book)
            if not cross:
                continue
            if o.side is Side.BUY:
                if book.ask is not None and book.ask <= o.limit_ticks:
                    crossing[Side.BUY].append(o)
            elif book.bid is not None and book.bid >= o.limit_ticks:
                crossing[Side.SELL].append(o)

        reports: list[ExecReport] = []
        for side, candidates in crossing.items():
            if not candidates:
                continue
            _, size, consumed = book.opposite(side)
            available = size - consumed
            for o in sorted(candidates, key=_priority):
                if available <= 0:
                    break
                qty = min(o.remaining, available)
                available -= qty
                consumed += qty
                self._passive_fill(o, qty, spec, quote.ts, reports)
            book.consume(side, consumed)
        return reports

    def on_trade(self, trade: TradeEvent) -> list[ExecReport]:
        spec = self.spec(trade.symbol)
        book = self.book(trade.symbol)
        px = spec.nearest_ticks(trade.price)
        book.last_trade = px

        orders = self._working.get(trade.symbol)
        if not orders:
            return []
        reports: list[ExecReport] = []

        # 1) Stops: trigger on prints at or through the stop price.
        triggered = [
            o
            for o in orders.values()
            if o.stop_ticks is not None
            and (
                (o.side is Side.BUY and px >= o.stop_ticks)
                or (o.side is Side.SELL and px <= o.stop_ticks)
            )
        ]
        for o in sorted(triggered, key=lambda w: w.arrival_seq):
            self._remove(o)
            self._aggress(o.req, o.remaining, book, spec, trade.ts, reports, start_ticks=px)

        # 2) Resting limits: trade-throughs and prints at our price.
        buys: list[WorkingOrder] = []
        sells: list[WorkingOrder] = []
        for o in orders.values():
            lim = o.limit_ticks
            if lim is None:
                continue
            if o.side is Side.BUY:
                if px < lim or (px == lim and trade.aggressor is not Side.BUY):
                    buys.append(o)
            elif px > lim or (px == lim and trade.aggressor is not Side.SELL):
                sells.append(o)

        queue = self.config.queue_model
        for candidates in (buys, sells):
            budget = trade.size
            for o in sorted(candidates, key=_priority):
                if o.limit_ticks == px:
                    wanted = queue.on_trade_at_price(o, trade.size)  # always advance the queue
                else:
                    wanted = o.remaining  # traded through our price
                qty = min(wanted, o.remaining, budget)
                if qty > 0:
                    budget -= qty
                    self._passive_fill(o, qty, spec, trade.ts, reports)
        return reports

    # ------------------------------------------------------------------ order entry
    def submit(self, request: OrderRequest, ts: int) -> list[ExecReport]:
        spec = self.spec(request.symbol)
        book = self.book(request.symbol)
        reports: list[ExecReport] = []

        def reject(reason: str) -> list[ExecReport]:
            return [ExecReport(ReportType.REJECTED, request.id, request.symbol, ts, reason=reason)]

        if request.qty <= 0:
            return reject("quantity must be positive")
        if request.side not in (Side.BUY, Side.SELL):
            return reject("side must be BUY or SELL")
        try:
            limit_ticks = (
                None if request.limit_price is None else spec.to_ticks(request.limit_price)
            )
            stop_ticks = None if request.stop_price is None else spec.to_ticks(request.stop_price)
        except ValueError as exc:
            return reject(str(exc))

        accepted = ExecReport(ReportType.ACCEPTED, request.id, request.symbol, ts)
        direction = request.side.sign

        if request.type is OrderType.MARKET:
            if book.opposite(request.side)[0] is None:
                return reject("no opposite quote (book empty)")
            reports.append(accepted)
            self._aggress(request, request.qty, book, spec, ts, reports)
            return reports

        if request.type is OrderType.LIMIT:
            if limit_ticks is None:
                return reject("limit order requires limit_price")
            reports.append(accepted)
            remaining = request.qty
            top = book.opposite(request.side)[0]
            if top is not None and (limit_ticks - top) * direction >= 0:
                remaining = self._aggress(
                    request, remaining, book, spec, ts, reports, limit_ticks=limit_ticks
                )
            if remaining > 0:
                wo = self._add(request, remaining, ts, limit_ticks=limit_ticks)
                if remaining < request.qty:
                    wo.queue_ahead = 0  # we cleared the opposite side up to our price
                else:
                    self.config.queue_model.on_join(wo, book)
            return reports

        if request.type is OrderType.STOP:
            if stop_ticks is None:
                return reject("stop order requires stop_price")
            last = book.last_trade
            if last is not None and (last - stop_ticks) * direction >= 0:
                if self.config.stop_through_market == "reject":
                    return reject("stop price is at or through the last trade")
                reports.append(accepted)
                self._aggress(request, request.qty, book, spec, ts, reports, start_ticks=last)
                return reports
            reports.append(accepted)
            self._add(request, request.qty, ts, stop_ticks=stop_ticks)
            return reports

        return reject(f"unsupported order type {request.type!r}")

    def cancel(self, order_id: int, ts: int) -> list[ExecReport]:
        symbol = self._symbol_of.get(order_id)
        if symbol is None:
            return [
                ExecReport(
                    ReportType.CANCEL_REJECTED,
                    order_id,
                    "",
                    ts,
                    reason="order is not working (filled, cancelled, or not yet live)",
                )
            ]
        self._remove(self._working[symbol][order_id])
        return [ExecReport(ReportType.CANCELLED, order_id, symbol, ts)]

    # ------------------------------------------------------------------ internals
    def _add(
        self,
        request: OrderRequest,
        remaining: int,
        ts: int,
        *,
        limit_ticks: int | None = None,
        stop_ticks: int | None = None,
    ) -> WorkingOrder:
        self._seq += 1
        wo = WorkingOrder(
            req=request,
            remaining=remaining,
            limit_ticks=limit_ticks,
            stop_ticks=stop_ticks,
            arrival_ts=ts,
            arrival_seq=self._seq,
        )
        self._working.setdefault(request.symbol, {})[request.id] = wo
        self._symbol_of[request.id] = request.symbol
        return wo

    def _remove(self, wo: WorkingOrder) -> None:
        self._working[wo.req.symbol].pop(wo.req.id, None)
        self._symbol_of.pop(wo.req.id, None)

    def _passive_fill(
        self, wo: WorkingOrder, qty: int, spec: ContractSpec, ts: int, reports: list[ExecReport]
    ) -> None:
        assert wo.limit_ticks is not None
        reports.append(
            ExecReport(
                ReportType.FILL,
                wo.req.id,
                wo.req.symbol,
                ts,
                fill_qty=qty,
                fill_price=spec.from_ticks(wo.limit_ticks),
                liquidity=Liquidity.MAKER,
            )
        )
        wo.remaining -= qty
        if wo.remaining <= 0:
            self._remove(wo)

    def _aggress(
        self,
        request: OrderRequest,
        qty: int,
        book: TopOfBook,
        spec: ContractSpec,
        ts: int,
        reports: list[ExecReport],
        *,
        start_ticks: int | None = None,
        limit_ticks: int | None = None,
    ) -> int:
        """Take liquidity from the opposite side. Returns the unfilled quantity."""
        side = request.side
        direction = side.sign
        top, top_size, consumed = book.opposite(side)
        real_top = top is not None
        if top is None:
            if start_ticks is None:
                return qty
            top, top_size, consumed = start_ticks, 0, 0
        depth = self.config.depth_per_level or max(1, top_size)

        # Skip levels better than the starting price (e.g. a stop triggered by a gap).
        skip_levels = 0 if start_ticks is None else max(0, (start_ticks - top) * direction)
        skip_units = 0 if skip_levels == 0 else top_size + (skip_levels - 1) * depth
        cursor = max(consumed, skip_units)
        max_level = None if limit_ticks is None else (limit_ticks - top) * direction
        if max_level is not None and max_level < 0:
            return qty

        slip = 0 if limit_ticks is not None else self.config.extra_slippage_ticks
        remaining = qty
        while remaining > 0:
            if cursor < top_size:
                level, level_end = 0, top_size
            else:
                level = 1 + (cursor - top_size) // depth
                level_end = top_size + level * depth
            if max_level is not None and level > max_level:
                break
            take = min(remaining, level_end - cursor)
            reports.append(
                ExecReport(
                    ReportType.FILL,
                    request.id,
                    request.symbol,
                    ts,
                    fill_qty=take,
                    fill_price=spec.from_ticks(top + (level + slip) * direction),
                    liquidity=Liquidity.TAKER,
                )
            )
            cursor += take
            remaining -= take
        if real_top:
            book.consume(side, cursor)
        return remaining


def _priority(o: WorkingOrder) -> tuple[int, int]:
    """Price-time priority: best price first, then earliest arrival."""
    assert o.limit_ticks is not None
    price_key = -o.limit_ticks if o.side is Side.BUY else o.limit_ticks
    return price_key, o.arrival_seq
