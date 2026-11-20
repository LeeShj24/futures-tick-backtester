"""Opening range breakout (ORB) example strategy.

Each session:

1. Build the opening range from trade prints in
   ``[session_open, session_open + range_minutes)``.
2. When the range is complete, place a buy stop ``entry_offset_ticks`` above the
   high and a sell stop below the low (one-cancels-other, managed by the
   strategy - so a fast whipsaw can fill both, as it could in reality).
3. On entry, place a protective stop beyond the other side of the range and,
   optionally, a profit target at ``target_r`` times the initial risk.
4. Cancel unfilled entries at ``entry_cutoff``; flatten everything at
   ``flatten_time``.

All decisions use only what the strategy has been told so far: positions come
from delivered fills, and order states from delivered reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from ftb.events import QuoteEvent, Side, TradeEvent, datetime_to_ns, ns_to_datetime
from ftb.orders import Order, OrderStatus
from ftb.strategy import Context, Strategy


@dataclass(frozen=True)
class ORBConfig:
    symbol: str = "NQ"
    qty: int = 1
    session_open: time = time(9, 30)
    range_minutes: int = 15
    entry_cutoff: time = time(11, 30)
    flatten_time: time = time(15, 55)
    entry_offset_ticks: int = 1
    stop_offset_ticks: int = 1
    target_r: float | None = 1.5
    min_range_ticks: int = 4
    max_range_ticks: int | None = None
    timezone: str = "America/New_York"


class _Phase(str, Enum):
    WAITING = "waiting"  # before/while building the range
    ARMED = "armed"  # entry stops working
    IN_POSITION = "in_position"
    DONE = "done"


@dataclass
class SessionLog:
    date: str
    range_high: float | None = None
    range_low: float | None = None
    range_ticks: float | None = None
    outcome: str = "no_range"
    notes: list[str] = field(default_factory=list)


class OpeningRangeBreakout(Strategy):
    def __init__(self, config: ORBConfig | None = None, **overrides) -> None:
        self.cfg = replace(config or ORBConfig(), **overrides)
        if self.cfg.qty < 1:
            raise ValueError("qty must be >= 1")
        self._tz = ZoneInfo(self.cfg.timezone)
        self.sessions: list[SessionLog] = []
        self._day_end = -1
        self._reset_day()

    @property
    def name(self) -> str:
        return f"ORB({self.cfg.symbol}, {self.cfg.range_minutes}m)"

    # ------------------------------------------------------------------ callbacks
    def on_trade(self, ctx: Context, trade: TradeEvent) -> None:
        if trade.symbol != self.cfg.symbol:
            return
        ts = trade.ts
        if ts >= self._day_end:
            self._new_day(ctx, ts)
        if self._phase is _Phase.WAITING and self._open_ns <= ts < self._range_end_ns:
            p = trade.price
            self._hi = p if self._hi is None or p > self._hi else self._hi
            self._lo = p if self._lo is None or p < self._lo else self._lo
            return
        self._on_clock(ctx, ts)

    def on_quote(self, ctx: Context, quote: QuoteEvent) -> None:
        if quote.symbol != self.cfg.symbol:
            return
        if quote.ts >= self._day_end:
            self._new_day(ctx, quote.ts)
        self._on_clock(ctx, quote.ts)

    def on_order_update(self, ctx: Context, order: Order) -> None:
        if order.symbol != self.cfg.symbol:
            return
        sym = self.cfg.symbol
        tag = order.tag

        if tag.startswith("entry"):
            if order.status is OrderStatus.FILLED:
                other = self._short_entry if order is self._long_entry else self._long_entry
                if other is not None and other.is_active:
                    ctx.cancel(other)
                pos = ctx.position(sym)
                if pos == 0:
                    # Both entries filled before the OCO cancel got through.
                    self._log.notes.append("both entries filled (whipsaw)")
                    self._log.outcome = "whipsaw"
                    ctx.cancel_all(sym)
                    self._phase = _Phase.DONE
                elif self._phase is not _Phase.DONE:
                    self._place_brackets(ctx, pos, order.avg_fill_price)
                    self._phase = _Phase.IN_POSITION
                    self._log.outcome = "long" if pos > 0 else "short"
            elif order.status is OrderStatus.REJECTED:
                self._log.notes.append(f"{tag} rejected: {order.reject_reason}")
            return

        if tag in ("stop", "target"):
            if order.status is OrderStatus.FILLED:
                other = self._target if tag == "stop" else self._stop
                if other is not None and other.is_active:
                    ctx.cancel(other)
                self._log.outcome += f"->{tag}"
                if ctx.position(sym) != 0:
                    ctx.flatten(sym, cancel_orders=False)
                self._phase = _Phase.DONE
            elif order.status is OrderStatus.REJECTED and tag == "stop":
                # Protective stop could not be placed (market already beyond it).
                self._log.notes.append(f"stop rejected: {order.reject_reason}; flattening")
                ctx.flatten(sym)
                self._phase = _Phase.DONE

    # ------------------------------------------------------------------ logic
    def _on_clock(self, ctx: Context, ts: int) -> None:
        if self._phase is _Phase.WAITING and ts >= self._range_end_ns:
            self._arm(ctx)
        if (
            self._phase is _Phase.ARMED
            and ts >= self._cutoff_ns
            and ctx.position(self.cfg.symbol) == 0
        ):
            ctx.cancel_all(self.cfg.symbol)
            self._log.outcome = "no_breakout"
            self._phase = _Phase.DONE
        if ts >= self._flatten_ns:
            if not self._flattened:
                self._flattened = True
                if ctx.position(self.cfg.symbol) != 0 or ctx.open_orders(self.cfg.symbol):
                    if ctx.position(self.cfg.symbol) != 0:
                        self._log.outcome += "->time_exit"
                    ctx.flatten(self.cfg.symbol)
                self._phase = _Phase.DONE
            elif ctx.position(self.cfg.symbol) != 0:
                ctx.flatten(self.cfg.symbol, cancel_orders=False)  # late fills arrived

    def _arm(self, ctx: Context) -> None:
        cfg = self.cfg
        if self._hi is None or self._lo is None:
            self._log.outcome = "no_range"
            self._phase = _Phase.DONE
            return
        tick = ctx.contract(cfg.symbol).tick_size
        range_ticks = (self._hi - self._lo) / tick
        self._log.range_high, self._log.range_low, self._log.range_ticks = (
            self._hi,
            self._lo,
            range_ticks,
        )
        if range_ticks < cfg.min_range_ticks or (
            cfg.max_range_ticks is not None and range_ticks > cfg.max_range_ticks
        ):
            self._log.outcome = "range_filtered"
            self._phase = _Phase.DONE
            return
        self._long_entry = ctx.submit_stop(
            cfg.symbol,
            Side.BUY,
            cfg.qty,
            self._hi + cfg.entry_offset_ticks * tick,
            tag="entry_long",
        )
        self._short_entry = ctx.submit_stop(
            cfg.symbol,
            Side.SELL,
            cfg.qty,
            self._lo - cfg.entry_offset_ticks * tick,
            tag="entry_short",
        )
        self._log.outcome = "armed"
        self._phase = _Phase.ARMED

    def _place_brackets(self, ctx: Context, pos: int, entry_price: float) -> None:
        cfg = self.cfg
        spec = ctx.contract(cfg.symbol)
        tick = spec.tick_size
        assert self._hi is not None and self._lo is not None
        for o in (self._stop, self._target):
            if o is not None and o.is_active:
                ctx.cancel(o)
        qty = abs(pos)
        if pos > 0:
            stop_px = self._lo - cfg.stop_offset_ticks * tick
            exit_side = Side.SELL
            risk = entry_price - stop_px
            target_px = entry_price + cfg.target_r * risk if cfg.target_r else None
        else:
            stop_px = self._hi + cfg.stop_offset_ticks * tick
            exit_side = Side.BUY
            risk = stop_px - entry_price
            target_px = entry_price - cfg.target_r * risk if cfg.target_r else None
        self._stop = ctx.submit_stop(cfg.symbol, exit_side, qty, stop_px, tag="stop")
        self._target = (
            ctx.submit_limit(cfg.symbol, exit_side, qty, spec.round_price(target_px), tag="target")
            if target_px is not None
            else None
        )

    # ------------------------------------------------------------------ sessions
    def _reset_day(self) -> None:
        self._phase = _Phase.WAITING
        self._hi: float | None = None
        self._lo: float | None = None
        self._long_entry: Order | None = None
        self._short_entry: Order | None = None
        self._stop: Order | None = None
        self._target: Order | None = None
        self._flattened = False
        self._open_ns = self._range_end_ns = self._cutoff_ns = self._flatten_ns = 0
        self._log = SessionLog(date="")

    def _new_day(self, ctx: Context, ts: int) -> None:
        if ctx.position(self.cfg.symbol) != 0:  # should not happen: flattened each day
            ctx.flatten(self.cfg.symbol)
        self._reset_day()
        day = ns_to_datetime(ts, self._tz).date()

        def at(t: time, d=day) -> int:
            return datetime_to_ns(datetime.combine(d, t, self._tz))

        self._open_ns = at(self.cfg.session_open)
        self._range_end_ns = self._open_ns + self.cfg.range_minutes * 60 * 1_000_000_000
        self._cutoff_ns = at(self.cfg.entry_cutoff)
        self._flatten_ns = at(self.cfg.flatten_time)
        self._day_end = at(time(0), day + timedelta(days=1))
        if ts > self._open_ns:
            # Data starts after the open: the range would be incomplete.
            self._phase = _Phase.DONE
            self._log = SessionLog(date=day.isoformat(), outcome="late_start")
        else:
            self._log = SessionLog(date=day.isoformat())
        self.sessions.append(self._log)
