"""Positions, PnL in dollars, commissions, equity curve and round-trip trades.

The portfolio is the *ground truth* ledger: it is updated at exchange time,
the moment a fill happens (the strategy learns about it later, after report
latency). Equity is marked to market on every market event, so the running
peak and maximum drawdown are exact at tick resolution even though the stored
equity curve is sampled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ftb.contracts import ContractRegistry
from ftb.events import NS_PER_SECOND, QuoteEvent, Side, TradeEvent
from ftb.orders import Fill

MarkMethod = Literal["liquidation", "mid", "last"]


@dataclass(slots=True)
class Position:
    symbol: str
    qty: int = 0  # signed: + long, - short
    avg_price: float = 0.0
    realized_pnl: float = 0.0  # gross, USD
    commissions: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.realized_pnl - self.commissions + self.unrealized_pnl


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """A completed trade: from flat to flat (a reversal closes one and opens another)."""

    symbol: str
    direction: Side
    qty: int  # maximum absolute position held
    entry_ts: int
    exit_ts: int
    entry_price: float  # average of entry fills
    exit_price: float  # average of exit fills
    gross_pnl: float
    commissions: float
    net_pnl: float
    mae: float  # most adverse gross open PnL during the trade (<= 0)
    mfe: float  # most favourable gross open PnL during the trade (>= 0)
    pnl_ticks: float  # gross PnL per contract, in ticks
    n_fills: int

    @property
    def duration_s(self) -> float:
        return (self.exit_ts - self.entry_ts) / NS_PER_SECOND


@dataclass(frozen=True, slots=True)
class EquityPoint:
    ts: int
    equity: float
    realized_pnl: float
    commissions: float
    unrealized_pnl: float
    net_position: int
    drawdown: float


@dataclass(slots=True)
class _OpenTrade:
    symbol: str
    direction: Side
    entry_ts: int
    entry_qty: int = 0
    entry_notional: float = 0.0
    exit_qty: int = 0
    exit_notional: float = 0.0
    realized: float = 0.0
    commissions: float = 0.0
    max_qty: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    n_fills: int = 0
    last_ts: int = 0


@dataclass(slots=True)
class _Marks:
    bid: float = math.nan
    ask: float = math.nan
    last: float = math.nan


@dataclass
class Portfolio:
    contracts: ContractRegistry
    initial_capital: float = 0.0
    mark_method: MarkMethod = "liquidation"
    sample_interval_ns: int = 60 * NS_PER_SECOND
    record_on_fill: bool = True

    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[RoundTrip] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    realized_pnl: float = 0.0
    commissions: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = field(init=False)
    max_drawdown: float = 0.0
    max_drawdown_ts: int = 0

    _open: dict[str, _OpenTrade] = field(default_factory=dict, repr=False)
    _marks: dict[str, _Marks] = field(default_factory=dict, repr=False)
    _next_sample: int = field(default=0, repr=False)
    _now: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.mark_method not in ("liquidation", "mid", "last"):
            raise ValueError("mark_method must be 'liquidation', 'mid' or 'last'")
        if self.sample_interval_ns <= 0:
            raise ValueError("sample_interval_ns must be > 0")
        self.peak_equity = self.initial_capital

    # ------------------------------------------------------------------ queries
    @property
    def equity(self) -> float:
        return self.initial_capital + self.realized_pnl - self.commissions + self.unrealized_pnl

    @property
    def net_pnl(self) -> float:
        return self.equity - self.initial_capital

    def position(self, symbol: str) -> Position:
        pos = self.positions.get(symbol)
        if pos is None:
            pos = self.positions[symbol] = Position(symbol)
        return pos

    def open_trade_symbols(self) -> list[str]:
        return list(self._open)

    # ------------------------------------------------------------------ market data
    def on_quote(self, quote: QuoteEvent) -> None:
        marks = self._marks.get(quote.symbol)
        if marks is None:
            marks = self._marks[quote.symbol] = _Marks()
        marks.bid = quote.bid_price
        marks.ask = quote.ask_price
        self._now = quote.ts
        pos = self.positions.get(quote.symbol)
        if pos is not None and pos.qty != 0:
            self._revalue(pos)

    def on_trade(self, trade: TradeEvent) -> None:
        marks = self._marks.get(trade.symbol)
        if marks is None:
            marks = self._marks[trade.symbol] = _Marks()
        marks.last = trade.price
        self._now = trade.ts
        pos = self.positions.get(trade.symbol)
        if pos is not None and pos.qty != 0:
            self._revalue(pos)

    def mark_price(self, symbol: str, qty: int) -> float:
        """Price used to value ``qty`` contracts of ``symbol`` (NaN if unknown)."""
        m = self._marks.get(symbol)
        if m is None:
            return math.nan
        if self.mark_method == "liquidation":
            p = m.bid if qty > 0 else m.ask
            return m.last if math.isnan(p) else p
        if self.mark_method == "mid":
            if math.isnan(m.bid) or math.isnan(m.ask):
                return m.last
            return (m.bid + m.ask) / 2.0
        if math.isnan(m.last):
            return (m.bid + m.ask) / 2.0
        return m.last

    def _revalue(self, pos: Position) -> None:
        if pos.qty == 0:
            new_unrealized = 0.0
        else:
            mark = self.mark_price(pos.symbol, pos.qty)
            if math.isnan(mark):
                new_unrealized = 0.0
            else:
                pv = self.contracts.resolve(pos.symbol).point_value
                new_unrealized = (mark - pos.avg_price) * pos.qty * pv
        self.unrealized_pnl += new_unrealized - pos.unrealized_pnl
        pos.unrealized_pnl = new_unrealized

        trade = self._open.get(pos.symbol)
        if trade is not None:
            running = trade.realized + new_unrealized
            if running < trade.mae:
                trade.mae = running
            if running > trade.mfe:
                trade.mfe = running
        self._update_drawdown()

    def _update_drawdown(self) -> None:
        equity = self.equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = self.peak_equity - equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_ts = self._now

    # ------------------------------------------------------------------ fills
    def on_fill(self, fill: Fill) -> None:
        spec = self.contracts.resolve(fill.symbol)
        pos = self.position(fill.symbol)
        self._now = fill.ts
        self.fills.append(fill)
        pos.commissions += fill.commission
        self.commissions += fill.commission
        signed = fill.signed_qty
        fill_sign = 1 if signed > 0 else -1

        if pos.qty == 0 or (pos.qty > 0) == (signed > 0):
            # Opening or adding to a position.
            new_qty = pos.qty + signed
            pos.avg_price = (pos.avg_price * abs(pos.qty) + fill.price * fill.qty) / abs(new_qty)
            pos.qty = new_qty
            trade = self._open.get(fill.symbol)
            if trade is None:
                trade = self._open[fill.symbol] = _OpenTrade(fill.symbol, Side(fill_sign), fill.ts)
            self._add_entry(trade, fill.price, fill.qty, fill.commission, fill.ts, abs(new_qty))
        else:
            # Reducing, closing or reversing.
            trade = self._open[fill.symbol]
            close_qty = min(fill.qty, abs(pos.qty))
            open_qty = fill.qty - close_qty
            pos_sign = 1 if pos.qty > 0 else -1
            pnl = (fill.price - pos.avg_price) * close_qty * pos_sign * spec.point_value
            pos.realized_pnl += pnl
            self.realized_pnl += pnl
            close_commission = fill.commission * close_qty / fill.qty
            trade.exit_qty += close_qty
            trade.exit_notional += fill.price * close_qty
            trade.realized += pnl
            trade.commissions += close_commission
            trade.n_fills += 1
            trade.last_ts = fill.ts
            pos.qty -= pos_sign * close_qty

            # Position value changes with the fill: drop old unrealized now.
            self.unrealized_pnl -= pos.unrealized_pnl
            pos.unrealized_pnl = 0.0

            if pos.qty == 0:
                pos.avg_price = 0.0
                self._close_trade(trade, spec.tick_value)
                if open_qty > 0:
                    pos.qty = fill_sign * open_qty
                    pos.avg_price = fill.price
                    new_trade = self._open[fill.symbol] = _OpenTrade(
                        fill.symbol, Side(fill_sign), fill.ts
                    )
                    self._add_entry(
                        new_trade,
                        fill.price,
                        open_qty,
                        fill.commission - close_commission,
                        fill.ts,
                        open_qty,
                    )

        self._revalue(pos)
        if self.record_on_fill:
            self.sample(fill.ts)

    @staticmethod
    def _add_entry(
        trade: _OpenTrade, price: float, qty: int, commission: float, ts: int, abs_pos: int
    ) -> None:
        trade.entry_qty += qty
        trade.entry_notional += price * qty
        trade.commissions += commission
        trade.n_fills += 1
        trade.last_ts = ts
        trade.max_qty = max(trade.max_qty, abs_pos)

    def _close_trade(self, trade: _OpenTrade, tick_value: float) -> None:
        trade.mae = min(trade.mae, trade.realized)
        trade.mfe = max(trade.mfe, trade.realized)
        self._open.pop(trade.symbol, None)
        self.trades.append(
            RoundTrip(
                symbol=trade.symbol,
                direction=trade.direction,
                qty=trade.max_qty,
                entry_ts=trade.entry_ts,
                exit_ts=trade.last_ts,
                entry_price=trade.entry_notional / trade.entry_qty,
                exit_price=trade.exit_notional / trade.exit_qty,
                gross_pnl=trade.realized,
                commissions=trade.commissions,
                net_pnl=trade.realized - trade.commissions,
                mae=trade.mae,
                mfe=trade.mfe,
                pnl_ticks=trade.realized / (tick_value * trade.exit_qty),
                n_fills=trade.n_fills,
            )
        )

    # ------------------------------------------------------------------ equity curve
    def maybe_sample(self, ts: int) -> None:
        if ts >= self._next_sample:
            self.sample(ts)

    def sample(self, ts: int) -> None:
        self.equity_curve.append(
            EquityPoint(
                ts=ts,
                equity=self.equity,
                realized_pnl=self.realized_pnl,
                commissions=self.commissions,
                unrealized_pnl=self.unrealized_pnl,
                net_position=sum(p.qty for p in self.positions.values()),
                drawdown=self.peak_equity - self.equity,
            )
        )
        interval = self.sample_interval_ns
        self._next_sample = (ts // interval + 1) * interval
