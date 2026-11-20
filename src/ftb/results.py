"""Backtest result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ftb.accounting import EquityPoint, Position, RoundTrip
from ftb.contracts import ContractRegistry
from ftb.orders import Fill, Order

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BacktestResult:
    strategy_name: str
    initial_capital: float
    final_equity: float
    max_drawdown: float  # exact (tick-level), USD
    fills: list[Fill]
    trades: list[RoundTrip]
    equity_curve: list[EquityPoint]
    orders: list[Order]
    positions: dict[str, Position]
    contracts: ContractRegistry = field(repr=False)
    halted: str | None = None
    halt_ts: int | None = None
    events_processed: int = 0
    start_ts: int | None = None
    end_ts: int | None = None
    elapsed_s: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.final_equity - self.initial_capital

    @property
    def open_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if p.qty != 0}

    def trades_frame(self, tz: str = "America/New_York") -> pd.DataFrame:
        from ftb.analytics.stats import trades_to_frame

        return trades_to_frame(self.trades, tz=tz)

    def equity_frame(self, tz: str = "America/New_York") -> pd.DataFrame:
        from ftb.analytics.stats import equity_to_frame

        return equity_to_frame(self.equity_curve, tz=tz)

    def fills_frame(self, tz: str = "America/New_York") -> pd.DataFrame:
        from ftb.analytics.stats import fills_to_frame

        return fills_to_frame(self.fills, tz=tz)

    def summary(self, tz: str = "America/New_York") -> dict[str, Any]:
        from ftb.analytics.stats import summarize

        return summarize(self, tz=tz)
