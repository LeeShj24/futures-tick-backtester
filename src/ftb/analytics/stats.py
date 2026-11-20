"""Trade statistics and DataFrame conversions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ftb.accounting import EquityPoint, RoundTrip
from ftb.orders import Fill

if TYPE_CHECKING:
    from ftb.results import BacktestResult


def _to_local(ts: Sequence[int] | np.ndarray, tz: str) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(ts, dtype=np.int64), unit="ns", utc=True).tz_convert(tz)


def trades_to_frame(trades: Sequence[RoundTrip], tz: str = "America/New_York") -> pd.DataFrame:
    columns = [
        "symbol",
        "direction",
        "qty",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "pnl_ticks",
        "gross_pnl",
        "commissions",
        "net_pnl",
        "mae",
        "mfe",
        "duration_s",
        "n_fills",
    ]
    if not trades:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(
        {
            "symbol": [t.symbol for t in trades],
            "direction": ["long" if t.direction.sign > 0 else "short" for t in trades],
            "qty": [t.qty for t in trades],
            "entry_time": _to_local([t.entry_ts for t in trades], tz),
            "exit_time": _to_local([t.exit_ts for t in trades], tz),
            "entry_price": [t.entry_price for t in trades],
            "exit_price": [t.exit_price for t in trades],
            "pnl_ticks": [t.pnl_ticks for t in trades],
            "gross_pnl": [t.gross_pnl for t in trades],
            "commissions": [t.commissions for t in trades],
            "net_pnl": [t.net_pnl for t in trades],
            "mae": [t.mae for t in trades],
            "mfe": [t.mfe for t in trades],
            "duration_s": [t.duration_s for t in trades],
            "n_fills": [t.n_fills for t in trades],
        }
    )
    return df[columns]


def equity_to_frame(points: Sequence[EquityPoint], tz: str = "America/New_York") -> pd.DataFrame:
    if not points:
        return pd.DataFrame(
            columns=[
                "equity",
                "realized_pnl",
                "commissions",
                "unrealized_pnl",
                "net_position",
                "drawdown",
            ]
        )
    df = pd.DataFrame(
        {
            "equity": [p.equity for p in points],
            "realized_pnl": [p.realized_pnl for p in points],
            "commissions": [p.commissions for p in points],
            "unrealized_pnl": [p.unrealized_pnl for p in points],
            "net_position": [p.net_position for p in points],
            "drawdown": [p.drawdown for p in points],
        },
        index=_to_local([p.ts for p in points], tz),
    )
    df.index.name = "time"
    return df


def fills_to_frame(fills: Sequence[Fill], tz: str = "America/New_York") -> pd.DataFrame:
    if not fills:
        return pd.DataFrame(
            columns=[
                "time",
                "order_id",
                "symbol",
                "side",
                "qty",
                "price",
                "liquidity",
                "commission",
                "tag",
            ]
        )
    return pd.DataFrame(
        {
            "time": _to_local([f.ts for f in fills], tz),
            "order_id": [f.order_id for f in fills],
            "symbol": [f.symbol for f in fills],
            "side": [f.side.name for f in fills],
            "qty": [f.qty for f in fills],
            "price": [f.price for f in fills],
            "liquidity": [f.liquidity.value for f in fills],
            "commission": [f.commission for f in fills],
            "tag": [f.tag for f in fills],
        }
    )


def daily_pnl(points: Sequence[EquityPoint], initial_capital: float, tz: str) -> pd.Series:
    """Net PnL per session date (local ``tz``), from the end-of-day equity."""
    if not points:
        return pd.Series(dtype=float)
    eq = equity_to_frame(points, tz)["equity"]
    eod = eq.groupby(eq.index.date).last()
    prev = eod.shift(1)
    prev.iloc[0] = initial_capital
    return eod - prev


def summarize(result: BacktestResult, tz: str = "America/New_York") -> dict[str, Any]:
    trades = result.trades
    net = np.array([t.net_pnl for t in trades], dtype=float)
    wins = net[net > 0]
    losses = net[net < 0]
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    daily = daily_pnl(result.equity_curve, result.initial_capital, tz)
    daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else math.nan
    sharpe = (
        float(daily.mean() / daily_std * math.sqrt(252))
        if len(daily) > 1 and daily_std > 0
        else math.nan
    )
    contracts_traded = sum(f.qty for f in result.fills)

    summary: dict[str, Any] = {
        "strategy": result.strategy_name,
        "events_processed": result.events_processed,
        "initial_capital": result.initial_capital,
        "final_equity": result.final_equity,
        "net_pnl": result.net_pnl,
        "gross_pnl_closed": float(sum(t.gross_pnl for t in trades)),
        "commissions_total": float(sum(f.commission for f in result.fills)),
        "max_drawdown": result.max_drawdown,
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(trades)) if trades else math.nan,
        "avg_trade": float(net.mean()) if trades else math.nan,
        "avg_win": float(wins.mean()) if wins.size else math.nan,
        "avg_loss": float(losses.mean()) if losses.size else math.nan,
        "largest_win": float(net.max()) if trades else math.nan,
        "largest_loss": float(net.min()) if trades else math.nan,
        "profit_factor": (gross_profit / gross_loss)
        if gross_loss > 0
        else math.inf
        if gross_profit > 0
        else math.nan,
        "avg_pnl_ticks": float(np.mean([t.pnl_ticks for t in trades])) if trades else math.nan,
        "avg_mae": float(np.mean([t.mae for t in trades])) if trades else math.nan,
        "avg_mfe": float(np.mean([t.mfe for t in trades])) if trades else math.nan,
        "avg_duration_s": float(np.mean([t.duration_s for t in trades])) if trades else math.nan,
        "contracts_traded": contracts_traded,
        "maker_fill_share": (
            sum(f.qty for f in result.fills if f.liquidity.value == "maker") / contracts_traded
            if contracts_traded
            else math.nan
        ),
        "sessions": len(daily),
        "best_day": float(daily.max()) if len(daily) else math.nan,
        "worst_day": float(daily.min()) if len(daily) else math.nan,
        "daily_sharpe_annualized": sharpe,
        "open_positions": {s: p.qty for s, p in result.open_positions.items()},
        "halted": result.halted,
        "elapsed_s": result.elapsed_s,
    }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """Render a summary dict as an aligned text table."""
    money = {
        "initial_capital",
        "final_equity",
        "net_pnl",
        "gross_pnl_closed",
        "commissions_total",
        "max_drawdown",
        "avg_trade",
        "avg_win",
        "avg_loss",
        "largest_win",
        "largest_loss",
        "avg_mae",
        "avg_mfe",
        "best_day",
        "worst_day",
    }
    pct = {"win_rate", "maker_fill_share"}
    width = max(len(k) for k in summary)
    lines = []
    for key, value in summary.items():
        if isinstance(value, float) and key in money:
            text = "n/a" if math.isnan(value) else f"${value:,.2f}"
        elif isinstance(value, float) and key in pct:
            text = "n/a" if math.isnan(value) else f"{value:.1%}"
        elif isinstance(value, float):
            text = "n/a" if math.isnan(value) else f"{value:,.3f}"
        else:
            text = str(value)
        lines.append(f"{key.ljust(width)}  {text}")
    return "\n".join(lines)
