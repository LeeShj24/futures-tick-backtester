"""Statistics and charts."""

from ftb.analytics.stats import (
    daily_pnl,
    equity_to_frame,
    fills_to_frame,
    format_summary,
    summarize,
    trades_to_frame,
)

__all__ = [
    "daily_pnl",
    "equity_to_frame",
    "fills_to_frame",
    "format_summary",
    "plot_equity",
    "summarize",
    "trades_to_frame",
]


def plot_equity(*args, **kwargs):
    """Lazy wrapper so importing ftb never requires matplotlib."""
    from ftb.analytics.plot import plot_equity as _plot

    return _plot(*args, **kwargs)
