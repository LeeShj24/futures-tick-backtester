"""Equity / drawdown chart (matplotlib, static PNG).

Two stacked panels sharing the time axis - never a dual y-axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ftb.results import BacktestResult

# Reference light palette (validated categorical slot 1 + slot 8, neutral inks).
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3de"
_EQUITY = "#2a78d6"
_DRAWDOWN = "#e34948"


def _usd(value: float) -> str:
    return f"-${-value:,.0f}" if value < 0 else f"${value:,.0f}"


def plot_equity(
    result: BacktestResult,
    path: str | Path,
    *,
    title: str | None = None,
    tz: str = "America/New_York",
    dpi: int = 150,
) -> Path:
    """Save an equity curve + drawdown chart to ``path`` and return the path.

    The x-axis is the equity-sample index, i.e. trading time only (overnight and
    weekend gaps are removed); ticks mark session starts.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib: pip install 'futures-tick-backtester[plot]'"
        ) from exc

    eq = result.equity_frame(tz=tz)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(11, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08},
        facecolor=_SURFACE,
    )
    for ax in (ax_eq, ax_dd):
        ax.set_facecolor(_SURFACE)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_TEXT_SECONDARY, labelsize=9, length=0)

    money = FuncFormatter(lambda v, _: _usd(v))
    if eq.empty:
        ax_eq.text(0.5, 0.5, "no equity samples", ha="center", va="center", color=_TEXT_SECONDARY)
    else:
        x = range(len(eq))
        ax_eq.plot(x, eq["equity"].to_numpy(), color=_EQUITY, linewidth=2, solid_capstyle="round")
        ax_eq.axhline(
            result.initial_capital, color=_TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (3, 3))
        )
        dd = -eq["drawdown"].to_numpy()
        ax_dd.fill_between(x, dd, 0, color=_DRAWDOWN, alpha=0.35, linewidth=0)
        ax_dd.plot(x, dd, color=_DRAWDOWN, linewidth=1.2)

        # Session boundaries as x tick labels.
        dates = eq.index.date
        starts = [i for i in range(len(dates)) if i == 0 or dates[i] != dates[i - 1]]
        step = max(1, len(starts) // 10)
        ticks = starts[::step]
        ax_dd.set_xticks(ticks)
        ax_dd.set_xticklabels([dates[i].strftime("%b %d") for i in ticks])
        for i in starts[1:]:
            ax_eq.axvline(i, color=_GRID, linewidth=0.6)
            ax_dd.axvline(i, color=_GRID, linewidth=0.6)
        ax_eq.set_xlim(0, len(eq) - 1)

        last = eq["equity"].iloc[-1]
        ax_eq.annotate(
            _usd(last),
            xy=(len(eq) - 1, last),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_TEXT,
        )

    ax_eq.yaxis.set_major_formatter(money)
    ax_dd.yaxis.set_major_formatter(money)
    ax_eq.set_ylabel("Equity", color=_TEXT_SECONDARY, fontsize=9)
    ax_dd.set_ylabel("Drawdown", color=_TEXT_SECONDARY, fontsize=9)

    s = result.summary(tz=tz)
    heading = title or f"{result.strategy_name} - equity curve"
    subtitle = (
        f"net {_usd(s['net_pnl'])}  |  {s['n_trades']} trades  |  "
        f"commissions {_usd(s['commissions_total'])}  |  max drawdown {_usd(s['max_drawdown'])}"
    )
    fig.text(0.07, 0.955, heading, fontsize=13, color=_TEXT, weight="bold", ha="left")
    fig.text(0.07, 0.918, subtitle, fontsize=9.5, color=_TEXT_SECONDARY, ha="left")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.89, bottom=0.07)
    fig.savefig(path, dpi=dpi, facecolor=_SURFACE)
    plt.close(fig)
    return path
