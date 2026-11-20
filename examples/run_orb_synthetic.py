"""Run the opening-range breakout on synthetic NQ tick data.

The synthetic data is written in Databento MBP-1 layout and loaded back through
the real Databento loader, so the full pipeline is exercised:

    synthetic generator -> Parquet -> load_mbp1 -> Backtester -> stats + chart

Usage::

    python examples/run_orb_synthetic.py --days 10 --out examples/output

Synthetic prices are a driftless random walk: expect the strategy to lose
roughly its trading costs. That is the point - an honest engine should not
manufacture an edge out of noise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from ftb import (
    Backtester,
    ConstantLatency,
    ExecutionConfig,
    MaxContracts,
    SyntheticConfig,
    TopOfBookExecution,
    VolumeQueue,
    generate_mbp1,
    load_mbp1,
)
from ftb.analytics import format_summary, plot_equity
from ftb.strategies import OpeningRangeBreakout, ORBConfig


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--days", type=int, default=10, help="number of synthetic sessions")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--symbol", default="NQ", choices=["NQ", "ES", "MNQ", "MES"])
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--latency-ms", type=float, default=5.0, help="one-way latency")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1) Synthetic data in Databento MBP-1 layout, saved to Parquet.
    start_price = {"NQ": 18_000.0, "MNQ": 18_000.0, "ES": 5_000.0, "MES": 5_000.0}[args.symbol]
    synth = SyntheticConfig(
        symbol=args.symbol, start_price=start_price, days=args.days, seed=args.seed
    )
    frame = generate_mbp1(synth)
    data_path = out / f"synthetic_{args.symbol.lower()}_mbp1.parquet"
    frame.to_parquet(data_path, index=False)
    print(f"generated {len(frame):,} MBP-1 rows over {args.days} sessions -> {data_path}")

    # 2) Load through the Databento loader (lazy iterator, sorted by ts_recv).
    events = load_mbp1(data_path, symbol=args.symbol)

    # 3) Configure and run.
    strategy = OpeningRangeBreakout(ORBConfig(symbol=args.symbol, qty=args.qty, target_r=1.5))
    backtester = Backtester(
        strategy,
        events,
        execution=TopOfBookExecution(ExecutionConfig(queue_model=VolumeQueue(ahead_fraction=1.0))),
        latency=ConstantLatency(order_ms=args.latency_ms, report_ms=args.latency_ms),
        risk_rules=[MaxContracts(limit=2 * args.qty)],
        initial_capital=50_000.0,
        mark_method="liquidation",
        equity_interval_s=60.0,
    )
    result = backtester.run()

    # 4) Report.
    summary = result.summary()
    print()
    print(format_summary(summary))
    print(f"\nthroughput: {result.events_processed / max(result.elapsed_s, 1e-9):,.0f} events/s")

    chart = plot_equity(
        result, out / "equity.png", title=f"Opening range breakout - synthetic {args.symbol}"
    )
    result.trades_frame().to_csv(out / "trades.csv", index=False)
    result.fills_frame().to_csv(out / "fills.csv", index=False)
    pd.DataFrame([s.__dict__ for s in strategy.sessions]).to_csv(out / "sessions.csv", index=False)
    with open(out / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(summary), fh, indent=2, default=str)
    print(f"\nsaved: {chart}, {out / 'trades.csv'}, {out / 'fills.csv'}, {out / 'stats.json'}")
    return summary


if __name__ == "__main__":
    main()
