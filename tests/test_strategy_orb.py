"""Opening range breakout on synthetic data + the example script."""

import importlib.util
import json
from datetime import time
from pathlib import Path

import pandas as pd
import pytest

from ftb import ConstantLatency, generate_events
from ftb.strategies import OpeningRangeBreakout, ORBConfig
from helpers import run


@pytest.fixture(scope="module")
def orb_run():
    events = generate_events(days=4, events_per_second=2.0, seed=11)
    strat = OpeningRangeBreakout(ORBConfig(symbol="NQ", qty=1))
    result = run(events, strat, latency=ConstantLatency(order_ms=5, report_ms=5))
    return strat, result


def test_orb_trades_and_ends_flat(orb_run):
    strat, result = orb_run
    assert len(strat.sessions) == 4
    assert result.trades, "expected at least one breakout on 4 synthetic sessions"
    assert result.open_positions == {}


def test_orb_never_enters_before_range_end_or_after_cutoff(orb_run):
    _, result = orb_run
    fills = result.fills_frame()
    entries = fills[fills["tag"].str.startswith("entry")]
    local = entries["time"].dt.time
    assert (local >= time(9, 45)).all()
    assert (local < time(11, 30)).all()


def test_orb_at_most_one_position_per_session(orb_run):
    _, result = orb_run
    trades = result.trades_frame()
    per_day = trades.groupby(trades["entry_time"].dt.date).size()
    assert (per_day <= 2).all()  # 1 normally; 2 only on a whipsaw


def test_orb_exits_before_close(orb_run):
    _, result = orb_run
    trades = result.trades_frame()
    assert (trades["exit_time"].dt.time <= time(16, 0)).all()


def test_example_script(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples" / "run_orb_synthetic.py"
    spec = importlib.util.spec_from_file_location("run_orb_synthetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.main(["--days", "2", "--out", str(tmp_path)])
    for name in ("equity.png", "trades.csv", "fills.csv", "stats.json", "sessions.csv"):
        assert (tmp_path / name).stat().st_size > 0
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["net_pnl"] == pytest.approx(summary["net_pnl"])
    assert (tmp_path / "equity.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert len(trades) == summary["n_trades"]
