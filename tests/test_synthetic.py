"""Synthetic data generator."""

from datetime import date

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ftb import generate_events, generate_mbp1, load_mbp1
from ftb.data import trading_days


def small(**kw):
    return generate_mbp1(days=1, events_per_second=0.5, **kw)


def test_deterministic_for_a_seed():
    assert_frame_equal(small(seed=1), small(seed=1))
    assert not small(seed=1).equals(small(seed=2))


def test_book_invariants():
    df = small(seed=4)
    assert (np.diff(df["ts_recv"].to_numpy()) >= 0).all()
    assert (df["bid_px_00"] < df["ask_px_00"]).all()
    for col in ("bid_px_00", "ask_px_00"):
        ticks = df[col] / 0.25
        assert np.allclose(ticks, ticks.round())
    assert (df["bid_sz_00"] > 0).all() and (df["ask_sz_00"] > 0).all()
    trades = df[df["action"] == "T"]
    buys, sells = trades[trades["side"] == "B"], trades[trades["side"] == "A"]
    assert len(buys) and len(sells)
    # Trade rows carry the pre-trade book: buys lift the ask, sells hit the bid.
    assert (buys["price"] == buys["ask_px_00"]).all()
    assert (sells["price"] == sells["bid_px_00"]).all()


def test_events_stay_inside_the_session():
    df = small(seed=2)
    local = pd.to_datetime(df["ts_recv"], unit="ns", utc=True).dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    assert minutes.min() >= 9 * 60 + 30
    assert minutes.max() < 16 * 60


def test_parquet_round_trip_through_databento_loader(tmp_path):
    df = small(seed=9)
    path = tmp_path / "synthetic.parquet"
    df.to_parquet(path, index=False)
    assert list(load_mbp1(path, symbol="NQ")) == generate_events(
        days=1, events_per_second=0.5, seed=9
    )


def test_weekends_are_skipped():
    assert trading_days(date(2024, 1, 5), 2) == [date(2024, 1, 5), date(2024, 1, 8)]
    df = generate_mbp1(start_date=date(2024, 1, 5), days=2, events_per_second=0.2)
    days = (
        pd.to_datetime(df["ts_recv"], unit="ns", utc=True).dt.tz_convert("America/New_York").dt.date
    )
    assert sorted(set(days)) == [date(2024, 1, 5), date(2024, 1, 8)]
