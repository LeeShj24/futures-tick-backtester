"""Databento MBP-1 loader."""

import math

import numpy as np
import pandas as pd
import pytest

from ftb import QuoteEvent, Side, TradeEvent, load_mbp1
from ftb.data import merge_streams, normalize_mbp1_frame
from ftb.data.databento import UNDEF_PRICE
from helpers import T0

SCALE = 1_000_000_000


def raw_rows():
    """Rows in Databento's raw encoding: int ns timestamps, fixed-point int prices."""
    p = lambda x: round(x * SCALE)  # noqa: E731
    return pd.DataFrame(
        {
            "ts_recv": [T0, T0 + 10, T0 + 10, T0 + 20, T0 + 30, T0 + 40, T0 + 50],
            "ts_event": [T0 - 5, T0 + 5, T0 + 6, T0 + 15, T0 + 25, T0 + 35, T0 + 45],
            "rtype": 1,
            "publisher_id": 1,
            "instrument_id": 42,
            "action": ["A", "T", "C", "M", "M", "R", "F"],
            "side": ["B", "B", "A", "A", "A", "N", "B"],
            "depth": 0,
            "price": [
                p(18000.0),
                p(18000.25),
                p(18000.25),
                p(18000.5),
                p(18000.5),
                UNDEF_PRICE,
                p(1.0),
            ],
            "size": [3, 2, 1, 4, 4, 0, 1],
            "flags": 0,
            "bid_px_00": [
                p(18000.0),
                p(18000.0),
                p(18000.0),
                p(18000.0),
                p(18000.0),
                UNDEF_PRICE,
                0,
            ],
            "ask_px_00": [
                UNDEF_PRICE,
                p(18000.25),
                p(18000.25),
                p(18000.5),
                p(18000.5),
                UNDEF_PRICE,
                0,
            ],
            "bid_sz_00": [3, 3, 3, 3, 3, 0, 0],
            "ask_sz_00": [0, 5, 3, 4, 4, 0, 0],
            "symbol": "NQZ4",
        }
    )


def test_load_csv_with_fixed_point_prices(tmp_path):
    path = tmp_path / "mbp1.csv"
    raw_rows().to_csv(path, index=False)
    events = list(load_mbp1(path))
    assert [type(e).__name__ for e in events] == [
        "QuoteEvent",
        "TradeEvent",
        "QuoteEvent",
        "QuoteEvent",
        "QuoteEvent",  # 'R' book clear
    ]  # duplicate 'M' deduped, 'F' ignored
    first, trade, after, _, cleared = events
    assert first.symbol == "NQZ4"
    assert first.bid_price == 18000.0 and first.bid_size == 3
    assert math.isnan(first.ask_price) and first.ask_size == 0
    assert isinstance(trade, TradeEvent)
    assert (trade.ts, trade.price, trade.size, trade.aggressor) == (T0 + 10, 18000.25, 2, Side.BUY)
    assert after.ask_size == 3
    assert math.isnan(cleared.bid_price) and math.isnan(cleared.ask_price)


def test_symbol_override_and_ts_event(tmp_path):
    path = tmp_path / "mbp1.csv"
    raw_rows().to_csv(path, index=False)
    events = list(load_mbp1(path, symbol="NQ", ts_field="ts_event"))
    assert {e.symbol for e in events} == {"NQ"}
    assert events[0].ts == T0 - 5


def test_parquet_with_pretty_prices_and_datetime_index(tmp_path):
    df = raw_rows()
    for col in ("price", "bid_px_00", "ask_px_00"):
        raw = df[col].astype("int64")
        df[col] = np.where(raw == UNDEF_PRICE, np.nan, raw / SCALE)
    df["ts_event"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
    df.index = pd.to_datetime(df.pop("ts_recv"), unit="ns", utc=True)
    df.index.name = "ts_recv"  # like DBNStore.to_df()
    path = tmp_path / "mbp1.parquet"
    df.to_parquet(path)
    events = list(load_mbp1(path, symbol="NQ"))
    assert events[0].ts == T0
    assert events[1].ts == T0 + 10
    assert events[1].price == 18000.25


def test_iso_timestamps_keep_nanoseconds(tmp_path):
    df = raw_rows().head(2)
    df["ts_recv"] = ["2024-01-02T14:30:00.000000001Z", "2024-01-02T14:30:00.000000010Z"]
    path = tmp_path / "iso.csv"
    df.to_csv(path, index=False)
    events = list(load_mbp1(path, symbol="NQ"))
    assert [e.ts for e in events] == [T0 + 1, T0 + 10]


def test_rows_are_sorted_stably():
    df = raw_rows().head(4).iloc[[3, 1, 2, 0]].reset_index(drop=True)
    norm = normalize_mbp1_frame(df)
    assert norm["ts"].tolist() == [T0, T0 + 10, T0 + 10, T0 + 20]
    # The two rows at T0+10 keep their file order (T then C).
    assert norm["action"].tolist() == ["A", "T", "C", "M"]


def test_multiple_instruments_require_a_filter():
    df = pd.concat([raw_rows(), raw_rows().assign(instrument_id=43, symbol="NQH5")])
    with pytest.raises(ValueError, match="several instruments"):
        normalize_mbp1_frame(df)
    assert len(normalize_mbp1_frame(df, instrument_id=43)) == len(raw_rows())
    assert len(normalize_mbp1_frame(df, raw_symbol="NQZ4")) == len(raw_rows())


def test_time_window():
    norm = normalize_mbp1_frame(raw_rows(), start=T0 + 10, end=T0 + 30)
    assert norm["ts"].tolist() == [T0 + 10, T0 + 10, T0 + 20]


def test_missing_columns_are_reported():
    with pytest.raises(ValueError, match="bid_px_00"):
        normalize_mbp1_frame(raw_rows().drop(columns=["bid_px_00"]))


def test_unsupported_extension(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="unsupported"):
        load_mbp1(path, symbol="NQ")


def test_merge_streams_by_time_with_stable_ties():
    a = [QuoteEvent(1, "NQ", 1.0, 1, 2.0, 1), QuoteEvent(3, "NQ", 1.0, 1, 2.0, 1)]
    b = [QuoteEvent(1, "ES", 1.0, 1, 2.0, 1), QuoteEvent(2, "ES", 1.0, 1, 2.0, 1)]
    merged = list(merge_streams(a, b))
    assert [(e.ts, e.symbol) for e in merged] == [(1, "NQ"), (1, "ES"), (2, "ES"), (3, "NQ")]
