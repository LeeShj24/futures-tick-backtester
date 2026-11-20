"""Loader for Databento MBP-1 (top of book + trades) files.

Accepted inputs
---------------
* Parquet or CSV (optionally ``.gz``/``.zst`` compressed) exported from Databento,
  e.g. ``DBNStore.to_parquet()``, ``DBNStore.to_df().to_parquet()`` or a batch
  download in CSV encoding.
* Prices either as floats ("pretty" prices) or as raw fixed-point int64
  (1 unit = 1e-9), where ``INT64_MAX`` means "undefined".
* Timestamps either as integer nanoseconds, datetimes, or ISO-8601 strings.

Mapping to events
-----------------
* ``action == 'T'`` -> :class:`TradeEvent` (``side`` is the aggressor:
  ``'B'`` buy, ``'A'`` sell, ``'N'`` unknown).
* ``action in {'A', 'C', 'M'}`` -> :class:`QuoteEvent` built from the
  ``bid_px_00/ask_px_00/bid_sz_00/ask_sz_00`` fields (the book after the update).
* ``action == 'R'`` (book clear) -> a quote with both sides empty.
* ``action == 'F'`` is ignored (CME trades are reported as ``'T'``).

Book fields on trade rows are not turned into quotes: on CME the book update
that reflects a trade arrives in the following record(s).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from ftb.events import MarketEvent, QuoteEvent, Side, TradeEvent

UNDEF_PRICE = np.iinfo(np.int64).max
FIXED_PRICE_SCALE = 1e-9

REQUIRED_COLUMNS = (
    "action",
    "side",
    "price",
    "size",
    "bid_px_00",
    "ask_px_00",
    "bid_sz_00",
    "ask_sz_00",
)
_PRICE_COLUMNS = ("price", "bid_px_00", "ask_px_00")
_SIZE_COLUMNS = ("size", "bid_sz_00", "ask_sz_00")


def read_mbp1_frame(path: str | Path) -> pd.DataFrame:
    """Read a Parquet/CSV MBP-1 file into a DataFrame without normalising it."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    elif any(name.endswith(ext) for ext in (".csv", ".csv.gz", ".csv.zst", ".csv.bz2", ".csv.xz")):
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"unsupported file type: {path.name} (expected .parquet or .csv[.gz|.zst])"
        )
    # DBNStore.to_df() puts ts_recv in the index.
    if df.index.name in ("ts_recv", "ts_event"):
        df = df.reset_index()
    return df


def normalize_mbp1_frame(
    df: pd.DataFrame,
    *,
    ts_field: str = "ts_recv",
    instrument_id: int | None = None,
    raw_symbol: str | None = None,
    start: int | str | pd.Timestamp | None = None,
    end: int | str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Validate and normalise an MBP-1 frame.

    Returns a new frame with columns ``ts`` (int64 ns), ``action``/``side``
    (1-char str), float prices (NaN = undefined) and int sizes, sorted by ``ts``
    with the original order preserved for equal timestamps.
    """
    if ts_field not in df.columns:
        raise ValueError(f"timestamp column {ts_field!r} not found; columns: {list(df.columns)}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"MBP-1 data is missing required column(s): {missing}")

    df = df.copy()
    if raw_symbol is not None:
        if "symbol" not in df.columns:
            raise ValueError("raw_symbol filter requested but the data has no 'symbol' column")
        df = df[df["symbol"].astype(str) == raw_symbol]
    if instrument_id is not None:
        if "instrument_id" not in df.columns:
            raise ValueError("instrument_id filter requested but the data has no 'instrument_id'")
        df = df[df["instrument_id"].astype("int64") == int(instrument_id)]

    # Refuse to silently mix contracts (e.g. two expiries during a roll).
    for col in ("instrument_id", "symbol"):
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            values = sorted(map(str, df[col].dropna().unique()))[:10]
            raise ValueError(
                f"data contains several instruments ({col}: {values}); "
                "pass instrument_id=... or raw_symbol=... to select one"
            )

    out = pd.DataFrame(index=df.index)
    out["ts"] = _to_epoch_ns(df[ts_field])
    out["action"] = _to_char(df["action"])
    out["side"] = _to_char(df["side"])
    for col in _PRICE_COLUMNS:
        out[col] = _to_price(df[col])
    for col in _SIZE_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    if "symbol" in df.columns:
        out["symbol"] = df["symbol"].astype(str)

    out = out.sort_values("ts", kind="mergesort").reset_index(drop=True)
    if start is not None:
        out = out[out["ts"] >= _scalar_to_ns(start)]
    if end is not None:
        out = out[out["ts"] < _scalar_to_ns(end)]
    return out.reset_index(drop=True)


def iter_mbp1_events(
    frame: pd.DataFrame, symbol: str, *, dedupe_quotes: bool = True
) -> Iterator[MarketEvent]:
    """Turn a normalised frame into a time-ordered stream of events (lazily)."""
    ts = frame["ts"].to_numpy(dtype=np.int64)
    action = frame["action"].to_numpy(dtype=object)
    side = frame["side"].to_numpy(dtype=object)
    price = frame["price"].to_numpy(dtype=float)
    size = frame["size"].to_numpy(dtype=np.int64)
    bid_px = frame["bid_px_00"].to_numpy(dtype=float)
    ask_px = frame["ask_px_00"].to_numpy(dtype=float)
    bid_sz = frame["bid_sz_00"].to_numpy(dtype=np.int64)
    ask_sz = frame["ask_sz_00"].to_numpy(dtype=np.int64)

    aggressor = {"B": Side.BUY, "A": Side.SELL}
    last_quote: tuple[float, int, float, int] | None = None
    nan = math.nan

    for i in range(len(ts)):
        act = action[i]
        t = int(ts[i])
        if act == "T":
            p = float(price[i])
            if math.isnan(p):
                continue
            yield TradeEvent(t, symbol, p, int(size[i]), aggressor.get(side[i], Side.NONE))
        elif act in ("A", "C", "M", "R"):
            if act == "R":
                key = (nan, 0, nan, 0)
            else:
                bp, ap = float(bid_px[i]), float(ask_px[i])
                key = (
                    bp,
                    int(bid_sz[i]) if not math.isnan(bp) else 0,
                    ap,
                    int(ask_sz[i]) if not math.isnan(ap) else 0,
                )
            if dedupe_quotes and last_quote is not None and _same_quote(key, last_quote):
                continue
            last_quote = key
            yield QuoteEvent(t, symbol, key[0], key[1], key[2], key[3])
        # 'F' and anything else: ignored


def load_mbp1(
    path: str | Path,
    symbol: str | None = None,
    *,
    ts_field: str = "ts_recv",
    instrument_id: int | None = None,
    raw_symbol: str | None = None,
    start: int | str | pd.Timestamp | None = None,
    end: int | str | pd.Timestamp | None = None,
    dedupe_quotes: bool = True,
) -> Iterator[MarketEvent]:
    """Load a Databento MBP-1 file as a time-ordered event iterator.

    Parameters
    ----------
    symbol:
        Name given to the events (e.g. ``"NQ"`` or ``"NQZ4"``); it must resolve
        in the :class:`~ftb.contracts.ContractRegistry`. Defaults to the file's
        ``symbol`` column.
    ts_field:
        ``"ts_recv"`` (default; when the data reached the capture server, the
        closest proxy for when a strategy could react) or ``"ts_event"``.
    """
    frame = normalize_mbp1_frame(
        read_mbp1_frame(path),
        ts_field=ts_field,
        instrument_id=instrument_id,
        raw_symbol=raw_symbol,
        start=start,
        end=end,
    )
    if symbol is None:
        if "symbol" not in frame.columns or frame.empty:
            raise ValueError("symbol=... is required when the data has no 'symbol' column")
        symbol = str(frame["symbol"].iloc[0])
    return iter_mbp1_events(frame, symbol, dedupe_quotes=dedupe_quotes)


# ---------------------------------------------------------------------------- helpers


def _same_quote(a: tuple[float, int, float, int], b: tuple[float, int, float, int]) -> bool:
    def eq(x: float, y: float) -> bool:
        return x == y or (math.isnan(x) and math.isnan(y))

    return eq(a[0], b[0]) and a[1] == b[1] and eq(a[2], b[2]) and a[3] == b[3]


def _to_char(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int64").map(chr).astype(object)
    return series.astype(str).str.slice(0, 1).astype(object)


def _to_price(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        raw = series.astype("int64")
        values = raw.astype("float64") * FIXED_PRICE_SCALE
        values[raw == UNDEF_PRICE] = np.nan
        return values
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _to_epoch_ns(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        return series.astype("int64")
    if pd.api.types.is_float_dtype(series):
        if series.isna().any():
            raise ValueError("timestamp column contains missing values")
        return series.astype("int64")
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        dt = series.dt.tz_convert("UTC")
    elif pd.api.types.is_datetime64_dtype(series):
        dt = series.dt.tz_localize("UTC")
    else:
        dt = pd.to_datetime(series, utc=True, format="ISO8601")
    if dt.isna().any():
        raise ValueError("timestamp column contains missing/unparseable values")
    return dt.dt.as_unit("ns").astype("int64")


def _scalar_to_ns(value: int | str | pd.Timestamp) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.as_unit("ns").value)
