"""Synthetic top-of-book + trades generator (for tests and examples).

The generator produces a DataFrame in the **Databento MBP-1 column layout**, so
synthetic data flows through exactly the same loader path as real data.

The model is a toy, not a market simulator:

* events arrive as a Poisson process (rate modulated by a per-minute volatility
  regime);
* a trade lifts the ask / hits the bid and depletes displayed size; when a level
  is exhausted the book moves by one tick;
* quote updates add/cancel size, occasionally shift the whole book by a few
  ticks, and tighten the spread back to one tick.

Prices follow a driftless random walk: any strategy edge found on this data is
noise by construction. That makes it a good sanity check that costs, slippage
and latency are being charged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ftb.data.databento import iter_mbp1_events, normalize_mbp1_frame
from ftb.events import NS_PER_SECOND, MarketEvent, datetime_to_ns


@dataclass(frozen=True)
class SyntheticConfig:
    symbol: str = "NQ"
    start_price: float = 18_000.0
    tick_size: float = 0.25
    start_date: date = date(2024, 1, 2)
    days: int = 1  # number of weekdays to generate (no holiday calendar)
    session_start: time = time(9, 30)
    session_end: time = time(16, 0)
    timezone: str = "America/New_York"
    events_per_second: float = 5.0
    trade_prob: float = 0.35  # share of events that are trades
    shift_prob: float = 0.05  # share of events where the whole book jumps
    max_jump_ticks: int = 3
    mean_level_size: float = 8.0
    mean_trade_size: float = 2.0
    vol_of_vol: float = 0.35  # sigma of the per-minute log-volatility AR(1)
    overnight_gap_ticks: float = 40.0
    seed: int = 42


def trading_days(start: date, n: int) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


class _Pool:
    """Block-drawn random numbers (numpy per-call overhead dominates otherwise)."""

    def __init__(self, draw, block: int = 65_536) -> None:
        self._draw = draw
        self._block = block
        self._buf = draw(block)
        self._i = 0

    def next(self):
        if self._i >= len(self._buf):
            self._buf = self._draw(self._block)
            self._i = 0
        v = self._buf[self._i]
        self._i += 1
        return v


def generate_mbp1(config: SyntheticConfig | None = None, **overrides) -> pd.DataFrame:
    """Generate synthetic MBP-1 rows (Databento column names, float prices, ns timestamps)."""
    cfg = replace(config or SyntheticConfig(), **overrides)
    if cfg.events_per_second <= 0:
        raise ValueError("events_per_second must be > 0")
    if not 0 <= cfg.trade_prob + cfg.shift_prob <= 1:
        raise ValueError("trade_prob + shift_prob must be within [0, 1]")

    rng = np.random.default_rng(cfg.seed)
    tz = ZoneInfo(cfg.timezone)
    tick = cfg.tick_size

    uniform = _Pool(lambda n: rng.random(n))
    expo = _Pool(lambda n: rng.exponential(1.0, n))
    trade_size = _Pool(lambda n: rng.geometric(1.0 / max(cfg.mean_trade_size, 1.0), n))
    level_size = _Pool(lambda n: 1 + rng.poisson(max(cfg.mean_level_size - 1.0, 0.0), n))
    jump = _Pool(lambda n: np.minimum(rng.geometric(0.65, n), max(cfg.max_jump_ticks, 1)))
    churn = _Pool(lambda n: rng.integers(1, 4, n))
    normal = _Pool(lambda n: rng.standard_normal(n), block=1024)

    ts_l: list[int] = []
    act_l: list[str] = []
    side_l: list[str] = []
    px_l: list[float] = []
    sz_l: list[int] = []
    bpx_l: list[int] = []
    apx_l: list[int] = []
    bsz_l: list[int] = []
    asz_l: list[int] = []

    bid = round(cfg.start_price / tick)
    ask = bid + 1
    bid_sz = int(level_size.next())
    ask_sz = int(level_size.next())

    def emit(t: int, action: str, side: str, price_ticks: int | None, size: int) -> None:
        ts_l.append(t)
        act_l.append(action)
        side_l.append(side)
        px_l.append(np.nan if price_ticks is None else price_ticks * tick)
        sz_l.append(size)
        bpx_l.append(bid)
        apx_l.append(ask)
        bsz_l.append(bid_sz)
        asz_l.append(ask_sz)

    base_dt = 1.0 / cfg.events_per_second
    phi = 0.9
    for day_index, day in enumerate(trading_days(cfg.start_date, cfg.days)):
        t = datetime_to_ns(datetime.combine(day, cfg.session_start, tz))
        t_end = datetime_to_ns(datetime.combine(day, cfg.session_end, tz))
        if day_index > 0:
            gap = round(float(normal.next()) * cfg.overnight_gap_ticks)
            bid += gap
            ask = bid + 1
        bid_sz, ask_sz = int(level_size.next()), int(level_size.next())
        emit(t, "A", "N", None, 0)

        log_vol = 0.0
        vol_mult = 1.0
        next_regime = t + 60 * NS_PER_SECOND
        while True:
            if t >= next_regime:
                log_vol = phi * log_vol + cfg.vol_of_vol * float(normal.next())
                vol_mult = float(np.exp(log_vol))
                next_regime += 60 * NS_PER_SECOND
            t += max(1, int(float(expo.next()) * base_dt / np.sqrt(vol_mult) * NS_PER_SECOND))
            if t >= t_end:
                break

            u = float(uniform.next())
            p_shift = min(0.5, cfg.shift_prob * vol_mult)
            if u < cfg.trade_prob:
                buy = float(uniform.next()) < 0.5
                if buy:
                    size = min(int(trade_size.next()), ask_sz)
                    emit(t, "T", "B", ask, size)
                    ask_sz -= size
                    if ask_sz <= 0:
                        ask += 1
                        ask_sz = int(level_size.next())
                        if float(uniform.next()) < 0.6:
                            bid = ask - 1
                            bid_sz = int(level_size.next())
                    emit(t, "C", "A", None, 0)
                else:
                    size = min(int(trade_size.next()), bid_sz)
                    emit(t, "T", "A", bid, size)
                    bid_sz -= size
                    if bid_sz <= 0:
                        bid -= 1
                        bid_sz = int(level_size.next())
                        if float(uniform.next()) < 0.6:
                            ask = bid + 1
                            ask_sz = int(level_size.next())
                    emit(t, "C", "B", None, 0)
            elif u < cfg.trade_prob + p_shift:
                direction = 1 if float(uniform.next()) < 0.5 else -1
                k = int(jump.next())
                bid += direction * k
                ask += direction * k
                bid_sz, ask_sz = int(level_size.next()), int(level_size.next())
                emit(t, "A", "N", None, 0)
            else:
                on_bid = float(uniform.next()) < 0.5
                delta = int(churn.next()) * (1 if float(uniform.next()) < 0.52 else -1)
                if on_bid:
                    bid_sz += delta
                    if bid_sz <= 0:
                        bid -= 1
                        bid_sz = int(level_size.next())
                else:
                    ask_sz += delta
                    if ask_sz <= 0:
                        ask += 1
                        ask_sz = int(level_size.next())
                spread = ask - bid
                if spread > 1 and (float(uniform.next()) < 0.6 or spread > 4):
                    if float(uniform.next()) < 0.5:
                        bid += 1
                        bid_sz = 1 + int(churn.next())
                    else:
                        ask -= 1
                        ask_sz = 1 + int(churn.next())
                emit(t, "M", "B" if on_bid else "A", None, 0)

    ts = np.asarray(ts_l, dtype=np.int64)
    df = pd.DataFrame(
        {
            "ts_recv": ts,
            "ts_event": ts - 25_000,  # pretend 25us exchange -> capture latency
            "rtype": 1,
            "publisher_id": 1,
            "instrument_id": 1,
            "action": act_l,
            "side": side_l,
            "depth": 0,
            "price": np.asarray(px_l, dtype=float),
            "size": np.asarray(sz_l, dtype=np.int64),
            "bid_px_00": np.asarray(bpx_l, dtype=float) * tick,
            "ask_px_00": np.asarray(apx_l, dtype=float) * tick,
            "bid_sz_00": np.asarray(bsz_l, dtype=np.int64),
            "ask_sz_00": np.asarray(asz_l, dtype=np.int64),
            "symbol": cfg.symbol,
        }
    )
    return df


def iter_synthetic_events(
    config: SyntheticConfig | None = None, **overrides
) -> Iterator[MarketEvent]:
    cfg = replace(config or SyntheticConfig(), **overrides)
    frame = normalize_mbp1_frame(generate_mbp1(cfg))
    return iter_mbp1_events(frame, cfg.symbol)


def generate_events(config: SyntheticConfig | None = None, **overrides) -> list[MarketEvent]:
    """Convenience wrapper returning a list (fine for small test datasets)."""
    return list(iter_synthetic_events(config, **overrides))
