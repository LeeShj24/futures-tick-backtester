"""Market data events.

Timestamps are integer nanoseconds since the Unix epoch (UTC), the same unit
Databento uses, so no precision is lost and ordering is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum

NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000


class Side(IntEnum):
    """Order side or trade aggressor. ``int(side)`` is the position sign."""

    BUY = 1
    SELL = -1
    NONE = 0  # only valid as a trade aggressor (unknown)

    @property
    def sign(self) -> int:
        return int(self)

    @property
    def opposite(self) -> Side:
        return Side(-int(self))

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class MarketEvent:
    ts: int
    symbol: str


@dataclass(frozen=True, slots=True)
class TradeEvent(MarketEvent):
    """A print on the tape. ``aggressor`` is the side that crossed the spread."""

    price: float
    size: int
    aggressor: Side = Side.NONE


@dataclass(frozen=True, slots=True)
class QuoteEvent(MarketEvent):
    """Top of book after an update. A missing side has NaN price and size 0."""

    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int

    @property
    def has_bid(self) -> bool:
        return not math.isnan(self.bid_price)

    @property
    def has_ask(self) -> bool:
        return not math.isnan(self.ask_price)

    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price


def ns_to_datetime(ts: int, tz=timezone.utc) -> datetime:
    """Convert epoch nanoseconds to an aware datetime (microsecond precision)."""
    return datetime.fromtimestamp(ts / NS_PER_SECOND, tz=tz)


def datetime_to_ns(dt: datetime) -> int:
    """Convert an aware datetime to epoch nanoseconds."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    return (delta.days * 86_400 + delta.seconds) * NS_PER_SECOND + delta.microseconds * 1_000
