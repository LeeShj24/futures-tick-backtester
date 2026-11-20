"""Helpers for combining event streams."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator

from ftb.events import MarketEvent


class DataOrderError(ValueError):
    """Raised when market data is not in non-decreasing timestamp order."""


def merge_streams(*streams: Iterable[MarketEvent]) -> Iterator[MarketEvent]:
    """Merge already-sorted streams (e.g. several instruments) by timestamp.

    Ties keep the order of the input streams, so the merge is deterministic.
    """
    return heapq.merge(*streams, key=lambda e: e.ts)


def check_sorted(stream: Iterable[MarketEvent]) -> Iterator[MarketEvent]:
    """Pass events through, raising :class:`DataOrderError` on a timestamp regression."""
    last = None
    for i, event in enumerate(stream):
        if last is not None and event.ts < last:
            raise DataOrderError(f"event #{i} at ts={event.ts} is earlier than previous ts={last}")
        last = event.ts
        yield event
