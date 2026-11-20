"""Market data loading and generation."""

from ftb.data.databento import (
    iter_mbp1_events,
    load_mbp1,
    normalize_mbp1_frame,
    read_mbp1_frame,
)
from ftb.data.stream import DataOrderError, check_sorted, merge_streams
from ftb.data.synthetic import (
    SyntheticConfig,
    generate_events,
    generate_mbp1,
    iter_synthetic_events,
    trading_days,
)

__all__ = [
    "DataOrderError",
    "SyntheticConfig",
    "check_sorted",
    "generate_events",
    "generate_mbp1",
    "iter_mbp1_events",
    "iter_synthetic_events",
    "load_mbp1",
    "merge_streams",
    "normalize_mbp1_frame",
    "read_mbp1_frame",
    "trading_days",
]
