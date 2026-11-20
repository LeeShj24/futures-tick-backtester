"""Latency models.

Three delays are modelled separately:

* ``order``  - strategy decision -> order live at the exchange;
* ``cancel`` - strategy decision -> cancel processed at the exchange;
* ``report`` - exchange event (ack/fill/cancel) -> strategy learns about it.

Market data is assumed to reach the strategy at the data timestamp
(``ts_recv`` for Databento); fold any extra feed delay into ``order``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ftb.events import NS_PER_MS


class LatencyModel(ABC):
    @abstractmethod
    def order_delay(self, ts: int) -> int:
        """Nanoseconds between submitting at ``ts`` and the order being live."""

    @abstractmethod
    def cancel_delay(self, ts: int) -> int:
        """Nanoseconds between requesting a cancel at ``ts`` and it being processed."""

    @abstractmethod
    def report_delay(self, ts: int) -> int:
        """Nanoseconds between an exchange event at ``ts`` and the strategy seeing it."""


@dataclass(frozen=True)
class ConstantLatency(LatencyModel):
    """Fixed delays in milliseconds. ``cancel_ms`` defaults to ``order_ms``."""

    order_ms: float = 5.0
    report_ms: float = 5.0
    cancel_ms: float | None = None

    def __post_init__(self) -> None:
        for name in ("order_ms", "report_ms", "cancel_ms"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")

    def order_delay(self, ts: int) -> int:
        return round(self.order_ms * NS_PER_MS)

    def cancel_delay(self, ts: int) -> int:
        ms = self.order_ms if self.cancel_ms is None else self.cancel_ms
        return round(ms * NS_PER_MS)

    def report_delay(self, ts: int) -> int:
        return round(self.report_ms * NS_PER_MS)


ZERO_LATENCY = ConstantLatency(order_ms=0.0, report_ms=0.0)
