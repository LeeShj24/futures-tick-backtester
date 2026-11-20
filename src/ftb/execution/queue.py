"""Queue-position models for passive limit orders resting *at the touch*.

Trade-throughs (a print at a worse price than our limit) and quote crosses are
handled by the execution model and are independent of the queue model. The
queue model only answers: *a print happened exactly at my price - how much of
it reaches me?*

With MBP-1 data we cannot see individual orders, so every model here is an
approximation. An MBO model (exact queue position from order-level data) can
be added later by implementing :class:`QueueModel`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from ftb.events import Side
from ftb.execution.book import TopOfBook, WorkingOrder


class QueueModel(ABC):
    name: str = "queue"

    def on_join(self, order: WorkingOrder, book: TopOfBook) -> None:  # noqa: B027
        """Called when a limit order starts resting."""

    def on_quote(self, order: WorkingOrder, book: TopOfBook) -> None:  # noqa: B027
        """Called for every resting order after each top-of-book update."""

    @abstractmethod
    def on_trade_at_price(self, order: WorkingOrder, trade_size: int) -> int:
        """A print of ``trade_size`` at exactly our limit price (on our side).

        Return how many contracts of ``order`` this print fills (the execution
        model additionally caps it by the print's size).
        """


class TradeThroughOnly(QueueModel):
    """Most conservative: never fill at the touch; only trade-throughs and crosses fill."""

    name = "trade_through_only"

    def on_trade_at_price(self, order: WorkingOrder, trade_size: int) -> int:
        return 0


class FillOnTouch(QueueModel):
    """Optimistic: any print at our price fills us (up to the print's size).

    Provided for comparison only - this is how many backtesters overstate
    limit-order strategies.
    """

    name = "fill_on_touch"

    def on_trade_at_price(self, order: WorkingOrder, trade_size: int) -> int:
        return order.remaining


class VolumeQueue(QueueModel):
    """Volume-based FIFO approximation.

    * Joining at the best price: ``ceil(displayed_size * ahead_fraction)``
      contracts are assumed ahead of us (``ahead_fraction=1`` means we are last
      in line, the conservative default).
    * Joining at a better price than the best displayed: nobody is ahead.
    * Joining behind the best price: unknown; estimated when our price becomes
      the best price (using the displayed size at that moment).
    * Prints at our price (from the opposite-side aggressor) reduce the queue
      ahead; volume beyond it fills us.
    * Displayed size can only shrink what is ahead of us: if the level shows
      fewer contracts than our estimate, the estimate is clamped down.
    """

    name = "volume_queue"

    def __init__(self, ahead_fraction: float = 1.0) -> None:
        if not ahead_fraction >= 0.0:
            raise ValueError("ahead_fraction must be >= 0")
        self.ahead_fraction = ahead_fraction

    def _estimate(self, size: int) -> int:
        return math.ceil(size * self.ahead_fraction)

    def _relative(self, order: WorkingOrder, book: TopOfBook) -> tuple[int | None, int]:
        """(diff, displayed). diff > 0: our price is better than the best displayed."""
        price, size = book.best(order.side)
        if price is None:
            return None, 0
        assert order.limit_ticks is not None
        diff = (order.limit_ticks - price) * (1 if order.side is Side.BUY else -1)
        return diff, size

    def on_join(self, order: WorkingOrder, book: TopOfBook) -> None:
        diff, size = self._relative(order, book)
        if diff is None or diff > 0:
            order.queue_ahead = 0
        elif diff == 0:
            order.queue_ahead = self._estimate(size)
        else:
            order.queue_ahead = None

    def on_quote(self, order: WorkingOrder, book: TopOfBook) -> None:
        diff, size = self._relative(order, book)
        if diff is None or diff > 0:
            # Nobody displayed at (or better than) our price.
            order.queue_ahead = 0
        elif diff == 0:
            if order.queue_ahead is None:
                order.queue_ahead = self._estimate(size)
            else:
                order.queue_ahead = min(order.queue_ahead, size)
        # diff < 0: best price is better than ours; nothing observable at our level.

    def on_trade_at_price(self, order: WorkingOrder, trade_size: int) -> int:
        if order.queue_ahead is None:
            return 0
        order.queue_ahead -= trade_size
        if order.queue_ahead >= 0:
            return 0
        filled = min(order.remaining, -order.queue_ahead)
        order.queue_ahead = 0
        return filled
