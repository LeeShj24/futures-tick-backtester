"""Exchange-side state shared by execution and queue models."""

from __future__ import annotations

from dataclasses import dataclass

from ftb.events import Side
from ftb.orders import OrderRequest


@dataclass(slots=True)
class TopOfBook:
    """Best bid/ask in integer ticks, as known to the simulated exchange.

    ``consumed_*`` counts contracts our own aggressive orders have already taken
    from that side since the last quote update, so two market orders sent
    back-to-back do not both get the same displayed liquidity.
    """

    bid: int | None = None
    bid_size: int = 0
    ask: int | None = None
    ask_size: int = 0
    last_trade: int | None = None
    consumed_bid: int = 0
    consumed_ask: int = 0
    ts: int = 0

    def best(self, side: Side) -> tuple[int | None, int]:
        """(price, size) of the book side a *resting* order of ``side`` joins."""
        return (self.bid, self.bid_size) if side is Side.BUY else (self.ask, self.ask_size)

    def opposite(self, side: Side) -> tuple[int | None, int, int]:
        """(price, size, consumed) of the side an *aggressive* order of ``side`` hits."""
        if side is Side.BUY:
            return self.ask, self.ask_size, self.consumed_ask
        return self.bid, self.bid_size, self.consumed_bid

    def consume(self, side: Side, units: int) -> None:
        if side is Side.BUY:
            self.consumed_ask = units
        else:
            self.consumed_bid = units


@dataclass(slots=True)
class WorkingOrder:
    """Exchange-side record of a live limit or (untriggered) stop order."""

    req: OrderRequest
    remaining: int
    limit_ticks: int | None = None
    stop_ticks: int | None = None
    arrival_ts: int = 0
    arrival_seq: int = 0
    # Estimated contracts ahead of us at our price (None = unknown yet).
    queue_ahead: int | None = None

    @property
    def side(self) -> Side:
        return self.req.side
