"""Orders, execution reports and fills.

There are deliberately two representations of an order:

* :class:`OrderRequest` - the immutable message that travels to the simulated
  exchange (after latency).
* :class:`Order` - the *strategy's view*. It is only updated when an execution
  report reaches the strategy (after report latency), so a strategy can never
  learn about a fill before it could have in reality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ftb.events import Side


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"  # stop-market: becomes a market order when triggered


class OrderStatus(str, Enum):
    PENDING = "pending"  # sent, not yet acknowledged by the exchange
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


class Liquidity(str, Enum):
    MAKER = "maker"  # resting order was hit
    TAKER = "taker"  # we crossed the spread


class ReportType(str, Enum):
    ACCEPTED = "accepted"
    FILL = "fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    CANCEL_REJECTED = "cancel_rejected"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    id: int
    symbol: str
    side: Side
    qty: int
    type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    submit_ts: int = 0
    tag: str = ""


@dataclass(frozen=True, slots=True)
class ExecReport:
    """Exchange -> engine message. ``ts`` is exchange time."""

    type: ReportType
    order_id: int
    symbol: str
    ts: int
    fill_qty: int = 0
    fill_price: float = math.nan
    liquidity: Liquidity | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: int
    symbol: str
    side: Side
    qty: int
    price: float
    ts: int  # exchange time of the fill
    liquidity: Liquidity
    commission: float
    tag: str = ""

    @property
    def signed_qty(self) -> int:
        return self.qty * self.side.sign


@dataclass(slots=True)
class Order:
    """Strategy-side view of an order (see module docstring)."""

    id: int
    symbol: str
    side: Side
    qty: int
    type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str = ""
    submit_ts: int = 0
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: float = math.nan
    cancel_requested: bool = False
    reject_reason: str = ""
    last_update_ts: int = 0
    fills: list[Fill] = field(default_factory=list, repr=False)

    @property
    def remaining(self) -> int:
        return self.qty - self.filled_qty

    @property
    def is_active(self) -> bool:
        return not self.status.is_terminal

    def to_request(self) -> OrderRequest:
        return OrderRequest(
            id=self.id,
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            type=self.type,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            submit_ts=self.submit_ts,
            tag=self.tag,
        )
