"""Risk-rule hooks.

Two hooks, mirroring where real risk checks live:

* :meth:`RiskRule.check_order` - pre-trade check at submission, using only
  what the strategy knows (``Context``). Returning a string rejects the order.
* :meth:`RiskRule.on_update` - called after every market event and every fill
  with the ground-truth :class:`~ftb.accounting.Portfolio` (like a broker or
  prop-firm risk system). Returning a string halts trading: working orders are
  cancelled, positions are flattened, and only reducing orders are accepted.

Prop-firm rules (trailing drawdown, daily loss limit, ...) plug in here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ftb.accounting import Portfolio
    from ftb.orders import Order
    from ftb.strategy import Context


class RiskRule:
    name: str = "risk_rule"

    def check_order(self, order: Order, ctx: Context) -> str | None:
        return None

    def on_update(self, ts: int, portfolio: Portfolio) -> str | None:
        return None

    @property
    def monitors_updates(self) -> bool:
        return type(self).on_update is not RiskRule.on_update


class MaxContracts(RiskRule):
    """Reject orders that could take the position beyond ``limit`` contracts.

    Worst case = known position + every working order on the same side filling.
    """

    name = "max_contracts"

    def __init__(self, limit: int, symbol: str | None = None) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.limit = limit
        self.symbol = symbol

    def check_order(self, order: Order, ctx: Context) -> str | None:
        if self.symbol is not None and (
            ctx.contract(order.symbol).symbol != ctx.contract(self.symbol).symbol
        ):
            return None
        pending = sum(o.remaining for o in ctx.open_orders(order.symbol) if o.side is order.side)
        worst = ctx.position(order.symbol) + order.side.sign * (pending + order.qty)
        if abs(worst) > self.limit:
            return f"worst-case position {worst:+d} exceeds limit {self.limit}"
        return None
