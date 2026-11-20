"""Execution-model interface (the simulated exchange)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ftb.contracts import ContractRegistry, ContractSpec
from ftb.events import QuoteEvent, TradeEvent
from ftb.orders import ExecReport, OrderRequest


class ExecutionModel(ABC):
    """A simulated exchange.

    The engine guarantees that calls arrive in exchange-time order: market data
    via :meth:`on_quote` / :meth:`on_trade`, and order messages via
    :meth:`submit` / :meth:`cancel` *after* their latency. Implementations
    return :class:`ExecReport` objects stamped with exchange time; the engine
    takes care of accounting and of delaying reports to the strategy.

    To add an order-book (MBO) model later, implement this interface with a
    full depth book and an exact queue-position tracker.
    """

    def __init__(self) -> None:
        self.contracts: ContractRegistry | None = None
        self._specs: dict[str, ContractSpec] = {}

    def bind(self, contracts: ContractRegistry) -> None:
        self.contracts = contracts
        self._specs.clear()

    def spec(self, symbol: str) -> ContractSpec:
        spec = self._specs.get(symbol)
        if spec is None:
            if self.contracts is None:
                raise RuntimeError("execution model is not bound to a ContractRegistry")
            spec = self.contracts.resolve(symbol)
            self._specs[symbol] = spec
        return spec

    @abstractmethod
    def on_quote(self, quote: QuoteEvent) -> list[ExecReport]: ...

    @abstractmethod
    def on_trade(self, trade: TradeEvent) -> list[ExecReport]: ...

    @abstractmethod
    def submit(self, request: OrderRequest, ts: int) -> list[ExecReport]: ...

    @abstractmethod
    def cancel(self, order_id: int, ts: int) -> list[ExecReport]: ...
