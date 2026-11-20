"""futures-tick-backtester: an honest, event-driven tick backtester for CME futures."""

from ftb.accounting import EquityPoint, Portfolio, Position, RoundTrip
from ftb.contracts import (
    ContractRegistry,
    ContractSpec,
    FeeSchedule,
    default_contracts,
    load_contracts,
)
from ftb.data import DataOrderError, SyntheticConfig, generate_events, generate_mbp1, load_mbp1
from ftb.engine import Backtester
from ftb.events import MarketEvent, QuoteEvent, Side, TradeEvent
from ftb.execution import (
    ZERO_LATENCY,
    ConstantLatency,
    ExecutionConfig,
    ExecutionModel,
    FillOnTouch,
    LatencyModel,
    QueueModel,
    TopOfBookExecution,
    TradeThroughOnly,
    VolumeQueue,
)
from ftb.orders import Fill, Liquidity, Order, OrderStatus, OrderType
from ftb.results import BacktestResult
from ftb.risk import MaxContracts, RiskRule
from ftb.strategy import Context, Strategy

__version__ = "0.1.0"

__all__ = [
    "ZERO_LATENCY",
    "BacktestResult",
    "Backtester",
    "ConstantLatency",
    "Context",
    "ContractRegistry",
    "ContractSpec",
    "DataOrderError",
    "EquityPoint",
    "ExecutionConfig",
    "ExecutionModel",
    "FeeSchedule",
    "Fill",
    "FillOnTouch",
    "LatencyModel",
    "Liquidity",
    "MarketEvent",
    "MaxContracts",
    "Order",
    "OrderStatus",
    "OrderType",
    "Portfolio",
    "Position",
    "QueueModel",
    "QuoteEvent",
    "RiskRule",
    "RoundTrip",
    "Side",
    "Strategy",
    "SyntheticConfig",
    "TopOfBookExecution",
    "TradeEvent",
    "TradeThroughOnly",
    "VolumeQueue",
    "__version__",
    "default_contracts",
    "generate_events",
    "generate_mbp1",
    "load_contracts",
    "load_mbp1",
]
