"""Contract specifications (tick size, point value, fees) and a symbol registry.

Prices are floats in the public API, but every comparison the simulator makes is
done in *integer ticks* (``ContractSpec.to_ticks``) so that float rounding can
never decide whether an order fills.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

_MONTH_CODES = "FGHJKMNQUVXZ"
# Outright futures symbol: ROOT + month code + 1-2 digit year, e.g. NQZ4, MESH25.
_OUTRIGHT_RE = re.compile(rf"^(?P<root>[A-Z0-9]+?)(?P<month>[{_MONTH_CODES}])(?P<year>\d{{1,2}})$")
_FEE_KEYS = ("exchange", "broker", "regulatory")


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Per-contract, per-side fees in USD."""

    exchange: float = 0.0
    broker: float = 0.0
    regulatory: float = 0.0

    def __post_init__(self) -> None:
        for name in _FEE_KEYS:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"fee '{name}' must be a finite number >= 0, got {value!r}")

    @property
    def per_side(self) -> float:
        """Total cost of one contract, one side."""
        return self.exchange + self.broker + self.regulatory

    @property
    def round_turn(self) -> float:
        return 2.0 * self.per_side


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Static description of a futures contract."""

    symbol: str
    tick_size: float
    point_value: float
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    description: str = ""
    exchange: str = "CME"
    currency: str = "USD"
    _decimals: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("contract symbol must be non-empty")
        if not (math.isfinite(self.tick_size) and self.tick_size > 0):
            raise ValueError(f"{self.symbol}: tick_size must be > 0, got {self.tick_size!r}")
        if not (math.isfinite(self.point_value) and self.point_value > 0):
            raise ValueError(f"{self.symbol}: point_value must be > 0, got {self.point_value!r}")
        exponent = Decimal(str(self.tick_size)).normalize().as_tuple().exponent
        decimals = max(0, -exponent) if isinstance(exponent, int) else 0
        object.__setattr__(self, "_decimals", decimals)

    # ------------------------------------------------------------------ prices
    @property
    def tick_value(self) -> float:
        """Dollar value of a one-tick move for one contract."""
        return self.tick_size * self.point_value

    def to_ticks(self, price: float) -> int:
        """Convert a price to integer ticks. Raises if it is not on the tick grid."""
        raw = price / self.tick_size
        ticks = round(raw)
        if abs(raw - ticks) > 1e-6:
            raise ValueError(
                f"{self.symbol}: price {price!r} is not a multiple of tick size {self.tick_size}"
            )
        return int(ticks)

    def nearest_ticks(self, price: float) -> int:
        """Convert a price to the nearest integer tick (never raises; for market data)."""
        return round(price / self.tick_size)

    def from_ticks(self, ticks: int) -> float:
        return round(ticks * self.tick_size, self._decimals)

    def round_price(self, price: float) -> float:
        """Snap a price to the nearest valid tick."""
        return self.from_ticks(self.nearest_ticks(price))

    def is_on_grid(self, price: float) -> bool:
        raw = price / self.tick_size
        return abs(raw - round(raw)) <= 1e-6

    # ------------------------------------------------------------------ money
    def commission(self, qty: int) -> float:
        """All-in fees for ``qty`` contracts on one side."""
        return self.fees.per_side * abs(qty)

    def pnl(self, entry_price: float, exit_price: float, signed_qty: int) -> float:
        """Gross dollar PnL of ``signed_qty`` contracts (positive = long)."""
        return (exit_price - entry_price) * signed_qty * self.point_value

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, symbol: str, data: Mapping[str, Any]) -> ContractSpec:
        known = {"tick_size", "point_value", "fees", "description", "exchange", "currency"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{symbol}: unknown contract field(s) {sorted(unknown)}")
        for required in ("tick_size", "point_value"):
            if required not in data:
                raise ValueError(f"{symbol}: missing required field '{required}'")
        fees_raw = data.get("fees") or {}
        unknown_fees = set(fees_raw) - set(_FEE_KEYS)
        if unknown_fees:
            raise ValueError(
                f"{symbol}: unknown fee key(s) {sorted(unknown_fees)}; allowed: {list(_FEE_KEYS)}"
            )
        fees = FeeSchedule(**{k: float(v) for k, v in fees_raw.items()})
        return cls(
            symbol=symbol,
            tick_size=float(data["tick_size"]),
            point_value=float(data["point_value"]),
            fees=fees,
            description=str(data.get("description", "")),
            exchange=str(data.get("exchange", "CME")),
            currency=str(data.get("currency", "USD")),
        )


class ContractRegistry(Mapping[str, ContractSpec]):
    """Maps symbols to :class:`ContractSpec`.

    Lookup accepts the root (``NQ``), an outright (``NQZ4``, ``MNQH25``) or a
    Databento continuous/parent symbol (``NQ.c.0``, ``NQ.FUT``).
    """

    def __init__(self, specs: Iterable[ContractSpec] | Mapping[str, ContractSpec] = ()) -> None:
        items = specs.values() if isinstance(specs, Mapping) else specs
        self._specs: dict[str, ContractSpec] = {s.symbol: s for s in items}
        self._cache: dict[str, ContractSpec] = {}

    # Mapping interface ---------------------------------------------------
    def __getitem__(self, symbol: str) -> ContractSpec:
        return self.resolve(symbol)

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, symbol: object) -> bool:
        if not isinstance(symbol, str):
            return False
        try:
            self.resolve(symbol)
        except KeyError:
            return False
        return True

    def __repr__(self) -> str:
        return f"ContractRegistry({sorted(self._specs)})"

    # Resolution ----------------------------------------------------------
    def root_of(self, symbol: str) -> str:
        return self.resolve(symbol).symbol

    def resolve(self, symbol: str) -> ContractSpec:
        cached = self._cache.get(symbol)
        if cached is not None:
            return cached
        spec = self._resolve_uncached(symbol)
        self._cache[symbol] = spec
        return spec

    def _resolve_uncached(self, symbol: str) -> ContractSpec:
        if symbol in self._specs:
            return self._specs[symbol]
        base = symbol.split(".", 1)[0].upper()  # NQ.c.0 -> NQ, NQ.FUT -> NQ
        if base in self._specs:
            return self._specs[base]
        match = _OUTRIGHT_RE.match(base)
        if match and match.group("root") in self._specs:
            return self._specs[match.group("root")]
        raise KeyError(
            f"no contract spec for symbol {symbol!r}; known roots: {sorted(self._specs)}"
        )

    # Construction ---------------------------------------------------------
    def merged(self, other: ContractRegistry) -> ContractRegistry:
        """Return a new registry where ``other`` overrides/extends ``self``."""
        return ContractRegistry({**self._specs, **other._specs})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContractRegistry:
        if "contracts" not in data or not isinstance(data["contracts"], Mapping):
            raise ValueError("contract config must have a top-level 'contracts' mapping")
        return cls(ContractSpec.from_dict(sym, spec) for sym, spec in data["contracts"].items())

    @classmethod
    def from_yaml(cls, path: str | Path) -> ContractRegistry:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})


def default_contracts() -> ContractRegistry:
    """The bundled specs for ES, NQ, MES and MNQ."""
    text = resources.files("ftb").joinpath("default_contracts.yaml").read_text(encoding="utf-8")
    return ContractRegistry.from_dict(yaml.safe_load(text))


def load_contracts(
    path: str | Path | None = None, *, include_defaults: bool = True
) -> ContractRegistry:
    """Load contract specs.

    With no ``path`` the bundled defaults are returned. With a ``path`` the YAML
    file overrides/extends the defaults (or replaces them if
    ``include_defaults=False``).
    """
    base = default_contracts() if include_defaults or path is None else ContractRegistry()
    if path is None:
        return base
    return base.merged(ContractRegistry.from_yaml(path))
