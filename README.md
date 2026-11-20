# futures-tick-backtester

An **honest**, event-driven, tick-level backtesting engine for CME equity-index
futures (NQ, ES, MNQ, MES).

Most backtests look better than live trading because they quietly assume
things that never happen: fills at the touch, zero latency, free trading, bar
data that hides the order of highs and lows. This project makes each of those
assumptions explicit, configurable, and conservative by default.

> Status: **MVP / alpha (v0.1)**. The core loop, the top-of-book execution
> model, accounting and the Databento MBP-1 loader are tested. APIs may change.

## What "honest" means here

| Concern | What the engine does |
|---|---|
| Look-ahead | One clock. The data iterator is pulled **one event at a time** and never peeked. The strategy only sees a `Context` built from events already delivered. A *prefix-invariance* test checks that changing the future never changes past decisions. |
| Same-timestamp ties | An order arriving at time `T` is processed **after** every market event stamped `T`, so it can never trade on the snapshot it reacted to if the book moved in the same instant. |
| Latency | Separate order, cancel and report latencies (ms). Orders match the book **at arrival**; the strategy learns about fills **later**, so its position/order view lags the exchange, as in reality. Cancels can lose the race against fills. |
| Market orders | Fill at the opposite best price; size beyond the displayed quantity walks a virtual ladder (one tick per level). Liquidity you took is not available again until the next quote update. |
| Limit orders | No fill on touch by default: fills require a trade-through, a quote cross, or traded volume at your price exceeding the estimated queue ahead of you (configurable). |
| Stops | Trigger on **trades**, then execute as market orders from the worse of the opposite quote and the triggering print (no "filled exactly at my stop" on gaps). Stops already through the market are rejected by default. |
| Costs | Exchange + broker + regulatory fees per contract per side, from YAML. |
| Valuation | Open positions are marked at **liquidation** prices by default (long at bid, short at ask). Drawdown is tracked on **every event**, not only on sampled equity points. |
| Data hygiene | Out-of-order data raises. Files mixing several instruments (e.g. during a roll) are refused unless you select one. No market data is ever committed to the repo. |

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/your-org/futures-tick-backtester.git
```

```bash
cd futures-tick-backtester && python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

`matplotlib` is only needed for charts (`pip install -e ".[plot]"`).

## Quick start

Run the example: an opening range breakout on synthetic NQ ticks. The synthetic
data is written as Databento MBP-1 Parquet and loaded back through the real
loader, so the whole pipeline is exercised.

```bash
python examples/run_orb_synthetic.py --days 10 --out examples/output
```

It prints trade statistics and writes `equity.png`, `trades.csv`, `fills.csv`,
`sessions.csv` and `stats.json` to `examples/output/`. Synthetic prices are a
driftless random walk, so the expected result is a small loss of roughly the
trading costs: an honest engine should not manufacture an edge out of noise.

### Writing a strategy

```python
from ftb import Backtester, ConstantLatency, Side, Strategy, load_mbp1


class FadeLargePrints(Strategy):
    """Toy example: fade any print of 50+ contracts, take profit 4 ticks away."""

    def on_trade(self, ctx, trade):
        if (
            trade.size >= 50
            and ctx.position(trade.symbol) == 0
            and not ctx.open_orders(trade.symbol)
        ):
            side = Side.SELL if trade.aggressor is Side.BUY else Side.BUY
            ctx.submit_market(trade.symbol, side, 1, tag="entry")

    def on_fill(self, ctx, fill):
        if fill.tag == "entry":
            tick = ctx.contract(fill.symbol).tick_size
            target = fill.price + 4 * tick * fill.side.sign
            ctx.submit_limit(fill.symbol, fill.side.opposite, fill.qty, target, tag="exit")


events = load_mbp1("data/NQZ4_mbp1.parquet", symbol="NQ")
result = Backtester(
    FadeLargePrints(),
    events,
    latency=ConstantLatency(order_ms=5, report_ms=5),
    initial_capital=50_000,
).run()

print(result.summary())
result.trades_frame().to_csv("trades.csv")
```

Strategy callbacks: `on_start`, `on_quote`, `on_trade`, `on_fill`,
`on_order_update`, `on_halt`, `on_end`. The `Context` offers `now`,
`last_quote`, `last_trade`, `position`, `open_orders`, `contract`,
`submit_market`/`submit_limit`/`submit_stop` (plus `buy`/`sell`), `cancel`,
`cancel_all` and `flatten`. Prices must be on the tick grid
(`ctx.contract(sym).round_price(p)` helps); invalid input raises immediately.

## Data

### Databento MBP-1

Export MBP-1 (top of book + trades) from Databento as Parquet or CSV, for
example with `DBNStore.to_parquet(...)`, `DBNStore.to_df().to_parquet(...)`, or a
CSV batch download. The loader accepts:

* float ("pretty") prices or raw fixed-point int64 prices (1e-9 units,
  `INT64_MAX` = undefined);
* integer-nanosecond, datetime or ISO-8601 timestamps (`ts_recv` may be the
  index, as in `to_df()`).

```python
from ftb import load_mbp1

events = load_mbp1(
    "NQ_2024-03.parquet",
    symbol="NQ",  # name used for the events; must resolve to a contract spec
    raw_symbol="NQH4",  # or instrument_id=...; required if the file holds several
    ts_field="ts_recv",  # default; "ts_event" is also accepted
    start="2024-03-01T14:30Z",
    end="2024-03-01T21:00Z",
)
```

Row mapping: `action == 'T'` becomes a `TradeEvent` (`side` is the aggressor);
`A`/`C`/`M` become `QuoteEvent`s from the `*_00` book fields (consecutive
duplicates dropped); `R` (book clear) becomes an empty quote; `F` is ignored.
Several instruments can be combined with `ftb.data.merge_streams`.

### Synthetic data

```python
from ftb import SyntheticConfig, generate_events, generate_mbp1

frame = generate_mbp1(SyntheticConfig(symbol="NQ", days=5, seed=1))  # Databento columns
events = generate_events(days=1, events_per_second=2, seed=1)  # list of events
```

Deterministic per seed. A toy model (Poisson arrivals, volatility regimes,
trades that deplete displayed size), meant for tests and examples only.

## Contract specs

Bundled defaults live in `src/ftb/default_contracts.yaml`. Fees are **per
contract per side** and are illustrative: put in your own numbers.

```yaml
contracts:
  NQ:
    description: E-mini Nasdaq-100
    tick_size: 0.25
    point_value: 20.0
    fees: {exchange: 1.38, broker: 0.50, regulatory: 0.02}
```

```python
from ftb import load_contracts

contracts = load_contracts("my_contracts.yaml")  # overrides/extends the defaults
Backtester(strategy, events, contracts=contracts)
```

Symbols resolve from roots (`NQ`), outrights (`NQZ4`, `MNQH25`) and Databento
parent/continuous symbols (`NQ.FUT`, `NQ.c.0`). Unknown symbols raise, as do
unknown YAML keys (a typo never silently becomes a zero fee).

## Execution model v1 (top of book)

All price comparisons are done in integer ticks.

| Order | Rule |
|---|---|
| Market | Fill at the opposite best. Beyond the displayed size, each further level is one tick worse and holds `depth_per_level` contracts (default: the displayed size). Optional `extra_slippage_ticks`. Rejected if the opposite side is empty. |
| Limit, marketable on arrival | Take liquidity up to the limit (with price improvement); the remainder rests. |
| Limit, resting | Fills at the limit price (maker) on: a print through the price (up to the print's size); a print at the price from the opposite aggressor once the queue ahead is exhausted; or the opposite quote crossing the price (up to its displayed size). |
| Stop (stop-market) | Triggers on a trade at or through the stop; executes as a market order starting from the worse of the opposite best and the triggering print. Already through the last trade on arrival: rejected (`stop_through_market="reject"`) or executed (`"trigger"`). |

Queue models for prints at your price (`ExecutionConfig(queue_model=...)`):

* `VolumeQueue(ahead_fraction=1.0)` (default): when joining the best price,
  `ceil(displayed * ahead_fraction)` contracts are assumed ahead of you. Prints at
  your price reduce it; the estimate is clamped down when the displayed size
  shrinks; joining inside the spread means nobody is ahead.
* `TradeThroughOnly()`: never fills at the touch (most conservative).
* `FillOnTouch()`: fills on any print at your price (optimistic; provided to
  measure how much a strategy depends on that assumption).

```python
from ftb import ConstantLatency, ExecutionConfig, TopOfBookExecution, VolumeQueue

execution = TopOfBookExecution(
    ExecutionConfig(
        queue_model=VolumeQueue(ahead_fraction=1.0),
        depth_per_level=None,
        extra_slippage_ticks=0,
        fill_on_quote_cross=True,
        stop_through_market="reject",
    )
)
latency = ConstantLatency(order_ms=5, report_ms=5, cancel_ms=None)  # cancel defaults to order_ms
```

## Accounting and results

`Backtester.run()` returns a `BacktestResult` with fills, round-trip trades
(entry/exit averages, gross/net PnL, commissions, MAE/MFE, PnL in ticks),
the equity curve (sampled every `equity_interval_s`, plus at every fill), the
exact tick-level maximum drawdown, final positions, and every order.

* `result.summary()`: win rate, profit factor, average win/loss, drawdown,
  commissions, maker share, daily PnL statistics.
* `result.trades_frame()`, `equity_frame()`, `fills_frame()`: pandas DataFrames
  with exchange-local timestamps.
* `ftb.analytics.plot_equity(result, "equity.png")`: equity + drawdown chart.

Realized PnL uses average cost. A reversal (long 1, sell 3) closes one round
trip and opens another, splitting the fill's commission between them.

## Risk hooks

```python
from ftb import MaxContracts, RiskRule


class DailyStop(RiskRule):
    name = "daily_stop"

    def on_update(self, ts, portfolio):
        if portfolio.net_pnl < -1_000:
            return "lost $1,000"
        return None


Backtester(strategy, events, risk_rules=[MaxContracts(2), DailyStop()])
```

* `check_order(order, ctx)`: pre-trade check with the strategy's knowledge;
  returning a string rejects the order.
* `on_update(ts, portfolio)`: runs after every market event and every fill with
  the ground-truth ledger (like a broker or prop-firm risk system). Returning a
  string halts trading: working orders are cancelled, positions are flattened,
  and only reducing orders are accepted afterwards.

## Architecture

```
 Parquet / CSV ─► load_mbp1 ─► MarketEvent stream (sorted; pulled one event at a time)
                                     │
 ┌──────────────────────── Backtester: one clock (epoch ns) ─────────────────────────┐
 │                                   ▼                                               │
 │   ExecutionModel (simulated exchange) ──ExecReport──► Portfolio (ground truth:    │
 │   TopOfBookExecution + QueueModel                     PnL, fees, equity, trades)  │
 │          ▲                                  │                   │                 │
 │          │ order / cancel                   │ report            ▼                 │
 │          │ after latency                    │ after latency   RiskRule.on_update  │
 │          │                                  ▼                                     │
 │   heap of scheduled actions (ties at the same timestamp go to market data first)  │
 │          ▲                                  │                                     │
 │          │ submit / cancel                  │ callbacks                           │
 │          │ (RiskRule.check_order)           ▼                                     │
 │          └───────────── Strategy ◄──── Context (past & present only)              │
 └───────────────────────────────────────────────────────────────────────────────────┘
```

| Module | Responsibility |
|---|---|
| `ftb.contracts` | `ContractSpec` (tick size, point value, fees, tick conversion), `ContractRegistry`, YAML loading |
| `ftb.events` | `TradeEvent`, `QuoteEvent`, `Side`, time helpers (epoch ns) |
| `ftb.orders` | `OrderRequest` (to the exchange), `Order` (strategy view), `ExecReport`, `Fill` |
| `ftb.data` | Databento MBP-1 loader, synthetic generator, stream merging |
| `ftb.execution` | `ExecutionModel` interface, `TopOfBookExecution`, queue models, latency models |
| `ftb.engine` | `Backtester`: the clock, scheduling, order lifecycle, risk enforcement |
| `ftb.strategy` | `Strategy` base class and `Context` |
| `ftb.accounting` | `Portfolio`, `Position`, `RoundTrip`, `EquityPoint` |
| `ftb.risk` | `RiskRule` hooks, `MaxContracts` |
| `ftb.analytics` | statistics, DataFrames, equity chart |
| `ftb.strategies` | example strategies (`OpeningRangeBreakout`) |

### Extension points (roadmap)

* **Full order book (MBO) and exact queue position**: implement
  `ExecutionModel` with a depth book fed by MBO events (a new `MarketEvent`
  subclass) and a queue tracker; strategies and accounting do not change.
* **Prop-firm rules**: trailing drawdown, daily loss limit, max contracts and
  consistency rules as `RiskRule`s. `on_update` already receives the tick-level
  ledger, and a halt already cancels, flattens and blocks new risk.
* **Candle vs tick comparison report**: run the same strategy on bars built
  from the same trades and diff the trade lists and equity curves.
* Also planned: session/holiday calendars, contract roll handling, stop-limit
  orders, randomized latency, and a faster core for multi-year runs.

## Known approximations

What the v1 model does **not** know:

* MBP-1 shows only the best level; depth beyond it is assumed, and queue position
  is estimated (cancellations are assumed to come from behind you unless the
  displayed size proves otherwise).
* Your orders never move the replayed market (no market impact beyond the
  consumed-liquidity bookkeeping).
* Market data reaches the strategy at `ts_recv`; extra feed delay should be
  folded into `order_ms`.
* Stops are modelled as exchange-held stop-market orders; CME protections
  (price bands, velocity logic, stop-with-protection) are not modelled.
* Orders still in flight when the data ends are rejected, and open positions are
  reported marked to market, never closed at a price that was not available.

## Development

```bash
pytest
```

```bash
ruff check . && ruff format --check .
```

The test suite covers fills (market, limit, stop), latency and cancel races,
PnL and commissions, the data loaders, and the no-look-ahead guarantee
(prefix invariance, no prefetching, fill timing).

Troubleshooting: on macOS with Python 3.12+, `import ftb` can fail after
`pip install -e .` if the virtualenv's `.pth` files carry the "hidden" file flag
(Python skips hidden `.pth` files). Run `chflags -R nohidden .venv`, or put
`src` on `PYTHONPATH`. `pytest` is unaffected (it adds `src` itself).

## License

MIT - see [LICENSE](LICENSE).
