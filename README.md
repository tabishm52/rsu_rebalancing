# rsu-rebalancing

Backtest a one-way **threshold rebalancing** strategy for concentrated employer-stock
(RSU) compensation — and see how it would have stacked up against simply holding
everything or diversifying everything immediately.

Built as a small, typed Python package plus an interactive [marimo](https://marimo.io)
notebook. It's also a learning project for financial analysis with pandas, so the code
favors clarity over cleverness.

> **Not investment advice.** This is a historical simulation for curiosity and learning.
> Past performance says nothing about the future, and the model omits real-world details
> (wash sales, tax brackets, dividends as cash, transaction costs, slippage).

## The strategy

Each year you're granted an **award** of **employer stock**: its dollar value is fixed
into a share count at the award-date price, and those shares then vest in equal annual
tranches over the next few years (default four). Because the count is locked at award, the
*dollars* delivered at each vest float with the share price — a stock that ran up since the
award over-delivers, concentrating you further just as the rest of your employer holdings
peak. Awards can be backfilled before the window so it opens with a mature, overlapping
vesting stream. Twice a quarter — shortly after the trading window opens and just before the next
blackout period — you check what fraction of your *total* stock holdings is in employer
stock. If it exceeds a **threshold** (e.g., **1/3**), you sell the excess employer stock
down to that threshold and buy a diversified **index** with the proceeds.

It's a *one-way* rebalance: you only ever sell employer stock, never buy it. The effect
is self-correcting concentration control — you sell **more** when your employer
outperforms the market and **less** when it lags, and a fresh grant usually triggers a
sale at the next rebalance.

The backtest compares that strategy against two baselines:

| Strategy | What it does |
| --- | --- |
| **Threshold N%** | The real strategy: trim employer stock to N% on each rebalance day. |
| **Hold everything** | Never sell — maximum concentration. |
| **Sell all at vest** | Convert every grant straight to the index — full diversification. |

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Quickstart

### Interactive notebook

```bash
uv run marimo edit notebooks/rsu_backtest.py
```

Set the employer/index tickers, date range, annual award, vesting years, threshold, trade-day timing,
and tax rate; the backtest re-runs reactively with charts (concentration over time,
growth of $1) and a return/risk comparison table.

### From Python

```python
from rsu_rebalancing import GrantConfig, StrategyConfig, BacktestConfig, run_backtest, comparison_table

strategy = StrategyConfig(employer_ticker="AAPL", index_ticker="VTI", threshold=1/3)
schedule = GrantConfig(annual_dollars=100_000, start_year=2015, end_year=2023)
backtest = BacktestConfig(start="2015-01-01", end="2024-12-31", risk_free_rate=0.02)

results = run_backtest(strategy, schedule, backtest)
print(comparison_table(results, risk_free_rate=backtest.risk_free_rate))
```

Illustrative output (AAPL vs VTI, $100k/yr, 2015–2024 — *not a recommendation*):

| Metric | Threshold 33% | Hold everything | Sell all at vest |
| --- | ---: | ---: | ---: |
| Final value | $2,741,009 | $4,843,469 | $2,019,094 |
| Total vested contributions | $900,000 | $900,000 | $900,000 |
| Ann. return (TWR) | 16.4% | 24.3% | 12.2% |
| Ann. volatility | 19.9% | 28.2% | 17.9% |
| Max drawdown | −33.4% | −38.5% | −35.0% |
| Sharpe | 0.72 | 0.79 | 0.57 |
| End employer % | 33.1% | 100.0% | 0.0% |

Reading it: holding all AAPL won outright here *because AAPL beat the market* — but with
the most volatility and the deepest drawdown. The 33% threshold captured most of the
upside of staying partly invested in the employer while cutting concentration risk
roughly in half, and beat full diversification on both return and risk-adjusted return.
Try a flat or declining employer to see the threshold strategy protect the downside.

## How returns are measured

Because grants add money over time, raw portfolio value is **not** a clean return series
— a jump on a grant day is a deposit, not performance. Risk and return here use a
**time-weighted return** that strips out each day's contribution
(`metrics.time_weighted_returns`). Final dollar value is still reported directly, and is
a fair head-to-head number because every strategy receives the identical grant stream.

## Parameters

`StrategyConfig`:

- `employer_ticker`, `index_ticker` — symbols (index defaults to `VTI`).
- `threshold` — target max employer fraction (e.g. `1/3`).
- `rebalances_per_quarter` — how many evenly spaced rebalances to place in each quarter
  (default `2`).
- `tax_config` — a `TaxConfig` of tax rates. Capital gains on realized employer-stock sales
  are taxed by holding period: `short_term_rate` (default `0.48`) for a lot sold within
  `long_term_days` (default `365`) of its vest, `long_term_rate` (default `0.28`) for one
  held longer. Separately, `ordinary_income_rate` (default `0.443`) taxes each vest as
  ordinary income via sell-to-cover (see `GrantConfig`). Each rate is a single effective
  figure (fold in any state/NIIT surcharge); the defaults model a California single filer
  (see the `TaxConfig` docstring for the breakdown). Cost basis is the vest-day price, so
  trimming soon after a grant realizes little gain. Each rebalance sells the lowest-tax lots
  first (minimizing realized tax for the shares sold); with cap-gains taxes off this is just
  FIFO.

`GrantConfig`: `annual_dollars` (the per-award **gross** value, priced into shares at the
award date; vest-time withholding via `ordinary_income_rate` grosses the kept count down),
`start_year`, `end_year` (the award/employment span — set `start_year` before the window to
backfill), and optional `grant_month`/`grant_day` (default first trading day on/after
March 1) and `vesting_years` (equal annual tranches per award, default `4`).

`BacktestConfig`: `start`, `end`, `risk_free_rate` (for the Sharpe ratio).

## Data & caching

Prices come from [yfinance](https://github.com/ranaroussi/yfinance) (adjusted close, so
splits and dividends are handled). Fetches are memoized in-memory for the session, so
moving a slider in the notebook doesn't re-hit the network.

## Development

```bash
uv run pytest -q                                   # tests (no network needed)
uv run ruff check . && uv run ruff format --check .  # lint + format
uv run mypy                                        # lenient type check
```

See [CLAUDE.md](CLAUDE.md) for the project layout and conventions.

## License

MIT
