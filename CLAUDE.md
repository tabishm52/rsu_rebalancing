# CLAUDE.md

Working agreement for AI assistants (and humans) in this repo. The first four sections
are general engineering discipline; the last is project-specific orientation.

## 1. Think before coding

- Surface assumptions, ambiguities, and trade-offs explicitly before writing code.
- When a request has more than one reasonable interpretation, name them and ask, rather
  than silently picking one.
- Push back on oversimplified or hand-wavy solutions. Question requests that smell wrong.
- When uncertain, stop and clarify instead of guessing.

## 2. Simplicity first

- Minimum code that solves the problem. Nothing speculative.
- No unrequested features, premature abstractions, or just-in-case error handling.
- Self-check: would an experienced engineer find this overcomplicated? Bias to concise.

## 3. Surgical changes

- Preserve adjacent code and formatting. Only touch what the task requires.
- Remove an import or variable only when your edit made it unnecessary.
- Match the existing style and naming of the file you are editing.
- Flag unrelated dead code rather than deleting it as a drive-by.

## 4. Goal-driven execution

- Turn vague goals into concrete, testable checkpoints.
- Outline a multi-step plan with success criteria before large changes.
- "Done" means: tests pass, `ruff` is clean, and the behavior is verified by running it.

---

## Project: rsu-rebalancing

Backtests a one-way threshold rebalancing strategy for concentrated RSU compensation.
See [README.md](README.md) for the strategy and quickstart.

### Layout

Flat package (no `src/`):

- `rsu_rebalancing/` — the library
  - `config.py` — frozen dataclasses (`GrantSchedule`, `StrategyConfig`, `SimConfig`)
  - `data.py` — yfinance access, in-memory memoized, optional disk cache via env var
  - `calendar.py` — pure functions mapping grants/rebalances onto trading days
  - `portfolio.py` — holdings, FIFO tax lots, the sell-to-fraction primitive
  - `strategy.py` — the threshold rule + `HoldEverything` / `SellAllAtVest` baselines
  - `simulate.py` — the day-by-day engine; `run_backtest` is the top-level entry point
  - `metrics.py` — time-weighted returns and risk stats
- `notebooks/` — marimo notebooks (`rebalance_sim.py` is the app; `exploration.py` is a scratchpad)
- `tests/` — pytest, network-free (synthetic prices / trading days)

### Commands

```bash
uv sync --extra dev                 # set up the environment
uv run pytest -q                    # tests (no network)
uv run ruff check . && uv run ruff format --check .
uv run mypy                         # lenient; see below
uv run marimo edit notebooks/rebalance_sim.py
```

### Conventions & gotchas

- **uv + ruff** are the toolchain. Ruff (lint + format) is the quality gate.
- **mypy is lenient and not a gate.** Type checkers catch little inside pandas
  pipelines; annotations are for readability and editor help. Keep it green when cheap,
  but don't contort code to satisfy it.
- **Returns are time-weighted.** Grants are deposits, not performance. Any new metric
  must strip contributions (see `metrics.time_weighted_returns`) or it will be wrong.
- **Threshold targeting is pre-tax.** `sell_employer_to_fraction` sizes the trade on
  market value, then pays tax out of proceeds. With taxes on, the post-trade employer
  fraction sits slightly above the target by design — keep this intentional.
- **Tests must not hit the network.** Use synthetic `bdate_range` days and hand-built
  price frames; call `run_rule` (not `run_backtest`) so no fetch happens.
- **Data is fetched live by default.** Set `RSU_REBALANCING_CACHE_DIR` to cache to disk;
  the cache is never auto-refreshed — delete files to refetch.

### Commit messages

[Conventional Commits](https://www.conventionalcommits.org): `type(scope): description`.
Common types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build` (deps /
`pyproject.toml`), `ci`, `style`, `chore`.

- **Scope names the module the change is confined to** (`fix(portfolio)`,
  `feat(strategy)`, `refactor(notebooks)`); omit it for cross-cutting changes. Scope by
  whatever the module is called at commit time — a scope that later moves in the planned
  reorg is fine, since the message is a snapshot of intent, not a live pointer.
- **The sim notebook (`notebooks/rebalance_sim.py`) is a deliverable.** Pick its type by
  intent like source: new control/panel/analysis → `feat`, wrong calc / broken cell →
  `fix`, same-output rework (e.g. swapping plot libs) → `refactor`, cosmetic styling →
  `style`. Scope it `(notebooks)` so it reads as the demo surface, not the engine.
- **The scratchpad (`notebooks/exploration.py`) is throwaway** — its changes stay `chore`.
