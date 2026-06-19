# CLAUDE.md

Working agreement for AI assistants (and humans) in this repo. "Working principles" is
general engineering discipline (adapted from [Andrej Karpathy's CLAUDE.md](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md));
"Project: rsu-rebalancing" is project-specific orientation.

## Working principles

### Think before coding

- Surface assumptions, ambiguities, and trade-offs explicitly before writing code.
- When a request has more than one reasonable interpretation, name them and ask, rather
  than silently picking one.
- Push back on oversimplified or hand-wavy solutions. Question requests that smell wrong.
- When uncertain, stop and clarify instead of guessing.

### Simplicity first

- Minimum code that solves the problem. Nothing speculative.
- No unrequested features, premature abstractions, or just-in-case error handling.
- Self-check: would an experienced engineer find this overcomplicated? Bias to concise.

### Surgical changes

- Preserve adjacent code and formatting. Only touch what the task requires.
- Remove an import or variable only when your edit made it unnecessary.
- Match the existing style and naming of the file you are editing.
- Flag unrelated dead code rather than deleting it as a drive-by.

### Goal-driven execution

- Turn vague goals into concrete, testable checkpoints.
- Outline a multi-step plan with success criteria before large changes.
- "Done" means: tests pass, `ruff` is clean, and the behavior is verified by running it.

---

## Project: rsu-rebalancing

Backtests a one-way threshold rebalancing strategy for concentrated RSU compensation.
See [README.md](README.md) for the strategy and quickstart.

### Layout

- `rsu_rebalancing/` — the library
  - `config.py` — frozen dataclasses (`GrantConfig`, `StrategyConfig`, `BacktestConfig`)
  - `data.py` — yfinance access, in-memory memoized
  - `calendar.py` — pure functions mapping rebalances onto trading days
  - `vesting.py` — expands the award schedule into per-day vesting share counts
  - `portfolio.py` — holdings, tax lots (sold lowest-tax-first), the sell-to-fraction primitive
  - `strategy.py` — the threshold rule + `HoldEverything` / `SellAllAtVest` baselines
  - `backtest.py` — the day-by-day engine; `run_backtest` is the top-level entry point
  - `metrics.py` — time-weighted returns and risk stats
- `notebooks/` — marimo notebooks (`rsu_backtest.py` is the app; `exploration.py` is a scratchpad)
- `tests/` — pytest, network-free (synthetic prices / trading days)
- `assets/` — `generate_assets.py` renders the README's example charts/tables from the
  notebook defaults (run by hand; hits the network)

### Commands

```bash
uv sync --extra dev                 # set up the environment
uv run pytest -q                    # tests (no network)
uv run ruff check . && uv run ruff format --check .
uv run mypy                         # lenient; see below
uv run marimo edit notebooks/rsu_backtest.py
```

### Conventions & gotchas

- **uv + ruff** are the toolchain. Ruff (lint + format) is the quality gate.
- **mypy is lenient and not a gate.** Annotations are for readability and editor help;
  keep it green when cheap, but don't contort code to satisfy it.
- **Returns are time-weighted.** Grants are deposits, not performance. Any new metric
  must strip contributions (see `metrics.time_weighted_returns`) or it will be wrong.
- **README example blocks are generated.** The charts and the `<!-- BEGIN summary:* -->`
  tables come from `assets/generate_assets.py`; don't hand-edit them. Regenerate
  (`uv run python assets/generate_assets.py`) whenever a change moves the numbers or the
  figure — the notebook defaults, the `SCENARIOS`, or engine/metrics/plot output. Nothing
  hits the network in tests or CI, so this won't fail loudly when stale. Commit the
  refreshed README + PNGs **in the same commit** as the change that moved them, so no
  commit leaves the README contradicting the engine; a standalone refresh (e.g. newer
  price data, no code change) is its own `docs:` commit.
- **Threshold targeting is pre-tax.** `sell_employer_to_fraction` sizes the trade on
  market value, then pays tax out of proceeds. With taxes on, the post-trade employer
  fraction sits slightly above the target by design — keep this intentional.
- **Data is fetched live** and memoized in-memory for the session only.

### Testing

- **Tests must not hit the network.** Use synthetic `bdate_range` days and hand-built
  price frames; call `run_rule` (not `run_backtest`) so no fetch happens.
- **Arrange-Act-Assert, blank line between phases.** Act assigns the result to a variable
  (`got = fn(...)`); assert on it separately — don't fuse the call into the assert. The
  act+assert fusion inside a `with pytest.raises(...)` block is the one fine exception.
- **Test our seams, not the library's.** For thin wrappers (e.g. the `quantstats` calls
  in `metrics.py`), assert only that we wire the library up right and honor our own
  overrides/guards — not that the library's output matches a formula typed by hand. Cover
  our own logic (time-weighting that strips contributions), our explicit overrides (sharpe
  zero-vol → NaN), and our guards (empty series → NaN). Delete working-but-stale tests
  that only pin a dependency's internals.

### Commit messages

[Conventional Commits](https://www.conventionalcommits.org): `type(scope): description`.
Common types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build` (deps /
`pyproject.toml`), `ci`, `style`, `chore`.

Single-user repo: commit straight to `main`, don't branch first. Still only commit or
push when asked.

- **Scope names the module the change is confined to** (`fix(portfolio)`,
  `feat(strategy)`, `refactor(notebooks)`); omit it for cross-cutting changes.
- **The sim notebook (`notebooks/rsu_backtest.py`) is a deliverable.** Pick its type by
  intent like source: new control/panel/analysis → `feat`, wrong calc / broken cell →
  `fix`, same-output rework (e.g. swapping plot libs) → `refactor`, cosmetic styling →
  `style`. Scope it `(notebooks)` so it reads as the demo surface, not the engine.
- **The scratchpad (`notebooks/exploration.py`) is throwaway** — its changes stay `chore`.
