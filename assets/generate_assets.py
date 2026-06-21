"""Render the README assets from the notebook's default backtest.

Runs the marimo notebook's default configuration headlessly, once per scenario in
``SCENARIOS`` (the default differing only by employer ticker), and for each one:

- writes ``assets/growth-<slug>.png`` — the growth-of-100 performance chart.
- rewrites the ``<!-- BEGIN summary:<slug> -->`` / ``<!-- END summary:<slug> -->``
  block in ``README.md`` with the return/risk comparison table as Markdown.

Everything else (grants, window, taxes) comes from the notebook's own
``build_backtest_controls`` / ``build_configs`` (the single source of truth), so
regenerating here cannot drift from what the notebook shows. Prices come from the
checked-in fixture, so this is deterministic and offline; run it by hand when the
defaults, scenarios, or engine change:

    uv run python assets/generate_assets.py

Pass ``--refresh`` to first re-fetch the fixture from yfinance (the one networked
path), refreshing the README against newer price data:

    uv run python assets/generate_assets.py --refresh
"""

import argparse
import dataclasses
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: must run before rsu_app pulls in pyplot

from rsu_app import (  # noqa: E402 - import after the Agg backend is selected
    build_backtest_controls,
    build_configs,
    build_performance_figure,
    format_returns_table,
)
from rsu_app.fixtures import patch_yf, refresh  # noqa: E402
from rsu_rebalancing import comparison_table, run_backtest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent
README = ROOT / "README.md"


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A README exhibit: the notebook defaults with one employer ticker swapped in.

    ``slug`` keys both the chart file (``growth-<slug>.png``) and the README marker
    block (``summary:<slug>``).
    """

    slug: str
    employer: str


# AAPL beat the market over the window; INTC lagged it. The two runs bracket the
# strategy's behaviour — see the README's "Illustrative example" section.
SCENARIOS = (
    Scenario(slug="aapl", employer="AAPL"),
    Scenario(slug="intc", employer="INTC"),
)


def to_markdown(table: pd.DataFrame) -> str:
    """Render a formatted (string-valued) table as a right-aligned Markdown table."""
    header = "| Metric | " + " | ".join(table.columns) + " |"
    rule = "| --- | " + " | ".join(["---:"] * len(table.columns)) + " |"
    rows = [f"| {metric} | " + " | ".join(values) + " |" for metric, values in table.iterrows()]
    return "\n".join([header, rule, *rows])


def inject_summary(slug: str, table_md: str) -> None:
    """Replace the README's ``summary:<slug>`` marker block with the rendered table."""
    block = re.compile(
        rf"(?P<begin><!-- BEGIN summary:{slug}[^\n]*-->).*?(?P<end><!-- END summary:{slug} -->)",
        re.DOTALL,
    )
    readme = README.read_text()

    new_readme, count = block.subn(rf"\g<begin>\n\n{table_md}\n\n\g<end>", readme)
    if count != 1:
        raise RuntimeError(f"expected one 'summary:{slug}' marker block in {README}, found {count}")
    README.write_text(new_readme)


def render(scenario: Scenario, elements) -> str:
    """Run one scenario, write its chart, inject its table, and return the basis label."""
    strategy_cfg, grant_cfg, backtest_cfg, basis = build_configs(elements)
    strategy_cfg = dataclasses.replace(strategy_cfg, employer_ticker=scenario.employer)
    results = run_backtest(strategy_cfg, grant_cfg, backtest_cfg)

    fig = build_performance_figure(results, strategy_cfg, backtest_cfg, basis)
    fig.savefig(ASSETS / f"growth-{scenario.slug}.png", dpi=150, bbox_inches="tight")

    table = comparison_table(
        results,
        risk_free_rate=backtest_cfg.risk_free_rate,
        after_tax=backtest_cfg.after_tax_performance,
    )
    inject_summary(scenario.slug, to_markdown(format_returns_table(table)))
    return basis


def main() -> None:
    """Render every scenario's chart and summary table into README.md."""
    parser = argparse.ArgumentParser(description="Regenerate the README assets.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch the price fixture from yfinance before rendering (hits the network)",
    )
    args = parser.parse_args()

    elements, _ = build_backtest_controls()

    if args.refresh:
        # The fixture only needs to cover what the scenarios exercise: every scenario employer
        # plus the index, from the first (backfilled) award date through the backtest end.
        strategy_cfg, grant_cfg, backtest_cfg, _ = build_configs(elements)
        tickers = list(dict.fromkeys([*(s.employer for s in SCENARIOS), strategy_cfg.index_ticker]))
        refresh(tickers, grant_cfg.earliest_grant_date(backtest_cfg.start), backtest_cfg.end)

    # Render against the checked-in fixture so the README is deterministic and offline.
    with patch_yf():
        for scenario in SCENARIOS:
            basis = render(scenario, elements)
            print(f"Rendered {scenario.employer}: growth-{scenario.slug}.png + table ({basis}).")


if __name__ == "__main__":
    main()
