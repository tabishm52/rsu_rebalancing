"""The backtest notebook's control panel and its assembly into engine configs."""

from typing import Any

import marimo as mo
import pandas as pd

from rsu_rebalancing import (
    BacktestConfig,
    GrantConfig,
    StrategyConfig,
    TaxConfig,
)


def build_backtest_controls() -> tuple[mo.ui.dictionary, mo.Html]:
    """Construct the control panel.

    Tuning and reporting defaults come from the config dataclasses; the notebook owns UI
    presentation (widget type, ranges, percent units) and seeds the required policy inputs
    (employer, grant size, dates, threshold).

    Returns:
        elements: All widgets in a single ``mo.ui.dictionary``
        layout: The arranged panel to display in a cell.
    """
    # Index the dictionary (``elements["<name>"]``) to pull a widget into the layout below.
    elements = mo.ui.dictionary(
        {
            "employer": mo.ui.text(value="AAPL", label="Employer ticker"),
            "index": mo.ui.text(value=StrategyConfig.index_ticker, label="Index ticker"),
            "start": mo.ui.text(value="2015-01-01", label="Start date"),
            "end": mo.ui.text(value="2024-12-31", label="End date"),
            "annual_dollars": mo.ui.number(
                value=100_000, start=0, stop=1_000_000, step=25_000, label="First-year grant $"
            ),
            "vesting_years": mo.ui.slider(
                start=1,
                stop=6,
                value=GrantConfig.vesting_years,
                step=1,
                label="Vesting years",
                show_value=True,
            ),
            "backfill": mo.ui.switch(
                value=GrantConfig.backfill,
                label="Backfill grants before window (mature employee, not new hire)",
            ),
            "grant_growth": mo.ui.slider(
                start=0,
                stop=10,
                value=round(GrantConfig.grant_growth_rate * 100),
                step=1,
                label="Grant growth %/yr",
                show_value=True,
            ),
            "threshold": mo.ui.slider(
                start=5,
                stop=100,
                value=33,
                step=1,
                label="Rebalance threshold %",
                show_value=True,
            ),
            "rebalances": mo.ui.slider(
                start=1,
                stop=3,
                value=StrategyConfig.rebalances_per_quarter,
                step=1,
                label="Rebalances per quarter",
                show_value=True,
            ),
            "rebalance_band": mo.ui.slider(
                start=0,
                stop=10,
                value=round(StrategyConfig.rebalance_band * 100),
                step=1,
                label="Hysteresis band %",
                show_value=True,
            ),
            "short_term_tax": mo.ui.slider(
                start=0,
                stop=60,
                value=round(TaxConfig.short_term_rate * 100),
                step=1,
                label="Short-term cap-gains tax %",
                show_value=True,
            ),
            "long_term_tax": mo.ui.slider(
                start=0,
                stop=40,
                value=round(TaxConfig.long_term_rate * 100),
                step=1,
                label="Long-term cap-gains tax %",
                show_value=True,
            ),
            "vest_withholding": mo.ui.slider(
                start=0,
                stop=60,
                value=round(TaxConfig.ordinary_income_rate * 100),
                step=1,
                label="Vest withholding %",
                show_value=True,
            ),
            "risk_free": mo.ui.slider(
                start=0,
                stop=8,
                value=round(BacktestConfig.risk_free_rate * 100),
                step=1,
                label="Risk-free % (for Sharpe)",
                show_value=True,
            ),
            "after_tax_perf": mo.ui.switch(
                value=BacktestConfig.after_tax_performance, label="Analyze performance after tax"
            ),
        }
    )

    # The everyday knobs sit up top; the fussy details (exact tax rates, risk-free) tuck
    # into a collapsed accordion so they're available without crowding the common path.
    general = mo.vstack(
        [
            mo.hstack([elements["employer"], elements["index"]], justify="start"),
            mo.hstack([elements["start"], elements["end"]], justify="start"),
            mo.hstack([elements["annual_dollars"], elements["grant_growth"]], justify="start"),
            elements["threshold"],
            elements["after_tax_perf"],
        ]
    )
    advanced = mo.vstack(
        [
            elements["backfill"],
            mo.hstack([elements["vest_withholding"], elements["vesting_years"]], justify="start"),
            mo.hstack([elements["rebalances"], elements["rebalance_band"]], justify="start"),
            mo.hstack([elements["short_term_tax"], elements["long_term_tax"]], justify="start"),
            mo.hstack([elements["risk_free"]], justify="start"),
        ]
    )
    layout = mo.vstack([general, mo.accordion({"Extra settings": advanced})])

    return elements, layout


def build_configs(
    elements: mo.ui.dictionary,
) -> tuple[StrategyConfig, GrantConfig, BacktestConfig, str]:
    """Assemble the three library configs (plus the pre/after-tax basis label) from the controls.

    ``basis`` is derived here so the after-tax toggle is read in one place; the figure and
    table cells both title themselves with it rather than each re-deriving the string.
    """
    inputs: dict[str, Any] = elements.value
    start_ts = pd.Timestamp(inputs["start"])
    end_ts = pd.Timestamp(inputs["end"])

    tax_config = TaxConfig(
        short_term_rate=inputs["short_term_tax"] / 100.0,
        long_term_rate=inputs["long_term_tax"] / 100.0,
        ordinary_income_rate=inputs["vest_withholding"] / 100.0,
    )

    strategy_cfg = StrategyConfig(
        employer_ticker=inputs["employer"],
        index_ticker=inputs["index"],
        threshold=inputs["threshold"] / 100.0,
        rebalance_band=inputs["rebalance_band"] / 100.0,
        rebalances_per_quarter=int(inputs["rebalances"]),
        tax_config=tax_config,
    )

    grant_cfg = GrantConfig(
        grant_dollars=inputs["annual_dollars"],
        backfill=inputs["backfill"],
        vesting_years=int(inputs["vesting_years"]),
        grant_growth_rate=inputs["grant_growth"] / 100.0,
    )

    backtest_cfg = BacktestConfig(
        start=start_ts,
        end=end_ts,
        risk_free_rate=inputs["risk_free"] / 100.0,
        after_tax_performance=inputs["after_tax_perf"],
    )
    basis = "after-tax" if backtest_cfg.after_tax_performance else "pre-tax"
    return strategy_cfg, grant_cfg, backtest_cfg, basis
