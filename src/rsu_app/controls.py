"""The backtest notebook's control panel and its assembly into engine configs."""

from dataclasses import dataclass

import marimo as mo
import pandas as pd

from rsu_rebalancing import (
    BacktestConfig,
    GrantConfig,
    StrategyConfig,
    TaxConfig,
)


@dataclass
class BacktestControls:
    """The notebook's input widgets plus their assembled layout.

    Holding the ``mo.ui`` elements on a dataclass lets the notebook display
    ``controls.layout`` in one cell and read ``controls.threshold.value`` (etc.) in the
    others. marimo syncs any element that appears in displayed output, so reactivity is
    preserved even though the widgets are built here rather than in a notebook cell.
    """

    employer: mo.ui.text
    index: mo.ui.text
    start: mo.ui.text
    end: mo.ui.text
    annual_dollars: mo.ui.number
    vesting_years: mo.ui.slider
    backfill: mo.ui.switch
    grant_growth: mo.ui.slider
    threshold: mo.ui.slider
    rebalances: mo.ui.slider
    rebalance_band: mo.ui.slider
    short_term_tax: mo.ui.slider
    long_term_tax: mo.ui.slider
    vest_withholding: mo.ui.slider
    risk_free: mo.ui.slider
    after_tax_perf: mo.ui.switch
    layout: mo.Html


def build_backtest_controls() -> BacktestControls:
    """Construct the control panel.

    Tuning and reporting defaults come from the config dataclasses; the notebook owns UI
    presentation (widget type, ranges, percent units) and seeds the required policy
    inputs (employer, grant size, dates, threshold).
    """
    employer = mo.ui.text(value="AAPL", label="Employer ticker")
    index = mo.ui.text(value=StrategyConfig.index_ticker, label="Index ticker")
    start = mo.ui.text(value="2015-01-01", label="Start date")
    end = mo.ui.text(value="2024-12-31", label="End date")
    annual_dollars = mo.ui.number(
        value=100_000, start=0, stop=1_000_000, step=25_000, label="First-year grant $"
    )
    vesting_years = mo.ui.slider(
        start=1,
        stop=6,
        value=GrantConfig.vesting_years,
        step=1,
        label="Vesting years",
        show_value=True,
    )
    backfill = mo.ui.switch(
        value=True, label="Backfill grants before window (mature employee, not new hire)"
    )
    grant_growth = mo.ui.slider(
        start=0,
        stop=10,
        value=round(GrantConfig.grant_growth_rate * 100),
        step=1,
        label="Grant growth %/yr",
        show_value=True,
    )
    threshold = mo.ui.slider(
        start=5,
        stop=100,
        value=33,
        step=1,
        label="Rebalance threshold %",
        show_value=True,
    )
    rebalances = mo.ui.slider(
        start=1,
        stop=3,
        value=StrategyConfig.rebalances_per_quarter,
        step=1,
        label="Rebalances per quarter",
        show_value=True,
    )
    rebalance_band = mo.ui.slider(
        start=0,
        stop=10,
        value=round(StrategyConfig.rebalance_band * 100),
        step=1,
        label="Hysteresis band %",
        show_value=True,
    )
    short_term_tax = mo.ui.slider(
        start=0,
        stop=60,
        value=round(TaxConfig.short_term_rate * 100),
        step=1,
        label="Short-term cap-gains tax %",
        show_value=True,
    )
    long_term_tax = mo.ui.slider(
        start=0,
        stop=40,
        value=round(TaxConfig.long_term_rate * 100),
        step=1,
        label="Long-term cap-gains tax %",
        show_value=True,
    )
    vest_withholding = mo.ui.slider(
        start=0,
        stop=60,
        value=round(TaxConfig.ordinary_income_rate * 100),
        step=1,
        label="Vest withholding %",
        show_value=True,
    )
    risk_free = mo.ui.slider(
        start=0,
        stop=8,
        value=round(BacktestConfig.risk_free_rate * 100),
        step=1,
        label="Risk-free % (for Sharpe)",
        show_value=True,
    )
    after_tax_perf = mo.ui.switch(
        value=BacktestConfig.after_tax_performance, label="Analyze performance after tax"
    )

    # The everyday knobs sit up top; the fussy details (exact tax rates, risk-free) tuck
    # into a collapsed accordion so they're available without crowding the common path.
    general = mo.vstack(
        [
            mo.hstack([employer, index], justify="start"),
            mo.hstack([start, end], justify="start"),
            mo.hstack([annual_dollars, grant_growth], justify="start"),
            threshold,
            after_tax_perf,
        ]
    )
    advanced = mo.vstack(
        [
            backfill,
            mo.hstack([vest_withholding, vesting_years], justify="start"),
            mo.hstack([rebalances, rebalance_band], justify="start"),
            mo.hstack([short_term_tax, long_term_tax], justify="start"),
            mo.hstack([risk_free], justify="start"),
        ]
    )
    layout = mo.vstack([general, mo.accordion({"Extra settings": advanced})])

    return BacktestControls(
        employer=employer,
        index=index,
        start=start,
        end=end,
        annual_dollars=annual_dollars,
        vesting_years=vesting_years,
        backfill=backfill,
        grant_growth=grant_growth,
        threshold=threshold,
        rebalances=rebalances,
        rebalance_band=rebalance_band,
        short_term_tax=short_term_tax,
        long_term_tax=long_term_tax,
        vest_withholding=vest_withholding,
        risk_free=risk_free,
        after_tax_perf=after_tax_perf,
        layout=layout,
    )


def build_configs(
    c: BacktestControls,
) -> tuple[StrategyConfig, GrantConfig, BacktestConfig, str]:
    """Assemble the three library configs (plus the pre/after-tax basis label) from the controls.

    ``basis`` is derived here so the after-tax toggle is read in one place; the figure and
    table cells both title themselves with it rather than each re-deriving the string.
    """
    start_ts = pd.Timestamp(c.start.value)
    end_ts = pd.Timestamp(c.end.value)

    tax_config = TaxConfig(
        short_term_rate=c.short_term_tax.value / 100.0,
        long_term_rate=c.long_term_tax.value / 100.0,
        ordinary_income_rate=c.vest_withholding.value / 100.0,
    )

    strategy_cfg = StrategyConfig(
        employer_ticker=c.employer.value,
        index_ticker=c.index.value,
        threshold=c.threshold.value / 100.0,
        rebalance_band=c.rebalance_band.value / 100.0,
        rebalances_per_quarter=c.rebalances.value,
        tax_config=tax_config,
    )

    # Backfill makes grants begin vesting_years before the window so its first year opens
    # at steady-state overlapping vests (a mature employee). Otherwise, the first grant
    # lands at the window start (a new hire ramping up).
    grant_start_year = start_ts.year - (c.vesting_years.value if c.backfill.value else 0)
    grant_cfg = GrantConfig(
        grant_dollars=c.annual_dollars.value,
        start_year=grant_start_year,
        end_year=end_ts.year,
        vesting_years=c.vesting_years.value,
        grant_growth_rate=c.grant_growth.value / 100.0,
    )

    backtest_cfg = BacktestConfig(
        start=start_ts,
        end=end_ts,
        risk_free_rate=c.risk_free.value / 100.0,
        after_tax_performance=c.after_tax_perf.value,
    )
    basis = "after-tax" if backtest_cfg.after_tax_performance else "pre-tax"
    return strategy_cfg, grant_cfg, backtest_cfg, basis
