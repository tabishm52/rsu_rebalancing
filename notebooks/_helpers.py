"""Presentation helpers for the marimo backtest notebook.

This module owns the mechanical bulk of ``rsu_backtest.py`` — widget construction,
config assembly, matplotlib figure building, and table formatting — so the notebook
cells stay short and read as intent, not plumbing. It is presentation-only and
deliberately lives beside the notebooks rather than in the ``rsu_rebalancing`` package,
which stays a clean computational API.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import marimo as mo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rsu_rebalancing import (
    BacktestConfig,
    GrantConfig,
    StrategyConfig,
    TaxConfig,
    get_price_frame,
    growth_of_one,
    time_weighted_returns,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from rsu_rebalancing import BacktestResult

sns.set_theme()


# --- Controls --------------------------------------------------------------


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


# --- Figures ---------------------------------------------------------------


def build_concentration_figure(result: BacktestResult, threshold: float) -> Figure:
    """Employer fraction of holdings over time, with the threshold and sale events."""
    frac = result.employer_fraction

    trades = result.trades
    sale_dates = trades.loc[trades["kind"].isin(["rebalance", "liquidate"]), "date"]
    pre_sale_frac = frac.shift(1).loc[sale_dates]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frac.index, frac.values, color="#d62728")
    ax.axhline(threshold, linestyle="--", color="gray")
    ax.scatter(
        sale_dates,
        pre_sale_frac,
        marker="*",
        s=120,
        color="#1f77b4",
        zorder=3,
        label="rebalance events",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Employer fraction of holdings")
    ax.legend()

    fig.tight_layout()
    return fig


def outperformance_spans(employer, index, lookback=63, min_days=90, gap_days=10):
    """Date spans where the index has out-returned the employer stock.

    A day is "index-ahead" when the index's return over the trailing ``lookback`` *trading*
    days exceeds the employer's. Consecutive index-ahead days form an episode; episodes
    separated by a gap of at most ``gap_days`` *calendar* days are merged, and only the
    merged spans lasting at least ``min_days`` *calendar* days survive, dropping daily
    noise. Returns ``(start, end)`` pairs. This shades the relative *declines* (employer
    losing ground), not the recoveries.

    Note ``lookback`` counts trading days, while ``min_days`` and ``gap_days`` count
    calendar days. The defaults for ``lookback`` and ``min_days`` match — 63 trading days
    is ~90 calendar days.
    """
    ahead = employer.pct_change(lookback) < index.pct_change(lookback)
    runs = (ahead != ahead.shift()).cumsum()
    segments = [(run.index[0], run.index[-1]) for _, run in ahead.groupby(runs) if run.iloc[0]]
    spans = []
    for start, end in segments:
        if spans and (start - spans[-1][1]).days <= gap_days:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return [(start, end) for start, end in spans if (end - start).days >= min_days]


def build_performance_figure(
    results: dict[str, BacktestResult],
    strategy_cfg: StrategyConfig,
    backtest_cfg: BacktestConfig,
    basis: str,
) -> Figure:
    """Growth-of-100 curves per strategy, shading where the index out-returned employer.

    Each curve is labelled by ``BacktestResult.description`` (e.g. "Threshold: 33% AAPL /
    67% VTI"), so the legend stands alone; the generic result names stay as-is for the
    table and lookups. ``basis`` titles the y-axis (see ``build_configs``).
    """
    after_tax = backtest_cfg.after_tax_performance
    perf = {
        res.description: (res.net_of_tax if after_tax else res.market) for res in results.values()
    }
    curves = {label: growth_of_one(time_weighted_returns(p)) for label, p in perf.items()}
    growth_df = pd.DataFrame(curves)

    # Shade stretches where the diversified index out-returned the employer stock — the
    # episodes the threshold strategy's downside protection is meant to catch.
    prices = get_price_frame(
        [strategy_cfg.employer_ticker, strategy_cfg.index_ticker],
        backtest_cfg.start,
        backtest_cfg.end,
    )
    spans = outperformance_spans(
        prices[strategy_cfg.employer_ticker.upper()],
        prices[strategy_cfg.index_ticker.upper()],
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    (growth_df * 100).plot(ax=ax)
    band = None
    for start, end in spans:
        band = ax.axvspan(start, end, color="#d62728", alpha=0.12, zorder=0)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Index, start = 100 ({basis}, time-weighted)")
    handles, labels = ax.get_legend_handles_labels()
    if band is not None:
        handles.append(band)
        labels.append("index outperforming employer")
    ax.legend(handles, labels, loc="upper left")
    fig.tight_layout()
    return fig


# --- Tables ----------------------------------------------------------------


def format_returns_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format the raw returns table into display strings (rows = metrics)."""
    formatters = {
        "Final portfolio value": "${:,.0f}".format,
        "Liquidation value (net of tax)": "${:,.0f}".format,
        "Vested contributions (net of tax)": "${:,.0f}".format,
        "Taxes paid": "${:,.0f}".format,
        "Annualized return (TWR)": "{:.2%}".format,
        "Annualized volatility": "{:.2%}".format,
        "Max drawdown": "{:.2%}".format,
        "Sharpe ratio": "{:.2f}".format,
        "End employer %": "{:.1%}".format,
    }
    # Build an object-dtype frame of formatted strings (rows = metrics, cols = strategies).
    return pd.DataFrame({row: table.loc[row].map(fmt) for row, fmt in formatters.items()}).T


def format_trade_log(trades: pd.DataFrame) -> pd.DataFrame:
    """Round the trade log to display precision."""
    return trades.assign(
        date=trades["date"].dt.date,
        employer_shares=trades["employer_shares"].round(1),
        employer_price=trades["employer_price"].round(2),
        traded_value=trades["traded_value"].round(2),
        tax_paid=trades["tax_paid"].round(2),
        index_dollars_in=trades["index_dollars_in"].round(2),
    )
