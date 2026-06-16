"""Interactive backtest of the one-way threshold rebalancing strategy.

Run with: ``uv run marimo edit notebooks/rebalance_sim.py``

Set the parameters in the controls panel and the simulation re-runs reactively,
comparing the threshold strategy against the hold-everything and sell-all-at-vest
baselines on both return and risk.
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from rsu_rebalancing import (
        GrantSchedule,
        SimConfig,
        StrategyConfig,
        TaxConfig,
        comparison_table,
        growth_of_one,
        run_backtest,
        time_weighted_returns,
    )

    sns.set_theme()
    return (
        GrantSchedule,
        SimConfig,
        StrategyConfig,
        TaxConfig,
        comparison_table,
        growth_of_one,
        mo,
        pd,
        plt,
        run_backtest,
        time_weighted_returns,
    )


@app.cell
def _(mo):
    mo.md("""
    # RSU threshold rebalancing — backtest

    Each year a fixed-dollar grant of **employer stock** vests. Twice a quarter the
    strategy trims employer stock back down to a **threshold** fraction of total
    holdings, reinvesting the proceeds in a diversified **index**. Compare it against
    *holding everything* and *selling everything at vest*.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Controls
    """)
    return


@app.cell
def _(SimConfig, StrategyConfig, TaxConfig, mo):
    # Defaults come from the config dataclasses (single source of truth); the notebook
    # only owns UI presentation (widget type, ranges, percent units).
    employer = mo.ui.text(value="AAPL", label="Employer ticker")
    index = mo.ui.text(value=StrategyConfig.index_ticker, label="Index ticker")
    start = mo.ui.text(value="2015-01-01", label="Start date")
    end = mo.ui.text(value="2024-12-31", label="End date")
    annual_dollars = mo.ui.number(
        value=100_000, start=1, stop=10_000_000, step=1_000, label="Annual grant $"
    )
    threshold = mo.ui.slider(
        start=5,
        stop=100,
        value=round(StrategyConfig.threshold * 100),
        step=1,
        label="Threshold %",
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
    taxes_on = mo.ui.switch(value=True, label="Apply capital-gains taxes")
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
    risk_free = mo.ui.slider(
        start=0, stop=8, value=2, step=1, label="Risk-free % (for Sharpe)", show_value=True
    )
    after_tax_perf = mo.ui.switch(
        value=SimConfig.after_tax_performance, label="Measure performance after tax"
    )

    # The everyday knobs sit up top; the fussy details (exact tax rates, risk-free) tuck
    # into a collapsed accordion so they're available without crowding the common path.
    general = mo.vstack(
        [
            mo.hstack([employer, index], justify="start"),
            mo.hstack([start, end], justify="start"),
            annual_dollars,
            threshold,
            taxes_on,
            after_tax_perf,
        ]
    )
    advanced = mo.vstack(
        [
            mo.hstack([rebalances], justify="start"),
            mo.hstack([short_term_tax, long_term_tax], justify="start"),
            mo.hstack([risk_free], justify="start"),
        ]
    )
    controls = mo.vstack([general, mo.accordion({"Extra settings": advanced})])
    controls
    return (
        after_tax_perf,
        annual_dollars,
        employer,
        end,
        index,
        long_term_tax,
        rebalances,
        risk_free,
        short_term_tax,
        start,
        taxes_on,
        threshold,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Run the backtest
    """)
    return


@app.cell
def _(
    GrantSchedule,
    SimConfig,
    StrategyConfig,
    TaxConfig,
    after_tax_perf,
    annual_dollars,
    employer,
    end,
    index,
    long_term_tax,
    mo,
    pd,
    rebalances,
    risk_free,
    run_backtest,
    short_term_tax,
    start,
    taxes_on,
    threshold,
):
    start_ts = pd.Timestamp(start.value)
    end_ts = pd.Timestamp(end.value)

    # The toggle gates the fine-tuned rates: off means a zero-rate (tax-free) backtest.
    if taxes_on.value:
        tax_config = TaxConfig(
            short_term_rate=short_term_tax.value / 100.0,
            long_term_rate=long_term_tax.value / 100.0,
        )
    else:
        tax_config = TaxConfig(short_term_rate=0.0, long_term_rate=0.0)

    strategy_cfg = StrategyConfig(
        employer_ticker=employer.value,
        index_ticker=index.value,
        threshold=threshold.value / 100.0,
        rebalances_per_quarter=rebalances.value,
        tax_config=tax_config,
    )
    schedule = GrantSchedule(
        annual_dollars=annual_dollars.value,
        start_year=start_ts.year,
        end_year=end_ts.year,
    )
    sim_cfg = SimConfig(
        start=start_ts,
        end=end_ts,
        risk_free_rate=risk_free.value / 100.0,
        after_tax_performance=after_tax_perf.value,
    )

    try:
        results = run_backtest(strategy_cfg, schedule, sim_cfg)
        error = None
    except Exception as exc:  # noqa: BLE001 - surface any data/config error in the UI
        results = None
        error = str(exc)

    mo.stop(error is not None, mo.md(f"⚠️ **Could not run:** {error}"))
    threshold_name = next(name for name in results if name.startswith("Threshold"))
    results
    return results, sim_cfg, strategy_cfg, threshold_name


@app.cell
def _(mo):
    mo.md("""
    ## Results
    """)
    return


@app.cell
def _(mo, plt, results, strategy_cfg, threshold_name):
    frac = results[threshold_name].employer_fraction

    frac_fig, frac_ax = plt.subplots(figsize=(12, 5))
    frac_ax.plot(frac.index, frac.values, color="#d62728")
    frac_ax.axhline(strategy_cfg.threshold, linestyle="--", color="gray")
    frac_ax.set_xlabel("Date")
    frac_ax.set_ylabel("Employer fraction of holdings")
    frac_fig.tight_layout()

    mo.vstack(
        [
            mo.md("### Employer concentration (threshold strategy)"),
            frac_fig,
        ]
    )
    return


@app.cell
def _(growth_of_one, mo, pd, plt, results, sim_cfg, time_weighted_returns):
    after_tax = sim_cfg.after_tax_performance
    perf = {name: (res.net_of_tax if after_tax else res.market) for name, res in results.items()}
    curves = {name: growth_of_one(time_weighted_returns(p)) for name, p in perf.items()}
    growth_df = pd.DataFrame(curves)

    _basis = "after-tax" if after_tax else "pre-tax"
    growth_fig, growth_ax = plt.subplots(figsize=(12, 5))
    (growth_df * 100).plot(ax=growth_ax)
    growth_ax.set_xlabel("Date")
    growth_ax.set_ylabel(f"Index, start = 100 ({_basis}, time-weighted)")
    growth_ax.legend(title="strategy")
    growth_fig.tight_layout()

    mo.vstack([mo.md(f"### Investment performance ({_basis}, external flows removed)"), growth_fig])
    return


@app.cell
def _(comparison_table, mo, pd, results, sim_cfg):
    table = comparison_table(
        results,
        risk_free_rate=sim_cfg.risk_free_rate,
        after_tax=sim_cfg.after_tax_performance,
    )

    formatters = {
        "Final portfolio value": "${:,.0f}".format,
        "Liquidation value (net of tax)": "${:,.0f}".format,
        "Total contributed": "${:,.0f}".format,
        "Ann. return (TWR)": "{:.2%}".format,
        "Ann. volatility": "{:.2%}".format,
        "Max drawdown": "{:.2%}".format,
        "Sharpe": "{:.2f}".format,
        "End employer %": "{:.1%}".format,
    }
    # Build an object-dtype frame of formatted strings (rows = metrics, cols = strategies).
    styled = pd.DataFrame({row: table.loc[row].map(fmt) for row, fmt in formatters.items()}).T

    _basis = "after-tax" if sim_cfg.after_tax_performance else "pre-tax"
    mo.vstack(
        [mo.md(f"### Return & risk comparison ({_basis})"), mo.ui.table(styled, selection=None)]
    )
    return


@app.cell
def _(mo, results, threshold_name):
    trades = results[threshold_name].trades
    display = trades.assign(
        date=trades["date"].dt.date,
        employer_shares=trades["employer_shares"].round(1),
        employer_price=trades["employer_price"].round(2),
        gross_value=trades["gross_value"].round(2),
        tax_paid=trades["tax_paid"].round(2),
        index_dollars_in=trades["index_dollars_in"].round(2),
    )
    mo.vstack(
        [
            mo.md(f"### Trade log — {threshold_name}"),
            mo.ui.table(display, selection=None, pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
