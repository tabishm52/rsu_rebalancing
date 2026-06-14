"""Interactive backtest of the one-way threshold rebalancing strategy.

Run with: ``uv run marimo edit notebooks/rebalance_sim.py``

Set the parameters in the controls panel and the simulation re-runs reactively,
comparing the threshold strategy against the hold-everything and sell-all-at-vest
baselines on both return and risk.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    from rsu_rebalancing import (
        GrantSchedule,
        SimConfig,
        StrategyConfig,
        comparison_table,
        growth_of_one,
        run_backtest,
        time_weighted_returns,
    )

    return (
        GrantSchedule,
        SimConfig,
        StrategyConfig,
        alt,
        comparison_table,
        growth_of_one,
        mo,
        pd,
        run_backtest,
        time_weighted_returns,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # RSU threshold rebalancing — backtest

        Each year a fixed-dollar grant of **employer stock** vests. Twice a quarter the
        strategy trims employer stock back down to a **threshold** fraction of total
        holdings, reinvesting the proceeds in a diversified **index**. Compare it against
        *holding everything* and *selling everything at vest*.
        """
    )
    return


@app.cell
def _(mo):
    # --- Controls ---------------------------------------------------------------
    employer = mo.ui.text(value="AAPL", label="Employer ticker")
    index = mo.ui.text(value="VTI", label="Index ticker")
    start = mo.ui.text(value="2015-01-01", label="Start date")
    end = mo.ui.text(value="2024-12-31", label="End date")
    annual_dollars = mo.ui.number(
        value=100_000, start=0, stop=10_000_000, step=1_000, label="Annual grant $"
    )
    threshold = mo.ui.slider(
        start=5, stop=100, value=33, step=1, label="Threshold %", show_value=True
    )
    days_after = mo.ui.slider(
        start=1, stop=20, value=5, step=1, label="Trade: Nth day after Q start", show_value=True
    )
    days_before = mo.ui.slider(
        start=1, stop=20, value=5, step=1, label="Trade: Nth day before Q end", show_value=True
    )
    tax_rate = mo.ui.slider(
        start=0, stop=50, value=0, step=1, label="Capital-gains tax %", show_value=True
    )
    risk_free = mo.ui.slider(
        start=0, stop=8, value=2, step=1, label="Risk-free % (for Sharpe)", show_value=True
    )

    controls = mo.vstack(
        [
            mo.hstack([employer, index], justify="start"),
            mo.hstack([start, end], justify="start"),
            annual_dollars,
            threshold,
            mo.hstack([days_after, days_before], justify="start"),
            mo.hstack([tax_rate, risk_free], justify="start"),
        ]
    )
    controls
    return (
        annual_dollars,
        controls,
        days_after,
        days_before,
        employer,
        end,
        index,
        risk_free,
        start,
        tax_rate,
        threshold,
    )


@app.cell
def _(
    GrantSchedule,
    SimConfig,
    StrategyConfig,
    annual_dollars,
    days_after,
    days_before,
    employer,
    end,
    index,
    mo,
    pd,
    risk_free,
    run_backtest,
    start,
    tax_rate,
    threshold,
):
    # --- Run the backtest -------------------------------------------------------
    start_ts = pd.Timestamp(start.value)
    end_ts = pd.Timestamp(end.value)

    strategy_cfg = StrategyConfig(
        employer_ticker=employer.value,
        index_ticker=index.value,
        threshold=threshold.value / 100.0,
        days_after_quarter_start=days_after.value,
        days_before_quarter_end=days_before.value,
        capital_gains_rate=tax_rate.value / 100.0,
    )
    schedule = GrantSchedule(
        annual_dollars=annual_dollars.value,
        start_year=start_ts.year,
        end_year=end_ts.year,
    )
    sim_cfg = SimConfig(start=start_ts, end=end_ts, risk_free_rate=risk_free.value / 100.0)

    try:
        results = run_backtest(strategy_cfg, schedule, sim_cfg)
        error = None
    except Exception as exc:  # noqa: BLE001 - surface any data/config error in the UI
        results = None
        error = str(exc)

    mo.stop(error is not None, mo.md(f"⚠️ **Could not run:** {error}"))
    threshold_name = next(name for name in results if name.startswith("Threshold"))
    results
    return (
        end_ts,
        error,
        results,
        schedule,
        sim_cfg,
        start_ts,
        strategy_cfg,
        threshold_name,
    )


@app.cell
def _(alt, mo, results, strategy_cfg, threshold_name):
    # --- Chart 1: employer concentration over time vs the threshold line --------
    frac = results[threshold_name].employer_fraction
    frac_df = frac.reset_index()
    frac_df.columns = ["Date", "employer_fraction"]

    line = (
        alt.Chart(frac_df)
        .mark_line(color="#d62728")
        .encode(x="Date:T", y=alt.Y("employer_fraction:Q", title="Employer fraction of holdings"))
    )
    rule = (
        alt.Chart(alt.Data(values=[{"y": strategy_cfg.threshold}]))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y="y:Q")
    )
    mo.vstack(
        [
            mo.md("### Employer concentration (threshold strategy)"),
            (line + rule).properties(height=260),
        ]
    )
    return frac, frac_df, line, rule


@app.cell
def _(alt, growth_of_one, mo, pd, results, time_weighted_returns):
    # --- Chart 2: growth of $1 (time-weighted) for all strategies ----------------
    curves = {
        name: growth_of_one(time_weighted_returns(res.values, res.contributions))
        for name, res in results.items()
    }
    growth_df = pd.DataFrame(curves)
    growth_long = growth_df.reset_index().melt(
        id_vars=growth_df.index.name or "index", var_name="strategy", value_name="growth"
    )
    growth_long.columns = ["Date", "strategy", "growth"]

    growth_chart = (
        alt.Chart(growth_long)
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("growth:Q", title="Growth of $1 (time-weighted)"),
            color="strategy:N",
        )
        .properties(height=300)
    )
    mo.vstack([mo.md("### Investment performance (contributions removed)"), growth_chart])
    return curves, growth_chart, growth_df, growth_long


@app.cell
def _(comparison_table, mo, pd, results, sim_cfg):
    # --- Metrics comparison table -----------------------------------------------
    table = comparison_table(results, risk_free_rate=sim_cfg.risk_free_rate)

    formatters = {
        "Final value": "${:,.0f}".format,
        "Total contributed": "${:,.0f}".format,
        "Ann. return (TWR)": "{:.2%}".format,
        "Ann. volatility": "{:.2%}".format,
        "Max drawdown": "{:.2%}".format,
        "Sharpe": "{:.2f}".format,
        "End employer %": "{:.1%}".format,
    }
    # Build an object-dtype frame of formatted strings (rows = metrics, cols = strategies).
    styled = pd.DataFrame({row: table.loc[row].map(fmt) for row, fmt in formatters.items()}).T

    mo.vstack([mo.md("### Return & risk comparison"), mo.ui.table(styled, selection=None)])
    return formatters, styled, table


@app.cell
def _(mo, results, threshold_name):
    # --- Trade log for the threshold strategy -----------------------------------
    trades = results[threshold_name].trades
    mo.vstack(
        [
            mo.md(f"### Trade log — {threshold_name}"),
            mo.ui.table(trades, selection=None, pagination=True),
        ]
    )
    return (trades,)


if __name__ == "__main__":
    app.run()
