"""Interactive backtest of the one-way threshold rebalancing strategy.

Run with: ``uv run marimo edit notebooks/rsu_backtest.py``

Set the parameters in the controls panel and the backtest re-runs reactively,
comparing the threshold strategy against the hold-everything and sell-all-at-vest
baselines on both return and risk.
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # RSU threshold rebalancing — backtest

    Each year you receive a grant of employer stock: its dollar value is converted
    to a fixed share count at the grant-date price, which then vests in equal annual
    tranches over the following few years. The threshold rebalancing strategy regularly
    trims employer stock back down to a specified fraction of total holdings, reinvesting
    the proceeds in a diversified index. Compare the threshold rebalancing strategy against
    *holding everything* and *selling everything at vest*.
    """)
    return


@app.cell
def imports():
    import marimo as mo

    from rsu_app import (
        build_backtest_controls,
        build_concentration_figure,
        build_configs,
        build_performance_figure,
        format_returns_table,
        format_trade_log,
    )
    from rsu_rebalancing import comparison_table, run_backtest

    return (
        build_backtest_controls,
        build_concentration_figure,
        build_configs,
        build_performance_figure,
        comparison_table,
        format_returns_table,
        format_trade_log,
        mo,
        run_backtest,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Controls
    """)
    return


@app.cell
def controls(build_backtest_controls):
    elements, layout = build_backtest_controls()
    layout
    return (elements,)


@app.cell
def backtest(build_configs, elements, mo, run_backtest):
    # Run the backtest
    try:
        strategy_cfg, grant_cfg, backtest_cfg = build_configs(elements)
        results = run_backtest(strategy_cfg, grant_cfg, backtest_cfg)
        error = None
    except Exception as exc:  # noqa: BLE001 - surface any data/config error in the UI
        results = None
        error = str(exc)

    mo.stop(error is not None, mo.md(f"⚠️ **Could not run:** {error}"))
    # mo.stop halts the cell when results is None; pyright can't see that guard.
    threshold_name = next(
        name
        for name in results  # pyright: ignore[reportOptionalIterable]
        if name.startswith("Threshold")
    )
    return backtest_cfg, results, strategy_cfg, threshold_name


@app.cell
def _(mo):
    mo.md("""
    ## Results
    """)
    return


@app.cell
def concentration_plot(
    build_concentration_figure,
    mo,
    results,
    strategy_cfg,
    threshold_name,
):
    conc_fig = build_concentration_figure(results[threshold_name], strategy_cfg.threshold)
    mo.vstack(
        [
            mo.md("### Employer concentration (threshold strategy)"),
            mo.mpl.interactive(conc_fig),
        ]
    )
    return


@app.cell
def performance_plot(
    backtest_cfg,
    build_performance_figure,
    mo,
    results,
    strategy_cfg,
):
    perf_fig = build_performance_figure(results, strategy_cfg, backtest_cfg)
    mo.vstack(
        [
            mo.md("### Investment performance (external flows removed)"),
            mo.mpl.interactive(perf_fig),
        ]
    )
    return


@app.cell
def comparison(
    backtest_cfg,
    comparison_table,
    format_returns_table,
    mo,
    results,
):
    returns_table = comparison_table(
        results,
        risk_free_rate=backtest_cfg.risk_free_rate,
    )
    styled = format_returns_table(returns_table)

    mo.vstack([mo.md("### Return & risk comparison"), mo.ui.table(styled, selection=None)])
    return


@app.cell
def trade_log(format_trade_log, mo, results, threshold_name):
    trade_log = format_trade_log(results[threshold_name].trades)
    mo.vstack(
        [
            mo.md(f"### Trade log — {threshold_name}"),
            mo.ui.table(trade_log, selection=None, pagination=True),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
