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

    Each year you receive a **grant** of **employer stock**: its dollar value is converted
    to a fixed share count at the grant-date price, which then vests in equal annual
    tranches over the following few years (so the *dollars* delivered at each vest float
    with the stock). Twice a quarter the strategy trims employer stock back down to a
    **threshold** fraction of total holdings, reinvesting the proceeds in a diversified
    **index**. Compare it against *holding everything* and *selling everything at vest*.
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from rsu_rebalancing import (
        BacktestConfig,
        GrantConfig,
        StrategyConfig,
        TaxConfig,
        comparison_table,
        get_price_frame,
        growth_of_one,
        run_backtest,
        time_weighted_returns,
    )

    sns.set_theme()
    return (
        BacktestConfig,
        GrantConfig,
        StrategyConfig,
        TaxConfig,
        comparison_table,
        get_price_frame,
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
    ## Controls
    """)
    return


@app.cell(hide_code=True)
def controls(BacktestConfig, GrantConfig, StrategyConfig, TaxConfig, mo):
    # Tuning and reporting defaults come from the config dataclasses (single source of
    # truth); the notebook owns UI presentation (widget type, ranges, percent units) and
    # seeds the required policy inputs (employer, grant size, dates, threshold), which the
    # library deliberately refuses to default.
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
    # Drives the sell-to-cover haircut on vesting shares (TaxConfig.ordinary_income_rate).
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
            mo.hstack([rebalances, rebalance_band], justify="start"),
            mo.hstack([short_term_tax, long_term_tax], justify="start"),
            mo.hstack([vest_withholding, vesting_years], justify="start"),
            mo.hstack([risk_free], justify="start"),
        ]
    )
    controls = mo.vstack([general, mo.accordion({"Extra settings": advanced})])
    controls
    return (
        after_tax_perf,
        annual_dollars,
        backfill,
        employer,
        end,
        grant_growth,
        index,
        long_term_tax,
        rebalance_band,
        rebalances,
        risk_free,
        short_term_tax,
        start,
        threshold,
        vest_withholding,
        vesting_years,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Run the backtest
    """)
    return


@app.cell
def backtest(
    BacktestConfig,
    GrantConfig,
    StrategyConfig,
    TaxConfig,
    after_tax_perf,
    annual_dollars,
    backfill,
    employer,
    end,
    grant_growth,
    index,
    long_term_tax,
    mo,
    pd,
    rebalance_band,
    rebalances,
    risk_free,
    run_backtest,
    short_term_tax,
    start,
    threshold,
    vest_withholding,
    vesting_years,
):
    start_ts = pd.Timestamp(start.value)
    end_ts = pd.Timestamp(end.value)

    # The three rate sliders are the only tax control: drag them all to zero for a
    # tax-free backtest.
    tax_config = TaxConfig(
        short_term_rate=short_term_tax.value / 100.0,
        long_term_rate=long_term_tax.value / 100.0,
        ordinary_income_rate=vest_withholding.value / 100.0,
    )

    strategy_cfg = StrategyConfig(
        employer_ticker=employer.value,
        index_ticker=index.value,
        threshold=threshold.value / 100.0,
        rebalance_band=rebalance_band.value / 100.0,
        rebalances_per_quarter=rebalances.value,
        tax_config=tax_config,
    )
    # Backfill makes grants begin vesting_years before the window so its first year opens
    # at steady-state overlapping vests (a mature employee); otherwise the first grant lands
    # at the window start (a new hire ramping up). Either way, pre-window vests are dropped.
    grant_start_year = start_ts.year - (vesting_years.value if backfill.value else 0)
    schedule = GrantConfig(
        grant_dollars=annual_dollars.value,
        start_year=grant_start_year,
        end_year=end_ts.year,
        vesting_years=vesting_years.value,
        grant_growth_rate=grant_growth.value / 100.0,
    )
    backtest_cfg = BacktestConfig(
        start=start_ts,
        end=end_ts,
        risk_free_rate=risk_free.value / 100.0,
        after_tax_performance=after_tax_perf.value,
    )

    try:
        results = run_backtest(strategy_cfg, schedule, backtest_cfg)
        error = None
    except Exception as exc:  # noqa: BLE001 - surface any data/config error in the UI
        results = None
        error = str(exc)

    mo.stop(error is not None, mo.md(f"⚠️ **Could not run:** {error}"))
    threshold_name = next(name for name in results if name.startswith("Threshold"))
    results
    return backtest_cfg, results, strategy_cfg, threshold_name


@app.cell
def _(mo):
    mo.md("""
    ## Results
    """)
    return


@app.cell
def concentration_plot(mo, plt, results, strategy_cfg, threshold_name):
    result = results[threshold_name]
    frac = result.employer_fraction

    _trades = result.trades
    sale_dates = _trades.loc[_trades["kind"].isin(["rebalance", "liquidate"]), "date"]
    pre_sale_frac = frac.shift(1).loc[sale_dates]

    frac_fig, frac_ax = plt.subplots(figsize=(12, 5))
    frac_ax.plot(frac.index, frac.values, color="#d62728")
    frac_ax.axhline(strategy_cfg.threshold, linestyle="--", color="gray")
    frac_ax.scatter(
        sale_dates,
        pre_sale_frac,
        marker="*",
        s=120,
        color="#1f77b4",
        zorder=3,
        label="rebalance events",
    )
    frac_ax.set_xlabel("Date")
    frac_ax.set_ylabel("Employer fraction of holdings")
    frac_ax.legend()
    frac_fig.tight_layout()

    mo.vstack(
        [
            mo.md("### Employer concentration (threshold strategy)"),
            mo.mpl.interactive(frac_fig),
        ]
    )
    return


@app.function
def outperformance_spans(employer, index, lookback=63, min_days=90, gap_days=10):
    """Date spans where the index has out-returned the employer stock.

    A day is "index-ahead" when the index's return over the trailing ``lookback``
    trading days exceeds the employer's — i.e. diversification has been paying off
    lately. Consecutive index-ahead days form an episode; episodes separated by a gap of
    at most ``gap_days`` calendar days are merged (so one good week doesn't split a
    protection phase), and only the merged spans lasting at least ``min_days`` survive,
    dropping daily noise. Returns ``(start, end)`` pairs.

    This shades the relative *declines* (employer losing ground), not the recoveries:
    an underwater/drawdown definition would also redden the long climbs back and, for a
    strong long-term outperformer, cover almost the whole window.
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


@app.cell
def performance_plot(
    backtest_cfg,
    get_price_frame,
    growth_of_one,
    mo,
    pd,
    plt,
    results,
    strategy_cfg,
    time_weighted_returns,
):
    # Spell out each strategy's target composition so the legend stands alone (e.g.
    # "Threshold: 33% AAPL / 67% VTI"); derived from config, so it tracks the ticker and
    # threshold controls. The generic result names stay as-is for the table and lookups.
    _emp, _idx, _thr = (
        strategy_cfg.employer_ticker,
        strategy_cfg.index_ticker,
        strategy_cfg.threshold,
    )

    def _label(name):
        if name == "Hold everything":
            return f"Hold everything: 100% {_emp}"
        if name == "Sell all at vest":
            return f"Sell all at vest: 100% {_idx}"
        if name.startswith("Threshold"):
            return f"Threshold: {_thr:.0%} {_emp} / {1 - _thr:.0%} {_idx}"
        return name

    after_tax = backtest_cfg.after_tax_performance
    perf = {name: (res.net_of_tax if after_tax else res.market) for name, res in results.items()}
    curves = {_label(name): growth_of_one(time_weighted_returns(p)) for name, p in perf.items()}
    growth_df = pd.DataFrame(curves)

    # Shade stretches where the diversified index out-returned the employer stock — the
    # episodes the threshold strategy's downside protection is meant to catch.
    _prices = get_price_frame(
        [strategy_cfg.employer_ticker, strategy_cfg.index_ticker],
        backtest_cfg.start,
        backtest_cfg.end,
    )
    _spans = outperformance_spans(
        _prices[strategy_cfg.employer_ticker.upper()],
        _prices[strategy_cfg.index_ticker.upper()],
    )

    _basis = "after-tax" if after_tax else "pre-tax"
    growth_fig, growth_ax = plt.subplots(figsize=(12, 5))
    (growth_df * 100).plot(ax=growth_ax)
    _band = None
    for _start, _end in _spans:
        _band = growth_ax.axvspan(_start, _end, color="#d62728", alpha=0.12, zorder=0)
    growth_ax.set_xlabel("Date")
    growth_ax.set_ylabel(f"Index, start = 100 ({_basis}, time-weighted)")
    _handles, _labels = growth_ax.get_legend_handles_labels()
    if _band is not None:
        _handles.append(_band)
        _labels.append("index outperforming employer")
    growth_ax.legend(_handles, _labels, loc="upper left")
    growth_fig.tight_layout()

    mo.vstack(
        [
            mo.md(f"### Investment performance ({_basis}, external flows removed)"),
            mo.mpl.interactive(growth_fig),
        ]
    )
    return


@app.cell
def comparison(backtest_cfg, comparison_table, mo, pd, results):
    table = comparison_table(
        results,
        risk_free_rate=backtest_cfg.risk_free_rate,
        after_tax=backtest_cfg.after_tax_performance,
    )

    formatters = {
        "Final portfolio value": "${:,.0f}".format,
        "Liquidation value (net of tax)": "${:,.0f}".format,
        "Total vested contributions": "${:,.0f}".format,
        "Ann. return (TWR)": "{:.2%}".format,
        "Ann. volatility": "{:.2%}".format,
        "Max drawdown": "{:.2%}".format,
        "Sharpe": "{:.2f}".format,
        "End employer %": "{:.1%}".format,
    }
    # Build an object-dtype frame of formatted strings (rows = metrics, cols = strategies).
    styled = pd.DataFrame({row: table.loc[row].map(fmt) for row, fmt in formatters.items()}).T

    _basis = "after-tax" if backtest_cfg.after_tax_performance else "pre-tax"
    mo.vstack(
        [mo.md(f"### Return & risk comparison ({_basis})"), mo.ui.table(styled, selection=None)]
    )
    return


@app.cell
def trade_log(mo, results, threshold_name):
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
