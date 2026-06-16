"""Exploratory scratchpad: getting a feel for prices with pandas + yfinance.

Run with: ``uv run marimo edit notebooks/exploration.py``

This is a learning notebook, not part of the package API -- a place to poke at the
data layer and basic financial idioms (returns, normalization, rolling stats) in raw
pandas and via quantstats, before the structured backtest in ``rsu_backtest.py``.
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # Exploration scratchpad

    A place to learn pandas financial idioms against real prices. Edit freely.
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import quantstats as qs
    import seaborn as sns

    from rsu_rebalancing import (
        annualized_return,
        annualized_volatility,
        get_price_frame,
        max_drawdown,
        sharpe_ratio,
    )

    sns.set_theme()
    return (
        annualized_return,
        annualized_volatility,
        get_price_frame,
        max_drawdown,
        mo,
        pd,
        plt,
        qs,
        sharpe_ratio,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Controls
    """)
    return


@app.cell(hide_code=True)
def controls(mo):
    ticker_a = mo.ui.text(value="AAPL", label="First ticker")
    ticker_b = mo.ui.text(value="VTI", label="Second ticker")
    start = mo.ui.text(value="2015-01-01", label="Start date")
    end = mo.ui.text(value="2024-12-31", label="End date")

    controls = mo.vstack(
        [
            mo.hstack([ticker_a, ticker_b], justify="start"),
            mo.hstack([start, end], justify="start"),
        ]
    )
    controls
    return end, start, ticker_a, ticker_b


@app.cell
def _(mo):
    mo.md("""
    ## Price data
    """)
    return


@app.cell
def fetch_prices(end, get_price_frame, mo, start, ticker_a, ticker_b):
    # Two tickers, aligned on common trading days.
    try:
        prices = get_price_frame([ticker_a.value, ticker_b.value], start.value, end.value)
        error = None
    except Exception as exc:  # noqa: BLE001 - surface any data/config error in the UI
        prices = None
        error = str(exc)

    mo.stop(error is not None, mo.md(f"⚠️ **Could not fetch:** {error}"))
    prices
    return (prices,)


@app.cell
def _(mo):
    mo.md("""
    ## Cumulative performance
    """)
    return


@app.cell
def growth_plot(mo, plt, prices):
    # Growth chart, rebased to 100 at the start date.
    fig, ax = plt.subplots(figsize=(12, 6))
    normalized = prices / prices.iloc[0]
    (normalized * 100).plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index (start = 100)")
    ax.legend(title="ticker")
    fig.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Daily returns
    """)
    return


@app.cell
def returns(prices):
    # Daily returns by hand -- the series the summary table below is built from.
    daily_returns = prices.pct_change().dropna()
    daily_returns
    return (daily_returns,)


@app.cell
def _(mo):
    mo.md("""
    ## Headline stats (metrics.py)
    """)
    return


@app.cell
def headline_stats(
    annualized_return,
    annualized_volatility,
    daily_returns,
    max_drawdown,
    mo,
    pd,
    sharpe_ratio,
):
    headline = pd.DataFrame(
        {
            "Ann. return": daily_returns.apply(annualized_return),
            "Ann. volatility": daily_returns.apply(annualized_volatility),
            "Sharpe": daily_returns.apply(sharpe_ratio),
            "Max drawdown": daily_returns.apply(max_drawdown),
        }
    ).T

    formatters = {
        "Ann. return": "{:.2%}".format,
        "Ann. volatility": "{:.2%}".format,
        "Sharpe": "{:.2f}".format,
        "Max drawdown": "{:.2%}".format,
    }
    styled = pd.DataFrame({row: headline.loc[row].map(fmt) for row, fmt in formatters.items()}).T

    mo.ui.table(styled, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Full tearsheet per ticker (quantstats)
    """)
    return


@app.cell
def tearsheet(daily_returns, pd, qs):
    tearsheet = pd.DataFrame(
        {
            col: qs.reports.metrics(daily_returns[col], mode="basic", display=False)["Strategy"]
            for col in daily_returns.columns
        }
    )
    tearsheet
    return


if __name__ == "__main__":
    app.run()
