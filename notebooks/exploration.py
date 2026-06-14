"""Exploratory scratchpad: getting a feel for prices with pandas + yfinance.

Run with: ``uv run marimo edit notebooks/exploration.py``

This is a learning notebook, not part of the package API -- a place to poke at the
data layer and basic pandas financial idioms (returns, normalization, rolling stats)
before the structured simulation in ``rebalance_sim.py``.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    from rsu_rebalancing import get_price_frame

    return alt, get_price_frame, mo, pd


@app.cell
def _(mo):
    mo.md(
        """
        # Exploration scratchpad

        A place to learn pandas financial idioms against real prices. Edit freely.
        """
    )
    return


@app.cell
def _(get_price_frame):
    # One employer stock vs a diversified index, aligned on trading days.
    prices = get_price_frame(["AAPL", "VTI"], "2015-01-01", "2024-12-31")
    prices.tail()
    return (prices,)


@app.cell
def _(prices):
    # Normalize each series to start at 1.0 to compare growth on the same scale.
    normalized = prices / prices.iloc[0]
    normalized.tail()
    return (normalized,)


@app.cell
def _(alt, normalized, pd):
    # Growth-of-$1 chart.
    long = normalized.reset_index().melt("Date", var_name="ticker", value_name="growth")
    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("growth:Q", title="Growth of $1"),
            color="ticker:N",
        )
        .properties(height=300)
    )
    chart
    return chart, long


@app.cell
def _(prices):
    # Daily returns and annualized volatility -- core pandas financial idioms.
    daily_returns = prices.pct_change().dropna()
    annualized_vol = daily_returns.std() * (252**0.5)
    annualized_vol
    return annualized_vol, daily_returns


@app.cell
def _(daily_returns):
    # 63-day (~one quarter) rolling volatility, annualized.
    rolling_vol = daily_returns.rolling(63).std() * (252**0.5)
    rolling_vol.tail()
    return (rolling_vol,)


if __name__ == "__main__":
    app.run()
