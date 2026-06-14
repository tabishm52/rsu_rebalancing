"""Exploratory scratchpad: getting a feel for prices with pandas + yfinance.

Run with: ``uv run marimo edit notebooks/exploration.py``

This is a learning notebook, not part of the package API -- a place to poke at the
data layer and basic pandas financial idioms (returns, normalization, rolling stats)
before the structured simulation in ``rebalance_sim.py``.
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns

    from rsu_rebalancing import get_price_frame

    sns.set_theme()
    return get_price_frame, mo, plt


@app.cell
def _(mo):
    mo.md("""
    # Exploration scratchpad

    A place to learn pandas financial idioms against real prices. Edit freely.
    """)
    return


@app.cell
def _(mo):
    mo.md("## Controls")
    return


@app.cell
def _(mo):
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
    mo.md("## Price data")
    return


@app.cell
def _(end, get_price_frame, mo, start, ticker_a, ticker_b):
    # Two tickers, aligned on common trading days.
    try:
        prices = get_price_frame([ticker_a.value, ticker_b.value], start.value, end.value)
        error = None
    except Exception as exc:  # noqa: BLE001 - surface any data/config error in the UI
        prices = None
        error = str(exc)

    mo.stop(error is not None, mo.md(f"⚠️ **Could not fetch:** {error}"))
    prices.tail()
    return (prices,)


@app.cell
def _(prices):
    # Normalize each series to start at 1.0 to compare growth on the same scale.
    normalized = prices / prices.iloc[0]
    normalized.tail()
    return (normalized,)


@app.cell
def _(mo):
    mo.md("## Cumulative performance")
    return


@app.cell
def _(normalized, plt):
    # Growth chart, rebased to 100 at the start date.
    fig, ax = plt.subplots(figsize=(12, 6))
    (normalized * 100).plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index (start = 100)")
    ax.legend(title="ticker")
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("## Returns & volatility")
    return


@app.cell
def _(prices):
    # Daily returns and annualized volatility -- core pandas financial idioms.
    daily_returns = prices.pct_change().dropna()
    annualized_vol = daily_returns.std() * (252**0.5)
    annualized_vol
    return (daily_returns,)


@app.cell
def _(daily_returns):
    # 63-day (~one quarter) rolling volatility, annualized.
    rolling_vol = daily_returns.rolling(63).std() * (252**0.5)
    rolling_vol.tail()
    return


if __name__ == "__main__":
    app.run()
