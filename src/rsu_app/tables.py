"""Display formatting for the backtest notebook's results tables."""

import pandas as pd


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
