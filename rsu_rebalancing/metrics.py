"""Return and risk metrics for a backtest run.

Because grants add money over time, the raw portfolio value is *not* a clean return
series -- a jump on a grant day is a deposit, not performance. These functions compute
a **time-weighted return** (TWR): each day's return strips out that day's contribution,
so risk and return reflect the strategy's investment performance, not the contribution
schedule. Final dollar value is still reported directly, and is a fair head-to-head
number because every strategy receives the identical grant stream.
"""

import numpy as np
import pandas as pd
import quantstats as qs

from .simulate import PerfSeries, SimResult

TRADING_DAYS_PER_YEAR = 252


def time_weighted_returns(perf: PerfSeries) -> pd.Series:
    """Daily time-weighted returns on a performance basis, removing the effect of flows.

    The return for day *t* is ``(value_t - flow_t) / value_{t-1} - 1`` so the external cash
    flow on day *t* is not counted as performance. Flows are signed: a grant (deposit) is
    positive, tax paid out (a withdrawal) is negative.

    Args:
        perf: A performance basis (``values`` paired with the ``flows`` to strip).

    Returns:
        A Series of daily returns, starting from the second day.
    """
    prev = perf.values.shift(1)
    returns = (perf.values - perf.flows) / prev - 1.0
    return returns.iloc[1:].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def growth_of_one(returns: pd.Series) -> pd.Series:
    """Cumulative growth of $1 invested, given a daily return series.

    Args:
        returns: A daily return series.

    Returns:
        The curve ``(1 + returns).cumprod()``. Note it starts at ``1 + returns[0]``,
        with no leading 1.0 baseline.
    """
    return (1.0 + returns).cumprod()


def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized (CAGR-equivalent) time-weighted return.

    Args:
        returns: Daily time-weighted returns.

    Returns:
        The CAGR, annualized at 252 trading days per year, or NaN if ``returns`` is empty.
    """
    if returns.empty:
        return float("nan")
    return float(qs.stats.cagr(returns, periods=TRADING_DAYS_PER_YEAR))


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns.

    Args:
        returns: Daily time-weighted returns.

    Returns:
        The sample standard deviation (``ddof=1``) scaled by ``sqrt(252)``.
    """
    return float(qs.stats.volatility(returns, periods=TRADING_DAYS_PER_YEAR, annualize=True))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline, measured from initial capital (a negative number).

    Args:
        returns: Daily time-weighted returns. Must carry a ``DatetimeIndex``, as the
            series from :func:`time_weighted_returns` does.

    Returns:
        The maximum drawdown as a negative fraction, or NaN if ``returns`` is empty.
    """
    if returns.empty:
        return float("nan")
    return float(qs.stats.max_drawdown(returns))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio of the return series.

    Args:
        returns: Daily time-weighted returns.
        risk_free_rate: Annual risk-free rate to subtract from the return.

    Returns:
        The Sharpe ratio, or NaN if volatility is zero.
    """
    if annualized_volatility(returns) == 0.0:
        return float("nan")  # qs.stats.sharpe yields inf here; we report NaN instead
    return float(qs.stats.sharpe(returns, rf=risk_free_rate, periods=TRADING_DAYS_PER_YEAR))


def summarize(result: SimResult, risk_free_rate: float, after_tax: bool) -> pd.Series:
    """Build a one-row summary of return and risk metrics for a strategy.

    Args:
        result: A completed strategy run.
        risk_free_rate: Annual risk-free rate for the Sharpe ratio.
        after_tax: Measure return and risk on the net-of-tax basis when True, else on the
            raw market-value basis.

    Returns:
        A Series of labelled metrics, suitable as one column of a comparison table.
    """
    perf = result.net_of_tax if after_tax else result.market
    returns = time_weighted_returns(perf)
    return pd.Series(
        {
            "Final portfolio value": float(result.market.values.iloc[-1]),
            "Liquidation value (net of tax)": float(result.net_of_tax.values.iloc[-1]),
            "Total contributed": float(result.gross_grants.sum()),
            "Ann. return (TWR)": annualized_return(returns),
            "Ann. volatility": annualized_volatility(returns),
            "Max drawdown": max_drawdown(returns),
            "Sharpe": sharpe_ratio(returns, risk_free_rate),
            "End employer %": float(result.employer_fraction.iloc[-1]),
        },
        name=result.name,
    )


def comparison_table(
    results: dict[str, SimResult], risk_free_rate: float, after_tax: bool
) -> pd.DataFrame:
    """Stack per-strategy summaries into one comparison table (strategies as columns)."""
    return pd.DataFrame(
        {
            name: summarize(result, risk_free_rate, after_tax=after_tax)
            for name, result in results.items()
        }
    )
