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

from .backtest import BacktestResult, PerfSeries

TRADING_DAYS_PER_YEAR = 252


def _as_scalar(value: float | pd.Series) -> float:
    """Narrow a quantstats stat (typed ``float | Series``) to a scalar float."""
    if isinstance(value, pd.Series):
        raise TypeError(f"expected a scalar stat, got a Series of length {len(value)}")
    return float(value)


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
    """Cumulative growth of $1: the curve ``(1 + returns).cumprod()``.

    Starts at ``1 + returns[0]``, with no leading 1.0 baseline.
    """
    return (1.0 + returns).cumprod()


def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized (CAGR-equivalent) return, at 252 days/year; NaN if empty."""
    if returns.empty:
        return float("nan")
    return _as_scalar(qs.stats.cagr(returns, periods=TRADING_DAYS_PER_YEAR))


def annualized_volatility(returns: pd.Series) -> float:
    """Sample standard deviation (``ddof=1``) of daily returns, scaled by ``sqrt(252)``."""
    return _as_scalar(qs.stats.volatility(returns, periods=TRADING_DAYS_PER_YEAR, annualize=True))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline from initial capital (negative); NaN if empty.

    ``returns`` must carry a ``DatetimeIndex``, as :func:`time_weighted_returns` produces.
    """
    if returns.empty:
        return float("nan")
    return _as_scalar(qs.stats.max_drawdown(returns))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio, net of ``risk_free_rate``; NaN if volatility is zero."""
    if annualized_volatility(returns) == 0.0:
        return float("nan")  # qs.stats.sharpe yields inf here; we report NaN instead
    return _as_scalar(qs.stats.sharpe(returns, rf=risk_free_rate, periods=TRADING_DAYS_PER_YEAR))


def summarize(result: BacktestResult, risk_free_rate: float) -> pd.Series:
    """Build a one-row summary of return and risk metrics for a strategy.

    Return and risk are measured on the raw market-value basis; the after-tax outcome is
    captured separately by the end-of-run liquidation value and taxes-paid rows below.

    Args:
        result: A completed strategy run.
        risk_free_rate: Annual risk-free rate for the Sharpe ratio.

    Returns:
        A Series of labelled metrics, suitable as one column of a comparison table.
    """
    returns = time_weighted_returns(result.market)
    return pd.Series(
        {
            "Final portfolio value": float(result.market.values.iloc[-1]),
            "Liquidation value (net of tax)": result.liquidation_value,
            "Vested contributions (net of tax)": float(result.vested_contributions.sum()),
            "Taxes paid": float(result.taxes_paid.sum()),
            "Annualized return (TWR)": annualized_return(returns),
            "Annualized volatility": annualized_volatility(returns),
            "Max drawdown": max_drawdown(returns),
            "Sharpe ratio": sharpe_ratio(returns, risk_free_rate),
            "End employer %": float(result.employer_fraction.iloc[-1]),
        },
        name=result.name,
    )


def comparison_table(results: dict[str, BacktestResult], risk_free_rate: float) -> pd.DataFrame:
    """Stack per-strategy summaries into one comparison table (strategies as columns)."""
    return pd.DataFrame(
        {name: summarize(result, risk_free_rate) for name, result in results.items()}
    )
