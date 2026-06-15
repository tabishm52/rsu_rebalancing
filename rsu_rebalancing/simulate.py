"""The simulation engine: walk the price timeline applying a rule, day by day.

This ties together prices (:mod:`rsu_rebalancing.data`), the calendar
(:mod:`rsu_rebalancing.calendar`), the portfolio, and a rule
(:mod:`rsu_rebalancing.strategy`) to produce a per-day value series and a trade log.
"""

from dataclasses import asdict, dataclass

import pandas as pd

from .calendar import grant_trade_dates, rebalance_trade_dates
from .config import GrantSchedule, SimConfig, StrategyConfig
from .data import get_price_frame
from .portfolio import Portfolio
from .strategy import HoldEverything, RebalanceRule, SellAllAtVest, ThresholdRebalance, TradingDay


@dataclass
class SimResult:
    """The output of one strategy run over the backtest window.

    Attributes:
        name: Human-readable strategy name.
        values: Daily total portfolio value, indexed by trading date.
        employer_fraction: Daily employer-stock share of total holdings.
        contributions: External cash added each day (grant dollars; 0 otherwise).
            Identical across strategies, but carried here so metrics are self-contained.
        trades: Audit log of every grant and sale, one row each.
        final_portfolio: The portfolio state at the end of the run.
    """

    name: str
    values: pd.Series
    employer_fraction: pd.Series
    contributions: pd.Series
    trades: pd.DataFrame
    final_portfolio: Portfolio


def run_rule(
    prices: pd.DataFrame,
    employer_ticker: str,
    index_ticker: str,
    grants: dict[pd.Timestamp, float],
    rebalance_days: list[pd.Timestamp],
    rule: RebalanceRule,
) -> SimResult:
    """Run a single rule over a price frame.

    Args:
        prices: Aligned daily prices with one column per ticker.
        employer_ticker: Column name of the employer stock.
        index_ticker: Column name of the diversified index.
        grants: Mapping of trading day to grant dollars.
        rebalance_days: Trading days on which the rule may rebalance.
        rule: The strategy logic to apply each day.

    Returns:
        A :class:`SimResult` for this rule.
    """
    employer = prices[employer_ticker.upper()]
    index = prices[index_ticker.upper()]
    rebalance_set = set(rebalance_days)

    portfolio = Portfolio()
    records = []
    values: dict[pd.Timestamp, float] = {}
    fractions: dict[pd.Timestamp, float] = {}
    contributions: dict[pd.Timestamp, float] = {}

    for date in prices.index:
        emp_price = float(employer.loc[date])
        idx_price = float(index.loc[date])
        grant_dollars = grants.get(date)

        day = TradingDay(
            date=date,
            employer_price=emp_price,
            index_price=idx_price,
            grant_dollars=grant_dollars,
            is_rebalance_day=date in rebalance_set,
        )
        records.extend(rule.step(portfolio, day))

        values[date] = portfolio.total_value(emp_price, idx_price)
        fractions[date] = portfolio.employer_fraction(emp_price, idx_price)
        contributions[date] = grant_dollars or 0.0

    trades = pd.DataFrame([asdict(r) for r in records])
    return SimResult(
        name=rule.name,
        values=pd.Series(values, name=rule.name),
        employer_fraction=pd.Series(fractions, name=rule.name),
        contributions=pd.Series(contributions, name="contributions"),
        trades=trades,
        final_portfolio=portfolio,
    )


def run_backtest(
    strategy: StrategyConfig,
    schedule: GrantSchedule,
    sim: SimConfig,
) -> dict[str, SimResult]:
    """Run the threshold strategy and both baselines over identical grants and prices.

    Args:
        strategy: Strategy parameters (tickers, threshold, trade-day offsets, tax).
        schedule: The annual grant stream.
        sim: The date window and risk-free rate.

    Returns:
        A dict keyed by strategy name, mapping to each :class:`SimResult`. The
        threshold strategy's result is always present under its threshold-labelled key.
    """
    prices = get_price_frame([strategy.employer_ticker, strategy.index_ticker], sim.start, sim.end)
    # Trim the leading window before every ticker is trading: a ticker that IPO'd after
    # sim.start has NaNs that ffill can't backfill, which would corrupt valuations. After
    # get_price_frame's ffill the only NaNs left are leading, so dropna trims just those.
    prices = prices.dropna()

    trading_days = pd.DatetimeIndex(prices.index)

    grants = grant_trade_dates(trading_days, schedule)
    rebalance_days = rebalance_trade_dates(
        trading_days, strategy.rebalances_per_quarter, sim.start, sim.end
    )

    rules: list[RebalanceRule] = [
        ThresholdRebalance(strategy.threshold, strategy.capital_gains_rate),
        HoldEverything(),
        SellAllAtVest(strategy.capital_gains_rate),
    ]
    return {
        rule.name: run_rule(
            prices,
            strategy.employer_ticker,
            strategy.index_ticker,
            grants,
            rebalance_days,
            rule,
        )
        for rule in rules
    }
