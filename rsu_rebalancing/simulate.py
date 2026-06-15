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
        values: Daily total portfolio value, indexed by trading date. May begin with
            zero-value days before the first grant lands; ``metrics.time_weighted_returns``
            neutralizes these, so they carry no spurious return.
        employer_fraction: Daily employer-stock share of total holdings.
        trades: Audit log of every grant and sale, one row each.
        final_portfolio: The portfolio state at the end of the run.
    """

    name: str
    values: pd.Series
    employer_fraction: pd.Series
    trades: pd.DataFrame
    final_portfolio: Portfolio

    @property
    def contributions(self) -> pd.Series:
        """External inflows per day, recovered from the trade log.

        Grants are the only inflow; rebalances move money internally and tax is a cost,
        not a withdrawal. So contributions are the grant rows' gross value, summed per
        day and aligned to ``values.index`` (zero-filled). Recomputed on each access;
        cheap over the in-memory log, and never stale.
        """
        grants = self.trades.loc[self.trades["kind"] == "grant", ["date", "gross_value"]]
        by_day = grants.groupby("date")["gross_value"].sum()
        return by_day.reindex(self.values.index, fill_value=0.0)


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
        employer_ticker: Employer-stock column in ``prices`` (assumed upper-cased).
        index_ticker: Diversified-index column in ``prices`` (assumed upper-cased).
        grants: Mapping of trading day to grant dollars.
        rebalance_days: Trading days on which the rule may rebalance.
        rule: The strategy logic to apply each day.

    Returns:
        A :class:`SimResult` for this rule.
    """
    employer = prices[employer_ticker]
    index = prices[index_ticker]
    rebalance_set = set(rebalance_days)

    portfolio = Portfolio()
    records = []
    values: dict[pd.Timestamp, float] = {}
    fractions: dict[pd.Timestamp, float] = {}

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

    trades = pd.DataFrame([asdict(r) for r in records])
    return SimResult(
        name=rule.name,
        values=pd.Series(values, name=rule.name),
        employer_fraction=pd.Series(fractions, name=rule.name),
        trades=trades,
        final_portfolio=portfolio,
    )


def run_backtest(
    strategy: StrategyConfig,
    schedule: GrantSchedule,
    sim: SimConfig,
) -> dict[str, SimResult]:
    """Run the threshold strategy and both baselines over identical grants and prices.

    :func:`~rsu_rebalancing.data.get_price_frame` trims the window to where every ticker
    trades, so a ticker that IPO'd after ``sim.start`` shifts the start forward and any
    grants nominally before it collapse onto its first trading day -- intentional, and
    identical across strategies.

    Args:
        strategy: Strategy parameters (tickers, threshold, trade-day offsets, tax).
        schedule: The annual grant stream.
        sim: The date window and risk-free rate.

    Returns:
        A dict keyed by strategy name, mapping to each :class:`SimResult`. The
        threshold strategy's result is always present under its threshold-labelled key.
    """
    prices = get_price_frame([strategy.employer_ticker, strategy.index_ticker], sim.start, sim.end)

    trading_days = pd.DatetimeIndex(prices.index)

    grants = grant_trade_dates(trading_days, schedule)
    rebalance_days = rebalance_trade_dates(
        trading_days, strategy.rebalances_per_quarter, sim.start, sim.end
    )

    rules: list[RebalanceRule] = [
        ThresholdRebalance(strategy.threshold, strategy.tax_config),
        HoldEverything(),
        SellAllAtVest(strategy.tax_config),
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
