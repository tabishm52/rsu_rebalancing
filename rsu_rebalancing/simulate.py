"""The simulation engine: walk the price timeline applying a rule, day by day.

This ties together prices (:mod:`rsu_rebalancing.data`), the calendar
(:mod:`rsu_rebalancing.calendar`), the portfolio, and a rule
(:mod:`rsu_rebalancing.strategy`) to produce a per-day value series and a trade log.
"""

from dataclasses import asdict, dataclass, field

import pandas as pd

from .calendar import grant_trade_dates, rebalance_trade_dates
from .config import GrantSchedule, SimConfig, StrategyConfig, TaxConfig
from .data import get_price_frame
from .portfolio import Portfolio
from .strategy import HoldEverything, RebalanceRule, SellAllAtVest, ThresholdRebalance, TradingDay


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float)


@dataclass
class PerfSeries:
    """A performance basis: a daily value series paired with the flows to strip from it.

    Time-weighted returns are ``(values - flows) / values.shift(1) - 1``, so the two must
    travel together. A *flow* is a non-market change in value -- money in or out, or (on
    the net basis) a tax-status change -- that is not investment performance.

    Attributes:
        values: Daily portfolio value on this basis (market or net-of-tax).
        flows: Daily non-market change in ``values``, signed (inflow positive).
    """

    values: pd.Series = field(default_factory=_empty_series)
    flows: pd.Series = field(default_factory=_empty_series)


@dataclass
class SimResult:
    """The output of one strategy run over the backtest window.

    Attributes:
        name: Human-readable strategy name.
        market: Raw market-value basis. ``market.values`` may begin with zero-value days
            before the first grant lands; ``metrics.time_weighted_returns`` neutralizes
            these, so they carry no spurious return.
        net_of_tax: Net-of-tax (liquidation-value) basis. Puts strategies that realized gains
            along the way on an even footing with those carrying large unrealized gains,
            and strips the short-to-long-term tax-status drift as a flow.
        employer_fraction: Daily employer-stock share of total holdings.
        trades: Audit log of every grant and sale, one row each.
        final_portfolio: The portfolio state at the end of the run.
    """

    name: str
    market: PerfSeries = field(default_factory=PerfSeries)
    net_of_tax: PerfSeries = field(default_factory=PerfSeries)
    employer_fraction: pd.Series = field(default_factory=_empty_series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_portfolio: Portfolio = field(default_factory=Portfolio)

    @property
    def gross_grants(self) -> pd.Series:
        """Gross grant dollars per day, recovered from the trade log.

        Grant rows' gross value, summed per day and aligned to ``market.values.index``
        (zero-filled). This is the total-contributed figure for reporting -- distinct from
        ``market.flows``, which also nets out tax paid. Recomputed on each access; cheap
        over the in-memory log, and never stale.
        """
        grants = self.trades.loc[self.trades["kind"] == "grant", ["date", "gross_value"]]
        by_day = grants.groupby("date")["gross_value"].sum()
        return by_day.reindex(self.market.values.index, fill_value=0.0)


def run_rule(
    prices: pd.DataFrame,
    employer_ticker: str,
    index_ticker: str,
    grants: dict[pd.Timestamp, float],
    rebalance_days: list[pd.Timestamp],
    rule: RebalanceRule,
    tax_config: TaxConfig | None = None,
) -> SimResult:
    """Run a single rule over a price frame.

    Args:
        prices: Aligned daily prices with one column per ticker.
        employer_ticker: Employer-stock column in ``prices`` (assumed upper-cased).
        index_ticker: Diversified-index column in ``prices`` (assumed upper-cased).
        grants: Mapping of trading day to grant dollars.
        rebalance_days: Trading days on which the rule may rebalance.
        rule: The strategy logic to apply each day.
        tax_config: Rates for the hypothetical end-of-run liquidation tax (the same rates
            across all strategies, for a fair after-tax comparison).

    Returns:
        A :class:`SimResult` for this rule.
    """
    if tax_config is None:
        tax_config = TaxConfig()

    employer = prices[employer_ticker]
    index = prices[index_ticker]
    rebalance_set = set(rebalance_days)

    portfolio = Portfolio()
    records = []
    market_values: dict[pd.Timestamp, float] = {}
    market_flows: dict[pd.Timestamp, float] = {}
    net_values: dict[pd.Timestamp, float] = {}
    net_flows: dict[pd.Timestamp, float] = {}
    fractions: dict[pd.Timestamp, float] = {}
    prev_date: pd.Timestamp | None = None

    for date in prices.index:
        emp_price = float(employer.loc[date])
        idx_price = float(index.loc[date])
        grant_dollars = grants.get(date)

        # Compute portfolio value at today's prices before the step; the net-of-tax value
        # uses yesterday's date so a change from short- to long-term tax status is seen as a flow
        ref_date = prev_date if prev_date is not None else date
        market_before = portfolio.market_value(emp_price, idx_price)
        net_before = portfolio.liquidation_value(emp_price, idx_price, ref_date, tax_config)

        # Step the strategy one day forward: apply any grants or rebalance, mutating the portfolio
        day = TradingDay(
            date=date,
            employer_price=emp_price,
            index_price=idx_price,
            grant_dollars=grant_dollars,
            is_rebalance_day=date in rebalance_set,
        )
        records.extend(rule.step(portfolio, day))

        # Compute portfolio value after the step at today's prices
        market_after = portfolio.market_value(emp_price, idx_price)
        net_after = portfolio.liquidation_value(emp_price, idx_price, date, tax_config)

        # Non-market flows = value after the step - value before the step
        market_values[date] = market_after
        market_flows[date] = market_after - market_before
        net_values[date] = net_after
        net_flows[date] = net_after - net_before
        fractions[date] = portfolio.employer_fraction(emp_price, idx_price)
        prev_date = date

    trades = pd.DataFrame([asdict(r) for r in records])
    return SimResult(
        name=rule.name,
        market=PerfSeries(
            values=pd.Series(market_values, name=rule.name),
            flows=pd.Series(market_flows, name=rule.name),
        ),
        net_of_tax=PerfSeries(
            values=pd.Series(net_values, name=rule.name),
            flows=pd.Series(net_flows, name=rule.name),
        ),
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
            strategy.tax_config,
        )
        for rule in rules
    }
