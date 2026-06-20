"""The backtest engine: walk the price timeline applying a rule, day by day.

This ties together prices (:mod:`rsu_rebalancing.data`), the calendar
(:mod:`rsu_rebalancing.calendar`), the portfolio, and a rule
(:mod:`rsu_rebalancing.strategy`) to produce a per-day value series and a trade log.
"""

from dataclasses import asdict, dataclass, field

import pandas as pd

from .calendar import rebalance_trade_dates
from .config import BacktestConfig, GrantConfig, StrategyConfig, TaxConfig
from .data import get_price_frame, get_prices
from .portfolio import Portfolio
from .strategy import HoldEverything, RebalanceRule, SellAllAtVest, ThresholdRebalance, TradingDay
from .vesting import VestingSchedule, build_vesting_schedule


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
class BacktestResult:
    """The output of one strategy run over the backtest window.

    Attributes:
        name: Human-readable strategy name.
        description: Standalone legend label spelling out the target mix for this run.
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
    description: str
    market: PerfSeries = field(default_factory=PerfSeries)
    net_of_tax: PerfSeries = field(default_factory=PerfSeries)
    employer_fraction: pd.Series = field(default_factory=_empty_series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_portfolio: Portfolio = field(default_factory=Portfolio)

    @property
    def vested_contributions(self) -> pd.Series:
        """Net vested contribution dollars per day, recovered from the trade log.

        The dollar value of the shares actually vested into the portfolio, summed per day
        and aligned to ``market.values.index`` (zero-filled). This is net of vest-time
        withholding (sell-to-cover), and excludes any subsequent gains or losses. Recomputed
        on each access.
        """
        grants = self.trades.loc[self.trades["kind"] == "grant", ["date", "traded_value"]]
        by_day = grants.groupby("date")["traded_value"].sum()
        return by_day.reindex(self.market.values.index, fill_value=0.0)

    @property
    def taxes_paid(self) -> pd.Series:
        """Taxes actually paid per day, recovered from the trade log.

        Every row's ``tax_paid``, summed per day and aligned to ``market.values.index``
        (zero-filled). Recomputed on each access.
        """
        by_day = self.trades.groupby("date")["tax_paid"].sum()
        return by_day.reindex(self.market.values.index, fill_value=0.0)


def run_rule(
    prices: pd.DataFrame,
    employer_ticker: str,
    index_ticker: str,
    vesting: VestingSchedule,
    rebalance_days: list[pd.Timestamp],
    rule: RebalanceRule,
    tax_config: TaxConfig | None = None,
) -> BacktestResult:
    """Run a single rule over a price frame.

    Args:
        prices: Aligned daily prices with one column per ticker.
        employer_ticker: Employer-stock column in ``prices`` (assumed upper-cased).
        index_ticker: Diversified-index column in ``prices`` (assumed upper-cased).
        vesting: The shares vesting on each trading day.
        rebalance_days: Trading days on which the rule may rebalance.
        rule: The strategy logic to apply each day.
        tax_config: Rates for the hypothetical end-of-run liquidation tax (the same rates
            across all strategies, for a fair after-tax comparison).

    Returns:
        A :class:`BacktestResult` for this rule.
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
        grant_shares = vesting.get(date)

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
            grant_shares=grant_shares,
            is_rebalance_day=date in rebalance_set,
        )
        records.extend(rule.step(portfolio, day, tax_config))

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
    return BacktestResult(
        name=rule.name,
        description=rule.describe(employer_ticker, index_ticker),
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
    grant_config: GrantConfig,
    backtest: BacktestConfig,
) -> dict[str, BacktestResult]:
    """Run the threshold strategy and both baselines over identical grants and prices.

    :func:`~rsu_rebalancing.data.get_price_frame` trims the simulation window to where both
    tickers trade, so one that IPO'd after ``backtest.start`` shifts the start forward --
    intentional, and identical across strategies.

    Args:
        strategy: Strategy parameters (tickers, threshold, trade-day offsets, tax).
        grant_config: The annual award stream and its vesting terms.
        backtest: The date window and risk-free rate.

    Returns:
        A dict keyed by strategy name, mapping to each :class:`BacktestResult`. The
        threshold strategy's result is always present under its threshold-labelled key.

    Raises:
        ValueError: If a grant predates the employer's price history (see
            :func:`~rsu_rebalancing.vesting.build_vesting_schedule`).
    """
    prices = get_price_frame(
        [strategy.employer_ticker, strategy.index_ticker], backtest.start, backtest.end
    )

    trading_days = pd.DatetimeIndex(prices.index)

    # Grants can predate the window; fetch employer prices back to the earliest grant so
    # each grant's share count can be locked at its grant-date price.
    award_prices = get_prices(
        strategy.employer_ticker, grant_config.earliest_grant_date, backtest.end
    )
    vesting = build_vesting_schedule(grant_config, award_prices, trading_days)
    rebalance_days = rebalance_trade_dates(
        trading_days, strategy.rebalances_per_quarter, backtest.start, backtest.end
    )

    rules: list[RebalanceRule] = [
        HoldEverything(),
        ThresholdRebalance(strategy.threshold, strategy.rebalance_band),
        SellAllAtVest(),
    ]
    return {
        rule.name: run_rule(
            prices,
            strategy.employer_ticker,
            strategy.index_ticker,
            vesting,
            rebalance_days,
            rule,
            strategy.tax_config,
        )
        for rule in rules
    }
