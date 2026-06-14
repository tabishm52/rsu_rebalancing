"""Rebalancing rules: the threshold strategy and two comparison baselines.

Each rule decides what trades to make on a given day, given the portfolio and the
day's market facts (:class:`TradingDay`). They share the same simulation engine
(:mod:`rsu_rebalancing.simulate`), so they differ only in this small ``step`` method.
"""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .portfolio import Portfolio, TradeRecord


@dataclass(frozen=True)
class TradingDay:
    """The immutable market facts of one trading day, passed to each rule's ``step``.

    Attributes:
        date: The trading date.
        employer_price: Employer share price on this day.
        index_price: Index share price on this day.
        grant_dollars: Dollars vesting today, or ``None`` if this is not a grant day.
        is_rebalance_day: Whether the rule may rebalance today.
    """

    date: pd.Timestamp
    employer_price: float
    index_price: float
    grant_dollars: float | None
    is_rebalance_day: bool


class RebalanceRule(Protocol):
    """A strategy's per-day decision logic."""

    name: str  # human-readable label; keys the results dict and labels plots/tables

    def step(self, portfolio: Portfolio, day: TradingDay) -> list[TradeRecord]:
        """Apply this rule's trades for one day, mutating ``portfolio`` in place.

        Args:
            portfolio: Holdings to act on; grants and sales mutate it in place.
            day: The day's market facts (prices, any grant, rebalance flag).

        Returns:
            One :class:`TradeRecord` per trade executed today, in execution order
            (a grant before any sale); empty if the rule did nothing.
        """
        ...


class ThresholdRebalance:
    """Trim employer stock to a target fraction on each rebalance day."""

    def __init__(self, threshold: float, capital_gains_rate: float = 0.0) -> None:
        """Store the target fraction and tax rate.

        Args:
            threshold: Target maximum employer fraction; rebalances trim down to this.
            capital_gains_rate: Tax rate applied to realized gains on each sale.
        """
        self.threshold = threshold
        self.capital_gains_rate = capital_gains_rate
        self.name = f"Threshold {threshold:.0%}"

    def step(self, portfolio: Portfolio, day: TradingDay) -> list[TradeRecord]:
        """Vest any grant, then trim to the threshold if this is a rebalance day."""
        trades: list[TradeRecord] = []

        if day.grant_dollars is not None:
            trades.append(portfolio.add_grant(day.date, day.grant_dollars, day.employer_price))

        if day.is_rebalance_day:
            trade = portfolio.sell_employer_to_fraction(
                day.date,
                self.threshold,
                day.employer_price,
                day.index_price,
                self.capital_gains_rate,
            )
            if trade is not None:
                trades.append(trade)

        return trades


class HoldEverything:
    """Never sell: employer grants simply accumulate (maximum concentration baseline).

    Behaviorally this is the degenerate ``ThresholdRebalance(threshold=1.0)`` case (trim
    to 100% will never sell), but it is kept as its own rule so the baseline is explicit.
    """

    name = "Hold everything"

    def step(self, portfolio: Portfolio, day: TradingDay) -> list[TradeRecord]:
        """Vest any grant; never sell."""
        if day.grant_dollars is None:
            return []
        return [portfolio.add_grant(day.date, day.grant_dollars, day.employer_price)]


class SellAllAtVest:
    """Diversify fully: convert each grant to the index immediately.

    The full-diversification baseline -- holds no employer stock at all.
    """

    name = "Sell all at vest"

    def __init__(self, capital_gains_rate: float = 0.0) -> None:
        """Store the tax rate (gains are ~0 at vest, so tax is usually negligible).

        Args:
            capital_gains_rate: Tax rate applied to realized gains on each sale.
        """
        self.capital_gains_rate = capital_gains_rate

    def step(self, portfolio: Portfolio, day: TradingDay) -> list[TradeRecord]:
        """Vest any grant, then sell the entire employer position into the index."""
        if day.grant_dollars is None:
            return []

        trades = [portfolio.add_grant(day.date, day.grant_dollars, day.employer_price)]

        trade = portfolio.sell_employer_to_fraction(
            day.date, 0.0, day.employer_price, day.index_price, self.capital_gains_rate
        )
        if trade is not None:
            trades.append(trade)

        return trades
