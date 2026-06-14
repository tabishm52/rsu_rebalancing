"""Rebalancing rules: the threshold strategy and two comparison baselines.

Each rule decides what trades to make on a given day, given the portfolio and whether
that day is a grant day and/or a rebalance day. They share the same simulation engine
(:mod:`rsu_rebalancing.simulate`), so they differ only in this small ``step`` method.
"""

from typing import Protocol

import pandas as pd

from .portfolio import Portfolio, TradeRecord


class RebalanceRule(Protocol):
    """A strategy's per-day decision logic."""

    name: str  # human-readable label; keys the results dict and labels plots/tables

    def step(
        self,
        portfolio: Portfolio,
        date: pd.Timestamp,
        employer_price: float,
        index_price: float,
        grant_dollars: float | None,
        is_rebalance_day: bool,
    ) -> list[TradeRecord]:
        """Apply the day's trades to ``portfolio`` and return their audit rows."""
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

    def step(
        self,
        portfolio: Portfolio,
        date: pd.Timestamp,
        employer_price: float,
        index_price: float,
        grant_dollars: float | None,
        is_rebalance_day: bool,
    ) -> list[TradeRecord]:
        """Vest any grant, then trim to the threshold if this is a rebalance day."""
        trades: list[TradeRecord] = []

        if grant_dollars is not None:
            trades.append(portfolio.add_grant(date, grant_dollars, employer_price))

        if is_rebalance_day:
            trade = portfolio.sell_employer_to_fraction(
                date, self.threshold, employer_price, index_price, self.capital_gains_rate
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

    def step(
        self,
        portfolio: Portfolio,
        date: pd.Timestamp,
        employer_price: float,
        index_price: float,
        grant_dollars: float | None,
        is_rebalance_day: bool,
    ) -> list[TradeRecord]:
        """Vest any grant; never sell."""
        if grant_dollars is None:
            return []
        return [portfolio.add_grant(date, grant_dollars, employer_price)]


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

    def step(
        self,
        portfolio: Portfolio,
        date: pd.Timestamp,
        employer_price: float,
        index_price: float,
        grant_dollars: float | None,
        is_rebalance_day: bool,
    ) -> list[TradeRecord]:
        """Vest any grant, then sell the entire employer position into the index."""
        if grant_dollars is None:
            return []

        trades = [portfolio.add_grant(date, grant_dollars, employer_price)]

        trade = portfolio.sell_employer_to_fraction(
            date, 0.0, employer_price, index_price, self.capital_gains_rate
        )
        if trade is not None:
            trades.append(trade)

        return trades
