"""Rebalancing rules: the threshold strategy and two comparison baselines.

Each rule decides what trades to make on a given day, given the portfolio and the
day's market facts (:class:`TradingDay`). They share the same backtest engine
(:mod:`rsu_rebalancing.backtest`), so they differ only in this small ``step`` method.
"""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .config import TaxConfig
from .portfolio import Portfolio, TradeRecord


@dataclass(frozen=True)
class TradingDay:
    """The immutable market facts of one trading day, passed to each rule's ``step``.

    Attributes:
        date: The trading date.
        employer_price: Employer share price on this day.
        index_price: Index share price on this day.
        grant_shares: Employer shares vesting today, or ``None`` if not a vest day.
        is_rebalance_day: Whether the rule may rebalance today.
    """

    date: pd.Timestamp
    employer_price: float
    index_price: float
    grant_shares: float | None
    is_rebalance_day: bool


class RebalanceRule(Protocol):
    """A strategy's per-day decision logic."""

    name: str  # human-readable label; keys the results dict and labels plots/tables

    def describe(self, employer_ticker: str, index_ticker: str) -> str:
        """A standalone legend label spelling out the target mix.

        Tickers are passed in rather than stored on the rule: a rule is ticker-agnostic
        (the same threshold runs against any employer/index pair), so the label is only
        resolved once the run binds it to a concrete pair.
        """
        ...

    def step(
        self, portfolio: Portfolio, day: TradingDay, tax_config: TaxConfig
    ) -> list[TradeRecord]:
        """Apply this rule's trades for one day, mutating ``portfolio`` in place.

        Args:
            portfolio: Holdings to act on; grants and sales mutate it in place.
            day: The day's market facts (prices, any grant, rebalance flag).
            tax_config: Tax rates for any trade the rule makes.

        Returns:
            One :class:`TradeRecord` per trade executed today, in execution order
            (a grant before any sale); empty if the rule did nothing.
        """
        ...


class ThresholdRebalance:
    """Trim employer stock to a target fraction on each rebalance day."""

    def __init__(self, threshold: float, band: float) -> None:
        """Store the target fraction and hysteresis band.

        Args:
            threshold: Target maximum employer fraction; rebalances trim down to this.
            band: One-way hysteresis band, in fraction points. A rebalance fires only
                once the employer fraction exceeds ``threshold + band``.
        """
        self.threshold = threshold
        self.band = band
        self.name = f"Threshold {threshold:.0%}"

    def describe(self, employer_ticker: str, index_ticker: str) -> str:
        """Legend label for the target split."""
        return (
            f"Threshold: {self.threshold:.0%} {employer_ticker} / "
            f"{1 - self.threshold:.0%} {index_ticker}"
        )

    def step(
        self, portfolio: Portfolio, day: TradingDay, tax_config: TaxConfig
    ) -> list[TradeRecord]:
        """Vest any grant, then trim to the threshold if past the band on a rebalance day."""
        trades: list[TradeRecord] = []

        if day.grant_shares is not None:
            trades.append(
                portfolio.add_grant(day.date, day.grant_shares, day.employer_price, tax_config)
            )

        if day.is_rebalance_day:
            fraction = portfolio.employer_fraction(day.employer_price, day.index_price)
            if fraction > self.threshold + self.band:
                trade = portfolio.sell_employer_to_fraction(
                    day.date,
                    self.threshold,
                    day.employer_price,
                    day.index_price,
                    tax_config,
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

    def describe(self, employer_ticker: str, index_ticker: str) -> str:
        """Legend label for the all-employer baseline."""
        return f"Hold everything: 100% {employer_ticker}"

    def step(
        self, portfolio: Portfolio, day: TradingDay, tax_config: TaxConfig
    ) -> list[TradeRecord]:
        """Vest any grant (withholding sell-to-cover); never sell otherwise."""
        if day.grant_shares is None:
            return []
        return [portfolio.add_grant(day.date, day.grant_shares, day.employer_price, tax_config)]


class SellAllAtVest:
    """Diversify fully: convert each grant to the index immediately.

    The full-diversification baseline -- holds no employer stock at all.
    """

    name = "Sell all at vest"

    def describe(self, employer_ticker: str, index_ticker: str) -> str:
        """Legend label for the fully-diversified baseline."""
        return f"Sell all at vest: 100% {index_ticker}"

    def step(
        self, portfolio: Portfolio, day: TradingDay, tax_config: TaxConfig
    ) -> list[TradeRecord]:
        """Vest any grant, then sell the entire employer position into the index."""
        if day.grant_shares is None:
            return []

        trades = [portfolio.add_grant(day.date, day.grant_shares, day.employer_price, tax_config)]

        trade = portfolio.sell_employer_to_fraction(
            day.date, 0.0, day.employer_price, day.index_price, tax_config
        )
        if trade is not None:
            trades.append(trade)

        return trades
