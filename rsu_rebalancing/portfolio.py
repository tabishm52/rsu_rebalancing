"""Portfolio bookkeeping: employer tax lots, an index position, and trade actions.

The portfolio holds two assets: concentrated employer stock (tracked as FIFO cost
lots so realized gains can be taxed) and a diversified index fund (proceeds of any
sale flow here). It is a plain mutable object driven by the simulation engine; it
holds no dates or prices of its own.
"""

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .config import TaxConfig

# Shares below this are treated as zero, to avoid float dust accumulating in lots.
_SHARE_EPS = 1e-9


@dataclass
class TaxLot:
    """A block of employer shares acquired at one cost basis (one vesting event).

    Attributes:
        shares: Number of employer shares in the lot.
        cost_per_share: Per-share cost basis (the vest-day price); the lot's total
            basis is ``shares * cost_per_share``.
        acquisition_date: Vest date of the lot, used to classify a sale's realized gain
            as short- or long-term by holding period.
    """

    shares: float
    cost_per_share: float
    acquisition_date: pd.Timestamp


@dataclass
class TradeRecord:
    """An audit-log row for a single grant or sale.

    Attributes:
        date: Trade date.
        kind: What the trade represents.
        employer_shares: Employer shares transacted, signed: ``+`` on a grant, ``-`` on a sale.
        employer_price: Employer share price at the trade.
        gross_value: Absolute dollar value transacted in employer stock.
        tax_paid: Capital-gains tax paid out of the sale proceeds.
        index_dollars_in: Net dollars moved into the index position.
    """

    date: pd.Timestamp
    kind: Literal["grant", "rebalance", "liquidate"]
    employer_shares: float
    employer_price: float
    gross_value: float
    tax_paid: float
    index_dollars_in: float


@dataclass
class Portfolio:
    """Holdings in employer stock (as cost lots) plus a diversified index position.

    Attributes:
        employer_lots: FIFO cost lots of employer stock, oldest first.
        index_shares: Shares of the diversified index. Tracked as a plain count with no
            cost basis, since the index is only ever bought, never sold.
    """

    employer_lots: list[TaxLot] = field(default_factory=list)
    index_shares: float = 0.0

    @property
    def employer_shares(self) -> float:
        """Total employer shares held across all lots."""
        return sum(lot.shares for lot in self.employer_lots)

    def employer_value(self, employer_price: float) -> float:
        """Market value of the employer position at ``employer_price``."""
        return self.employer_shares * employer_price

    def index_value(self, index_price: float) -> float:
        """Market value of the index position at ``index_price``."""
        return self.index_shares * index_price

    def total_value(self, employer_price: float, index_price: float) -> float:
        """Combined market value of both positions."""
        return self.employer_value(employer_price) + self.index_value(index_price)

    def employer_fraction(self, employer_price: float, index_price: float) -> float:
        """Fraction of total holdings in employer stock (0 when the portfolio is empty)."""
        total = self.total_value(employer_price, index_price)
        if total <= 0.0:
            return 0.0
        return self.employer_value(employer_price) / total

    def add_grant(self, date: pd.Timestamp, dollars: float, employer_price: float) -> TradeRecord:
        """Vest ``dollars`` of employer stock at ``employer_price`` as a new lot."""
        shares = dollars / employer_price
        self.employer_lots.append(
            TaxLot(shares=shares, cost_per_share=employer_price, acquisition_date=date)
        )
        return TradeRecord(
            date=date,
            kind="grant",
            employer_shares=shares,
            employer_price=employer_price,
            gross_value=dollars,
            tax_paid=0.0,
            index_dollars_in=0.0,
        )

    def _sell_employer_shares(
        self, shares: float, price: float, sale_date: pd.Timestamp, tax_config: TaxConfig
    ) -> tuple[float, float]:
        """Sell ``shares`` FIFO across lots; return (gross proceeds, tax owed).

        Each lot is taxed at the long- or short-term rate depending on how long it was
        held before ``sale_date``; gains are assessed per lot and losses are ignored.
        """
        remaining = shares
        proceeds = 0.0
        tax_owed = 0.0
        while remaining > _SHARE_EPS and self.employer_lots:
            # Take from the oldest lot: the whole lot, or just what's left to sell.
            lot = self.employer_lots[0]
            take = min(lot.shares, remaining)

            # Record the sale value and assess tax on this lot's realized gain at its
            # holding-period rate, ignoring losses.
            proceeds += take * price
            lot_gain = take * (price - lot.cost_per_share)
            long_term = (sale_date - lot.acquisition_date).days > tax_config.long_term_days
            rate = tax_config.long_term_rate if long_term else tax_config.short_term_rate
            tax_owed += max(lot_gain, 0.0) * rate

            # Shrink the lot and the outstanding ask.
            lot.shares -= take
            remaining -= take

            # Evict the lot once it's drained.
            if lot.shares <= _SHARE_EPS:
                self.employer_lots.pop(0)

        return proceeds, tax_owed

    def sell_employer_to_fraction(
        self,
        date: pd.Timestamp,
        target_fraction: float,
        employer_price: float,
        index_price: float,
        tax_config: TaxConfig,
    ) -> TradeRecord | None:
        """Trim employer stock down to ``target_fraction`` of holdings, buying the index.

        The excess is measured at current market prices (pre-tax). Any capital-gains
        tax on the realized gain is paid out of the proceeds, and the remainder is
        reinvested in the index. Selling everything corresponds to ``target_fraction=0``.

        Args:
            date: Trade date, recorded on the returned audit row and used as the sale
                date when classifying each lot's gain as short- or long-term.
            target_fraction: Desired maximum employer fraction after the trade.
            employer_price: Current employer share price.
            index_price: Current index share price.
            tax_config: Capital-gains tax rates applied to realized gains.

        Returns:
            A :class:`TradeRecord` if a sale happened, else ``None`` (already at or
            below target).
        """
        employer_val = self.employer_value(employer_price)
        total = employer_val + self.index_value(index_price)
        if total <= 0.0:
            return None

        # Dollar value of employer stock above the target fraction, sized pre-tax.
        sell_value = employer_val - target_fraction * total
        if sell_value <= _SHARE_EPS * employer_price:
            return None  # already at or below the target fraction

        shares_to_sell = sell_value / employer_price
        proceeds, tax_paid = self._sell_employer_shares(
            shares_to_sell, employer_price, date, tax_config
        )

        # Pay capital-gains tax out of the proceeds; reinvest the remainder in the index.
        net = proceeds - tax_paid
        self.index_shares += net / index_price

        return TradeRecord(
            date=date,
            kind="liquidate" if target_fraction == 0.0 else "rebalance",
            employer_shares=-shares_to_sell,
            employer_price=employer_price,
            gross_value=proceeds,
            tax_paid=tax_paid,
            index_dollars_in=net,
        )
