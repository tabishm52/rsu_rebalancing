"""Configuration objects for a rebalancing backtest.

These are small, frozen dataclasses that capture *what* to simulate (the grant
stream, the strategy parameters, and the date window). They hold no price data and
do no I/O, which keeps them trivial to construct in tests and notebooks.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GrantSchedule:
    """A stream of equal-dollar RSU grants, one per year on a fixed month/day.

    The dollar amount is the *grant value* on the vest date; the number of employer
    shares it buys depends on the share price that day.

    Attributes:
        annual_dollars: Dollar value granted each year.
        start_year: First calendar year to grant in (inclusive).
        end_year: Last calendar year to grant in (inclusive).
        grant_month: Month of the grant (1-12). Defaults to March.
        grant_day: Day of month for the nominal grant date. The simulator snaps this
            to the first trading day on or after it.
    """

    annual_dollars: float
    start_year: int
    end_year: int
    grant_month: int = 3
    grant_day: int = 1

    def nominal_grant_dates(self) -> list[pd.Timestamp]:
        """Return the nominal (calendar) grant date for each year in the range."""
        return [
            pd.Timestamp(year=year, month=self.grant_month, day=self.grant_day)
            for year in range(self.start_year, self.end_year + 1)
        ]


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters of the one-way threshold rebalancing strategy.

    Attributes:
        employer_ticker: Symbol of the concentrated employer stock.
        index_ticker: Symbol of the diversified fund that sale proceeds buy.
        threshold: Target maximum fraction of total holdings in employer stock
            (e.g. ``1/3``). Rebalances trim employer stock down to this fraction.
        rebalances_per_quarter: Number of evenly spaced rebalances to place in each
            quarter.
        capital_gains_rate: Tax rate applied to realized gains when selling employer
            stock. ``0.0`` disables taxes (cost basis = vest-day price).
    """

    employer_ticker: str
    index_ticker: str = "VTI"
    threshold: float = 1.0 / 3.0
    rebalances_per_quarter: int = 2
    capital_gains_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]; got {self.threshold}")
        if not 0.0 <= self.capital_gains_rate < 1.0:
            raise ValueError(f"capital_gains_rate must be in [0, 1); got {self.capital_gains_rate}")
        if self.rebalances_per_quarter < 1:
            raise ValueError(
                f"rebalances_per_quarter must be >= 1; got {self.rebalances_per_quarter}"
            )


@dataclass(frozen=True)
class SimConfig:
    """The simulation window and risk-free assumption.

    Attributes:
        start: First date of the backtest (inclusive).
        end: Last date of the backtest (inclusive).
        risk_free_rate: Annual risk-free rate used by the Sharpe ratio.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        """Normalize string/loose dates to tz-naive Timestamps and validate order."""
        object.__setattr__(self, "start", pd.Timestamp(self.start).normalize())
        object.__setattr__(self, "end", pd.Timestamp(self.end).normalize())
        if self.start >= self.end:
            raise ValueError(f"start ({self.start.date()}) must be before end ({self.end.date()})")
