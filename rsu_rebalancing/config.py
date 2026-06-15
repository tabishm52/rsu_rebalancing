"""Configuration objects for a rebalancing backtest.

These are small, frozen dataclasses that capture *what* to simulate (the grant
stream, the strategy parameters, and the date window). They hold no price data and
do no I/O, which keeps them trivial to construct in tests and notebooks.
"""

from dataclasses import dataclass, field

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

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.annual_dollars <= 0.0:
            raise ValueError(f"annual_dollars must be > 0; got {self.annual_dollars}")
        if self.start_year > self.end_year:
            raise ValueError(
                f"start_year ({self.start_year}) must be <= end_year ({self.end_year})"
            )
        # Probe month/day against a non-leap year so the grant date is valid in every
        # year (this rejects Feb 29, which only some years would accept).
        try:
            pd.Timestamp(year=2001, month=self.grant_month, day=self.grant_day)
        except ValueError as exc:
            raise ValueError(
                f"invalid grant_month/grant_day: {self.grant_month}/{self.grant_day}"
            ) from exc

    def nominal_grant_dates(self) -> list[pd.Timestamp]:
        """Return the nominal (calendar) grant date for each year in the range."""
        return [
            pd.Timestamp(year=year, month=self.grant_month, day=self.grant_day)
            for year in range(self.start_year, self.end_year + 1)
        ]


@dataclass(frozen=True)
class TaxConfig:
    """Capital-gains tax rates applied to realized employer-stock gains.

    Gains are taxed by holding period: a lot sold within ``long_term_days`` of its vest
    date is taxed at ``short_term_rate`` (ordinary income), and one held longer at the
    lower ``long_term_rate``. Each rate is a single effective figure, so fold any state
    or NIIT surcharge into it.

    The defaults model nominal marginal tax rates for a California single filer with AGI
    roughly between $260k and $375k (2024 brackets):

    - ``short_term_rate = 0.48``: 35% federal ordinary income + 9.3% California +
      3.8% net investment income tax.
    - ``long_term_rate = 0.28``: 15% federal long-term + 9.3% California + 3.8% net
      investment income tax.

    Attributes:
        short_term_rate: Rate on gains realized on lots held <= ``long_term_days``.
        long_term_rate: Rate on gains realized on lots held > ``long_term_days``.
        long_term_days: Holding-period boundary in days (US long-term is > 1 year).
    """

    short_term_rate: float = 0.48
    long_term_rate: float = 0.28
    long_term_days: int = 365

    def __post_init__(self) -> None:
        """Validate the rate ranges and the holding-period boundary."""
        if not 0.0 <= self.short_term_rate < 1.0:
            raise ValueError(f"short_term_rate must be in [0, 1); got {self.short_term_rate}")
        if not 0.0 <= self.long_term_rate < 1.0:
            raise ValueError(f"long_term_rate must be in [0, 1); got {self.long_term_rate}")
        if self.long_term_days <= 0:
            raise ValueError(f"long_term_days must be > 0; got {self.long_term_days}")


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
        tax_config: Capital-gains tax rates applied to realized gains.
    """

    employer_ticker: str
    index_ticker: str = "VTI"
    threshold: float = 1.0 / 3.0
    rebalances_per_quarter: int = 2
    tax_config: TaxConfig = field(default_factory=TaxConfig)

    def __post_init__(self) -> None:
        """Upper-case the tickers (the canonical form downstream) and validate ranges."""
        object.__setattr__(self, "employer_ticker", self.employer_ticker.upper())
        object.__setattr__(self, "index_ticker", self.index_ticker.upper())
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]; got {self.threshold}")
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
