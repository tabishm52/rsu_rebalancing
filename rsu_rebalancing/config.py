"""Configuration objects for a rebalancing backtest.

These are small, frozen dataclasses that capture *what* to backtest (the grant
stream, the strategy parameters, and the date window). They hold no price data and
do no I/O, which keeps them trivial to construct in tests and notebooks.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class GrantConfig:
    """Parameters of an annual RSU award stream.

    One award is granted each year; :func:`~rsu_rebalancing.vesting.build_vesting_schedule`
    expands these parameters against prices into the shares vesting on each trading day.

    Attributes:
        grant_dollars: Pre-tax value of the award granted in the backtest window's
            first year. Awards grow ``grant_growth_rate`` per year off that anchor.
        start_year: First calendar year to grant an award in (inclusive). Set this before
            the backtest window to backfill awards whose vests land inside it.
        end_year: Last calendar year to grant an award in (inclusive).
        grant_growth_rate: Annual growth of the award's dollar value, modeling wage
            inflation in the grant band. Compounds off the backtest window's first year.
        grant_month: Month of the award (1-12). Defaults to March.
        grant_day: Day of month for the nominal award date. The backtest snaps this
            to the first trading day on or after it.
        vesting_years: Number of equal annual tranches each award vests over (>= 1).
    """

    grant_dollars: float
    start_year: int
    end_year: int
    grant_growth_rate: float = 0.04
    grant_month: int = 3
    grant_day: int = 1
    vesting_years: int = 4

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.grant_dollars < 0.0:
            raise ValueError(f"grant_dollars must be >= 0; got {self.grant_dollars}")
        if self.grant_growth_rate <= -1.0:
            raise ValueError(f"grant_growth_rate must be > -1; got {self.grant_growth_rate}")
        if self.start_year > self.end_year:
            raise ValueError(
                f"start_year ({self.start_year}) must be <= end_year ({self.end_year})"
            )
        if self.vesting_years < 1:
            raise ValueError(f"vesting_years must be >= 1; got {self.vesting_years}")
        # Probe month/day against a non-leap year so the grant date is valid in every
        # year (this rejects Feb 29, which only some years would accept).
        try:
            pd.Timestamp(year=2001, month=self.grant_month, day=self.grant_day)
        except ValueError as exc:
            raise ValueError(
                f"invalid grant_month/grant_day: {self.grant_month}/{self.grant_day}"
            ) from exc

    def nominal_grant_dates(self) -> list[pd.Timestamp]:
        """Return the nominal (calendar) award date for each year, earliest first."""
        return [
            pd.Timestamp(year=year, month=self.grant_month, day=self.grant_day)
            for year in range(self.start_year, self.end_year + 1)
        ]

    @property
    def earliest_grant_date(self) -> pd.Timestamp:
        """Nominal date of the first award.

        Awards can predate the backtest window, so this is how far back employer prices
        must be fetched to lock each award's share count at its award-date price.
        """
        return self.nominal_grant_dates()[0]


@dataclass(frozen=True)
class TaxConfig:
    """Tax rates on RSU vest income and realized capital gains.

    The defaults model nominal marginal tax rates for a California single filer with AGI
    roughly between $260k and $375k (2024 brackets):

    - ``ordinary_income_rate = 0.443``: 35% federal ordinary income + 9.3% California.
    - ``short_term_rate = 0.483``: 35% federal ordinary income + 9.3% California +
      3.8% net investment income tax.
    - ``long_term_rate = 0.281``: 15% federal long-term + 9.3% California + 3.8% net
      investment income tax.

    Attributes:
        ordinary_income_rate: Rate on ordinary income, used for RSU vesting.
        short_term_rate: Rate on gains realized on lots held <= ``long_term_days``.
        long_term_rate: Rate on gains realized on lots held > ``long_term_days``.
        long_term_days: Holding-period boundary in days (US long-term is > 1 year).
    """

    ordinary_income_rate: float = 0.443
    short_term_rate: float = 0.483
    long_term_rate: float = 0.281
    long_term_days: int = 365

    def __post_init__(self) -> None:
        """Validate the rate ranges and the holding-period boundary."""
        if not 0.0 <= self.ordinary_income_rate < 1.0:
            raise ValueError(
                f"ordinary_income_rate must be in [0, 1); got {self.ordinary_income_rate}"
            )
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
        threshold: Target maximum fraction of total holdings in employer stock
            (e.g. ``1/3``). Rebalances trim employer stock down to this fraction.
            Required: the concentration target is a policy choice with no neutral
            default, so the caller must state it.
        index_ticker: Symbol of the diversified fund that sale proceeds buy.
        rebalance_band: One-way hysteresis band, in fraction points. A rebalance
            fires only once the employer fraction exceeds ``threshold + rebalance_band``.
        rebalances_per_quarter: Number of evenly spaced rebalances to place in each
            quarter.
        tax_config: Capital-gains tax rates applied to realized gains.
    """

    employer_ticker: str
    threshold: float
    index_ticker: str = "VTI"
    rebalance_band: float = 0.05
    rebalances_per_quarter: int = 2
    tax_config: TaxConfig = field(default_factory=TaxConfig)

    def __post_init__(self) -> None:
        """Upper-case the tickers (the canonical form downstream) and validate ranges."""
        object.__setattr__(self, "employer_ticker", self.employer_ticker.upper())
        object.__setattr__(self, "index_ticker", self.index_ticker.upper())
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]; got {self.threshold}")
        if self.rebalance_band < 0.0:
            raise ValueError(f"rebalance_band must be >= 0; got {self.rebalance_band}")
        if self.threshold + self.rebalance_band > 1.0:
            raise ValueError(
                f"threshold + rebalance_band must be <= 1; got "
                f"{self.threshold} + {self.rebalance_band}"
            )
        if self.rebalances_per_quarter < 1:
            raise ValueError(
                f"rebalances_per_quarter must be >= 1; got {self.rebalances_per_quarter}"
            )


@dataclass(frozen=True, init=False)
class BacktestConfig:
    """The backtest window and reporting assumptions (risk-free rate, performance basis).

    Loose dates are part of the contract: ``start`` and ``end`` accept anything
    ``pd.Timestamp`` does (e.g. ``"2020-01-01"``) and are stored as normalized, tz-naive
    Timestamps, so reads are always a ``Timestamp``.

    Attributes:
        start: First date of the backtest (inclusive).
        end: Last date of the backtest (inclusive).
        risk_free_rate: Annual risk-free rate used by the Sharpe ratio. Defaults to
            roughly the average 3-month Treasury yield over 2015-2024.
        after_tax_performance: When True, return and risk metrics are measured on
            net-of-tax liquidation value; when False, on raw market value.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    risk_free_rate: float
    after_tax_performance: bool

    def __init__(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        risk_free_rate: float = 0.02,
        after_tax_performance: bool = False,
    ) -> None:
        """Normalize loose dates to tz-naive Timestamps and validate order."""
        object.__setattr__(self, "start", pd.Timestamp(start).normalize())
        object.__setattr__(self, "end", pd.Timestamp(end).normalize())
        object.__setattr__(self, "risk_free_rate", risk_free_rate)
        object.__setattr__(self, "after_tax_performance", after_tax_performance)
        if self.start >= self.end:
            raise ValueError(f"start ({self.start.date()}) must be before end ({self.end.date()})")
