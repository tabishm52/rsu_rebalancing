"""The vesting schedule: employer shares vesting on each trading day.

Real RSUs fix a *share count* at award time (the year's dollar value divided by the
award-date price), then vest that fixed count in equal annual tranches over the following
years. So a vest delivers a known number of shares whose dollar value floats with the
price between award and vest -- the count, not the dollars, is locked.

:func:`build_vesting_schedule` expands a :class:`~rsu_rebalancing.config.GrantConfig`
against employer prices into that per-day share count. It is I/O-free (prices come in as an
argument), so it is fully testable with synthetic series.
"""

import pandas as pd

from .calendar import first_trading_day_on_or_after
from .config import GrantConfig

# A resolved vesting schedule: a mapping from trading day to the employer shares vesting
# that day. A plain dict; the alias just names the domain meaning at function boundaries.
VestingSchedule = dict[pd.Timestamp, float]

# An award snaps forward to the first trading day on or after it. A gap larger than a
# handful of days means no price existed near the award -- it predates the employer's price
# history (e.g. before its IPO), which we flag as an error.
_MAX_AWARD_SNAP_DAYS = 7


def build_vesting_schedule(
    config: GrantConfig,
    employer_prices: pd.Series,
    window_days: pd.DatetimeIndex,
) -> VestingSchedule:
    """Expand ``config`` into the employer shares vesting on each trading day.

    Each award's dollar value is converted to a share count at the award-date price, and
    that count is fixed; it then vests in equal annual tranches over the next
    ``config.vesting_years``. ``config.grant_dollars`` is the award value in the window's
    first year; awards grow ``config.grant_growth_rate`` per year off that anchor, so
    backfilled awards are worth proportionally less.

    Awards may predate the window (their award-date prices fetched from before it); vests
    landing before the window are dropped, so a window can open at a steady state of
    overlapping awards.

    Args:
        config: The award stream parameters.
        employer_prices: Employer close prices indexed by trading day, spanning the award
            dates (which may predate ``window_days``).
        window_days: Sorted trading days of the backtest window; each tranche is snapped
            onto these and dropped if it falls outside.

    Returns:
        A mapping from trading day to the shares vesting that day. Awards whose nominal date
        is past the last available price are dropped.

    Raises:
        ValueError: If an award predates the available price history (e.g. before the
            employer's IPO), so its share count can't be locked at an award-date price.
    """
    if len(window_days) == 0:
        return {}

    award_index = pd.DatetimeIndex(employer_prices.index)
    tranche_fraction = 1.0 / config.vesting_years
    anchor_year = window_days[0].year

    shares_by_day: VestingSchedule = {}
    for award_nominal in config.nominal_grant_dates(window_days[0], window_days[-1]):
        award_day = first_trading_day_on_or_after(award_index, award_nominal)
        if award_day is None:
            # Award granted after the price history ends (past the window); all its vests
            # are out of window too, so it contributes nothing.
            continue
        if (award_day - award_nominal).days > _MAX_AWARD_SNAP_DAYS:
            ticker = employer_prices.name or "the employer stock"
            raise ValueError(
                f"award on {award_nominal.date()} predates the price history for {ticker} "
                f"(first available price is {award_day.date()}); start the awards on or after "
                "the first trading day, or shorten the backfill."
            )

        years_from_anchor = award_nominal.year - anchor_year
        award_dollars = config.grant_dollars * (1.0 + config.grant_growth_rate) ** years_from_anchor
        total_shares = award_dollars / float(employer_prices.loc[award_day])

        for k in range(1, config.vesting_years + 1):
            vest_nominal = award_nominal + pd.DateOffset(years=k)
            # Drop vests before the window: snapping forward would wrongly pin them to the
            # first window day. (Past the window's end snaps to None, also dropped.)
            if vest_nominal < window_days[0]:
                continue
            vest_day = first_trading_day_on_or_after(window_days, vest_nominal)
            if vest_day is None:
                continue

            shares_by_day[vest_day] = (
                shares_by_day.get(vest_day, 0.0) + total_shares * tranche_fraction
            )

    return shares_by_day
