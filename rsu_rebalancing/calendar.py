"""Map nominal dates onto actual trading days.

Grants and rebalances are described in calendar terms ("first day of March", "twice per
quarter"), but trades can only happen on days the market is open. These pure functions
translate those rules against a known set of trading days, so they are fully testable
without any network access.
"""

import pandas as pd

from .config import GrantSchedule


def first_trading_day_on_or_after(
    trading_days: pd.DatetimeIndex, date: pd.Timestamp
) -> pd.Timestamp | None:
    """Return the earliest trading day that is on or after ``date``.

    Args:
        trading_days: Sorted index of available trading days.
        date: The nominal target date.

    Returns:
        The matching trading day, or ``None`` if ``date`` is past the last trading day.
    """
    pos = int(trading_days.searchsorted(date, side="left"))
    if pos >= len(trading_days):
        return None

    return trading_days[pos]


def grant_trade_dates(
    trading_days: pd.DatetimeIndex, schedule: GrantSchedule
) -> dict[pd.Timestamp, float]:
    """Snap each nominal grant date to a real trading day.

    Args:
        trading_days: Sorted index of available trading days.
        schedule: The grant stream.

    Returns:
        A mapping from trading day to the dollar value granted that day. Grants whose
        nominal date falls after the last trading day are dropped.
    """
    grants: dict[pd.Timestamp, float] = {}

    for nominal in schedule.nominal_grant_dates():
        day = first_trading_day_on_or_after(trading_days, nominal)
        if day is None:
            continue

        # Two grants can snap to the same trading day; accumulate their dollars.
        grants[day] = grants.get(day, 0.0) + schedule.annual_dollars

    return grants


def rebalance_trade_dates(
    trading_days: pd.DatetimeIndex,
    rebalances_per_quarter: int,
    sim_start: pd.Timestamp,
    sim_end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Place evenly spaced rebalance trade days within each calendar quarter.

    The trades sit at the centers of ``rebalances_per_quarter`` equal slices of the
    quarter's true calendar span: one trade lands mid-quarter, two at the quarter and
    three-quarter marks, three at the sixth marks, and so on. Any dates that land outside
    ``[sim_start, sim_end]`` are dropped.

    This spacing is deliberate: one or two rebalances a quarter sit near the middle and
    so roughly respect insider-trading blackout windows, which open near the start and
    end of a quarter. More than two no longer does — the outer trades drift toward the
    edges. A large count approaches a rebalance every trading day.

    Args:
        trading_days: Sorted index of available trading days.
        rebalances_per_quarter: Number of rebalances to place in each quarter (>= 1).
        sim_start: First date of the simulation window (inclusive).
        sim_end: Last date of the simulation window (inclusive).

    Returns:
        A sorted list of rebalance trade days, deduplicated.
    """
    dates: set[pd.Timestamp] = set()
    for quarter in trading_days.to_period("Q").unique():
        span = quarter.end_time - quarter.start_time

        for i in range(1, rebalances_per_quarter + 1):
            fraction = (2 * i - 1) / (2 * rebalances_per_quarter)
            target = (quarter.start_time + fraction * span).normalize()
            if not sim_start <= target <= sim_end:
                continue

            day = first_trading_day_on_or_after(trading_days, target)
            if day is not None:
                dates.add(day)

    return sorted(dates)
