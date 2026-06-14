"""Map nominal dates onto actual trading days.

Grants and rebalances are described in calendar terms ("first trading day of March",
"5th trading day of the quarter"), but trades can only happen on days the market is
open. These pure functions translate those rules against a known set of trading days,
so they are fully testable without any network access.
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
        if day is not None:
            grants[day] = grants.get(day, 0.0) + schedule.annual_dollars
    return grants


def rebalance_trade_dates(
    trading_days: pd.DatetimeIndex,
    days_after_quarter_start: int,
    days_before_quarter_end: int,
) -> list[pd.Timestamp]:
    """Compute the two rebalance trade days for each calendar quarter.

    The two days approximate trading just after a blackout opens and just before the
    next one closes: the Nth trading day from the start of the quarter and the Nth
    trading day from its end. Offsets are clamped to the days actually available in a
    quarter, and duplicate dates (when a quarter is short) are collapsed.

    Args:
        trading_days: Sorted index of available trading days.
        days_after_quarter_start: 1-based offset from the quarter's first trading day.
        days_before_quarter_end: 1-based offset from the quarter's last trading day.

    Returns:
        A sorted list of rebalance trade days.
    """
    dates: set[pd.Timestamp] = set()
    quarters = trading_days.to_period("Q")
    for _, group in pd.Series(trading_days, index=quarters).groupby(level=0):
        days = group.to_numpy()
        n = len(days)
        early = min(days_after_quarter_start - 1, n - 1)
        late = max(n - days_before_quarter_end, 0)
        dates.add(pd.Timestamp(days[early]))
        dates.add(pd.Timestamp(days[late]))
    return sorted(dates)
