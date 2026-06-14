"""Tests for calendar mapping of grants and rebalances onto trading days."""

from __future__ import annotations

import pandas as pd

from rsu_rebalancing.calendar import (
    first_trading_day_on_or_after,
    grant_trade_dates,
    rebalance_trade_dates,
)
from rsu_rebalancing.config import GrantSchedule

# A full year of US-business-day "trading days" (ignores holidays; fine for these tests).
TRADING_DAYS = pd.bdate_range("2020-01-01", "2021-12-31")


def test_first_trading_day_snaps_forward_over_weekend():
    # 2020-03-01 is a Sunday; the next trading day is Monday the 2nd.
    got = first_trading_day_on_or_after(TRADING_DAYS, pd.Timestamp("2020-03-01"))
    assert got == pd.Timestamp("2020-03-02")


def test_first_trading_day_returns_same_day_when_open():
    got = first_trading_day_on_or_after(TRADING_DAYS, pd.Timestamp("2020-03-02"))
    assert got == pd.Timestamp("2020-03-02")


def test_first_trading_day_past_end_is_none():
    assert first_trading_day_on_or_after(TRADING_DAYS, pd.Timestamp("2030-01-01")) is None


def test_grant_dates_snap_and_carry_dollars():
    schedule = GrantSchedule(annual_dollars=50_000, start_year=2020, end_year=2021)
    grants = grant_trade_dates(TRADING_DAYS, schedule)

    assert grants == {
        pd.Timestamp("2020-03-02"): 50_000,  # Mar 1 2020 was a Sunday
        pd.Timestamp("2021-03-01"): 50_000,  # Mar 1 2021 was a Monday
    }


def test_rebalance_first_and_last_trading_day_of_quarter():
    days = rebalance_trade_dates(
        TRADING_DAYS, days_after_quarter_start=1, days_before_quarter_end=1
    )

    # Q1 2020 opens 2020-01-01 and closes 2020-03-31; both are business days here.
    assert pd.Timestamp("2020-01-01") in days
    assert pd.Timestamp("2020-03-31") in days
    # Two days per quarter, 8 quarters across 2020-2021.
    assert len(days) == 16
    assert days == sorted(days)


def test_rebalance_offsets_count_into_the_quarter():
    days = rebalance_trade_dates(
        TRADING_DAYS, days_after_quarter_start=5, days_before_quarter_end=3
    )
    q1_days = [d for d in days if d.year == 2020 and d.quarter == 1]

    # 5th business day of 2020 is Jan 7; 3rd-from-last of Q1 is Mar 27.
    assert q1_days[0] == pd.Timestamp("2020-01-07")
    assert q1_days[1] == pd.Timestamp("2020-03-27")


def test_rebalance_single_day_quarter_collapses_to_one():
    # A quarter with a single trading day: both offsets clamp to that day and collapse.
    tiny = pd.DatetimeIndex([pd.Timestamp("2020-01-01")])
    days = rebalance_trade_dates(tiny, days_after_quarter_start=10, days_before_quarter_end=10)
    assert days == [pd.Timestamp("2020-01-01")]
