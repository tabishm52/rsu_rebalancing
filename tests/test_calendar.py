"""Tests for calendar mapping of rebalances onto trading days."""

import pandas as pd

from rsu_rebalancing.calendar import (
    first_trading_day_on_or_after,
    rebalance_trade_dates,
)

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


# The window spans the full TRADING_DAYS range, so no marks are clipped.
SIM_START = pd.Timestamp("2020-01-01")
SIM_END = pd.Timestamp("2021-12-31")


def test_one_rebalance_lands_midquarter():
    days = rebalance_trade_dates(TRADING_DAYS, 1, SIM_START, SIM_END)
    q1 = [d for d in days if d.year == 2020 and d.quarter == 1]

    # Mid-Q1 2020 is ~Feb 15 (a Saturday); snaps forward to Mon Feb 17.
    assert q1 == [pd.Timestamp("2020-02-17")]
    assert len(days) == 8  # one per quarter, 8 quarters across 2020-2021


def test_two_rebalances_land_at_quarter_and_three_quarter_marks():
    days = rebalance_trade_dates(TRADING_DAYS, 2, SIM_START, SIM_END)
    q1 = [d for d in days if d.year == 2020 and d.quarter == 1]

    # The quarter (~Jan 22) and three-quarter (~Mar 9) marks of Q1 2020.
    assert q1 == [pd.Timestamp("2020-01-23"), pd.Timestamp("2020-03-09")]
    assert len(days) == 16
    assert days == sorted(days)


def test_many_rebalances_devolve_to_every_trading_day():
    # A count far larger than the quarter's trading days resolves to each of them.
    days = rebalance_trade_dates(TRADING_DAYS, 500, SIM_START, SIM_END)
    q1_marks = [d for d in days if d.year == 2020 and d.quarter == 1]
    q1_trading = [d for d in TRADING_DAYS if d.year == 2020 and d.quarter == 1]

    assert q1_marks == q1_trading


def test_marks_outside_window_are_clipped():
    # Window opens mid-Q1, so Q1's quarter mark (~Jan 22) precedes it and is dropped;
    # only the three-quarter mark survives. Q2 is fully covered, so both marks land.
    stub = pd.bdate_range("2020-02-18", "2020-06-30")
    start, end = pd.Timestamp("2020-02-18"), pd.Timestamp("2020-06-30")

    days = rebalance_trade_dates(stub, 2, start, end)
    q1 = [d for d in days if d.year == 2020 and d.quarter == 1]
    q2 = [d for d in days if d.year == 2020 and d.quarter == 2]

    assert q1 == [pd.Timestamp("2020-03-09")]
    assert q2 == [pd.Timestamp("2020-04-23"), pd.Timestamp("2020-06-08")]


def test_empty_trading_days_yield_no_dates():
    empty = pd.DatetimeIndex([])

    assert rebalance_trade_dates(empty, 2, SIM_START, SIM_END) == []
