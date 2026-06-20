"""Tests for expanding a GrantConfig into a vesting schedule of per-day share counts."""

import pandas as pd
import pytest
from pytest import approx

from rsu_rebalancing.calendar import first_trading_day_on_or_after
from rsu_rebalancing.config import GrantConfig
from rsu_rebalancing.vesting import build_vesting_schedule

# Employer prices spanning the award dates (including the years a backfilled stream reaches
# before the window) and the window itself, so awards can be priced and their vests snapped.
PRICE_DAYS = pd.bdate_range("2016-01-01", "2026-01-01")
WINDOW = pd.bdate_range("2021-01-01", "2025-12-31")


def _flat_prices(value: float = 10.0) -> pd.Series:
    return pd.Series(value, index=PRICE_DAYS, name="EMP")


def _vest_day(year: int) -> pd.Timestamp:
    """The Mar 1 anniversary of ``year`` snapped onto WINDOW's trading days (e.g. off a weekend)."""
    day = first_trading_day_on_or_after(WINDOW, pd.Timestamp(f"{year}-03-01"))
    assert day is not None  # these anniversaries all sit inside WINDOW
    return day


def _config(
    backfill: bool = False,
    years: int = 4,
    growth: float = 0.0,
) -> GrantConfig:
    # Flat grants by default so the share-count tests isolate the award->vest lock; growth
    # gets its own test.
    return GrantConfig(
        grant_dollars=120_000,
        backfill=backfill,
        vesting_years=years,
        grant_growth_rate=growth,
    )


def test_earliest_grant_date_backfills_before_window():
    config = _config(backfill=True, years=4)

    got = config.earliest_grant_date(pd.Timestamp("2021-01-01"))

    assert got == pd.Timestamp("2017-03-01")  # 2021 window backfilled by 4 vesting years


def test_share_count_locked_at_award_price():
    # vesting_years=1 over a two-year window isolates one award (the 2022 award vests out of range).
    window = pd.bdate_range("2021-01-01", "2022-12-31")

    schedule = build_vesting_schedule(_config(years=1), _flat_prices(10.0), window)

    # $120k / $10 award price = 12,000 shares, vesting in one tranche.
    assert schedule == approx({pd.Timestamp("2022-03-01"): 12_000})


def test_lower_award_price_locks_more_shares():
    window = pd.bdate_range("2021-01-01", "2022-12-31")

    cheap = build_vesting_schedule(_config(years=1), _flat_prices(5.0), window)
    rich = build_vesting_schedule(_config(years=1), _flat_prices(20.0), window)

    # The count is fixed at award: a cheaper award price buys more shares, which then vest
    # unchanged regardless of later prices -- the award->vest lock that drives the drift.
    assert cheap[pd.Timestamp("2022-03-01")] == approx(24_000)
    assert rich[pd.Timestamp("2022-03-01")] == approx(6_000)


def test_no_backfill_ramps_up_from_window_start():
    schedule = build_vesting_schedule(_config(backfill=False, years=4), _flat_prices(10.0), WINDOW)

    # Awards 2021-2025, each 12,000 shares in four 3,000-share tranches: year one has no vests
    # yet and the overlapping tranches ramp up.
    assert schedule == approx(
        {
            _vest_day(2022): 3_000,
            _vest_day(2023): 6_000,
            _vest_day(2024): 9_000,
            _vest_day(2025): 12_000,
        }
    )


def test_backfill_opens_at_steady_state():
    schedule = build_vesting_schedule(_config(backfill=True, years=4), _flat_prices(10.0), WINDOW)

    # Year one already sits at the steady state of four overlapping tranches (12,000 shares/yr).
    assert schedule == approx({_vest_day(year): 12_000 for year in range(2021, 2026)})


def test_prewindow_tranches_are_dropped_not_piled_on_first_day():
    # A backfilled stream's anniversaries that predate the window must be dropped, not snapped
    # forward onto the first window day (which would wrongly pile shares there).
    window = pd.bdate_range("2021-06-01", "2024-12-31")

    schedule = build_vesting_schedule(_config(backfill=True, years=4), _flat_prices(10.0), window)

    # The Mar 1 2021 tranches fall before the Jun window start, so the earliest vest is 2022.
    assert min(schedule) == pd.Timestamp("2022-03-01")


def test_pre_ipo_award_raises():
    # The backfilled stream reaches back to 2017, before the employer's first 2022 quote, so
    # those awards' share counts can't be locked -- an error, not a silent snap to 2022's price.
    prices = pd.Series(10.0, index=pd.bdate_range("2022-01-01", "2026-01-01"), name="EMP")

    with pytest.raises(ValueError, match="predates the price history"):
        build_vesting_schedule(_config(backfill=True), prices, WINDOW)


def test_grant_value_grows_off_window_first_year():
    # The anchor-year (2021) award is worth the full $120k; the 2020 award is discounted by one
    # growth step (10%). vesting_years=1 lands them on 2022 and 2021 respectively.
    window = pd.bdate_range("2021-01-01", "2022-12-31")

    schedule = build_vesting_schedule(
        _config(backfill=True, years=1, growth=0.10), _flat_prices(10.0), window
    )

    assert schedule[pd.Timestamp("2021-03-01")] == approx(12_000 / 1.10)
    assert schedule[pd.Timestamp("2022-03-01")] == approx(12_000)
