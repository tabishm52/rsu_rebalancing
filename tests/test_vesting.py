"""Tests for expanding a GrantConfig into a vesting schedule of per-day share counts."""

import pandas as pd
import pytest
from pytest import approx

from rsu_rebalancing.config import GrantConfig, TaxConfig
from rsu_rebalancing.vesting import build_vesting_schedule

# Employer prices spanning the award dates (including years before the window) and the
# window itself, so awards can be priced and their vests snapped.
PRICE_DAYS = pd.bdate_range("2019-01-01", "2026-01-01")
WINDOW = pd.bdate_range("2021-01-01", "2025-12-31")

# Most tests isolate the award->vest share-count math from withholding, so they pass a
# zero ordinary-income rate; the haircut gets its own dedicated test.
NO_WITHHOLDING = TaxConfig(ordinary_income_rate=0.0)


def _flat_prices(value: float = 10.0) -> pd.Series:
    return pd.Series(value, index=PRICE_DAYS, name="EMP")


def _config(
    start_year: int = 2020,
    end_year: int = 2020,
    years: int = 4,
    growth: float = 0.0,
) -> GrantConfig:
    # Default to flat grants so the share-count math tests isolate the award->vest lock;
    # wage-inflation growth gets its own dedicated test.
    return GrantConfig(
        grant_dollars=120_000,
        start_year=start_year,
        end_year=end_year,
        vesting_years=years,
        grant_growth_rate=growth,
    )


def test_earliest_grant_date_is_first_award():
    config = GrantConfig(grant_dollars=120_000, start_year=2020, end_year=2023)

    got = config.earliest_grant_date

    assert got == pd.Timestamp("2020-03-01")


def test_share_count_locked_at_award_price():
    schedule = build_vesting_schedule(_config(), _flat_prices(10.0), WINDOW, NO_WITHHOLDING)

    # $120k / $10 award price = 12,000 shares, vesting 3,000/yr over four anniversaries.
    assert sum(schedule.values()) == approx(12_000)
    assert all(v == approx(3_000) for v in schedule.values())


def test_tranches_land_on_award_anniversaries():
    schedule = build_vesting_schedule(_config(), _flat_prices(10.0), WINDOW)

    # Award nominal Mar 1 2020; tranches on its next four (in-window) anniversaries.
    assert sorted(schedule) == [
        pd.Timestamp("2021-03-01"),
        pd.Timestamp("2022-03-01"),
        pd.Timestamp("2023-03-01"),
        pd.Timestamp("2024-03-01"),
    ]


def test_prewindow_vests_are_dropped():
    window = pd.bdate_range("2023-01-01", "2025-12-31")

    schedule = build_vesting_schedule(_config(), _flat_prices(10.0), window, NO_WITHHOLDING)

    # The 2021 and 2022 tranches fall before the window and are dropped; 2023 and 2024 stay.
    assert sorted(schedule) == [pd.Timestamp("2023-03-01"), pd.Timestamp("2024-03-01")]
    assert sum(schedule.values()) == approx(6_000)


def test_lower_award_price_locks_more_shares():
    cheap = build_vesting_schedule(_config(), _flat_prices(5.0), WINDOW, NO_WITHHOLDING)
    rich = build_vesting_schedule(_config(), _flat_prices(20.0), WINDOW, NO_WITHHOLDING)

    # The count is fixed at award: a cheaper award price buys more shares, which then vest
    # unchanged regardless of later prices -- the award->vest lock that drives the drift.
    assert sum(cheap.values()) == approx(24_000)
    assert sum(rich.values()) == approx(6_000)


def test_pre_ipo_award_raises():
    # Employer's first quote is 2022, but the award is nominally 2020 -- there's no price to
    # lock its share count, so this is an error rather than a silent snap to the 2022 price.
    prices = pd.Series(10.0, index=pd.bdate_range("2022-01-01", "2026-01-01"), name="EMP")

    with pytest.raises(ValueError, match="predates the price history"):
        build_vesting_schedule(_config(start_year=2020, end_year=2020), prices, WINDOW)


def test_overlapping_awards_accumulate_on_shared_vest_days():
    # Two consecutive awards put a tranche on the same anniversary day; their shares sum.
    config = _config(start_year=2019, end_year=2020)

    schedule = build_vesting_schedule(config, _flat_prices(10.0), WINDOW, NO_WITHHOLDING)

    # 2019 award's 3rd tranche and 2020 award's 2nd both land on Mar 1 2022: 3,000 + 3,000.
    assert schedule[pd.Timestamp("2022-03-01")] == approx(6_000)


def test_grant_value_grows_off_window_first_year():
    # WINDOW opens in 2021. The anchor-year award is worth the full $120k; an award one year
    # before the window is discounted by one growth step (here 10%).
    anchor = build_vesting_schedule(
        _config(start_year=2021, end_year=2021, growth=0.10),
        _flat_prices(10.0),
        WINDOW,
        NO_WITHHOLDING,
    )
    backfilled = build_vesting_schedule(
        _config(start_year=2020, end_year=2020, growth=0.10),
        _flat_prices(10.0),
        WINDOW,
        NO_WITHHOLDING,
    )

    assert sum(anchor.values()) == approx(12_000)
    assert sum(backfilled.values()) == approx(12_000 / 1.10)


def test_vest_withholding_grosses_down_kept_shares():
    # Sell-to-cover takes a flat fraction of the vesting shares: 12,000 locked shares kept at
    # (1 - 0.25) = 9,000, still split 2,250 across the four anniversaries.
    schedule = build_vesting_schedule(
        _config(), _flat_prices(10.0), WINDOW, TaxConfig(ordinary_income_rate=0.25)
    )

    assert sum(schedule.values()) == approx(9_000)
    assert all(v == approx(2_250) for v in schedule.values())
