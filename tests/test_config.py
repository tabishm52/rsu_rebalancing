"""Tests for config dataclass validation and the small behaviors they expose."""

import pandas as pd
import pytest

from rsu_rebalancing.config import BacktestConfig, GrantConfig, StrategyConfig, TaxConfig

# --- GrantConfig ---------------------------------------------------------


def test_grant_config_nominal_dates_one_per_year():
    schedule = GrantConfig(annual_dollars=50_000, start_year=2020, end_year=2022)

    dates = schedule.nominal_grant_dates()

    assert dates == [
        pd.Timestamp("2020-03-01"),
        pd.Timestamp("2021-03-01"),
        pd.Timestamp("2022-03-01"),
    ]


def test_grant_config_rejects_negative_dollars():
    with pytest.raises(ValueError, match="annual_dollars must be >= 0"):
        GrantConfig(annual_dollars=-1, start_year=2020, end_year=2021)


def test_grant_config_rejects_inverted_year_range():
    with pytest.raises(ValueError, match="start_year .* must be <= end_year"):
        GrantConfig(annual_dollars=50_000, start_year=2021, end_year=2020)


def test_grant_config_rejects_non_positive_vesting_years():
    with pytest.raises(ValueError, match="vesting_years must be >= 1"):
        GrantConfig(annual_dollars=50_000, start_year=2020, end_year=2021, vesting_years=0)


def test_grant_config_allows_feb_28():
    # Valid in every year; the non-leap probe should accept it.
    GrantConfig(annual_dollars=50_000, start_year=2020, end_year=2021, grant_month=2, grant_day=28)


def test_grant_config_rejects_feb_29():
    # Only valid in leap years, so the non-leap probe rejects it even though some
    # years in the range (2020) would accept it.
    with pytest.raises(ValueError, match="invalid grant_month/grant_day"):
        GrantConfig(
            annual_dollars=50_000, start_year=2020, end_year=2021, grant_month=2, grant_day=29
        )


def test_grant_config_rejects_out_of_range_month():
    with pytest.raises(ValueError, match="invalid grant_month/grant_day"):
        GrantConfig(annual_dollars=50_000, start_year=2020, end_year=2021, grant_month=13)


# --- StrategyConfig --------------------------------------------------------


def test_strategy_config_upper_cases_tickers():
    config = StrategyConfig(employer_ticker="aapl", index_ticker="vti", threshold=1 / 3)

    assert config.employer_ticker == "AAPL"
    assert config.index_ticker == "VTI"


@pytest.mark.parametrize("threshold", [0.0, 1.5])
def test_strategy_config_rejects_threshold_out_of_range(threshold):
    with pytest.raises(ValueError, match="threshold must be in"):
        StrategyConfig(employer_ticker="AAPL", threshold=threshold)


def test_strategy_config_rejects_zero_rebalances():
    with pytest.raises(ValueError, match="rebalances_per_quarter must be >= 1"):
        StrategyConfig(employer_ticker="AAPL", threshold=1 / 3, rebalances_per_quarter=0)


def test_strategy_config_rejects_negative_rebalance_band():
    with pytest.raises(ValueError, match="rebalance_band must be >= 0"):
        StrategyConfig(employer_ticker="AAPL", threshold=1 / 3, rebalance_band=-0.01)


def test_strategy_config_rejects_band_pushing_trigger_above_one():
    with pytest.raises(ValueError, match="threshold \\+ rebalance_band must be <= 1"):
        StrategyConfig(employer_ticker="AAPL", threshold=0.8, rebalance_band=0.3)


# --- TaxConfig -------------------------------------------------------------


@pytest.mark.parametrize("rate", [-0.1, 1.0])
def test_tax_config_rejects_short_term_rate_out_of_range(rate):
    with pytest.raises(ValueError, match="short_term_rate must be in"):
        TaxConfig(short_term_rate=rate)


@pytest.mark.parametrize("rate", [-0.1, 1.0])
def test_tax_config_rejects_long_term_rate_out_of_range(rate):
    with pytest.raises(ValueError, match="long_term_rate must be in"):
        TaxConfig(long_term_rate=rate)


def test_tax_config_rejects_non_positive_long_term_days():
    with pytest.raises(ValueError, match="long_term_days must be > 0"):
        TaxConfig(long_term_days=0)


# --- BacktestConfig --------------------------------------------------------


def test_backtest_config_normalizes_string_dates():
    cfg = BacktestConfig(start="2020-01-01 09:30", end="2024-12-31")

    assert cfg.start == pd.Timestamp("2020-01-01")
    assert cfg.end == pd.Timestamp("2024-12-31")
    assert cfg.start.tz is None


def test_backtest_config_rejects_start_not_before_end():
    with pytest.raises(ValueError, match="must be before"):
        BacktestConfig(start="2024-01-01", end="2024-01-01")
