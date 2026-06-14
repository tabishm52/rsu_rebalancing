"""Tests for return/risk metrics, especially contribution stripping (TWR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pytest import approx

from rsu_rebalancing.metrics import (
    annualized_volatility,
    growth_of_one,
    max_drawdown,
    sharpe_ratio,
    time_weighted_returns,
)

DATES = pd.bdate_range("2020-01-01", periods=4)


def test_time_weighted_returns_strip_contributions():
    # Day 0: deposit 100. Day 1: grows to 110 (+10%). Day 2: deposit 100 -> 220.
    # Day 3: grows to 231 (+5% on 220). The day-2 deposit must NOT count as a gain.
    values = pd.Series([100.0, 110.0, 220.0, 231.0], index=DATES)
    contributions = pd.Series([100.0, 0.0, 100.0, 0.0], index=DATES)

    returns = time_weighted_returns(values, contributions)

    assert list(returns.index) == list(DATES[1:])
    assert returns.iloc[0] == approx(0.10)  # 110 / 100 - 1
    assert returns.iloc[1] == approx((220 - 100) / 110 - 1)  # deposit removed
    assert returns.iloc[2] == approx(0.05)  # 231 / 220 - 1


def test_growth_of_one_compounds():
    returns = pd.Series([0.10, -0.05, 0.05])
    curve = growth_of_one(returns)
    assert curve.iloc[-1] == approx(1.10 * 0.95 * 1.05)


def test_max_drawdown_is_largest_peak_to_trough():
    # Up 20%, then down to 0.84 of peak (a 30% drawdown), then partial recovery.
    returns = pd.Series([0.20, -0.30, 0.10])
    # Peak growth = 1.20; trough = 1.20 * 0.70 = 0.84 -> drawdown = 0.84/1.20 - 1 = -0.30.
    assert max_drawdown(returns) == approx(-0.30)


def test_annualized_volatility_scales_by_sqrt_252():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, size=1000))
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert annualized_volatility(returns) == approx(expected)


def test_sharpe_zero_volatility_is_nan():
    returns = pd.Series([0.001, 0.001, 0.001])  # constant -> zero std
    assert np.isnan(sharpe_ratio(returns))
