"""Tests for return/risk metrics, especially contribution stripping (TWR).

The risk/return stats delegate to quantstats, so these tests cover only our own seams:
functions with no library underneath, our deliberate overrides, and our guards. We do
not re-derive or pin quantstats' own math.
"""

import numpy as np
import pandas as pd
from pytest import approx

from rsu_rebalancing.metrics import (
    annualized_return,
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


def test_time_weighted_returns_scrub_divide_by_zero():
    # A prior value of 0 makes the return undefined: 0/0 -> NaN, x/0 -> inf. Both must
    # be scrubbed to 0.0 so an empty-then-funded portfolio doesn't poison the series.
    values = pd.Series([0.0, 100.0, 0.0, 50.0], index=DATES)
    contributions = pd.Series([0.0, 100.0, 0.0, 0.0], index=DATES)

    returns = time_weighted_returns(values, contributions)

    assert returns.iloc[0] == 0.0  # (100 - 100) / 0 -> NaN -> 0
    assert returns.iloc[1] == approx(-1.0)  # (0 - 0) / 100 - 1
    assert returns.iloc[2] == 0.0  # (50 - 0) / 0 -> inf -> 0


def test_growth_of_one_compounds():
    returns = pd.Series([0.10, -0.05, 0.05])

    curve = growth_of_one(returns)

    assert curve.iloc[-1] == approx(1.10 * 0.95 * 1.05)


def test_annualized_return_empty_is_nan():
    assert np.isnan(annualized_return(pd.Series([], dtype=float)))


def test_max_drawdown_empty_is_nan():
    assert np.isnan(max_drawdown(pd.Series([], dtype=float)))


def test_sharpe_zero_volatility_is_nan():
    returns = pd.Series([0.001, 0.001, 0.001])  # constant -> zero std

    assert np.isnan(sharpe_ratio(returns))
