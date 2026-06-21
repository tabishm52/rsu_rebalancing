"""Tests for the offline price fixture's coverage guard."""

import pandas as pd
import pytest

from rsu_app.fixtures import _FixtureTicker


def _frame(start: str, end: str) -> pd.DataFrame:
    """A snapshot holding only the trading (business) days over ``[start, end]``."""
    index = pd.bdate_range(start, end)
    return pd.DataFrame({"AAPL": range(len(index))}, index=index)


def test_serves_request_whose_boundaries_land_on_non_trading_days():
    # Snapshot padded a week each side of a [2011-03-01, 2024-12-31]-style window; the
    # request's raw start (Sat) / end (Sun) fall between trading days inside that pad.
    ticker = _FixtureTicker("AAPL", _frame("2011-02-22", "2011-04-08"))

    # data.py bumps the inclusive end by a day, so a Sun end arrives as the next Mon.
    sat_start = pd.Timestamp("2011-03-05")
    sun_end_bumped = pd.Timestamp("2011-03-13") + pd.Timedelta(days=1)

    got = ticker.history(sat_start, sun_end_bumped)

    assert not got.empty


def test_request_before_coverage_raises():
    ticker = _FixtureTicker("AAPL", _frame("2011-03-01", "2011-03-31"))

    with pytest.raises(ValueError, match="outside the price fixture's coverage"):
        ticker.history(pd.Timestamp("2010-01-01"), pd.Timestamp("2011-03-10"))


def test_request_after_coverage_raises():
    ticker = _FixtureTicker("AAPL", _frame("2011-03-01", "2011-03-31"))

    with pytest.raises(ValueError, match="outside the price fixture's coverage"):
        ticker.history(pd.Timestamp("2011-03-10"), pd.Timestamp("2011-05-01"))


def test_unknown_ticker_raises():
    ticker = _FixtureTicker("MSFT", _frame("2011-03-01", "2011-03-31"))

    with pytest.raises(ValueError, match="not in the price fixture"):
        ticker.history(pd.Timestamp("2011-03-10"), pd.Timestamp("2011-03-20"))
