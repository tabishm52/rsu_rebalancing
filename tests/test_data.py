"""Tests for price access, with yfinance mocked so nothing hits the network."""

import pandas as pd
import pytest

from rsu_rebalancing import data


def _history(dates: list[str], closes: list[float]) -> pd.DataFrame:
    """Build a yfinance-shaped history frame: a Close column on a tz-aware index."""
    index = pd.to_datetime(dates).tz_localize("America/New_York")
    return pd.DataFrame({"Close": closes}, index=index)


class _FakeTicker:
    def __init__(self, parent: _FakeYF, symbol: str):
        self._parent = parent
        self._symbol = symbol

    def history(self, start: pd.Timestamp, end: pd.Timestamp, auto_adjust: bool) -> pd.DataFrame:
        self._parent.history_calls.append((self._symbol, start, end))
        return self._parent.frames[self._symbol]


class _FakeYF:
    """Stand-in for the yfinance module: serves canned frames and records calls."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.history_calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def Ticker(self, symbol: str) -> _FakeTicker:  # noqa: N802 (mirror yfinance's name)
        return _FakeTicker(self, symbol)


def _patch_yf(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]) -> _FakeYF:
    fake = _FakeYF(frames)
    monkeypatch.setattr(data, "yf", fake)
    return fake


@pytest.fixture(autouse=True)
def _clear_price_cache():
    """Drop memoized prices so call counts and mocks don't leak between tests."""
    data._get_prices_cached.cache_clear()
    yield
    data._get_prices_cached.cache_clear()


# --- get_prices ------------------------------------------------------------


def test_get_prices_returns_named_tz_naive_series(monkeypatch):
    _patch_yf(monkeypatch, {"AAA": _history(["2020-01-02", "2020-01-03"], [10.0, 11.0])})

    series = data.get_prices("AAA", "2020-01-02", "2020-01-03")

    assert series.name == "AAA"
    assert series.index.tz is None  # type: ignore[attr-defined]  # pandas-stubs types .index as Index, not DatetimeIndex
    assert series.tolist() == [10.0, 11.0]


def test_get_prices_uppercases_ticker(monkeypatch):
    fake = _patch_yf(monkeypatch, {"AAA": _history(["2020-01-02"], [10.0])})

    series = data.get_prices("aaa", "2020-01-02", "2020-01-02")

    assert series.name == "AAA"
    assert fake.history_calls[0][0] == "AAA"


def test_get_prices_makes_end_date_inclusive(monkeypatch):
    fake = _patch_yf(monkeypatch, {"AAA": _history(["2020-01-02"], [10.0])})

    data.get_prices("AAA", "2020-01-02", "2020-01-10")

    _, start, end = fake.history_calls[0]
    assert start == pd.Timestamp("2020-01-02")
    # yfinance end is exclusive, so the boundary is bumped a day past the request.
    assert end == pd.Timestamp("2020-01-11")


def test_get_prices_raises_on_empty_result(monkeypatch):
    _patch_yf(monkeypatch, {"AAA": pd.DataFrame()})

    with pytest.raises(ValueError, match="No price data returned for 'AAA'"):
        data.get_prices("AAA", "2020-01-02", "2020-01-03")


def test_get_prices_memoizes_repeat_requests(monkeypatch):
    fake = _patch_yf(monkeypatch, {"AAA": _history(["2020-01-02"], [10.0])})

    data.get_prices("AAA", "2020-01-02", "2020-01-02")
    data.get_prices("AAA", "2020-01-02", "2020-01-02")

    assert len(fake.history_calls) == 1


def test_get_prices_returns_a_defensive_copy(monkeypatch):
    _patch_yf(monkeypatch, {"AAA": _history(["2020-01-02"], [10.0])})

    first = data.get_prices("AAA", "2020-01-02", "2020-01-02")
    first.iloc[0] = 999.0
    second = data.get_prices("AAA", "2020-01-02", "2020-01-02")

    assert second.iloc[0] == 10.0


# --- get_price_frame -------------------------------------------------------


def test_get_price_frame_aligns_and_forward_fills_gaps(monkeypatch):
    _patch_yf(
        monkeypatch,
        {
            "AAA": _history(["2020-01-02", "2020-01-03", "2020-01-06"], [10.0, 11.0, 12.0]),
            "BBB": _history(["2020-01-02", "2020-01-06"], [20.0, 22.0]),
        },
    )

    frame = data.get_price_frame(["AAA", "BBB"], "2020-01-02", "2020-01-06")

    assert list(frame.columns) == ["AAA", "BBB"]
    # BBB has no 01-03 quote; the prior day's value is carried forward.
    assert frame.loc[pd.Timestamp("2020-01-03"), "BBB"] == 20.0


def test_get_price_frame_trims_leading_pre_ipo_window(monkeypatch):
    _patch_yf(
        monkeypatch,
        {
            "AAA": _history(["2020-01-02", "2020-01-03"], [10.0, 11.0]),
            "BBB": _history(["2020-01-03"], [21.0]),
        },
    )

    frame = data.get_price_frame(["AAA", "BBB"], "2020-01-02", "2020-01-03")

    # BBB started trading after the window opened; ffill cannot backfill the leading gap,
    # so the frame is trimmed to the first day both tickers trade.
    assert frame.index[0] == pd.Timestamp("2020-01-03")
    assert not frame.isna().to_numpy().any()
