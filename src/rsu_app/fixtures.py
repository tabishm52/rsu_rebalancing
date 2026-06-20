"""Frozen price fixture for deterministic, network-free backtests.

The README assets, the notebook smoke test, and the offline test suite run against a
checked-in snapshot of AAPL/INTC/VTI prices instead of live yfinance, so results are
reproducible and CI never touches the network. ``patch_yf`` swaps the snapshot in behind
``rsu_rebalancing.data``; ``refresh`` re-fetches it from yfinance.
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

# Employers exercised by the README scenarios (AAPL/INTC) plus the default index (VTI),
# over a window generous enough for the notebook defaults' grant award-lookback (~2011).
FIXTURE_TICKERS = ("AAPL", "INTC", "VTI")
FIXTURE_START = "2010-01-01"
FIXTURE_END = "2025-01-01"

# Half-open coverage bound: yfinance's end is exclusive and data.py bumps the caller's
# inclusive end by a day, so the fixture can serve requests up to FIXTURE_END + 1 day.
_COVERED_START = pd.Timestamp(FIXTURE_START)
_COVERED_END = pd.Timestamp(FIXTURE_END) + pd.Timedelta(days=1)

_PARQUET = Path(__file__).resolve().parent / "_data" / "prices.parquet"


def load_fixture_frame() -> pd.DataFrame:
    """Return the checked-in price snapshot: a date-indexed frame, one column per ticker."""
    return pd.read_parquet(_PARQUET)


class _FixtureTicker:
    """Stand-in for ``yfinance.Ticker`` that serves one column of the snapshot."""

    def __init__(self, ticker: str, frame: pd.DataFrame) -> None:
        self._ticker = ticker.upper()
        self._frame = frame

    def history(
        self, start: pd.Timestamp, end: pd.Timestamp, auto_adjust: bool = True
    ) -> pd.DataFrame:
        """Return snapshot closes over ``[start, end)`` shaped like yfinance's output.

        Requests outside the snapshot's coverage raise rather than silently returning a
        truncated frame, so a default that drifts past the fixture fails loudly.
        """
        if self._ticker not in self._frame.columns:
            raise ValueError(
                f"{self._ticker!r} is not in the price fixture "
                f"({', '.join(map(str, self._frame.columns))}); "
                "add it to FIXTURE_TICKERS and run rsu_app.fixtures.refresh()."
            )
        if start < _COVERED_START or end > _COVERED_END:
            raise ValueError(
                f"requested {start.date()}..{end.date()} falls outside the price fixture's "
                f"coverage ({FIXTURE_START}..{FIXTURE_END}); widen FIXTURE_START/FIXTURE_END "
                "and run rsu_app.fixtures.refresh()."
            )

        close = self._frame[self._ticker]
        window = close.loc[(close.index >= start) & (close.index < end)]
        out = window.to_frame(name="Close")
        # yfinance returns a tz-aware index; data.py strips the tz, so the zone is moot.
        out = out.tz_localize("UTC")
        return out


class _FixtureYF:
    """Stand-in for the ``yfinance`` module, dispensing fixture-backed tickers."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def Ticker(self, ticker: str) -> _FixtureTicker:  # noqa: N802 - mirror yfinance's API
        """Return a fixture-backed stand-in for ``yfinance.Ticker(ticker)``."""
        return _FixtureTicker(ticker, self._frame)


@contextlib.contextmanager
def patch_yf() -> Iterator[None]:
    """Serve fixture prices in place of live yfinance for the duration of the context.

    Patches the ``yf`` handle in ``rsu_rebalancing.data`` and clears the price cache on
    both entry and exit, so neither real nor fixture data leaks across the boundary.
    """
    from rsu_rebalancing import data

    fake = _FixtureYF(load_fixture_frame())
    original = data.yf
    data._get_prices_cached.cache_clear()
    data.yf = fake
    try:
        yield
    finally:
        data.yf = original
        data._get_prices_cached.cache_clear()


def refresh() -> None:
    """Re-fetch the snapshot from live yfinance and overwrite the checked-in parquet."""
    from rsu_rebalancing import get_price_frame

    frame = get_price_frame(list(FIXTURE_TICKERS), FIXTURE_START, FIXTURE_END)
    _PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(_PARQUET)
