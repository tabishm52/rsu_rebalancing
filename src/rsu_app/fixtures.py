"""Frozen price fixture for deterministic, network-free backtests.

The README assets, the notebook smoke test, and the offline test suite run against a
checked-in snapshot of prices instead of live yfinance, so results are reproducible and CI
never touches the network.

The snapshot holds whatever tickers and date window ``refresh`` was last told to fetch.
When called from ``assets/generate_assets.py``, these are derived from the notebook
defaults. ``patch_yf`` swaps it in behind ``rsu_rebalancing.data`` and validates every
request against the snapshot's own coverage.
"""

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

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
                "add it to the tickers passed to rsu_app.fixtures.refresh() and re-fetch."
            )
        # Half-open coverage bound: yfinance's end is exclusive and data.py bumps the
        # caller's inclusive end by a day, so the snapshot serves up to its last date + 1.
        covered_start = self._frame.index.min()
        covered_end = self._frame.index.max() + pd.Timedelta(days=1)
        if start < covered_start or end > covered_end:
            raise ValueError(
                f"requested {start.date()}..{end.date()} falls outside the price fixture's "
                f"coverage ({covered_start.date()}..{self._frame.index.max().date()}); widen "
                "the window passed to rsu_app.fixtures.refresh() and re-fetch."
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


def refresh(tickers: Sequence[str], start: str | pd.Timestamp, end: str | pd.Timestamp) -> None:
    """Re-fetch the snapshot from live yfinance and overwrite the checked-in parquet.

    The fetched window is padded a week beyond ``start``/``end`` on each side. The coverage
    check compares raw request dates against the snapshot's first/last *trading* day, so the
    pad guarantees the snapshot brackets requests whose exact boundary lands on a weekend or
    holiday (e.g. a March-1 grant date). The backtest slices to its own window, so the pad
    adds no rows it actually reads.
    """
    from rsu_rebalancing import get_price_frame

    pad = pd.Timedelta(days=7)
    frame = get_price_frame(list(tickers), pd.Timestamp(start) - pad, pd.Timestamp(end) + pad)
    _PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(_PARQUET)
