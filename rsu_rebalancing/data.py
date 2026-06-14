"""Historical price access via yfinance.

Prices are fetched lazily and memoized in-memory for the session, so changing
unrelated parameters in the notebook does not re-hit the network. On-disk caching
is optional: set ``RSU_REBALANCING_CACHE_DIR`` (e.g. in a ``.env`` file) to persist
fetched series as parquet between sessions. The disk cache is keyed by ticker and
date range and is never auto-refreshed -- delete the file (or directory) to refetch.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR_ENV = "RSU_REBALANCING_CACHE_DIR"


def _cache_path(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> Path | None:
    """Return the parquet cache path for this query, or None if caching is disabled."""
    cache_dir = os.environ.get(CACHE_DIR_ENV)
    if not cache_dir:
        return None
    name = f"{ticker.upper()}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    return Path(cache_dir).expanduser() / name


def _download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch split/dividend-adjusted close prices for a single ticker from yfinance."""
    frame = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if frame.empty:
        raise ValueError(
            f"No price data returned for {ticker!r} between {start.date()} and {end.date()}. "
            "Check the ticker symbol and date range."
        )
    close = frame["Close"].copy()
    # Drop timezone so the index is plain calendar dates, easy to align and compare.
    close.index = close.index.tz_localize(None).normalize()
    close.name = ticker.upper()
    return close


@lru_cache(maxsize=64)
def _get_prices_cached(ticker: str, start_iso: str, end_iso: str) -> pd.Series:
    """Memoized core: keyed by hashable ISO date strings (see :func:`get_prices`)."""
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)

    path = _cache_path(ticker, start, end)
    if path is not None and path.exists():
        return pd.read_parquet(path).iloc[:, 0]

    series = _download(ticker, start, end)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_frame().to_parquet(path)
    return series


def get_prices(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.Series:
    """Return adjusted daily close prices for ``ticker`` over ``[start, end]``.

    Args:
        ticker: Stock or ETF symbol, e.g. ``"AAPL"`` or ``"VTI"``.
        start: First date to include (inclusive), as ``YYYY-MM-DD`` or a Timestamp.
        end: Last date to include (inclusive), as ``YYYY-MM-DD`` or a Timestamp.

    Returns:
        A float Series of adjusted close prices indexed by trading date (tz-naive),
        named after the ticker.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    # Copy so callers can't mutate the memoized object.
    return _get_prices_cached(ticker.upper(), start_ts.isoformat(), end_ts.isoformat()).copy()


def get_price_frame(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return aligned adjusted close prices for several tickers.

    Each ticker is fetched independently then aligned on the union of trading dates,
    with gaps forward-filled so every column has a value on every row.

    Args:
        tickers: Symbols to fetch.
        start: First date to include (inclusive).
        end: Last date to include (inclusive).

    Returns:
        A DataFrame indexed by trading date with one column per ticker.
    """
    columns = {t.upper(): get_prices(t, start, end) for t in tickers}
    frame = pd.DataFrame(columns)
    return frame.sort_index().ffill().dropna(how="all")
