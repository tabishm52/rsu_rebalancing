"""Historical price access via yfinance.

Prices are fetched lazily and memoized in-memory for the session, so changing
unrelated parameters in the notebook does not re-hit the network.
"""

from functools import lru_cache

import pandas as pd
import yfinance as yf


@lru_cache(maxsize=64)
def _get_prices_cached(ticker: str, start_iso: str, end_iso: str) -> pd.Series:
    """Fetch split/dividend-adjusted close prices for a single ticker, memoized.

    Keyed by hashable ISO date strings (see :func:`get_prices`).
    """
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
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
