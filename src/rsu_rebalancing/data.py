"""Historical price access via yfinance.

Fetched prices are memoized in-memory for the session, so repeating a request
(e.g. changing unrelated parameters in the notebook) does not re-hit the network.
"""

from functools import lru_cache
from typing import cast

import pandas as pd
import yfinance as yf


@lru_cache(maxsize=64)
def _get_prices_cached(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch adjusted close prices for one ticker, memoized; see :func:`get_prices`.

    Arguments are pre-normalized by :func:`get_prices` so they serve as stable cache
    keys. Dates use yfinance's half-open semantics (``start`` inclusive, ``end``
    exclusive) -- :func:`get_prices` pre-bumps ``end`` so its own contract stays inclusive.
    """
    frame = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if frame.empty:
        raise ValueError(
            f"No price data returned for {ticker!r} between {start.date()} and {end.date()}. "
            "Check the ticker symbol and date range."
        )

    close = frame["Close"].copy()
    # Drop timezone so the index is plain calendar dates, easy to align and compare.
    idx = cast(pd.DatetimeIndex, close.index)
    close.index = idx.tz_localize(None).normalize()
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

    Raises:
        ValueError: If no price data is returned for the ticker and range.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    # yfinance treats end as exclusive; bump a day so the caller's end date is included.
    fetch_end = end_ts + pd.Timedelta(days=1)

    # Copy so callers can't mutate the memoized object.
    return _get_prices_cached(ticker.upper(), start_ts, fetch_end).copy()


def get_price_frame(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return aligned adjusted close prices for several tickers.

    Each ticker is fetched independently then aligned on the union of trading dates, with
    interior gaps forward-filled so every column has a value on every row. Forward-fill
    cannot fill *leading* gaps, so the frame is trimmed to the first day on which every
    ticker is trading: a ticker that IPO'd after ``start`` shifts the start forward. The
    result is therefore free of NaNs and safe to value directly.

    Args:
        tickers: Symbols to fetch.
        start: First date to include (inclusive).
        end: Last date to include (inclusive).

    Returns:
        A DataFrame indexed by trading date with one column per ticker, beginning on the
        first day all tickers are trading.

    Raises:
        ValueError: If any ticker returns no price data for the range.
    """
    columns = {t.upper(): get_prices(t, start, end) for t in tickers}
    frame = pd.DataFrame(columns)

    # ffill fills interior gaps but not leading ones, so the only NaNs left are before a
    # late-IPO ticker's first quote; dropna trims exactly that pre-IPO window.
    return frame.sort_index().ffill().dropna()
