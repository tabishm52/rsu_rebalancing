"""Matplotlib figures for the backtest notebook."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rsu_rebalancing import (
    BacktestConfig,
    StrategyConfig,
    get_price_frame,
    growth_of_one,
    time_weighted_returns,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from rsu_rebalancing import BacktestResult

sns.set_theme()


def build_concentration_figure(result: BacktestResult, threshold: float) -> Figure:
    """Employer fraction of holdings over time, with the threshold and sale events."""
    frac = result.employer_fraction

    trades = result.trades
    sale_dates = trades.loc[trades["kind"].isin(["rebalance", "liquidate"]), "date"]
    pre_sale_frac = frac.shift(1).loc[sale_dates]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frac.index.to_numpy(), frac.to_numpy(), color="#d62728")
    ax.axhline(threshold, linestyle="--", color="gray")
    ax.scatter(
        sale_dates,
        pre_sale_frac,
        marker="*",
        s=120,
        color="#1f77b4",
        zorder=3,
        label="rebalance events",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Employer fraction of holdings")
    ax.legend()

    fig.tight_layout()
    return fig


def outperformance_spans(
    employer: pd.Series,
    index: pd.Series,
    lookback: int = 63,
    min_days: int = 90,
    gap_days: int = 10,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Date spans where the index has out-returned the employer stock.

    A day is "index-ahead" when the index's return over the trailing ``lookback`` *trading*
    days exceeds the employer's. Consecutive index-ahead days form an episode; episodes
    separated by a gap of at most ``gap_days`` *calendar* days are merged, and only the
    merged spans lasting at least ``min_days`` *calendar* days survive, dropping daily
    noise. Returns ``(start, end)`` pairs. This shades the relative *declines* (employer
    losing ground), not the recoveries.

    Note ``lookback`` counts trading days, while ``min_days`` and ``gap_days`` count
    calendar days. The defaults for ``lookback`` and ``min_days`` match — 63 trading days
    is ~90 calendar days.
    """
    ahead = employer.pct_change(lookback) < index.pct_change(lookback)
    runs = (ahead != ahead.shift()).cumsum()
    segments = [(run.index[0], run.index[-1]) for _, run in ahead.groupby(runs) if run.iloc[0]]
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in segments:
        if spans and (start - spans[-1][1]).days <= gap_days:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return [(start, end) for start, end in spans if (end - start).days >= min_days]


def build_performance_figure(
    results: dict[str, BacktestResult],
    strategy_cfg: StrategyConfig,
    backtest_cfg: BacktestConfig,
    basis: str,
) -> Figure:
    """Growth-of-100 curves per strategy, shading where the index out-returned employer.

    Each curve is labelled by ``BacktestResult.description`` (e.g. "Threshold: 33% AAPL /
    67% VTI"), so the legend stands alone; the generic result names stay as-is for the
    table and lookups. ``basis`` titles the y-axis (see ``build_configs``).
    """
    after_tax = backtest_cfg.after_tax_performance
    perf = {
        res.description: (res.net_of_tax if after_tax else res.market) for res in results.values()
    }
    curves = {label: growth_of_one(time_weighted_returns(p)) for label, p in perf.items()}
    growth_df = pd.DataFrame(curves)

    # Shade stretches where the diversified index out-returned the employer stock — the
    # episodes the threshold strategy's downside protection is meant to catch.
    prices = get_price_frame(
        [strategy_cfg.employer_ticker, strategy_cfg.index_ticker],
        backtest_cfg.start,
        backtest_cfg.end,
    )
    spans = outperformance_spans(
        prices[strategy_cfg.employer_ticker.upper()],
        prices[strategy_cfg.index_ticker.upper()],
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    (growth_df * 100).plot(ax=ax)
    band = None
    for start, end in spans:
        # axvspan accepts datetime-like x at runtime; the stub only admits float.
        band = ax.axvspan(start, end, color="#d62728", alpha=0.12, zorder=0)  # pyright: ignore[reportArgumentType]
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Index, start = 100 ({basis}, time-weighted)")
    handles, labels = ax.get_legend_handles_labels()
    if band is not None:
        handles.append(band)
        labels.append("index outperforming employer")
    ax.legend(handles, labels, loc="upper left")
    fig.tight_layout()
    return fig
