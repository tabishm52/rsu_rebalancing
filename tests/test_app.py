"""Unit tests for the rsu_app presentation layer (config assembly, outperformance spans)."""

import dataclasses

import marimo as mo
import numpy as np
import pandas as pd

from rsu_app import (
    build_backtest_controls,
    build_configs,
    outperformance_spans,
)


def test_build_configs_maps_default_controls():
    controls = build_backtest_controls()

    strategy_cfg, grant_cfg, backtest_cfg, basis = build_configs(controls)

    assert strategy_cfg.employer_ticker == "AAPL"
    assert strategy_cfg.threshold == 0.33  # 33% slider -> fraction
    first_grant = grant_cfg.nominal_grant_dates(backtest_cfg.start, backtest_cfg.end)[0]
    assert first_grant.year == 2011  # 2015 window, backfilled by 4 vesting years
    assert basis == "pre-tax"


def test_build_configs_backfill_off_anchors_grants_at_window_start():
    controls = dataclasses.replace(build_backtest_controls(), backfill=mo.ui.switch(value=False))

    _, grant_cfg, backtest_cfg, _ = build_configs(controls)

    first_grant = grant_cfg.nominal_grant_dates(backtest_cfg.start, backtest_cfg.end)[0]
    assert first_grant.year == 2015


def test_build_configs_after_tax_toggle_sets_basis():
    controls = dataclasses.replace(
        build_backtest_controls(), after_tax_perf=mo.ui.switch(value=True)
    )

    _, _, backtest_cfg, basis = build_configs(controls)

    assert backtest_cfg.after_tax_performance is True
    assert basis == "after-tax"


def _prices_index_ahead_on(ahead_days: set[int], periods: int) -> tuple[pd.Series, pd.Series]:
    """Build prices where the index out-returns the employer exactly on ``ahead_days``.

    With ``lookback=1`` a day is index-ahead iff that day's index return exceeds the
    employer's, so per-day moves give exact control over which days are flagged.
    """
    dates = pd.bdate_range("2020-01-01", periods=periods)
    employer_ret = [(-0.01 if i in ahead_days else 0.01) for i in range(periods)]
    index_ret = [(0.01 if i in ahead_days else 0.0) for i in range(periods)]
    employer_ret[0] = index_ret[0] = 0.0  # no return on the first day
    employer = pd.Series(100 * np.cumprod([1 + r for r in employer_ret]), index=dates)
    index = pd.Series(100 * np.cumprod([1 + r for r in index_ret]), index=dates)
    return employer, index


def test_outperformance_spans_bounds_a_single_run():
    employer, index = _prices_index_ahead_on({1, 2, 3}, periods=6)

    spans = outperformance_spans(employer, index, lookback=1, min_days=2, gap_days=0)

    assert spans == [(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06"))]


def test_outperformance_spans_merges_episodes_within_the_gap():
    # Two index-ahead episodes (Jan 2-6 and Jan 9-13) each span only 4 calendar days, but
    # the 3-day gap between them is within gap_days, so the merged span clears min_days.
    employer, index = _prices_index_ahead_on({1, 2, 3, 6, 7, 8}, periods=10)

    spans = outperformance_spans(employer, index, lookback=1, min_days=10, gap_days=3)

    assert spans == [(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-13"))]


def test_outperformance_spans_drops_short_episodes_beyond_the_gap():
    # Same episodes, but a tighter gap_days leaves them separate; neither alone clears min_days.
    employer, index = _prices_index_ahead_on({1, 2, 3, 6, 7, 8}, periods=10)

    spans = outperformance_spans(employer, index, lookback=1, min_days=10, gap_days=2)

    assert spans == []


def test_outperformance_spans_returns_empty_when_index_never_ahead():
    employer, index = _prices_index_ahead_on(set(), periods=5)

    spans = outperformance_spans(employer, index, lookback=1, min_days=2, gap_days=0)

    assert spans == []
