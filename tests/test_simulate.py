"""Orchestration tests for run_backtest, with prices mocked so no network is hit.

run_rule (the per-day engine) is exercised in test_strategy.py; here we pin only what
run_backtest adds on top: which rules it runs, how it keys them, and that every strategy
sees the identical grant stream.
"""

import pandas as pd

from rsu_rebalancing import simulate
from rsu_rebalancing.config import GrantSchedule, SimConfig, StrategyConfig

_DATES = pd.bdate_range("2020-01-01", "2020-12-31")
_PRICES = pd.DataFrame({"EMP": 10.0, "IDX": 100.0}, index=_DATES)


def _run_backtest(monkeypatch):
    monkeypatch.setattr(simulate, "get_price_frame", lambda *args, **kwargs: _PRICES)
    strategy = StrategyConfig(employer_ticker="EMP", index_ticker="IDX", threshold=1 / 3)
    schedule = GrantSchedule(annual_dollars=50_000, start_year=2020, end_year=2020)
    sim = SimConfig(start="2020-01-01", end="2020-12-31")

    return simulate.run_backtest(strategy, schedule, sim)


def test_run_backtest_keys_each_rule_by_name(monkeypatch):
    results = _run_backtest(monkeypatch)

    assert set(results) == {"Threshold 33%", "Hold everything", "Sell all at vest"}


def test_run_backtest_feeds_identical_grants_to_every_strategy(monkeypatch):
    results = _run_backtest(monkeypatch)

    contributions = [result.contributions for result in results.values()]
    for other in contributions[1:]:
        pd.testing.assert_series_equal(other, contributions[0])
