"""Orchestration tests for run_backtest, with prices mocked so no network is hit.

run_rule (the per-day engine) is exercised in test_strategy.py; here we pin only what
run_backtest adds on top: which rules it runs, how it keys them, and that every strategy
sees the identical grant stream.
"""

import pandas as pd
from pytest import approx

from rsu_rebalancing import simulate
from rsu_rebalancing.config import GrantSchedule, SimConfig, StrategyConfig
from rsu_rebalancing.portfolio import Portfolio
from rsu_rebalancing.simulate import SimResult

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

    grants = [result.gross_grants for result in results.values()]
    for other in grants[1:]:
        pd.testing.assert_series_equal(other, grants[0])


def test_gross_grants_count_only_grants():
    # gross_grants is the "total contributed" reporting figure: only grant gross_value,
    # summed per day and aligned (zero-filled) to values.index. Rebalances and tax are not
    # contributions.
    trades = pd.DataFrame(
        {
            "kind": ["grant", "grant", "rebalance"],
            "date": [_DATES[0], _DATES[0], _DATES[1]],
            "gross_value": [100.0, 50.0, 200.0],
            "tax_paid": [0.0, 0.0, 30.0],
        }
    )
    result = SimResult(
        name="x",
        values=pd.Series(0.0, index=_DATES[:3]),
        employer_fraction=pd.Series(0.0, index=_DATES[:3]),
        trades=trades,
        final_portfolio=Portfolio(),
    )

    grants = result.gross_grants

    assert list(grants.index) == list(_DATES[:3])
    assert grants.iloc[0] == approx(150.0)  # two grants on day 0, summed
    assert grants.iloc[1] == 0.0  # rebalance is not a contribution
    assert grants.iloc[2] == 0.0  # no trades -> filled with 0


def test_flows_net_grants_against_tax_paid():
    # flows is the TWR cash flow: grants in, tax out. Tax is a withdrawal, not negative
    # performance, so it is netted out of the daily flow.
    trades = pd.DataFrame(
        {
            "kind": ["grant", "grant", "rebalance"],
            "date": [_DATES[0], _DATES[0], _DATES[1]],
            "gross_value": [100.0, 50.0, 200.0],
            "tax_paid": [0.0, 0.0, 30.0],
        }
    )
    result = SimResult(
        name="x",
        values=pd.Series(0.0, index=_DATES[:3]),
        employer_fraction=pd.Series(0.0, index=_DATES[:3]),
        trades=trades,
        final_portfolio=Portfolio(),
    )

    flows = result.flows

    assert flows.iloc[0] == approx(150.0)  # two grants on day 0
    assert flows.iloc[1] == approx(-30.0)  # tax paid is an outflow
    assert flows.iloc[2] == 0.0  # no trades -> 0
