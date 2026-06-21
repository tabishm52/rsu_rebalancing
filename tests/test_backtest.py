"""Orchestration tests for run_backtest, with prices mocked so no network is hit.

run_rule (the per-day engine) is exercised in test_strategy.py; here we pin only what
run_backtest adds on top: which rules it runs, how it keys them, and that every strategy
sees the identical grant stream.
"""

import pandas as pd
from pytest import approx

from rsu_rebalancing import backtest
from rsu_rebalancing.backtest import BacktestResult, PerfSeries
from rsu_rebalancing.config import BacktestConfig, GrantConfig, StrategyConfig

_DATES = pd.bdate_range("2020-01-01", "2020-12-31")
_PRICES = pd.DataFrame({"EMP": 10.0, "IDX": 100.0}, index=_DATES)
# Employer prices reaching back to the 2019 award, so it can be priced into shares.
_AWARD_PRICES = pd.Series(10.0, index=pd.bdate_range("2019-01-01", "2020-12-31"), name="EMP")


def _run_backtest(monkeypatch):
    monkeypatch.setattr(backtest, "get_price_frame", lambda *args, **kwargs: _PRICES)
    monkeypatch.setattr(backtest, "get_prices", lambda *args, **kwargs: _AWARD_PRICES)
    strategy = StrategyConfig(employer_ticker="EMP", index_ticker="IDX", threshold=1 / 3)
    schedule = GrantConfig(grant_dollars=50_000, vesting_years=1)
    backtest_cfg = BacktestConfig(start="2020-01-01", end="2020-12-31")

    return backtest.run_backtest(strategy, schedule, backtest_cfg)


def test_run_backtest_keys_each_rule_by_name(monkeypatch):
    results = _run_backtest(monkeypatch)

    assert set(results) == {"Threshold 33%", "Hold everything", "Sell all at vest"}


def test_run_backtest_feeds_identical_grants_to_every_strategy(monkeypatch):
    results = _run_backtest(monkeypatch)

    grants = [result.vested_contributions for result in results.values()]
    for other in grants[1:]:
        pd.testing.assert_series_equal(other, grants[0])


def test_vested_contributions_count_only_grants():
    # vested_contributions is the "total contributed" reporting figure: only grants'
    # employer_shares * employer_price, summed per day and aligned (zero-filled) to
    # market.values.index. Rebalances and tax are not contributions.
    trades = pd.DataFrame(
        {
            "kind": ["grant", "grant", "rebalance"],
            "date": [_DATES[0], _DATES[0], _DATES[1]],
            "employer_shares": [100.0, 50.0, -200.0],
            "employer_price": [1.0, 1.0, 1.0],
            "tax_paid": [0.0, 0.0, 30.0],
        }
    )
    result = BacktestResult(
        name="x",
        description="x",
        market=PerfSeries(values=pd.Series(0.0, index=_DATES[:3])),
        trades=trades,
    )

    grants = result.vested_contributions

    assert list(grants.index) == list(_DATES[:3])
    assert grants.iloc[0] == approx(150.0)  # two grants on day 0, summed
    assert grants.iloc[1] == 0.0  # rebalance is not a contribution
    assert grants.iloc[2] == 0.0  # no trades -> filled with 0


def test_taxes_paid_sums_every_kind_per_day():
    # taxes_paid is the "total taxes actually paid" figure: every row's tax_paid (grant
    # withholding plus realized cap-gains on sales), summed per day and aligned (zero-filled)
    # to market.values.index. The hypothetical end-of-run liquidation tax is not included.
    trades = pd.DataFrame(
        {
            "kind": ["grant", "rebalance", "grant"],
            "date": [_DATES[0], _DATES[0], _DATES[1]],
            "tax_paid": [10.0, 30.0, 5.0],
        }
    )
    result = BacktestResult(
        name="x",
        description="x",
        market=PerfSeries(values=pd.Series(0.0, index=_DATES[:3])),
        trades=trades,
    )

    taxes = result.taxes_paid

    assert list(taxes.index) == list(_DATES[:3])
    assert taxes.iloc[0] == approx(40.0)  # grant withholding 10 + rebalance tax 30
    assert taxes.iloc[1] == approx(5.0)  # grant withholding on day 1
    assert taxes.iloc[2] == 0.0  # no trades -> filled with 0
