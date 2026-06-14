"""Tests for the rules running through the simulation engine on synthetic prices."""

import pandas as pd
from pytest import approx

from rsu_rebalancing.simulate import run_rule
from rsu_rebalancing.strategy import HoldEverything, SellAllAtVest, ThresholdRebalance

# A small synthetic price frame: employer doubles, index flat. No network needed.
DATES = pd.bdate_range("2020-01-01", periods=10)
PRICES = pd.DataFrame(
    {
        "EMP": [10, 11, 12, 13, 14, 15, 16, 17, 18, 20],
        "IDX": [100] * 10,
    },
    index=DATES,
)
GRANT_DAY = DATES[1]
GRANTS = {GRANT_DAY: 22_000.0}  # 2000 employer shares at $11
REBALANCE_DAY = DATES[5]  # employer = $15 here


def test_hold_everything_never_sells():
    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], HoldEverything())
    assert (result.trades["kind"] == "grant").all()
    # All value stays in employer stock; fraction is 1.0 throughout (after the grant).
    assert result.employer_fraction.loc[GRANT_DAY:].eq(1.0).all()
    assert result.final_portfolio.index_shares == 0.0


def test_sell_all_at_vest_holds_no_employer_after_grant():
    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], SellAllAtVest())
    # The grant is fully converted to the index the same day.
    assert result.employer_fraction.loc[GRANT_DAY:].eq(0.0).all()
    assert result.final_portfolio.employer_shares == approx(0.0, abs=1e-6)
    assert {"grant", "liquidate"} <= set(result.trades["kind"])


def test_threshold_trims_to_target_on_rebalance_day():
    rule = ThresholdRebalance(threshold=1 / 3)
    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], rule)

    # Before the rebalance, the (only) holding is employer stock -> fraction 1.0.
    assert result.employer_fraction.loc[GRANT_DAY] == 1.0
    # On/after the rebalance day, employer is trimmed to the threshold.
    assert result.employer_fraction.loc[REBALANCE_DAY] == approx(1 / 3)
    assert (result.trades["kind"] == "rebalance").sum() == 1


def test_contributions_recorded_only_on_grant_day():
    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], ThresholdRebalance(1 / 3))
    assert result.contributions.loc[GRANT_DAY] == 22_000.0
    assert result.contributions.drop(GRANT_DAY).eq(0.0).all()
    assert result.contributions.sum() == 22_000.0
