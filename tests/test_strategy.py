"""Tests for the rules running through the simulation engine on synthetic prices."""

import pandas as pd
from pytest import approx

from rsu_rebalancing.config import TaxConfig
from rsu_rebalancing.simulate import run_rule
from rsu_rebalancing.strategy import HoldEverything, SellAllAtVest, ThresholdRebalance

# A small synthetic price frame: employer doubles, index flat.
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
    # No tax, so the pre-tax-sized trim lands exactly on target.
    no_tax = TaxConfig(short_term_rate=0.0, long_term_rate=0.0)
    rule = ThresholdRebalance(threshold=1 / 3, tax_config=no_tax)

    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], rule)

    # Before the rebalance, the (only) holding is employer stock -> fraction 1.0.
    assert result.employer_fraction.loc[GRANT_DAY] == 1.0
    # On/after the rebalance day, employer is trimmed to the threshold.
    assert result.employer_fraction.loc[REBALANCE_DAY] == approx(1 / 3)
    assert (result.trades["kind"] == "rebalance").sum() == 1


def test_tax_leaves_employer_fraction_above_target():
    rule = ThresholdRebalance(threshold=1 / 3, tax_config=TaxConfig(short_term_rate=0.2))

    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], rule)

    # The sale is sized pre-tax to hit 1/3, but tax is then paid out of the proceeds, so
    # less reaches the index and the employer fraction lands a touch above the target.
    assert result.employer_fraction.loc[REBALANCE_DAY] > 1 / 3
    assert (result.trades["tax_paid"] > 0).any()


def test_rebalance_is_noop_when_already_below_target():
    # Employer halves after the first trim, dropping the fraction below the threshold, so
    # the second rebalance day finds nothing to sell.
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"EMP": [10, 10, 5, 5], "IDX": [100] * 4}, index=dates)
    grants = {dates[0]: 30_000.0}  # 3000 employer shares at $10
    rule = ThresholdRebalance(threshold=1 / 3)

    result = run_rule(prices, "EMP", "IDX", grants, [dates[1], dates[3]], rule)

    # The first rebalance trims to 1/3; by the second, the fallen price already puts
    # employer below 1/3, so only the one rebalance trade is recorded.
    assert (result.trades["kind"] == "rebalance").sum() == 1
    assert result.employer_fraction.loc[dates[3]] < 1 / 3


def test_grant_vests_before_rebalancing_on_a_shared_day():
    rule = ThresholdRebalance(threshold=1 / 3)

    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [GRANT_DAY], rule)

    # Grant and rebalance fall on the same day: the grant must vest first, so the trim
    # has a position to size against and lands on the target. (Were the order reversed,
    # the trim would find an empty portfolio and the grant would stay untrimmed at 1.0.)
    assert list(result.trades["kind"]) == ["grant", "rebalance"]
    assert result.employer_fraction.loc[GRANT_DAY] == approx(1 / 3)


def test_final_net_value_haircuts_unrealized_gains():
    # Employer doubles over the run, so Hold-everything ends with a large unrealized gain;
    # the net-of-tax liquidation value sits below the gross portfolio value.
    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], HoldEverything())

    assert result.final_net_value < result.values.iloc[-1]


def test_final_net_value_equals_gross_with_no_gain():
    # Flat prices: nothing has appreciated, so liquidation owes no tax and net == gross.
    flat = pd.DataFrame({"EMP": [11] * 10, "IDX": [100] * 10}, index=DATES)

    result = run_rule(flat, "EMP", "IDX", GRANTS, [REBALANCE_DAY], HoldEverything())

    assert result.final_net_value == approx(result.values.iloc[-1])
