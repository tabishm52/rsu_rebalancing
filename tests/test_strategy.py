"""Tests for the rules running through the backtest engine on synthetic prices."""

import pandas as pd
from pytest import approx

from rsu_rebalancing.backtest import run_rule
from rsu_rebalancing.config import TaxConfig
from rsu_rebalancing.metrics import time_weighted_returns
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
GRANTS = {GRANT_DAY: 2000.0}  # 2000 employer shares, vesting at $11
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


def test_rules_describe_their_target_mix():
    got = [
        HoldEverything().describe("EMP", "IDX"),
        ThresholdRebalance(threshold=1 / 3, band=0.0).describe("EMP", "IDX"),
        SellAllAtVest().describe("EMP", "IDX"),
    ]

    assert got == [
        "Hold everything: 100% EMP",
        "Threshold: 33% EMP / 67% IDX",
        "Sell all at vest: 100% IDX",
    ]


def test_run_rule_records_the_description_for_its_tickers():
    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], HoldEverything())

    assert result.description == "Hold everything: 100% EMP"


def test_threshold_trims_to_target_on_rebalance_day():
    # No tax, so the pre-tax-sized trim lands exactly on target.
    no_tax = TaxConfig(short_term_rate=0.0, long_term_rate=0.0)
    rule = ThresholdRebalance(threshold=1 / 3, band=0.0)

    result = run_rule(PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], rule, no_tax)

    # Before the rebalance, the (only) holding is employer stock -> fraction 1.0.
    assert result.employer_fraction.loc[GRANT_DAY] == 1.0
    # On/after the rebalance day, employer is trimmed to the threshold.
    assert result.employer_fraction.loc[REBALANCE_DAY] == approx(1 / 3)
    assert (result.trades["kind"] == "rebalance").sum() == 1


def test_tax_leaves_employer_fraction_above_target():
    rule = ThresholdRebalance(threshold=1 / 3, band=0.0)

    result = run_rule(
        PRICES, "EMP", "IDX", GRANTS, [REBALANCE_DAY], rule, TaxConfig(short_term_rate=0.2)
    )

    # The sale is sized pre-tax to hit 1/3, but tax is then paid out of the proceeds, so
    # less reaches the index and the employer fraction lands a touch above the target.
    assert result.employer_fraction.loc[REBALANCE_DAY] > 1 / 3
    assert (result.trades["tax_paid"] > 0).any()


def test_rebalance_is_noop_when_already_below_target():
    # Employer halves after the first trim, dropping the fraction below the threshold, so
    # the second rebalance day finds nothing to sell.
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"EMP": [10, 10, 5, 5], "IDX": [100] * 4}, index=dates)
    grants = {dates[0]: 3000.0}  # 3000 employer shares, vesting at $10
    rule = ThresholdRebalance(threshold=1 / 3, band=0.0)

    result = run_rule(prices, "EMP", "IDX", grants, [dates[1], dates[3]], rule)

    # The first rebalance trims to 1/3; by the second, the fallen price already puts
    # employer below 1/3, so only the one rebalance trade is recorded.
    assert (result.trades["kind"] == "rebalance").sum() == 1
    assert result.employer_fraction.loc[dates[3]] < 1 / 3


def test_hysteresis_band_suppresses_a_trim_just_above_target():
    # No tax, so the first trim lands employer exactly on 1/3; the price then rises just
    # enough to push the fraction to ~0.365 -- above target but inside the 0.05 band -- so
    # the second rebalance day leaves it alone.
    no_tax = TaxConfig(short_term_rate=0.0, long_term_rate=0.0)
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"EMP": [10, 10, 10, 11.5], "IDX": [100] * 4}, index=dates)
    grants = {dates[0]: 3000.0}  # 3000 employer shares, vesting at $10
    rule = ThresholdRebalance(threshold=1 / 3, band=0.05)

    result = run_rule(prices, "EMP", "IDX", grants, [dates[1], dates[3]], rule, no_tax)

    # Only the first rebalance fires; the second sits inside the band and is suppressed.
    assert (result.trades["kind"] == "rebalance").sum() == 1
    assert result.employer_fraction.loc[dates[3]] == approx(11.5 / 31.5)
    assert 1 / 3 < result.employer_fraction.loc[dates[3]] <= 1 / 3 + 0.05


def test_hysteresis_band_trims_once_target_is_breached():
    # Same setup, but the price rises far enough (fraction ~0.43) to clear the band, so the
    # second rebalance day fires and trims back down to the target.
    no_tax = TaxConfig(short_term_rate=0.0, long_term_rate=0.0)
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"EMP": [10, 10, 10, 15], "IDX": [100] * 4}, index=dates)
    grants = {dates[0]: 3000.0}  # 3000 employer shares, vesting at $10
    rule = ThresholdRebalance(threshold=1 / 3, band=0.05)

    result = run_rule(prices, "EMP", "IDX", grants, [dates[1], dates[3]], rule, no_tax)

    # Both rebalances fire; the second trims back to the target.
    assert (result.trades["kind"] == "rebalance").sum() == 2
    assert result.employer_fraction.loc[dates[3]] == approx(1 / 3)


def test_grant_vests_before_rebalancing_on_a_shared_day():
    rule = ThresholdRebalance(threshold=1 / 3, band=0.0)

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

    assert result.net_of_tax.values.iloc[-1] < result.market.values.iloc[-1]


def test_final_net_value_equals_gross_with_no_gain():
    # Flat prices: nothing has appreciated, so liquidation owes no tax and net == gross.
    flat = pd.DataFrame({"EMP": [11] * 10, "IDX": [100] * 10}, index=DATES)

    result = run_rule(flat, "EMP", "IDX", GRANTS, [REBALANCE_DAY], HoldEverything())

    assert result.net_of_tax.values.iloc[-1] == approx(result.market.values.iloc[-1])


# Grant at $10, the price rises to $15 and holds across the rebalance, then ticks to $18.
# The flat $15 around the rebalance isolates the tax flow from any price move.
_FLOW_DATES = pd.bdate_range("2020-01-01", periods=6)
_FLOW_PRICES = pd.DataFrame({"EMP": [10, 10, 15, 15, 15, 18], "IDX": [100] * 6}, index=_FLOW_DATES)
_FLOW_GRANTS = {_FLOW_DATES[1]: 2000.0}  # 2000 shares, vesting at $10
_FLOW_REBALANCE = _FLOW_DATES[3]  # price flat at $15 across the trade


def test_realized_tax_is_a_flow_not_a_return():
    # A rebalance realizes gains and pays tax while the price is flat. Tax leaving the
    # portfolio is a withdrawal, so the day's time-weighted return is ~0 -- not a loss the
    # size of the tax.
    tax = TaxConfig(short_term_rate=0.2)
    rule = ThresholdRebalance(threshold=1 / 3, band=0.0)

    result = run_rule(_FLOW_PRICES, "EMP", "IDX", _FLOW_GRANTS, [_FLOW_REBALANCE], rule, tax)
    returns = time_weighted_returns(result.market)

    assert result.market.flows.loc[_FLOW_REBALANCE] == approx(-4000 / 3)  # tax paid, an outflow
    assert returns.loc[_FLOW_REBALANCE] == approx(0.0, abs=1e-9)


def test_grant_is_a_flow_and_price_move_is_a_return():
    tax = TaxConfig(short_term_rate=0.2)
    rule = ThresholdRebalance(threshold=1 / 3, band=0.0)

    result = run_rule(_FLOW_PRICES, "EMP", "IDX", _FLOW_GRANTS, [_FLOW_REBALANCE], rule, tax)
    returns = time_weighted_returns(result.market)

    # The grant is an inflow, not performance; a trade-free price move ($15 -> $18) is.
    assert result.market.flows.loc[_FLOW_DATES[1]] == approx(20_000.0)
    # $2,000 employer gain (666.67 sh x $3) over a prior value of $28,666.67 -> 3/43.
    assert returns.loc[_FLOW_DATES[5]] == approx(3 / 43)


def test_net_basis_rebalance_is_neutral():
    # Realizing gains just prepays tax: at constant prices the net-of-tax value is
    # unchanged, so a rebalance contributes ~0 to the net flow (no spurious return). The
    # realized-tax rate and the liquidation mark are the one config the engine threads
    # through, so they match by construction.
    tax = TaxConfig(short_term_rate=0.2)
    rule = ThresholdRebalance(threshold=1 / 3, band=0.0)

    result = run_rule(_FLOW_PRICES, "EMP", "IDX", _FLOW_GRANTS, [_FLOW_REBALANCE], rule, tax)

    assert result.net_of_tax.flows.loc[_FLOW_REBALANCE] == approx(0.0, abs=1e-9)


def test_net_basis_strips_short_to_long_term_drift():
    # One lot, flat price above its cost, observed either side of the 1-year mark. The tax
    # rate drops short->long with no price move, so the net-of-tax value jumps -- but that
    # jump is a tax-status flow, not performance, so the day's return is ~0.
    dates = pd.DatetimeIndex(["2020-01-01", "2020-12-31", "2021-01-01"])
    prices = pd.DataFrame({"EMP": [10.0, 15.0, 15.0], "IDX": [100.0] * 3}, index=dates)
    grants = {dates[0]: 1000.0}  # 1000 sh, vesting at $10, $5/sh gain at $15
    tax = TaxConfig(short_term_rate=0.4, long_term_rate=0.2, long_term_days=365)

    result = run_rule(prices, "EMP", "IDX", grants, [], HoldEverything(), tax)
    returns = time_weighted_returns(result.net_of_tax)

    assert result.net_of_tax.values.loc[dates[1]] == approx(13_000.0)  # short-term: 15k - 0.4*5k
    assert result.net_of_tax.values.loc[dates[2]] == approx(14_000.0)  # long-term: 15k - 0.2*5k
    assert result.net_of_tax.flows.loc[dates[2]] == approx(1_000.0)  # the drift, stripped as a flow
    assert returns.loc[dates[2]] == approx(0.0, abs=1e-9)
