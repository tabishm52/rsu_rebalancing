"""Tests for portfolio bookkeeping, threshold sells, FIFO tax, and liquidation."""

import pandas as pd
from pytest import approx

from rsu_rebalancing.portfolio import Portfolio

DATE = pd.Timestamp("2020-03-02")


def test_grant_adds_shares_at_price():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=90_000, employer_price=9.0)
    assert pf.employer_shares == 10_000
    assert pf.employer_value(9.0) == 90_000
    assert pf.employer_fraction(9.0, index_price=100.0) == 1.0


def test_sell_to_fraction_hits_target_without_tax():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=90_000, employer_price=9.0)  # all employer, $90k
    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=1 / 3, employer_price=9.0, index_price=9.0, capital_gains_rate=0.0
    )
    assert trade is not None
    assert trade.tax_paid == 0.0
    # Pre-tax targeting with no tax leak: employer is exactly one third afterward.
    assert pf.employer_fraction(9.0, 9.0) == approx(1 / 3)


def test_sell_to_fraction_noop_when_below_target():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=10_000, employer_price=10.0)  # $10k employer
    pf.index_shares = 9_000 / 10.0  # plus $9k index -> employer is 52.6%, below 0.6
    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=0.6, employer_price=10.0, index_price=10.0, capital_gains_rate=0.0
    )
    assert trade is None


def test_fifo_tax_on_realized_gain():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=10_000, employer_price=10.0)  # 1000 shares, basis $10
    # Price doubles to $20: position now worth $20k, no index yet.
    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=0.5, employer_price=20.0, index_price=20.0, capital_gains_rate=0.20
    )
    assert trade is not None
    # Sell $10k of stock = 500 shares; gain = 500 * ($20 - $10) = $5,000; tax = $1,000.
    assert trade.gross_value == approx(10_000)
    assert trade.tax_paid == approx(1_000)
    # Net $9,000 reinvested in the index.
    assert trade.index_dollars_in == approx(9_000)
    assert pf.index_value(20.0) == approx(9_000)
    assert pf.employer_shares == approx(500)


def test_liquidate_sells_all_employer():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=50_000, employer_price=25.0)
    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=0.0, employer_price=25.0, index_price=50.0, capital_gains_rate=0.0
    )
    assert trade is not None
    assert trade.kind == "liquidate"
    assert pf.employer_shares == approx(0.0, abs=1e-6)
    assert pf.index_value(50.0) == approx(50_000)
