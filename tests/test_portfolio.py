"""Tests for portfolio bookkeeping, threshold sells, FIFO tax, and liquidation."""

import pandas as pd
from pytest import approx

from rsu_rebalancing.config import TaxConfig
from rsu_rebalancing.portfolio import Portfolio

DATE = pd.Timestamp("2020-03-02")


def test_grant_adds_shares_at_price():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=90_000, employer_price=9.0)

    assert pf.employer_shares == 10_000
    assert pf.employer_value(9.0) == 90_000
    assert pf.employer_fraction(9.0, index_price=100.0) == 1.0


def test_empty_portfolio_fraction_is_zero():
    pf = Portfolio()

    # Guard against dividing by zero total value -> fraction is 0, not an error.
    assert pf.employer_fraction(10.0, 10.0) == 0.0


def test_sell_to_fraction_hits_target_without_tax():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=90_000, employer_price=9.0)  # all employer, $90k

    no_tax = TaxConfig(short_term_rate=0.0, long_term_rate=0.0)
    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=1 / 3, employer_price=9.0, index_price=9.0, tax_config=no_tax
    )

    assert trade is not None
    assert trade.tax_paid == 0.0
    # Pre-tax targeting with no tax leak: employer is exactly one third afterward.
    assert pf.employer_fraction(9.0, 9.0) == approx(1 / 3)


def test_sell_to_fraction_hits_target_from_mixed_portfolio():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=90_000, employer_price=9.0)  # $90k employer
    pf.index_shares = 30_000 / 10.0  # plus $30k index -> employer is 75% of $120k

    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=1 / 3, employer_price=9.0, index_price=10.0, tax_config=TaxConfig()
    )

    assert trade is not None
    # Trimming an already-mixed portfolio (distinct employer/index prices) still
    # lands exactly on target: sell $50k of employer, total value is conserved.
    assert trade.gross_value == approx(50_000)
    assert pf.employer_fraction(9.0, 10.0) == approx(1 / 3)


def test_sell_to_fraction_noop_when_below_target():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=10_000, employer_price=10.0)  # $10k employer
    pf.index_shares = 9_000 / 10.0  # plus $9k index -> employer is 52.6%, below 0.6

    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=0.6, employer_price=10.0, index_price=10.0, tax_config=TaxConfig()
    )

    assert trade is None


def test_fifo_tax_on_realized_gain():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=10_000, employer_price=10.0)  # 1000 shares, basis $10

    # Price doubles to $20: position now worth $20k, no index yet. Same-day sale, so the
    # gain is short-term.
    trade = pf.sell_employer_to_fraction(
        DATE,
        target_fraction=0.5,
        employer_price=20.0,
        index_price=20.0,
        tax_config=TaxConfig(short_term_rate=0.20),
    )

    assert trade is not None
    # Sell $10k of stock = 500 shares; gain = 500 * ($20 - $10) = $5,000; tax = $1,000.
    assert trade.gross_value == approx(10_000)
    assert trade.tax_paid == approx(1_000)
    # Net $9,000 reinvested in the index.
    assert trade.index_dollars_in == approx(9_000)
    assert pf.index_value(20.0) == approx(9_000)
    assert pf.employer_shares == approx(500)


def test_fifo_tax_across_multiple_lots():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=10_000, employer_price=10.0)  # lot 1: 1000 sh @ $10
    pf.add_grant(DATE, dollars=20_000, employer_price=20.0)  # lot 2: 1000 sh @ $20

    # At $20, the position is worth $40k. Trim to 25% -> sell $30k = 1500 shares,
    # which spans lot 1 entirely (1000 sh) plus 500 sh of lot 2.
    trade = pf.sell_employer_to_fraction(
        DATE,
        target_fraction=0.25,
        employer_price=20.0,
        index_price=20.0,
        tax_config=TaxConfig(short_term_rate=0.20),
    )

    assert trade is not None
    assert trade.gross_value == approx(30_000)
    # FIFO: gain is realized against the oldest lot's $10 basis first.
    # lot 1: 1000 * ($20-$10) = $10,000; lot 2: 500 * ($20-$20) = $0. Tax = 20% * $10k.
    # (Under LIFO the gain would be only $5k and tax $1k, so this pins the ordering.)
    assert trade.tax_paid == approx(2_000)
    # The surviving lot is the newer $20-basis one, with 500 shares left.
    assert pf.employer_shares == approx(500)
    assert len(pf.employer_lots) == 1
    assert pf.employer_lots[0].cost_per_share == 20.0
    assert pf.employer_lots[0].shares == approx(500)


def test_liquidate_sells_all_employer():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=50_000, employer_price=25.0)

    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=0.0, employer_price=25.0, index_price=50.0, tax_config=TaxConfig()
    )

    assert trade is not None
    assert trade.kind == "liquidate"
    assert pf.employer_shares == approx(0.0, abs=1e-6)
    assert pf.index_value(50.0) == approx(50_000)


def test_long_term_gain_taxed_at_long_term_rate():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=10_000, employer_price=10.0)  # 1000 shares, basis $10
    sale_date = DATE + pd.Timedelta(days=400)  # held > 365 days -> long-term

    trade = pf.sell_employer_to_fraction(
        sale_date,
        target_fraction=0.5,
        employer_price=20.0,
        index_price=20.0,
        tax_config=TaxConfig(short_term_rate=0.40, long_term_rate=0.20),
    )

    assert trade is not None
    # Sell 500 sh; gain = 500 * ($20 - $10) = $5,000; held long, so taxed at 20% = $1,000.
    assert trade.tax_paid == approx(1_000)


def test_short_term_gain_taxed_at_short_term_rate():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=10_000, employer_price=10.0)  # 1000 shares, basis $10
    sale_date = DATE + pd.Timedelta(days=200)  # held <= 365 days -> short-term

    trade = pf.sell_employer_to_fraction(
        sale_date,
        target_fraction=0.5,
        employer_price=20.0,
        index_price=20.0,
        tax_config=TaxConfig(short_term_rate=0.40, long_term_rate=0.20),
    )

    assert trade is not None
    # Same $5,000 gain, but held short, so taxed at 40% = $2,000.
    assert trade.tax_paid == approx(2_000)


def test_mixed_holding_periods_taxed_per_lot():
    pf = Portfolio()
    pf.add_grant(DATE, dollars=10_000, employer_price=10.0)  # lot 1: 1000 sh @ $10 (old)
    later = DATE + pd.Timedelta(days=400)
    pf.add_grant(later, dollars=20_000, employer_price=20.0)  # lot 2: 1000 sh @ $20 (new)

    # Sell on a day that is long-term for lot 1 but short-term for lot 2.
    sale_date = later + pd.Timedelta(days=100)
    trade = pf.sell_employer_to_fraction(
        sale_date,
        target_fraction=0.25,
        employer_price=30.0,
        index_price=30.0,
        tax_config=TaxConfig(short_term_rate=0.40, long_term_rate=0.20),
    )

    assert trade is not None
    # At $30 the position is worth $60k; trim to 25% -> sell $45k = 1500 shares: all of
    # lot 1 (long-term) plus 500 sh of lot 2 (short-term).
    # lot 1: 1000 * ($30-$10) = $20,000 @ 20% = $4,000.
    # lot 2: 500 * ($30-$20) = $5,000 @ 40% = $2,000.
    assert trade.tax_paid == approx(6_000)
