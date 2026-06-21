"""Tests for portfolio bookkeeping, threshold sells, min-tax lot selection, and liquidation."""

import pandas as pd
from pytest import approx

from rsu_rebalancing.config import TaxConfig
from rsu_rebalancing.portfolio import Portfolio

DATE = pd.Timestamp("2020-03-02")

# Lot-setup grants withhold nothing, so a grant of N shares lands N shares to test sells
# against; the withholding haircut has its own dedicated test.
NO_WITHHOLDING = TaxConfig(ordinary_income_rate=0.0)


def test_grant_adds_shares_at_price():
    pf = Portfolio()
    pf.add_grant(DATE, shares=10_000, employer_price=9.0, tax_config=NO_WITHHOLDING)

    assert pf.employer_shares == 10_000
    assert pf.employer_value(9.0) == 90_000
    assert pf.employer_fraction(9.0, index_price=100.0) == 1.0


def test_grant_withholds_sell_to_cover_and_records_the_tax():
    pf = Portfolio()

    trade = pf.add_grant(
        DATE, shares=10_000, employer_price=10.0, tax_config=TaxConfig(ordinary_income_rate=0.25)
    )

    # Gross vest is 10,000 sh @ $10 = $100k; a 25% sell-to-cover slice is withheld, so 7,500
    # shares are kept and the $25k of withholding is documented on the row.
    assert pf.employer_shares == approx(7_500)
    assert trade.employer_shares == approx(7_500)
    assert trade.employer_shares * trade.employer_price == approx(75_000)
    assert trade.tax_paid == approx(25_000)


def test_empty_portfolio_fraction_is_zero():
    pf = Portfolio()

    # Guard against dividing by zero total value -> fraction is 0, not an error.
    assert pf.employer_fraction(10.0, 10.0) == 0.0


def test_sell_to_fraction_hits_target_without_tax():
    pf = Portfolio()
    pf.add_grant(
        DATE, shares=10_000, employer_price=9.0, tax_config=NO_WITHHOLDING
    )  # all employer, $90k

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
    pf.add_grant(
        DATE, shares=10_000, employer_price=9.0, tax_config=NO_WITHHOLDING
    )  # $90k employer
    pf.buy_index(DATE, dollars=30_000, index_price=10.0)  # +$30k index -> employer is 75% of $120k

    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=1 / 3, employer_price=9.0, index_price=10.0, tax_config=TaxConfig()
    )

    assert trade is not None
    # Trimming an already-mixed portfolio (distinct employer/index prices) still
    # lands exactly on target: sell $50k of employer, total value is conserved.
    assert abs(trade.employer_shares) * trade.employer_price == approx(50_000)
    assert pf.employer_fraction(9.0, 10.0) == approx(1 / 3)


def test_sell_to_fraction_noop_when_below_target():
    pf = Portfolio()
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # $10k employer
    pf.buy_index(DATE, dollars=9_000, index_price=10.0)  # +$9k index -> employer 52.6%, below 0.6

    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=0.6, employer_price=10.0, index_price=10.0, tax_config=TaxConfig()
    )

    assert trade is None


def test_tax_on_realized_gain():
    pf = Portfolio()
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # 1000 shares, basis $10

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
    assert abs(trade.employer_shares) * trade.employer_price == approx(10_000)
    assert trade.tax_paid == approx(1_000)
    # Net $9,000 reinvested in the index.
    assert trade.index_invested == approx(9_000)
    assert pf.index_value(20.0) == approx(9_000)
    assert pf.employer_shares == approx(500)


def test_min_tax_sells_lower_gain_lot_first():
    pf = Portfolio()
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # lot 1: 1000 sh @ $10
    pf.add_grant(
        DATE, shares=1_000, employer_price=20.0, tax_config=NO_WITHHOLDING
    )  # lot 2: 1000 sh @ $20

    # At $20, the position is worth $40k. Trim to 25% -> sell $30k = 1500 shares,
    # which spans lot 2 entirely (1000 sh) plus 500 sh of lot 1.
    trade = pf.sell_employer_to_fraction(
        DATE,
        target_fraction=0.25,
        employer_price=20.0,
        index_price=20.0,
        tax_config=TaxConfig(short_term_rate=0.20),
    )

    assert trade is not None
    assert abs(trade.employer_shares) * trade.employer_price == approx(30_000)
    # Min-tax: the zero-gain $20 lot sells first (no tax), then 500 sh of the $10 lot.
    # lot 2: 1000 * ($20-$20) = $0; lot 1: 500 * ($20-$10) = $5,000. Tax = 20% * $5k.
    # (Under FIFO the gain would be $10k and tax $2k, so this pins the ordering.)
    assert trade.tax_paid == approx(1_000)
    # The surviving lot is the older $10-basis one, with 500 shares left.
    assert pf.employer_shares == approx(500)
    assert len(pf.employer_lots) == 1
    assert pf.employer_lots[0].cost_per_share == 10.0
    assert pf.employer_lots[0].shares == approx(500)


def test_liquidate_sells_all_employer():
    pf = Portfolio()
    pf.add_grant(DATE, shares=2_000, employer_price=25.0, tax_config=NO_WITHHOLDING)

    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=0.0, employer_price=25.0, index_price=50.0, tax_config=TaxConfig()
    )

    assert trade is not None
    assert trade.kind == "liquidate"
    assert pf.employer_shares == approx(0.0, abs=1e-6)
    assert pf.index_value(50.0) == approx(50_000)


def test_long_term_gain_taxed_at_long_term_rate():
    pf = Portfolio()
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # 1000 shares, basis $10
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
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # 1000 shares, basis $10
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
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # lot 1: 1000 sh @ $10 (old)
    later = DATE + pd.Timedelta(days=400)
    pf.add_grant(
        later, shares=1_000, employer_price=20.0, tax_config=NO_WITHHOLDING
    )  # lot 2: 1000 sh @ $20 (new)

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


def test_min_tax_orders_by_rate_not_just_basis():
    pf = Portfolio()
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # lot 1: 1000 sh @ $10 (old)
    later = DATE + pd.Timedelta(days=400)
    pf.add_grant(
        later, shares=1_000, employer_price=18.0, tax_config=NO_WITHHOLDING
    )  # lot 2: 1000 sh @ $18 (new)

    # Sell on a day that is long-term for lot 1 but short-term for lot 2. Sell 500 sh.
    # lot 1 (lower basis, but long-term): ($20-$10) * 0.20 = $2.00/sh tax.
    # lot 2 (higher basis, but short-term): ($20-$18) * 0.40 = $0.80/sh tax.
    # Rate, not basis, makes lot 2 cheaper to sell, so min-tax drains it first.
    # Position is $40k; trim to 75% -> sell $10k = 500 sh.
    sale_date = later + pd.Timedelta(days=100)
    trade = pf.sell_employer_to_fraction(
        sale_date,
        target_fraction=0.75,
        employer_price=20.0,
        index_price=20.0,
        tax_config=TaxConfig(short_term_rate=0.40, long_term_rate=0.20),
    )

    assert trade is not None
    # 500 sh from lot 2: 500 * $0.80 = $400. The older, lower-basis lot is untouched.
    assert trade.tax_paid == approx(400)
    assert pf.employer_lots[0].cost_per_share == 10.0
    assert pf.employer_lots[0].shares == approx(1000)


def test_min_tax_prefers_fresh_grant_with_near_zero_gain():
    pf = Portfolio()
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # old lot: 1000 sh @ $10
    # The common real flow: a fresh grant vests, then a rebalance lands days later. The
    # fresh lot is short-term (higher rate) but has barely moved, so its per-share tax is
    # ~0 and min-tax drains it first -- ahead of the old, deeply-appreciated lot.
    grant_day = DATE + pd.Timedelta(days=400)
    pf.add_grant(
        grant_day, shares=1_000, employer_price=30.0, tax_config=NO_WITHHOLDING
    )  # fresh lot: 1000 sh @ $30

    # Sell 3 days after the fresh grant at its grant price (fresh gain = $0, short-term);
    # the old lot is long-term with a $20/sh gain. Position is $60k; trim to 75% -> sell
    # $15k = 500 sh.
    sale_date = grant_day + pd.Timedelta(days=3)
    trade = pf.sell_employer_to_fraction(
        sale_date,
        target_fraction=0.75,
        employer_price=30.0,
        index_price=30.0,
        tax_config=TaxConfig(short_term_rate=0.40, long_term_rate=0.20),
    )

    assert trade is not None
    # All 500 sh come from the fresh, zero-gain lot, so no tax despite the short-term rate.
    assert trade.tax_paid == approx(0.0)
    assert pf.employer_lots[0].cost_per_share == 10.0  # the old lot is untouched
    assert pf.employer_lots[0].shares == approx(1000)


def test_no_tax_devolves_to_fifo():
    pf = Portfolio()
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # lot 1: 1000 sh @ $10 (old)
    pf.add_grant(
        DATE, shares=1_000, employer_price=20.0, tax_config=NO_WITHHOLDING
    )  # lot 2: 1000 sh @ $20 (new)

    # With taxes off every lot ties at zero tax, so the stable sort preserves acquisition
    # order: the partial sell drains the oldest lot first, exactly like FIFO.
    no_tax = TaxConfig(short_term_rate=0.0, long_term_rate=0.0)
    trade = pf.sell_employer_to_fraction(
        DATE, target_fraction=0.75, employer_price=20.0, index_price=20.0, tax_config=no_tax
    )

    assert trade is not None
    # Position is $40k; trim to 75% -> sell $10k = 500 sh, all from the oldest lot.
    assert pf.employer_lots[0].cost_per_share == 10.0
    assert pf.employer_lots[0].shares == approx(500)
    assert pf.employer_lots[1].shares == approx(1000)


def test_liquidation_tax_spans_both_legs_holding_periods_and_losses():
    pf = Portfolio()
    later = DATE + pd.Timedelta(days=400)
    pf.add_grant(
        DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING
    )  # emp old: 1000 sh @ $10
    pf.add_grant(
        later, shares=1_000, employer_price=20.0, tax_config=NO_WITHHOLDING
    )  # emp fresh: 1000 sh @ $20
    pf.buy_index(DATE, dollars=5_000, index_price=5.0)  # idx old: 1000 sh @ $5
    pf.buy_index(later, dollars=12_000, index_price=12.0)  # idx fresh: 1000 sh @ $12

    # Liquidate where old lots are long-term and fresh lots short-term; index fresh sits
    # at a loss so it owes nothing.
    sale_date = later + pd.Timedelta(days=100)
    cfg = TaxConfig(short_term_rate=0.40, long_term_rate=0.20)
    tax = pf.liquidation_tax(employer_price=25.0, index_price=10.0, date=sale_date, tax_config=cfg)

    # emp old:   1000 * ($25-$10) * 0.20 = $3,000 (long-term gain)
    # emp fresh: 1000 * ($25-$20) * 0.40 = $2,000 (short-term gain)
    # idx old:   1000 * ($10-$5)  * 0.20 = $1,000 (long-term gain)
    # idx fresh: ($10-$12) < 0             = $0    (loss owes nothing)
    assert tax == approx(6_000)


def test_liquidation_value_is_market_value_less_tax():
    pf = Portfolio()
    pf.add_grant(DATE, shares=1_000, employer_price=10.0, tax_config=NO_WITHHOLDING)
    pf.buy_index(DATE, dollars=5_000, index_price=5.0)

    cfg = TaxConfig(short_term_rate=0.20)
    net = pf.liquidation_value(employer_price=20.0, index_price=10.0, date=DATE, tax_config=cfg)

    # Market value: 1000 emp @ $20 + 1000 idx @ $10 = $30,000.
    # Same-day sale, so both gains are short-term @ 20%:
    #   emp: 1000 * ($20 - $10) = $10,000 gain; idx: 1000 * ($10 - $5) = $5,000 gain.
    #   tax = 0.20 * $15,000 = $3,000. Net = $30,000 - $3,000 = $27,000.
    assert net == approx(27_000)
