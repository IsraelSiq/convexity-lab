"""Tests for DiscreteDividendOption in american.py

Identities tested
-----------------
1.  No dividends => price equals plain AmericanOption (same model, same result)
2.  Discrete dividend put > no-dividend put (dividends reduce spot => put more valuable)
3.  Discrete dividend call < no-dividend call (dividends reduce spot => call less valuable)
4.  Early exercise: American discrete div >= European discrete div
5.  Large dividend makes early exercise of call optimal before ex-date
6.  PV stripping: dividend outside [0,T] has no effect on price
7.  vs_continuous: returns valid dict with positive prices
8.  Multiple dividends: price moves monotonically (put up, call down)
9.  ValueError raised when PV(dividends) >= spot
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from american import AmericanOption, DiscreteDividendOption


# ---- 1. No dividends => same as AmericanOption ------------------------------

def test_no_dividends_equals_american():
    """With empty dividend list, discrete model must match plain CRR."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    plain = AmericanOption(S, K, T, r, sigma, q=0.0, n=300, kind="put").price()
    disc  = DiscreteDividendOption(S, K, T, r, sigma, dividends=(), n=300, kind="put").price()
    assert math.isclose(plain, disc, rel_tol=1e-3)


# ---- 2. Dividend put > no-dividend put --------------------------------------

def test_dividend_increases_put_value():
    """Cash dividend reduces expected spot => put becomes more valuable."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    no_div   = DiscreteDividendOption(S, K, T, r, sigma, dividends=(), n=300, kind="put").price()
    with_div = DiscreteDividendOption(S, K, T, r, sigma, dividends=((0.5, 5.0),), n=300, kind="put").price()
    assert with_div > no_div


# ---- 3. Dividend call < no-dividend call ------------------------------------

def test_dividend_decreases_call_value():
    """Cash dividend reduces expected spot => call becomes less valuable."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    no_div   = DiscreteDividendOption(S, K, T, r, sigma, dividends=(), n=300, kind="call").price()
    with_div = DiscreteDividendOption(S, K, T, r, sigma, dividends=((0.5, 5.0),), n=300, kind="call").price()
    assert with_div < no_div


# ---- 4. American >= European with discrete dividends ------------------------

def test_american_ge_european_discrete():
    """Early exercise right remains non-negative with discrete dividends."""
    divs = ((0.25, 2.0), (0.75, 2.0))
    am = DiscreteDividendOption(100, 100, 1.0, 0.05, 0.20, dividends=divs, n=200, kind="put").price()
    eu = DiscreteDividendOption(100, 100, 1.0, 0.05, 0.20, dividends=divs, n=200, kind="put", style="european").price()
    assert am >= eu - 1e-8


# ---- 5. Large dividend makes call early exercise optimal --------------------

def test_large_dividend_call_early_exercise():
    """Large imminent dividend => American call > European call."""
    divs = ((0.05, 15.0),)
    am = DiscreteDividendOption(100, 90, 0.5, 0.05, 0.25, dividends=divs, n=300, kind="call").price()
    eu = DiscreteDividendOption(100, 90, 0.5, 0.05, 0.25, dividends=divs, n=300, kind="call", style="european").price()
    assert am > eu


# ---- 6. Dividend outside [0,T] has no effect --------------------------------

def test_dividend_outside_expiry_ignored():
    """Dividend paid after expiry must not affect option price."""
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
    no_div    = DiscreteDividendOption(S, K, T, r, sigma, dividends=(), n=300, kind="put").price()
    after_exp = DiscreteDividendOption(S, K, T, r, sigma, dividends=((1.0, 5.0),), n=300, kind="put").price()
    assert math.isclose(no_div, after_exp, rel_tol=1e-6)


# ---- 7. vs_continuous returns valid dict ------------------------------------

def test_vs_continuous_put():
    """vs_continuous returns a dict with positive prices."""
    divs   = ((0.25, 3.0),)
    result = DiscreteDividendOption(100, 100, 1.0, 0.05, 0.25, dividends=divs, n=300, kind="put").vs_continuous()
    assert "discrete_price"   in result
    assert "continuous_price" in result
    assert "difference_pct"   in result
    assert result["discrete_price"]   > 0
    assert result["continuous_price"] > 0


# ---- 8. More dividends => higher put, lower call ----------------------------

@pytest.mark.parametrize("kind,direction", [("put", 1), ("call", -1)])
def test_more_dividends_monotone(kind, direction):
    """Adding more dividends increases put value and decreases call value."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    p0 = DiscreteDividendOption(S, K, T, r, sigma, dividends=(), n=200, kind=kind).price()
    p1 = DiscreteDividendOption(S, K, T, r, sigma, dividends=((0.5, 3.0),), n=200, kind=kind).price()
    p2 = DiscreteDividendOption(S, K, T, r, sigma, dividends=((0.33, 3.0), (0.67, 3.0)), n=200, kind=kind).price()
    assert direction * (p1 - p0) > 0
    assert direction * (p2 - p1) > 0


# ---- 9. ValueError when PV(dividends) >= spot -------------------------------

def test_raises_when_dividends_exceed_spot():
    """Dividend strip cannot exceed spot price."""
    with pytest.raises(ValueError):
        DiscreteDividendOption(10.0, 10.0, 1.0, 0.05, 0.20,
                               dividends=((0.5, 15.0),), n=100, kind="put").price()
