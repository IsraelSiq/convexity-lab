"""Tests for american.py

Every test is an algebraic identity the binomial tree must satisfy.

Run: pytest tests/test_american.py -v

Identities tested
-----------------
1.  American put >= European put (early exercise premium >= 0)
2.  American call == European call when q=0 (never optimal to exercise early)
3.  American call > European call when q > 0 (dividend makes early exercise possible)
4.  American put > BSM European put (ATM, standard params)
5.  Deep-ITM put: early exercise premium large (intrinsic >> continuation)
6.  With n=1000, European binomial price converges to BSM within 0.5%
7.  Early exercise premium >= 0 always (calls and puts, various params)
8.  American put price is decreasing in r (higher r -> lower PV of K -> less value to wait)
9.  American put price is increasing in sigma (convexity of payoff)
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from american import AmericanOption
from stock_convexity import Option as BSM


# ---- helpers ----------------------------------------------------------------

def bsm_put(S, K, T, r, sigma, q=0.0):
    return BSM(S, K, T, r, sigma, q, kind="put").price()

def bsm_call(S, K, T, r, sigma, q=0.0):
    return BSM(S, K, T, r, sigma, q, kind="call").price()


# ---- 1. American put >= European put ----------------------------------------

def test_american_put_ge_european_put():
    """Early exercise right has non-negative value."""
    a = AmericanOption(100, 100, 1.0, 0.05, 0.20, n=200, kind="put")
    e = AmericanOption(100, 100, 1.0, 0.05, 0.20, n=200, kind="put", style="european")
    assert a.price() >= e.price()


# ---- 2. American call == European call quando q=0 ---------------------------

def test_american_call_equals_european_call_no_dividend():
    """With no dividends it is never optimal to exercise a call early."""
    a = AmericanOption(100, 100, 1.0, 0.05, 0.20, q=0.0, n=300, kind="call")
    e = AmericanOption(100, 100, 1.0, 0.05, 0.20, q=0.0, n=300, kind="call", style="european")
    assert math.isclose(a.price(), e.price(), rel_tol=1e-4)


# ---- 3. American call > European call quando q > 0 --------------------------

def test_american_call_gt_european_call_with_dividend():
    """Dividend yield makes early exercise of calls potentially optimal."""
    a = AmericanOption(100, 100, 1.0, 0.05, 0.20, q=0.08, n=300, kind="call")
    e = AmericanOption(100, 100, 1.0, 0.05, 0.20, q=0.08, n=300, kind="call", style="european")
    assert a.price() > e.price()


# ---- 4. American put > BSM European put (ATM) -------------------------------

def test_american_put_exceeds_bsm_european():
    """Binomial American put must exceed BSM European put at same params."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    am = AmericanOption(S, K, T, r, sigma, n=400, kind="put").price()
    eu = bsm_put(S, K, T, r, sigma)
    assert am > eu


# ---- 5. Deep-ITM put: early exercise premium e grande -----------------------

def test_deep_itm_put_early_exercise_premium():
    """Deep ITM put: intrinsic >> continuation => large premium."""
    opt = AmericanOption(S=60, K=100, T=1.0, r=0.05, sigma=0.20, n=300, kind="put")
    prem = opt.early_exercise_premium()
    assert prem > 0.5


# ---- 6. Europeia binomial converge pro BSM (n=1000) -------------------------

def test_european_binomial_converges_to_bsm():
    """With n=1000 steps, binomial European price matches BSM within 0.5%."""
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25
    binom = AmericanOption(S, K, T, r, sigma, n=1000, kind="put", style="european").price()
    bsm   = bsm_put(S, K, T, r, sigma)
    assert math.isclose(binom, bsm, rel_tol=0.005)


# ---- 7. Early exercise premium >= 0 em grid de parametros ------------------

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,sigma,q", [
    (100, 100, 1.0, 0.05, 0.20, 0.0),
    (100, 110, 0.5, 0.03, 0.30, 0.02),
    (80,  100, 0.25, 0.08, 0.40, 0.0),
])
def test_early_exercise_premium_nonnegative(S, K, T, r, sigma, q, kind):
    """American >= European sempre (no-arbitrage)."""
    opt = AmericanOption(S, K, T, r, sigma, q=q, n=200, kind=kind)
    assert opt.early_exercise_premium() >= -1e-8


# ---- 8. American put decresce em r ------------------------------------------

def test_american_put_decreasing_in_r():
    """Higher r -> lower PV(K) -> less benefit to waiting -> lower put price."""
    p_lo = AmericanOption(100, 100, 1.0, r=0.01, sigma=0.20, n=200, kind="put").price()
    p_hi = AmericanOption(100, 100, 1.0, r=0.10, sigma=0.20, n=200, kind="put").price()
    assert p_lo > p_hi


# ---- 9. American put cresce em sigma ----------------------------------------

def test_american_put_increasing_in_sigma():
    """Higher vol -> higher option value (convexity of payoff)."""
    p_lo = AmericanOption(100, 100, 1.0, 0.05, sigma=0.10, n=200, kind="put").price()
    p_hi = AmericanOption(100, 100, 1.0, 0.05, sigma=0.50, n=200, kind="put").price()
    assert p_hi > p_lo
