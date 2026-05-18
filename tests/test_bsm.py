"""Tests for stock_convexity.

Run: pytest -q
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_convexity import (
    Option,
    implied_vol,
    monte_carlo_price,
    parity_check,
)


# ----- Hull textbook reference values (chapter 15, example 15.6) -----

HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA = 42.0, 40.0, 0.5, 0.10, 0.20
HULL_CALL_EXPECTED = 4.7594
HULL_PUT_EXPECTED = 0.8086


def test_hull_call_price():
    opt = Option(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, kind="call")
    assert math.isclose(opt.price(), HULL_CALL_EXPECTED, abs_tol=1e-3)


def test_hull_put_price():
    opt = Option(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, kind="put")
    assert math.isclose(opt.price(), HULL_PUT_EXPECTED, abs_tol=1e-3)


def test_put_call_parity():
    """C - P must equal S*exp(-qT) - K*exp(-rT) at machine precision."""
    res = parity_check(100.0, 100.0, 1.0, 0.05, 0.25, q=0.02)
    assert abs(res["residual"]) < 1e-12


def test_gamma_identical_for_call_and_put():
    c = Option(100, 100, 0.5, 0.05, 0.3, kind="call")
    p = Option(100, 100, 0.5, 0.05, 0.3, kind="put")
    assert math.isclose(c.gamma(), p.gamma(), abs_tol=1e-15)


def test_gamma_positive_long_options():
    """Long options always have positive convexity."""
    opt = Option(100, 110, 0.25, 0.04, 0.30)
    assert opt.gamma() > 0


def test_deep_itm_call_delta_approaches_one():
    opt = Option(S=200, K=50, T=0.1, r=0.05, sigma=0.2, kind="call")
    assert opt.delta() > 0.99


def test_deep_otm_call_delta_approaches_zero():
    opt = Option(S=50, K=200, T=0.1, r=0.05, sigma=0.2, kind="call")
    assert opt.delta() < 0.01


def test_monte_carlo_matches_closed_form():
    """MC price within 4 standard errors of BSM for 200k antithetic paths."""
    opt = Option(100, 105, 0.5, 0.04, 0.25, kind="call")
    mc, se = monte_carlo_price(opt, n_paths=200_000, antithetic=True, seed=7)
    assert abs(mc - opt.price()) < 4 * se


def test_antithetic_reduces_variance():
    """Antithetic SE should be strictly smaller than plain MC SE."""
    opt = Option(100, 100, 0.5, 0.04, 0.30, kind="call")
    _, se_plain = monte_carlo_price(opt, n_paths=100_000, antithetic=False, seed=1)
    _, se_anti = monte_carlo_price(opt, n_paths=100_000, antithetic=True, seed=1)
    assert se_anti < se_plain


@pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.35, 0.60, 0.90])
def test_implied_vol_round_trip(sigma_true):
    """Price an option then invert IV - should recover sigma."""
    S, K, T, r = 100.0, 105.0, 0.5, 0.04
    price = Option(S, K, T, r, sigma_true, kind="call").price()
    iv = implied_vol(price, S, K, T, r, kind="call")
    assert iv is not None
    assert abs(iv - sigma_true) < 1e-6
