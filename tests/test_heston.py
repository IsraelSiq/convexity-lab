"""Tests for the Heston stochastic vol module."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heston import HestonParams, heston_call, heston_put
from stock_convexity import Option, implied_vol


# ---------------------------------------------------------------------------
#  Sanity: degenerate Heston should approach Black-Scholes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_bsm", [0.15, 0.25, 0.40])
def test_heston_reduces_to_bsm_low_volvol(sigma_bsm):
    """As sigma_v -> 0 with v0 = theta, the Heston model degenerates to BSM
    with vol sqrt(theta).  Test with sigma_v = 1e-6 ~ effectively constant vol."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    p = HestonParams(v0=sigma_bsm ** 2, kappa=2.0, theta=sigma_bsm ** 2,
                     sigma_v=1e-6, rho=0.0)
    c_heston = heston_call(S, K, T, r, q=0.0, p=p)
    c_bsm    = Option(S, K, T, r, sigma_bsm).price()
    # Heston Fourier integration + BSM closed-form should agree to ~1e-3
    assert abs(c_heston - c_bsm) < 5e-3


# ---------------------------------------------------------------------------
#  Put-call parity holds for Heston too (model-free property)
# ---------------------------------------------------------------------------

def test_heston_put_call_parity():
    S, K, T, r, q = 100.0, 105.0, 0.5, 0.04, 0.01
    p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.4, rho=-0.5)
    c   = heston_call(S, K, T, r, q, p)
    pu  = heston_put(S, K, T, r, q, p)
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs((c - pu) - rhs) < 1e-6


# ---------------------------------------------------------------------------
#  Heston with negative correlation produces the observed equity vol smile
#  (downward skew: OTM puts more expensive than OTM calls)
# ---------------------------------------------------------------------------

def test_heston_produces_negative_skew():
    S, T, r, q = 100.0, 0.5, 0.04, 0.0
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7)

    iv_otm_put  = implied_vol(heston_call(S, 80,  T, r, q, p), S, 80,  T, r, kind="call")
    iv_atm      = implied_vol(heston_call(S, 100, T, r, q, p), S, 100, T, r, kind="call")
    iv_otm_call = implied_vol(heston_call(S, 120, T, r, q, p), S, 120, T, r, kind="call")

    assert iv_otm_put is not None and iv_atm is not None and iv_otm_call is not None
    # negative skew: left wing > right wing
    assert iv_otm_put > iv_otm_call
    # at-the-money is between the wings (roughly)
    assert iv_otm_put > iv_atm


# ---------------------------------------------------------------------------
#  Heston with zero correlation produces a symmetric smile
# ---------------------------------------------------------------------------

def test_heston_zero_correlation_symmetric_smile():
    """With rho = 0, the IV smile around ATM should be approximately symmetric."""
    S, T, r, q = 100.0, 0.5, 0.04, 0.0
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=0.0)

    iv_left  = implied_vol(heston_call(S, 85,  T, r, q, p), S, 85,  T, r, kind="call")
    iv_right = implied_vol(heston_call(S, 115, T, r, q, p), S, 115, T, r, kind="call")
    # zero correlation -> nearly symmetric smile (differ by < 1 vol point)
    assert abs(iv_left - iv_right) < 0.01


# ---------------------------------------------------------------------------
#  Feller condition flag
# ---------------------------------------------------------------------------

def test_feller_condition():
    p_ok        = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.5)
    p_violated  = HestonParams(v0=0.04, kappa=0.5, theta=0.04, sigma_v=0.50, rho=-0.5)
    assert p_ok.feller() is True
    assert p_violated.feller() is False
    assert p_ok.feller_margin() > 0
    assert p_violated.feller_margin() < 0
