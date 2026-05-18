"""heston.py - Heston (1993) stochastic volatility model.

European option pricing under the Heston SDE:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^1
    dv_t = kappa * (theta - v_t) dt + sigma_v * sqrt(v_t) dW_t^2
    d<W^1, W^2>_t = rho dt

Closed-form price via Fourier inversion of the characteristic function.
Uses the "little Heston trap" (Albrecher et al. 2007) to avoid branch
cuts in the complex logarithm that plague the original 1993 formulation.

References
----------
Heston, S. (1993).  "A Closed-Form Solution for Options with Stochastic
    Volatility with Applications to Bond and Currency Options."
    Review of Financial Studies 6(2): 327-343.

Albrecher, H. et al. (2007).  "The Little Heston Trap."
    Wilmott Magazine, January, 83-92.

Gatheral, J. (2006).  "The Volatility Surface", Ch. 2.  Wiley.

Usage
-----
    python heston.py
    python heston.py --plot
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, log, pi

import numpy as np
from scipy.integrate import quad


# ============================================================================
#  Parameters
# ============================================================================

@dataclass(frozen=True)
class HestonParams:
    """Five parameters of the Heston model.

    v0      : initial instantaneous variance (note: variance, not vol)
    kappa   : mean-reversion speed of the variance process
    theta   : long-run mean variance
    sigma_v : volatility of variance ("vol of vol")
    rho     : instantaneous correlation between asset and variance,  in [-1, 1]
    """
    v0:      float
    kappa:   float
    theta:   float
    sigma_v: float
    rho:     float

    def feller(self) -> bool:
        """Feller condition: 2*kappa*theta > sigma_v^2 ensures the variance
        process stays strictly positive almost surely.  Equity calibrations
        often violate this; the closed-form pricer is robust either way."""
        return 2 * self.kappa * self.theta > self.sigma_v ** 2

    def feller_margin(self) -> float:
        return 2 * self.kappa * self.theta - self.sigma_v ** 2


# ============================================================================
#  Characteristic function (little Heston trap)
# ============================================================================

def _char(u, j: int, S: float, T: float, r: float, q: float, p: HestonParams):
    """Heston characteristic function f_j(u) for j in {1, 2}.

    f_j(u) = E^{Q_j}[ exp(i * u * log S_T) ]

    Q_1 is the stock-numeraire measure, Q_2 the bond-numeraire.
    The little-trap form uses g = (b - i*rho*sigma*u - d) / (b - i*rho*sigma*u + d)
    together with exp(-d*T) so the complex log stays on the principal branch.
    """
    i = 1j
    if j == 1:
        u_j = 0.5
        b   = p.kappa - p.rho * p.sigma_v
    else:
        u_j = -0.5
        b   = p.kappa

    # discriminant
    inside = (p.rho * p.sigma_v * i * u - b) ** 2 - p.sigma_v ** 2 * (2 * u_j * i * u - u ** 2)
    d = np.sqrt(inside)

    num = b - p.rho * p.sigma_v * i * u - d
    den = b - p.rho * p.sigma_v * i * u + d
    g   = num / den

    exp_mdT = np.exp(-d * T)

    C = (r - q) * i * u * T + (p.kappa * p.theta / p.sigma_v ** 2) * (
        num * T - 2.0 * np.log((1.0 - g * exp_mdT) / (1.0 - g))
    )
    D = (num / p.sigma_v ** 2) * (1.0 - exp_mdT) / (1.0 - g * exp_mdT)

    return np.exp(C + D * p.v0 + i * u * log(S))


# ============================================================================
#  Pricing
# ============================================================================

def heston_call(S: float, K: float, T: float, r: float, q: float,
                p: HestonParams, upper: float = 200.0) -> float:
    """European call price under Heston via Fourier inversion.

    C = S * e^{-qT} * P_1 - K * e^{-rT} * P_2,    where

        P_j = 1/2 + (1/pi) * integral_0^inf  Re[ e^{-i u ln K} f_j(u) / (i u) ] du
    """
    def integrand(u: float, j: int) -> float:
        phi = _char(u, j, S, T, r, q, p)
        return float(np.real(np.exp(-1j * u * log(K)) * phi / (1j * u)))

    P1, _ = quad(lambda u: integrand(u, 1), 1e-10, upper, limit=500)
    P2, _ = quad(lambda u: integrand(u, 2), 1e-10, upper, limit=500)
    P1 = 0.5 + P1 / pi
    P2 = 0.5 + P2 / pi

    return S * exp(-q * T) * P1 - K * exp(-r * T) * P2


def heston_put(S: float, K: float, T: float, r: float, q: float,
               p: HestonParams, upper: float = 200.0) -> float:
    """European put via put-call parity from the call price."""
    c = heston_call(S, K, T, r, q, p, upper)
    return c - S * exp(-q * T) + K * exp(-r * T)


# ============================================================================
#  Implied-volatility smile from Heston prices
# ============================================================================

def heston_iv_smile(S: float, T: float, r: float, q: float, p: HestonParams,
                    moneyness: tuple = (0.7, 1.3), n: int = 50
                    ) -> tuple[np.ndarray, np.ndarray]:
    """For each strike in the moneyness grid, price the Heston call and invert
    Black-Scholes to extract the implied volatility.

    Returns (strikes, implied_vols), both shape (n,).  np.nan where IV solver
    failed to converge (deep wings).
    """
    from stock_convexity import implied_vol

    Ks  = np.linspace(moneyness[0] * S, moneyness[1] * S, n)
    ivs = np.full(n, np.nan)
    for i, K in enumerate(Ks):
        c = heston_call(S, float(K), T, r, q, p)
        if c <= 0:
            continue
        iv = implied_vol(c, S, float(K), T, r, q=q, kind="call")
        if iv is not None:
            ivs[i] = iv
    return Ks, ivs


# ============================================================================
#  Demo / CLI
# ============================================================================

def _print_smile_summary(Ks: np.ndarray, ivs: np.ndarray, S: float) -> None:
    mid = len(ivs) // 2
    print(f"    IV( K = 0.70 S ) = {ivs[0] * 100:6.2f}%   (OTM put wing)")
    print(f"    IV( K = 1.00 S ) = {ivs[mid] * 100:6.2f}%   (ATM)")
    print(f"    IV( K = 1.30 S ) = {ivs[-1] * 100:6.2f}%   (OTM call wing)")
    skew = ivs[0] - ivs[-1]
    print(f"    25-delta skew (left - right) = {skew * 100:+.2f}%")


def demo() -> None:
    ap = argparse.ArgumentParser(description="Heston stochastic vol demo")
    ap.add_argument("--plot", action="store_true", help="save heston_smile.png")
    args = ap.parse_args()

    S, T, r, q = 100.0, 0.5, 0.04, 0.0
    # Realistic equity index calibration (S&P 500-ish)
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7)

    print("=" * 76)
    print(" Heston stochastic volatility model")
    print("=" * 76)
    print(f"\n  Parameters")
    print(f"    v0      = {p.v0}        (initial variance ~> vol = {np.sqrt(p.v0)*100:.1f}%)")
    print(f"    kappa   = {p.kappa}        (mean-reversion speed)")
    print(f"    theta   = {p.theta}        (long-run variance ~> vol = {np.sqrt(p.theta)*100:.1f}%)")
    print(f"    sigma_v = {p.sigma_v}        (vol of vol)")
    print(f"    rho     = {p.rho}       (asset-vol correlation)")
    print(f"    Feller cond.  2 kappa theta > sigma_v^2 :  {p.feller()}   "
          f"(margin = {p.feller_margin():+.4f})")

    print(f"\n  ATM option  (S={S}, K={S}, T={T}y)")
    c_at = heston_call(S, S, T, r, q, p)
    p_at = heston_put(S, S, T, r, q, p)
    print(f"    Heston call = ${c_at:.4f}")
    print(f"    Heston put  = ${p_at:.4f}")

    print(f"\n  Implied vol smile (Heston -> BSM IV by inversion, 50 strikes)")
    Ks, ivs = heston_iv_smile(S, T, r, q, p, moneyness=(0.7, 1.3), n=50)
    _print_smile_summary(Ks, ivs, S)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n  [!] matplotlib not installed; skipping plot")
            return
        plt.figure(figsize=(9, 5))
        plt.plot(Ks / S, ivs * 100, "o-", color="steelblue",
                 markersize=5, label="Heston implied vol")
        plt.axhline(np.sqrt(p.theta) * 100, ls="--", color="grey",
                    label=fr"$\sqrt{{\theta}} = {np.sqrt(p.theta)*100:.1f}\%$  (long-run vol)")
        plt.xlabel("Moneyness  K / S")
        plt.ylabel("Implied volatility (%)")
        plt.title(fr"Heston volatility smile   T={T}y,  $\rho$={p.rho}, "
                  fr"$\sigma_v$={p.sigma_v}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("heston_smile.png", dpi=150, bbox_inches="tight")
        print(f"\n  [+] smile plot saved to heston_smile.png")


if __name__ == "__main__":
    demo()
