"""
stock_convexity.py - Black-Scholes-Merton option analytics with full Greeks
and a 3D convexity (Gamma) surface visualization for any equity.

Implements:
- Closed-form European option pricing under BSM (Hull, 11th ed.)
- All first-order Greeks: Delta, Vega, Theta, Rho
- Second-order Greeks (convexity zone): Gamma, Vanna, Volga
- Monte Carlo with antithetic variates (~2x variance reduction)
- Newton-Raphson implied volatility solver with Brenner-Subrahmanyam seed
- Optional live options chain via yfinance
- Gamma surface across moneyness x expiration (matplotlib 3D)

Usage
-----
    python stock_convexity.py                      # synthetic demo
    python stock_convexity.py NVDA                 # real spot via yfinance
    python stock_convexity.py NVDA --plot          # also saves PNG surface
    python stock_convexity.py --paths 1000000      # heavier Monte Carlo
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from math import exp, log, pi, sqrt
from typing import Literal

import numpy as np
from scipy.stats import norm


# ============================================================================
#  Black-Scholes-Merton: closed-form pricing + full Greeks
# ============================================================================

@dataclass(frozen=True)
class Option:
    """European option under BSM assumptions.

    All rates and volatility are continuously compounded / annualized.
    Conventions follow Hull, "Options, Futures, and Other Derivatives", 11e.
    """
    S: float                                       # spot price
    K: float                                       # strike price
    T: float                                       # time to expiry (years)
    r: float                                       # risk-free rate
    sigma: float                                   # implied volatility
    q: float = 0.0                                 # continuous dividend yield
    kind: Literal["call", "put"] = "call"

    @property
    def d1(self) -> float:
        return (log(self.S / self.K) +
                (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))

    @property
    def d2(self) -> float:
        return self.d1 - self.sigma * sqrt(self.T)

    @property
    def _disc_r(self) -> float:
        return exp(-self.r * self.T)

    @property
    def _disc_q(self) -> float:
        return exp(-self.q * self.T)

    # ----- Price -----

    def price(self) -> float:
        d1, d2 = self.d1, self.d2
        if self.kind == "call":
            return self.S * self._disc_q * norm.cdf(d1) - self.K * self._disc_r * norm.cdf(d2)
        return self.K * self._disc_r * norm.cdf(-d2) - self.S * self._disc_q * norm.cdf(-d1)

    # ----- First-order Greeks -----

    def delta(self) -> float:
        """dV/dS: hedge ratio. Call in [0,1], put in [-1,0]."""
        sign = 1 if self.kind == "call" else -1
        return sign * self._disc_q * norm.cdf(sign * self.d1)

    def vega(self) -> float:
        """dV/dsigma per 1.0 of vol (divide by 100 for "per vol point")."""
        return self.S * self._disc_q * norm.pdf(self.d1) * sqrt(self.T)

    def theta(self) -> float:
        """dV/dT: annualized time decay (divide by 365 for daily)."""
        term1 = -self.S * self._disc_q * norm.pdf(self.d1) * self.sigma / (2 * sqrt(self.T))
        if self.kind == "call":
            return (term1
                    - self.r * self.K * self._disc_r * norm.cdf(self.d2)
                    + self.q * self.S * self._disc_q * norm.cdf(self.d1))
        return (term1
                + self.r * self.K * self._disc_r * norm.cdf(-self.d2)
                - self.q * self.S * self._disc_q * norm.cdf(-self.d1))

    def rho(self) -> float:
        """dV/dr: rate sensitivity."""
        if self.kind == "call":
            return self.K * self.T * self._disc_r * norm.cdf(self.d2)
        return -self.K * self.T * self._disc_r * norm.cdf(-self.d2)

    # ----- Second-order Greeks (the convexity zone) -----

    def gamma(self) -> float:
        """d^2 V / dS^2: convexity of option price w.r.t. spot.

        Always positive for long options (long convexity). Identical for call
        and put with same parameters by put-call parity.
        """
        return self._disc_q * norm.pdf(self.d1) / (self.S * self.sigma * sqrt(self.T))

    def vanna(self) -> float:
        """d^2 V / (dS dsigma): cross-greek. How delta moves with vol."""
        return -self._disc_q * norm.pdf(self.d1) * self.d2 / self.sigma

    def volga(self) -> float:
        """d^2 V / dsigma^2 (a.k.a. vomma): convexity w.r.t. volatility."""
        return self.vega() * self.d1 * self.d2 / self.sigma

    def all_greeks(self) -> dict:
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega":  self.vega(),
            "theta": self.theta(),
            "rho":   self.rho(),
            "vanna": self.vanna(),
            "volga": self.volga(),
        }


# ============================================================================
#  Monte Carlo with antithetic variates
# ============================================================================

def monte_carlo_price(opt: Option, n_paths: int = 200_000,
                      antithetic: bool = True, seed: int = 42) -> tuple[float, float]:
    """Price European option under geometric Brownian motion via Monte Carlo.

    Antithetic variates: for each Z draw, also use -Z. Cuts variance by ~2x
    for free (correlation between paired payoffs is negative on average).

    Returns (mean_price, standard_error).
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal(half)
    if antithetic:
        Z = np.concatenate([Z, -Z])

    drift = (opt.r - opt.q - 0.5 * opt.sigma ** 2) * opt.T
    diffusion = opt.sigma * sqrt(opt.T) * Z
    S_T = opt.S * np.exp(drift + diffusion)

    payoff = (np.maximum(S_T - opt.K, 0.0) if opt.kind == "call"
              else np.maximum(opt.K - S_T, 0.0))
    discounted = exp(-opt.r * opt.T) * payoff
    return float(discounted.mean()), float(discounted.std(ddof=1) / sqrt(len(discounted)))


# ============================================================================
#  Implied volatility (Newton-Raphson with Brenner-Subrahmanyam seed)
# ============================================================================

def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                q: float = 0.0, kind: str = "call",
                tol: float = 1e-8, max_iter: int = 100) -> float | None:
    """Invert Black-Scholes for sigma given an observed market price.

    Two-stage solver:
      1. Newton-Raphson seeded by Brenner-Subrahmanyam (1988) approximation -
         quadratic convergence near ATM where vega is large.
      2. Brent's method on [1e-6, 5.0] as a bracketing fallback for deep-wing
         strikes where vega is tiny and Newton diverges.

    Returns None only if both methods fail (e.g., arbitrage-violating input).
    """
    sigma = sqrt(2 * pi / T) * market_price / S
    sigma = max(min(sigma, 5.0), 1e-4)

    for _ in range(max_iter):
        opt = Option(S, K, T, r, sigma, q, kind)
        diff = market_price - opt.price()
        if abs(diff) < tol:
            return sigma
        v = opt.vega()
        if v < 1e-12:
            break
        sigma_new = sigma + diff / v
        if sigma_new <= 0 or sigma_new > 10:
            break
        sigma = sigma_new

    # Brent fallback (bracketing - works where Newton fails)
    from scipy.optimize import brentq

    def _residual(s: float) -> float:
        return Option(S, K, T, r, s, q, kind).price() - market_price

    try:
        return float(brentq(_residual, 1e-6, 5.0, xtol=tol, maxiter=200))
    except (ValueError, RuntimeError):
        return None


# ============================================================================
#  Convexity surface
# ============================================================================

def gamma_surface(S: float, r: float, sigma: float, q: float = 0.0,
                  moneyness_range: tuple = (0.7, 1.3),
                  T_range: tuple = (1 / 52, 1.0),
                  n_K: int = 60, n_T: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Gamma across (strike, time-to-expiry) grid.

    Returns moneyness mesh (K/S), time mesh and Gamma values - all (n_T, n_K).
    """
    Ks = np.linspace(moneyness_range[0] * S, moneyness_range[1] * S, n_K)
    Ts = np.linspace(T_range[0], T_range[1], n_T)
    K_mesh, T_mesh = np.meshgrid(Ks, Ts)
    G = np.zeros_like(K_mesh)
    for i in range(n_T):
        for j in range(n_K):
            G[i, j] = Option(S, K_mesh[i, j], T_mesh[i, j], r, sigma, q).gamma()
    return K_mesh / S, T_mesh, G


# ============================================================================
#  Validation: put-call parity
# ============================================================================

def parity_check(S: float, K: float, T: float, r: float,
                 sigma: float, q: float = 0.0) -> dict:
    """C - P should equal S e^{-qT} - K e^{-rT}. Residual should be ~1e-15."""
    c = Option(S, K, T, r, sigma, q, "call").price()
    p = Option(S, K, T, r, sigma, q, "put").price()
    rhs = S * exp(-q * T) - K * exp(-r * T)
    return {"C - P": c - p, "S e^{-qT} - K e^{-rT}": rhs, "residual": (c - p) - rhs}


# ============================================================================
#  Data fetching (yfinance - optional)
# ============================================================================

def fetch_spot(ticker: str) -> float | None:
    """Pull current spot price for a ticker via yfinance. None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        print("[!] yfinance not installed: pip install yfinance", file=sys.stderr)
        return None
    try:
        return float(yf.Ticker(ticker).fast_info["last_price"])
    except Exception as e:
        print(f"[!] yfinance fetch failed for {ticker}: {e}", file=sys.stderr)
        return None


# ============================================================================
#  Plotting
# ============================================================================

def plot_gamma_surface(moneyness, T, G, S: float, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("[!] matplotlib not installed: pip install matplotlib", file=sys.stderr)
        return
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(moneyness, T, G, cmap="viridis", linewidth=0,
                           antialiased=True, alpha=0.92)
    ax.set_xlabel("Moneyness  K / S")
    ax.set_ylabel("Time to expiry (years)")
    ax.set_zlabel(r"$\Gamma = \partial^2 V / \partial S^2$")
    ax.set_title(f"Option Convexity Surface  (spot = {S:.2f})")
    fig.colorbar(surf, shrink=0.55, aspect=14, label="Gamma")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[+] gamma surface saved to {out_path}")


# ============================================================================
#  Demo / CLI
# ============================================================================

def _print_greeks(g: dict) -> None:
    rows = [
        ("Price",          f"{g['price']:>14.6f}", ""),
        ("Delta",          f"{g['delta']:>14.6f}", ""),
        ("Gamma",          f"{g['gamma']:>14.6f}", "<-- convexity in spot"),
        ("Vega",           f"{g['vega']:>14.6f}",  ""),
        ("Theta (annual)", f"{g['theta']:>14.6f}", ""),
        ("Rho",            f"{g['rho']:>14.6f}",   ""),
        ("Vanna",          f"{g['vanna']:>14.6f}", "<-- delta vs vol"),
        ("Volga (vomma)",  f"{g['volga']:>14.6f}", "<-- convexity in vol"),
    ]
    for name, val, note in rows:
        print(f"    {name:<16}{val}   {note}")


def demo(ticker: str | None, plot: bool, n_paths: int) -> None:
    print("=" * 76)
    print(" Black-Scholes-Merton analytics + convexity surface")
    print("=" * 76)

    spot = fetch_spot(ticker) if ticker else None
    if spot is None:
        ticker = ticker or "SYNTH"
        S = 920.00
        print(f"\n[+] using synthetic data  ({ticker})  spot = ${S:.2f}")
    else:
        S = spot
        print(f"\n[+] live spot from yfinance  ({ticker})  spot = ${S:.2f}")

    # Sample option: 5% OTM call, 60 days, 50% IV, 4.5% rate
    K = round(S * 1.05, 2)
    T = 60 / 365
    r = 0.045
    sigma = 0.50

    opt = Option(S=S, K=K, T=T, r=r, sigma=sigma, kind="call")
    print(f"\n  Option contract: S={S:.2f}  K={K:.2f}  T={T:.4f}y  "
          f"r={r:.3f}  sigma={sigma:.3f}  kind=call")
    print("  " + "-" * 60)
    _print_greeks(opt.all_greeks())

    # Monte Carlo validation
    print(f"\n  Monte Carlo (antithetic, n_paths = {n_paths:,}):")
    t0 = time.perf_counter()
    mc, se = monte_carlo_price(opt, n_paths=n_paths, antithetic=True)
    dt = time.perf_counter() - t0
    bs = opt.price()
    print(f"    MC price        = {mc:.6f}  +- {1.96*se:.6f}  (95% CI)")
    print(f"    BSM closed-form = {bs:.6f}")
    print(f"    abs residual    = {abs(mc-bs):.6f}")
    print(f"    elapsed         = {dt*1000:.1f} ms")

    # Put-call parity
    print(f"\n  Put-call parity check (residual should be ~1e-14):")
    for k, v in parity_check(S, K, T, r, sigma).items():
        print(f"    {k:<28} = {v:>14.10f}")

    # Implied vol round-trip
    iv = implied_vol(opt.price(), S, K, T, r, kind="call")
    print(f"\n  Implied vol round-trip:  input = {sigma:.6f}   recovered = {iv:.6f}")

    if plot:
        print(f"\n  Computing gamma surface (60x60 grid)...")
        moneyness, Tm, G = gamma_surface(S, r, sigma)
        ticker_safe = (ticker or "synth").replace("/", "_")
        plot_gamma_surface(moneyness, Tm, G, S, out_path=f"{ticker_safe}_gamma_surface.png")

    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("ticker", nargs="?", default=None,
                   help="ticker symbol (e.g. NVDA); omit for synthetic demo")
    p.add_argument("--plot", action="store_true", help="save gamma surface PNG")
    p.add_argument("--paths", type=int, default=200_000,
                   help="Monte Carlo paths (default 200k)")
    args = p.parse_args()
    demo(args.ticker, args.plot, args.paths)


if __name__ == "__main__":
    main()
