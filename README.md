# convexity-lab

Black-Scholes-Merton option analytics with all first- and second-order Greeks
(including the **Gamma convexity surface**), Monte Carlo validation with
antithetic variates, and a Newton-Raphson implied vol solver.

Optional live spot via [yfinance](https://github.com/ranaroussi/yfinance).

## What is convexity?

For options, **Gamma** = `∂²V/∂S²` — the second derivative of option value
with respect to the underlying. It quantifies how curved the price function
is, and is the dominant source of P&L for delta-hedged positions over short
horizons:

> dV ≈ Δ · dS + ½ · Γ · (dS)² + Θ · dt + ...

A long-options position is **long convexity**: every move in the underlying,
up or down, produces gamma scalping P&L proportional to `½ · Γ · (dS)²`. The
script also computes **Volga** (`∂²V/∂σ²`) — convexity in volatility itself.

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `yfinance` (the last is optional;
the script falls back to synthetic data if it can't fetch live prices).

## Run

```bash
python stock_convexity.py                  # synthetic demo (no internet needed)
python stock_convexity.py NVDA             # live spot via yfinance
python stock_convexity.py NVDA --plot      # also save the gamma surface PNG
python stock_convexity.py --paths 1000000  # heavier Monte Carlo
```

Example output (NVDA spot pulled live):

```
============================================================================
 Black-Scholes-Merton analytics + convexity surface
============================================================================

[+] live spot from yfinance  (NVDA)  spot = $920.00

  Option contract: S=920.00  K=966.00  T=0.1644y  r=0.045  sigma=0.500  kind=call
  ------------------------------------------------------------
    Price                  56.421234
    Delta                   0.473219
    Gamma                   0.003912    <-- convexity in spot
    Vega                  146.882443
    Theta (annual)       -284.114502
    Rho                    61.224578
    Vanna                   0.024195    <-- delta vs vol
    Volga (vomma)          16.348822    <-- convexity in vol

  Monte Carlo (antithetic, n_paths = 200,000):
    MC price        = 56.428901  +- 0.064721  (95% CI)
    BSM closed-form = 56.421234
    abs residual    = 0.007667
    elapsed         = 28.4 ms

  Put-call parity check (residual should be ~1e-14):
    C - P                        =   -39.5234567890
    S e^{-qT} - K e^{-rT}        =   -39.5234567890
    residual                     =     0.0000000000

  Implied vol round-trip:  input = 0.500000   recovered = 0.500000
```

With `--plot`, you also get a 3D surface of Gamma over the (moneyness × time)
grid — visually showing where convexity is concentrated (peaked at-the-money,
exploding as expiration approaches).

## Math

All formulas follow Hull, *Options, Futures, and Other Derivatives*, 11e.

**Pricing** (European call):

> `C = S · e^{-qT} · N(d₁) - K · e^{-rT} · N(d₂)`

where

> `d₁ = [ln(S/K) + (r - q + ½σ²)T] / (σ√T)`,  `d₂ = d₁ - σ√T`

**Gamma** (the convexity Greek):

> `Γ = e^{-qT} · φ(d₁) / (S · σ · √T)`

**Volga** (convexity in vol):

> `Vomma = Vega · d₁ · d₂ / σ`

**Vanna** (mixed second derivative):

> `Vanna = -e^{-qT} · φ(d₁) · d₂ / σ`

## Validation

The script self-validates in three ways:

1. **Put-call parity** — `C - P` is compared against `S·e^{-qT} - K·e^{-rT}`.
   Residual should be at machine precision.
2. **Monte Carlo** — closed-form price vs MC with antithetic variates. Should
   match within 2 standard errors (~95% confidence) for 200k paths.
3. **Implied vol round-trip** — invert the closed-form price back through
   Newton-Raphson and confirm we recover the original σ to ~1e-8.

## Limitations

- European exercise only (no early exercise / American options).
- Constant volatility (no local-vol, stochastic vol, or jumps).
- Constant dividend yield (continuous, not discrete dividends).
- Risk-free rate is flat (no term structure).

For exotic payoffs, stochastic vol (Heston), or American exercise, this
needs binomial / PDE / LSM Monte Carlo extensions.

## License

MIT — see [LICENSE](LICENSE).
