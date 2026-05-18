# convexity-lab

Option pricing toolkit with two complementary models:

- **Black-Scholes-Merton (constant vol)** — closed-form pricing, all first
  and second-order Greeks (including the **Gamma convexity surface**),
  Monte Carlo validation with antithetic variates, implied vol solver
  (Newton-Raphson + Brent fallback).
- **Heston stochastic volatility** — closed-form pricing via Fourier
  inversion of the characteristic function (with the *little Heston trap*
  to avoid branch cuts), implied vol smile generation.

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

Black-Scholes-Merton demo:
```bash
python stock_convexity.py                  # synthetic demo (no internet needed)
python stock_convexity.py NVDA             # live spot via yfinance
python stock_convexity.py NVDA --plot      # also save the gamma surface PNG
python stock_convexity.py --paths 1000000  # heavier Monte Carlo
```

Heston stochastic volatility demo:
```bash
python heston.py                           # print ATM Heston price + IV smile summary
python heston.py --plot                    # save heston_smile.png
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

**BSM pricing** (European call):

> `C = S · e^{-qT} · N(d₁) - K · e^{-rT} · N(d₂)`

where

> `d₁ = [ln(S/K) + (r - q + ½σ²)T] / (σ√T)`,  `d₂ = d₁ - σ√T`

**Gamma** (the convexity Greek):

> `Γ = e^{-qT} · φ(d₁) / (S · σ · √T)`

**Volga** (convexity in vol):

> `Vomma = Vega · d₁ · d₂ / σ`

**Vanna** (mixed second derivative):

> `Vanna = -e^{-qT} · φ(d₁) · d₂ / σ`

### Heston model

The Heston SDE is

> `dS_t = (r-q) S_t dt + √v_t · S_t · dW_t¹`
>
> `dv_t = κ(θ - v_t) dt + σ_v · √v_t · dW_t²`
>
> `d⟨W¹,W²⟩_t = ρ dt`

Five parameters: `v₀` (initial variance), `κ` (mean-reversion speed),
`θ` (long-run variance), `σ_v` (vol-of-vol), `ρ` (asset-vol correlation).
With `ρ < 0` and `σ_v > 0`, the model produces the **negative skew** observed
in equity index options (OTM puts more expensive than OTM calls).

Pricing uses Fourier inversion of two characteristic functions:

> `C = S · e^{-qT} · P₁ - K · e^{-rT} · P₂`,
>
> `P_j = ½ + (1/π) ∫₀^∞ Re[ e^{-i u ln K} · f_j(u) / (i u) ] du`

The "little trap" form (Albrecher et al. 2007) keeps `g = (b - iρσu - d)/(b - iρσu + d)`
and uses `exp(-dT)` inside the log, eliminating the branch-cut discontinuity
present in the original Heston (1993) formulation.

Sanity check: as `σ_v → 0` with `v₀ = θ`, the Heston price degenerates to
BSM with `σ = √θ` — verified by the test suite.

## Validation

The test suite (`pytest tests/`) covers **21 tests**:

**BSM (14 tests):**
1. Hull textbook reference values (example 15.6) matched to `1e-3`
2. Put-call parity at machine precision (`< 1e-12`)
3. Gamma identical for call & put with same params (model property)
4. Gamma always positive (long convexity)
5. Deep-ITM call Δ → 1, deep-OTM call Δ → 0 (boundary behavior)
6. Monte Carlo matches closed-form within 4 standard errors
7. Antithetic variates produce strictly lower SE than plain MC
8. Implied vol round-trip exact to `1e-6` across σ ∈ [0.10, 0.90]

**Heston (7 tests):**
9. Heston with `σ_v ≈ 0` and `v₀ = θ` degenerates to BSM (parametrized at 3 vols)
10. Put-call parity holds (model-free property)
11. Negative correlation produces negative skew (OTM put IV > OTM call IV)
12. Zero correlation produces approximately symmetric smile
13. Feller condition flag (`2κθ > σ_v²`) correctly identifies regimes

## Limitations

- European exercise only (no early exercise / American options).
- Constant volatility (no local-vol, stochastic vol, or jumps).
- Constant dividend yield (continuous, not discrete dividends).
- Risk-free rate is flat (no term structure).

For exotic payoffs, stochastic vol (Heston), or American exercise, this
needs binomial / PDE / LSM Monte Carlo extensions.

## License

MIT — see [LICENSE](LICENSE).
