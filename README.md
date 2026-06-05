# convexity-lab

Option pricing toolkit with three complementary models:

- **Black-Scholes-Merton (constant vol)** — closed-form pricing, all first
  and second-order Greeks (including the **Gamma convexity surface**),
  Monte Carlo validation with antithetic variates, implied vol solver
  (Newton-Raphson + Brent fallback).
- **Heston stochastic volatility** — closed-form pricing via Fourier
  inversion of the characteristic function (with the *little Heston trap*
  to avoid branch cuts), implied vol smile generation.
- **American options via CRR binomial tree** — early exercise for calls
  and puts, discrete cash dividends via the Escrowed Dividend method,
  convergence to BSM in the European limit.

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

American options — CRR binomial tree:
```python
from american import AmericanOption

# ATM American put — 1 year, 20% vol, 5% rate
opt = AmericanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, n=200, kind="put")
print(f"American put  : {opt.price():.4f}")
print(f"Early premium : {opt.early_exercise_premium():.4f}")
```

Discrete dividends — Escrowed Dividend method:
```python
from american import DiscreteDividendOption

# PETR4: R$1.22 dividend in ~1 month, Selic 10.65%
opt = DiscreteDividendOption(
    S=40.76, K=41.0, T=0.25, r=0.1065, sigma=0.35,
    dividends=((0.083, 1.22),),   # (ex-date in years, cash amount)
    n=300, kind="put"
)
result = opt.vs_continuous()
print(f"Discrete price    : R$ {result['discrete_price']:.4f}")
print(f"Continuous approx : R$ {result['continuous_price']:.4f}  (q={result['q_approx']:.2%})")
print(f"Difference        : {result['difference_pct']:+.2f}%")
# Output:
# Discrete price    : R$ 3.1027
# Continuous approx : R$ 2.9664  (q=11.97%)
# Difference        : +4.59%
```

With Brazilian high-dividend stocks (PETR4, VALE3, banks), the continuous
yield approximation **underprices puts by ~4-5%** — a model error that
discrete dividends correctly captures.

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

### American options — CRR binomial tree

The Cox-Ross-Rubinstein (1979) parameterisation of the recombining tree:

> `u = exp(σ√dt)`,  `d = 1/u`,  `p = (exp((r−q)dt) − d) / (u − d)`

At each interior node the holder compares continuation value against
immediate exercise and takes the maximum:

> `V(i,j) = max( e^{-r·dt} · [p·V(i+1,j+1) + (1−p)·V(i+1,j)] , intrinsic )`

### Discrete dividends — Escrowed Dividend method

The recombining tree is built on the *pure stochastic component* of the stock:

> `S* = S − PV(dividends)` where `PV = Σ Dᵢ · e^{-r·tᵢ}`

At each node, the PV of dividends not yet paid is added back to recover
the full stock price for intrinsic value calculations. This preserves
tree recombination — superior to the naive approach of subtracting each
dividend at the ex-date, which breaks recombination and misprices vol.

## Validation

The test suite (`pytest tests/`) covers **45 tests**:

**BSM (14 tests):**
1. Hull textbook reference values (example 15.6) matched to `1e-3`
2. Put-call parity at machine precision (`< 1e-12`)
3. Gamma identical for call & put with same params
4. Gamma always positive (long convexity)
5. Deep-ITM call Δ → 1, deep-OTM call Δ → 0
6. Monte Carlo matches closed-form within 4 standard errors
7. Antithetic variates produce strictly lower SE than plain MC
8. Implied vol round-trip exact to `1e-6` across σ ∈ [0.10, 0.90]

**Heston (7 tests):**
9. Heston degenerates to BSM when `σ_v → 0` (3 vol levels)
10. Put-call parity holds
11. Negative ρ produces negative skew
12. Zero ρ produces symmetric smile
13. Feller condition flag correct

**American / CRR binomial tree (14 tests):**
14. American put ≥ European put
15. American call = European call when `q = 0`
16. American call > European call when `q > 0`
17. American put > BSM European put (ATM)
18. Deep-ITM put: early exercise premium > 0
19. European binomial converges to BSM within 0.5% at `n = 1000`
20–25. Early exercise premium ≥ 0 across param grid (calls + puts)
26. American put decreasing in `r`
27. American put increasing in `σ`

**Discrete dividends — Escrowed Dividend (10 tests):**
28. No dividends → same price as plain AmericanOption
29. Dividend increases put value
30. Dividend decreases call value
31. American ≥ European with discrete dividends
32. Large imminent dividend makes call early exercise optimal
33. Dividend after expiry has no effect on price
34. `vs_continuous` returns valid dict with positive prices
35–36. More dividends → higher put, lower call (monotone)
37. `ValueError` raised when PV(dividends) ≥ spot

## Limitations

- ~~European exercise only~~ ✅ American options added via CRR binomial tree.
- ~~Constant dividend yield~~ ✅ Discrete dividends added via Escrowed Dividend method.
- Constant volatility (no local-vol or jumps — Heston covers stochastic vol).
- Risk-free rate is flat (no term structure).

## License

MIT — see [LICENSE](LICENSE).
