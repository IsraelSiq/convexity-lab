"""
american.py - Binomial tree pricing for American options (Cox-Ross-Rubinstein)

Implements:
- American call and put pricing via CRR recombining binomial tree
- Early exercise detection at every node via backward induction
- Convergence to BSM European price as n -> inf (verified by tests)
- Discrete dividends via the Escrowed Dividend method (Hull 11e, Ch. 21)

The CRR parameterisation (Cox, Ross, Rubinstein 1979):
    u  = exp(sigma * sqrt(dt))     # up factor
    d  = 1 / u                     # down factor (recombining tree)
    p  = (exp((r - q) * dt) - d) / (u - d)   # risk-neutral up probability

At each interior node the holder compares:
    continuation value  = exp(-r*dt) * (p * V_up + (1-p) * V_down)
    immediate exercise  = intrinsic value (max(S-K,0) or max(K-S,0))
and takes the maximum.  European pricing is identical except the
max() with intrinsic is skipped - useful for convergence tests.

Discrete dividends (Escrowed Dividend method):
    S* = S - PV(dividends)   # strip the PV of all future dividends from spot
    Build the recombining tree on S* (pure vol component), then at each node
    add back the PV of dividends not yet paid to recover the full stock price.
    This preserves tree recombination while correctly modelling discrete cash
    dividends - superior to the naive approach of reducing S by each dividend
    at the ex-date (which breaks recombination and misprices vol).

Reference: Hull, "Options, Futures, and Other Derivatives", 11e, Chapter 21.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AmericanOption:
    """American (or European) option priced by CRR binomial tree.

    Parameters
    ----------
    S     : float  - current spot price
    K     : float  - strike price
    T     : float  - time to expiry in years
    r     : float  - continuously compounded risk-free rate
    sigma : float  - annualised volatility
    q     : float  - continuous dividend yield (default 0)
    n     : int    - number of time steps (more steps = more accurate)
    kind  : str    - "call" or "put"
    style : str    - "american" (default) or "european"
    """
    S:     float
    K:     float
    T:     float
    r:     float
    sigma: float
    q:     float = 0.0
    n:     int   = 200
    kind:  Literal["call", "put"]          = "put"
    style: Literal["american", "european"] = "american"

    def price(self) -> float:
        """Price the option using a CRR recombining binomial tree.

        Time complexity  : O(n^2)
        Space complexity : O(n)  - only one slice needed via backward pass
        """
        dt   = self.T / self.n
        u    = math.exp(self.sigma * math.sqrt(dt))
        d    = 1.0 / u
        disc = math.exp(-self.r * dt)
        p    = (math.exp((self.r - self.q) * dt) - d) / (u - d)
        q_   = 1.0 - p

        ST = [self.S * (u ** j) * (d ** (self.n - j)) for j in range(self.n + 1)]

        if self.kind == "call":
            V = [max(s - self.K, 0.0) for s in ST]
        else:
            V = [max(self.K - s, 0.0) for s in ST]

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                s_ij         = self.S * (u ** j) * (d ** (i - j))
                continuation = disc * (p * V[j + 1] + q_ * V[j])
                if self.style == "american":
                    intrinsic = (max(s_ij - self.K, 0.0) if self.kind == "call"
                                 else max(self.K - s_ij, 0.0))
                    V[j] = max(continuation, intrinsic)
                else:
                    V[j] = continuation

        return V[0]

    def early_exercise_premium(self) -> float:
        """Value of the early exercise right = American price - European price.

        Always >= 0 by no-arbitrage: the right to exercise early cannot
        have negative value.
        """
        eur = AmericanOption(
            self.S, self.K, self.T, self.r, self.sigma,
            self.q, self.n, self.kind, style="european"
        )
        return self.price() - eur.price()


# ============================================================================
#  Discrete dividends - Escrowed Dividend method (Hull 11e, Ch. 21)
# ============================================================================

@dataclass(frozen=True)
class DiscreteDividendOption:
    """American (or European) option with discrete cash dividends.

    Uses the Escrowed Dividend method: the recombining CRR tree is built on
    S* = S - PV(future dividends), which contains only the stochastic
    component of the stock price.  At each node, the PV of dividends not yet
    paid is added back to recover the full stock price for intrinsic value
    calculations.

    Parameters
    ----------
    S         : float        - current spot price
    K         : float        - strike price
    T         : float        - time to expiry in years
    r         : float        - continuously compounded risk-free rate
    sigma     : float        - annualised volatility
    dividends : tuple        - tuple of (t_i, D_i): ex-dividend time (years)
                               and cash dividend amount.  Only dividends with
                               0 < t_i < T are used.
    n         : int          - number of time steps
    kind      : str          - "call" or "put"
    style     : str          - "american" or "european"

    Example
    -------
    # PETR4: ~R$1.22/quarter
    opt = DiscreteDividendOption(
        S=40.74, K=41.0, T=0.25, r=0.1065, sigma=0.35,
        dividends=((0.083, 1.22),),
        n=300, kind="put"
    )
    print(opt.price())
    """
    S:         float
    K:         float
    T:         float
    r:         float
    sigma:     float
    dividends: tuple = ()
    n:         int   = 300
    kind:      Literal["call", "put"]          = "put"
    style:     Literal["american", "european"] = "american"

    def _pv_dividends(self, t: float) -> float:
        """PV at time t of all dividends paid strictly after t and before T."""
        return sum(
            D * math.exp(-self.r * (t_i - t))
            for t_i, D in self.dividends
            if t < t_i < self.T
        )

    def price(self) -> float:
        """Price via Escrowed Dividend CRR tree."""
        dt     = self.T / self.n
        S_star = self.S - self._pv_dividends(0.0)
        if S_star <= 0:
            raise ValueError(f"PV of dividends ({self.S - S_star:.4f}) >= spot ({self.S:.4f})")

        u    = math.exp(self.sigma * math.sqrt(dt))
        d    = 1.0 / u
        disc = math.exp(-self.r * dt)
        p    = (math.exp(self.r * dt) - d) / (u - d)
        q_   = 1.0 - p

        ST_star = [S_star * (u ** j) * (d ** (self.n - j)) for j in range(self.n + 1)]
        ST_full = [s + self._pv_dividends(self.T) for s in ST_star]

        if self.kind == "call":
            V = [max(s - self.K, 0.0) for s in ST_full]
        else:
            V = [max(self.K - s, 0.0) for s in ST_full]

        for i in range(self.n - 1, -1, -1):
            t_i    = i * dt
            pv_rem = self._pv_dividends(t_i)
            for j in range(i + 1):
                s_star_ij = S_star * (u ** j) * (d ** (i - j))
                s_full_ij = s_star_ij + pv_rem
                continuation = disc * (p * V[j + 1] + q_ * V[j])
                if self.style == "american":
                    intrinsic = (max(s_full_ij - self.K, 0.0) if self.kind == "call"
                                 else max(self.K - s_full_ij, 0.0))
                    V[j] = max(continuation, intrinsic)
                else:
                    V[j] = continuation

        return V[0]

    def vs_continuous(self) -> dict:
        """Compare discrete-dividend price against continuous-yield equivalent."""
        total_div  = sum(D for t_i, D in self.dividends if 0 < t_i < self.T)
        q_approx   = total_div / (self.S * self.T) if self.T > 0 else 0.0
        cont_price = AmericanOption(
            self.S, self.K, self.T, self.r, self.sigma,
            q=q_approx, n=self.n, kind=self.kind, style=self.style
        ).price()
        disc_price = self.price()
        return {
            "discrete_price":   disc_price,
            "continuous_price": cont_price,
            "q_approx":         q_approx,
            "difference":       disc_price - cont_price,
            "difference_pct":   (disc_price - cont_price) / cont_price * 100,
        }
