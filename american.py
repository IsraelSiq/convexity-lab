"""
american.py - Binomial tree pricing for American options (Cox-Ross-Rubinstein)

Implements:
- American call and put pricing via CRR recombining binomial tree
- Early exercise detection at every node via backward induction
- Convergence to BSM European price as n -> inf (verified by tests)

The CRR parameterisation (Cox, Ross, Rubinstein 1979):
    u  = exp(sigma * sqrt(dt))     # up factor
    d  = 1 / u                     # down factor (recombining tree)
    p  = (exp((r - q) * dt) - d) / (u - d)   # risk-neutral up probability

At each interior node the holder compares:
    continuation value  = exp(-r*dt) * (p * V_up + (1-p) * V_down)
    immediate exercise  = intrinsic value (max(S-K,0) or max(K-S,0))
and takes the maximum.  European pricing is identical except the
max() with intrinsic is skipped - useful for convergence tests.

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

        # terminal asset prices
        ST = [self.S * (u ** j) * (d ** (self.n - j)) for j in range(self.n + 1)]

        # terminal payoffs
        if self.kind == "call":
            V = [max(s - self.K, 0.0) for s in ST]
        else:
            V = [max(self.K - s, 0.0) for s in ST]

        # backward induction
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
