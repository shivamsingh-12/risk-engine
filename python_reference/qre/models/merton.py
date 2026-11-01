"""Phase 8 reference: Merton (1976) jump-diffusion.

Under the risk-neutral measure:  dS/S = (r - lam k) dt + sigma dW + (J - 1) dN
    N is a Poisson process with intensity lam (jumps per year); a jump multiplies S by J,
    with ln J ~ Normal(mu_j, delta^2) and mean relative jump k = E[J] - 1 = exp(mu_j + delta^2/2) - 1.
    The -lam k drift correction keeps the discounted stock price a martingale.
Terminal value: ln S_T = ln S + (r - lam k - sigma^2/2) T + sigma W_T + sum of N_T log-jumps,
    and the sum of n log-jumps is Normal(n mu_j, n delta^2).
Closed-form European price: a Poisson-weighted sum of Black-Scholes prices,
    sum_n e^{-lam' T} (lam' T)^n / n! * BS(S, K, r_n, sigma_n, T)
    lam' = lam (1 + k),  sigma_n^2 = sigma^2 + n delta^2 / T,  r_n = r - lam k + n ln(1 + k) / T
lam = 0 recovers Black-Scholes.
"""
import math
from dataclasses import dataclass

import numpy as np

from qre.pricing import black_scholes as bs


@dataclass(frozen=True)
class MertonParams:
    spot: float
    strike: float
    rate: float
    vol: float          # diffusion volatility
    time: float
    lam: float          # jump intensity per year
    mu_j: float         # mean log jump
    delta: float        # sd of log jump
    kind: str = "call"

    @property
    def k(self):
        return math.exp(self.mu_j + 0.5 * self.delta ** 2) - 1.0


def _validate(p):
    if p.spot <= 0 or p.strike <= 0 or p.vol <= 0 or p.time <= 0:
        raise ValueError("spot, strike, vol and time must be positive")
    if p.lam < 0 or p.delta < 0:
        raise ValueError("lam and delta must be non-negative")
    if p.kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")


def merton_price(p, tol=1e-15, max_terms=200):
    _validate(p)
    if p.lam == 0.0:
        return bs.price(p.spot, p.strike, p.rate, p.vol, p.time, p.kind)
    lam_prime = p.lam * (1.0 + p.k)
    lt = lam_prime * p.time
    total = 0.0
    for n in range(max_terms):
        weight = math.exp(-lt + n * math.log(lt) - math.lgamma(n + 1))
        sigma_n = math.sqrt(p.vol ** 2 + n * p.delta ** 2 / p.time)
        r_n = p.rate - p.lam * p.k + n * math.log(1.0 + p.k) / p.time
        total += weight * bs.price(p.spot, p.strike, r_n, sigma_n, p.time, p.kind)
        if n > lt and weight < tol:
            break
    return total


def simulate_terminal(p, n_paths, seed=42):
    _validate(p)
    rng = np.random.default_rng(seed)
    n_jumps = rng.poisson(p.lam * p.time, n_paths)
    jumps = n_jumps * p.mu_j + np.sqrt(n_jumps) * p.delta * rng.standard_normal(n_paths)
    drift = (p.rate - p.lam * p.k - 0.5 * p.vol ** 2) * p.time
    return p.spot * np.exp(drift + p.vol * math.sqrt(p.time) * rng.standard_normal(n_paths) + jumps)


def merton_mc_price(p, n_paths, seed=42):
    """Monte Carlo price and standard error."""
    s_t = simulate_terminal(p, n_paths, seed)
    payoff = np.maximum(s_t - p.strike, 0.0) if p.kind == "call" else np.maximum(p.strike - s_t, 0.0)
    disc = math.exp(-p.rate * p.time) * payoff
    return float(disc.mean()), float(disc.std(ddof=1) / math.sqrt(n_paths))
