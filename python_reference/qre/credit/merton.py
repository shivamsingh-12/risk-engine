"""Phase 16 reference: the Merton (1974) structural model of credit risk.

A firm's assets V follow GBM, dV/V = mu dt + sigma dW, and it owes a single zero-coupon debt of face D at T.
At T the shareholders get max(V_T - D, 0) and the creditors min(V_T, D) = D - max(D - V_T, 0), so
    equity  E = call on V with strike D = V N(d1) - D e^{-rT} N(d2)
    debt    B = V - E = D e^{-rT} - put on V with strike D
    d1 = (ln(V/D) + (r + sigma^2/2) T) / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)
Default happens if V_T < D:
    risk-neutral PD    N(-d2)
    distance to default DD = (ln(V/D) + (mu - sigma^2/2) T) / (sigma sqrt(T)), real-world PD = N(-DD)
    credit spread      s = -(1/T) ln(B / (D e^{-rT})) = -(1/T) ln(N(d2) + V e^{rT} / D * N(-d1))
    recovery           E_Q[V_T | V_T < D] / D = V e^{rT} N(-d1) / (D N(-d2))
so spread = (risk-neutral PD) x (loss given default) to first order.

Only E is observed, not V or sigma. By Ito's lemma on E(V): sigma_E E = N(d1) sigma V (equity is levered assets).
Two equations, two unknowns: given E, sigma_E, D, r, T, solve for (V, sigma). solve_assets nests two 1-D
bracketed root searches: for a trial sigma, V solves E = C(V, sigma) on [E, E + D e^{-rT}] (C is increasing in V,
C(V) < V and C(V) >= V - D e^{-rT}); then sigma solves N(d1) sigma V = sigma_E E on (0, sigma_E] (sigma <= sigma_E
because the equity delta times V/E is >= 1).

KMV / Vassalou-Xing iteration (kmv_iterative). Using a historical sigma_E in the two equations assumes sigma_E is
constant, but it moves with leverage. Instead, from a daily equity series: guess sigma, invert every day's E_t into
V_t with that sigma, set sigma to the annualized volatility of the implied ln V_t returns, and repeat to a fixed
point. The drift mu is the annualized mean log asset return plus sigma^2 / 2.

Black and Cox (1976) first passage: default the first time V hits a barrier K (a covenant), not only at T.
With nu = mu - sigma^2/2 and b = ln(K/V) < 0,
    P(tau <= T) = N((b - nu T) / (sigma sqrt T)) + (K/V)^(2 nu / sigma^2) N((b + nu T) / (sigma sqrt T))
which is always at least the Merton PD with the same K = D.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

SQRT2 = math.sqrt(2.0)


def N(x):
    return 0.5 * math.erfc(-x / SQRT2)


def _check(V, D, sigma, T):
    if not (V > 0 and D > 0 and sigma > 0 and T > 0):
        raise ValueError("need V, D, sigma and T positive")


def d1_d2(V, D, r, sigma, T):
    _check(V, D, sigma, T)
    s = sigma * math.sqrt(T)
    d1 = (math.log(V / D) + (r + 0.5 * sigma * sigma) * T) / s
    return d1, d1 - s


def equity_value(V, D, r, sigma, T):
    d1, d2 = d1_d2(V, D, r, sigma, T)
    return V * N(d1) - D * math.exp(-r * T) * N(d2)


def debt_value(V, D, r, sigma, T):
    d1, d2 = d1_d2(V, D, r, sigma, T)
    return V * N(-d1) + D * math.exp(-r * T) * N(d2)        # = V - E, written without cancellation


def credit_spread(V, D, r, sigma, T):
    d1, d2 = d1_d2(V, D, r, sigma, T)
    return max(0.0, -math.log(N(d2) + V * math.exp(r * T) / D * N(-d1)) / T)    # clamp rounding above 1 for safe firms


def risk_neutral_pd(V, D, r, sigma, T):
    return N(-d1_d2(V, D, r, sigma, T)[1])


def distance_to_default(V, D, mu, sigma, T):
    _check(V, D, sigma, T)
    return (math.log(V / D) + (mu - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def physical_pd(V, D, mu, sigma, T):
    return N(-distance_to_default(V, D, mu, sigma, T))


def recovery_rate(V, D, r, sigma, T):
    """Risk-neutral expected recovery per unit of face value, given default."""
    d1, d2 = d1_d2(V, D, r, sigma, T)
    return V * math.exp(r * T) * N(-d1) / (D * N(-d2))


def equity_volatility(V, D, r, sigma, T):
    """sigma_E = N(d1) sigma V / E."""
    d1, _ = d1_d2(V, D, r, sigma, T)
    return N(d1) * sigma * V / equity_value(V, D, r, sigma, T)


def asset_value(E, D, r, sigma, T, xtol=1e-12):
    """V with C(V, sigma) = E."""
    if not E > 0:
        raise ValueError("equity value must be positive")
    lo, hi = E, E + D * math.exp(-r * T)
    f = lambda v: equity_value(v, D, r, sigma, T) - E
    if f(hi) <= 0.0:                     # sigma so small that equity is the forward value
        return hi
    return brentq(f, lo, hi, xtol=xtol * max(E, 1.0), rtol=4 * np.finfo(float).eps, maxiter=500)


def solve_assets(E, sigma_E, D, r, T):
    """(V, sigma) from the two Merton equations."""
    if not (E > 0 and sigma_E > 0 and D > 0 and T > 0):
        raise ValueError("need E, sigma_E, D and T positive")

    def g(s):
        v = asset_value(E, D, r, s, T)
        d1, _ = d1_d2(v, D, r, s, T)
        return N(d1) * s * v - sigma_E * E

    lo = 1e-8
    if g(lo) >= 0.0:
        raise ValueError("no asset volatility reproduces this equity volatility")
    s = brentq(g, lo, sigma_E, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500)
    return asset_value(E, D, r, s, T), s


@dataclass
class KmvResult:
    assets: np.ndarray        # implied V_t
    sigma: float              # asset volatility, annualized
    mu: float                 # asset drift, annualized
    iterations: int
    distance_to_default: float    # at the last date, horizon T
    pd: float                 # real-world N(-DD)


def kmv_iterative(equity, D, r, T=1.0, trading_days=252, tol=1e-10, max_iter=200):
    """Iterative asset-volatility estimate from a daily equity series with constant debt face D."""
    E = np.asarray(equity, dtype=float)
    if E.ndim != 1 or E.size < 20 or np.any(E <= 0):
        raise ValueError("need at least 20 positive equity values")
    rE = np.diff(np.log(E))
    sigma = float(np.std(rE, ddof=1) * math.sqrt(trading_days) * E[-1] / (E[-1] + D))   # leverage-scaled start
    for it in range(1, max_iter + 1):
        V = np.array([asset_value(e, D, r, sigma, T) for e in E])
        rV = np.diff(np.log(V))
        new = float(np.std(rV, ddof=1) * math.sqrt(trading_days))
        done = abs(new - sigma) < tol
        sigma = new
        if done:
            break
    else:
        raise RuntimeError("KMV iteration did not converge")
    V = np.array([asset_value(e, D, r, sigma, T) for e in E])
    mu = float(np.mean(np.diff(np.log(V))) * trading_days + 0.5 * sigma * sigma)
    dd = distance_to_default(V[-1], D, mu, sigma, T)
    return KmvResult(V, sigma, mu, it, dd, N(-dd))


def black_cox_pd(V, K, mu, sigma, T):
    """Probability that GBM assets starting at V hit the barrier K < V before T."""
    if not (0 < K < V and sigma > 0 and T > 0):
        raise ValueError("need 0 < K < V, sigma > 0, T > 0")
    nu = mu - 0.5 * sigma * sigma
    b = math.log(K / V)
    s = sigma * math.sqrt(T)
    return N((b - nu * T) / s) + (K / V) ** (2 * nu / (sigma * sigma)) * N((b + nu * T) / s)


def first_passage_mc(V, K, mu, sigma, T, n_paths=200_000, n_steps=252, seed=42):
    """Monte Carlo check of black_cox_pd. Daily monitoring misses crossings between dates, so the barrier is moved
    towards the path by exp(+0.5826 sigma sqrt(dt)) (Broadie-Glasserman-Kou continuity correction)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    k = math.log(K / V) + 0.5826 * sigma * math.sqrt(dt)
    x = np.zeros(n_paths)
    hit = np.zeros(n_paths, dtype=bool)
    for _ in range(n_steps):
        x += (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n_paths)
        hit |= x <= k
    p = hit.mean()
    return float(p), float(math.sqrt(p * (1 - p) / n_paths))
