"""Phase 9 reference: the Heston stochastic volatility model.

Risk-neutral dynamics:
    dS = r S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + sigma sqrt(v) dW2,     corr(dW1, dW2) = rho
    kappa: speed of mean reversion of the variance, theta: long-run variance, sigma: vol of vol,
    rho: spot/vol correlation (negative for equities -> downside skew), v0: current variance.
Feller condition 2 kappa theta >= sigma^2 keeps v strictly positive; when it fails v can touch 0.

Characteristic function of X = ln(S_T / F), F = S e^{rT}, in the "little Heston trap" form
(Albrecher et al. 2007), which avoids the branch-cut discontinuity of the original formula:
    b = kappa - rho sigma i u,  d = sqrt(b^2 + sigma^2 (i u + u^2)),  g = (b - d) / (b + d)
    A = kappa theta / sigma^2 [ (b - d) T - 2 ln((1 - g e^{-dT}) / (1 - g)) ]
    B = v0 / sigma^2 (b - d) (1 - e^{-dT}) / (1 - g e^{-dT})
    psi(u) = exp(A + B)
European call (Lewis 2001), a single real integral:
    C = S - sqrt(S K e^{-rT}) / pi * int_0^inf Re[ e^{i u k} psi(u - i/2) ] / (u^2 + 1/4) du,  k = ln(F / K)
Put by put-call parity.

Monte Carlo: full-truncation Euler (Lord, Koekkoek, van Dijk 2010). The variance can step below zero;
wherever it is used, v+ = max(v, 0) is used instead, which is the scheme with the smallest bias among the
simple fixes. Log-price step: ln S += (r - v+/2) h + sqrt(v+ h) Z1.

Calibration: minimize vega-weighted squared price errors, sum ((model - market) / vega)^2, which
approximates squared implied-vol errors without inverting Black-Scholes at every step. Parameters are
mapped to an unconstrained space: kappa, theta, sigma, v0 = exp(.), rho = tanh(.).
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

from qre.pricing import black_scholes as bs


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    def feller(self):
        return 2.0 * self.kappa * self.theta >= self.sigma ** 2


def _validate(p):
    if p.kappa <= 0 or p.theta <= 0 or p.sigma <= 0 or p.v0 <= 0:
        raise ValueError("kappa, theta, sigma and v0 must be positive")
    if not (-1.0 < p.rho < 1.0):
        raise ValueError("rho must be in (-1, 1)")


def char_func(u, p, T):
    """E[exp(i u ln(S_T / F))] for real or complex u (scalar or array)."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = p.kappa - p.rho * p.sigma * iu
    d = np.sqrt(b * b + p.sigma ** 2 * (iu + u * u))
    g = (b - d) / (b + d)
    edt = np.exp(-d * T)
    A = p.kappa * p.theta / p.sigma ** 2 * ((b - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    B = p.v0 / p.sigma ** 2 * (b - d) * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(A + B)


def heston_price(p, spot, strike, rate, T, kind="call"):
    _validate(p)
    if spot <= 0 or strike <= 0 or T <= 0:
        raise ValueError("spot, strike and T must be positive")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    k = math.log(spot * math.exp(rate * T) / strike)
    integrand = lambda u: (np.exp(1j * u * k) * char_func(u - 0.5j, p, T)).real / (u * u + 0.25)
    integral = quad(integrand, 0.0, np.inf, limit=2000, epsabs=1e-12, epsrel=1e-11)[0]
    call = spot - math.sqrt(spot * strike * math.exp(-rate * T)) / math.pi * integral
    if kind == "call":
        return call
    return call - spot + strike * math.exp(-rate * T)


def heston_mc_price(p, spot, strike, rate, T, steps, n_paths, kind="call", seed=42):
    """Full-truncation Euler Monte Carlo. Returns (price, standard error)."""
    _validate(p)
    rng = np.random.default_rng(seed)
    h = T / steps
    x = np.full(n_paths, math.log(spot))
    v = np.full(n_paths, p.v0)
    sq = math.sqrt(1.0 - p.rho ** 2)
    for _ in range(steps):
        z2 = rng.standard_normal(n_paths)
        z1 = p.rho * z2 + sq * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        x += (rate - 0.5 * vp) * h + np.sqrt(vp * h) * z1
        v += p.kappa * (p.theta - vp) * h + p.sigma * np.sqrt(vp * h) * z2
    s_t = np.exp(x)
    payoff = np.maximum(s_t - strike, 0.0) if kind == "call" else np.maximum(strike - s_t, 0.0)
    disc = math.exp(-rate * T) * payoff
    return float(disc.mean()), float(disc.std(ddof=1) / math.sqrt(n_paths))


def implied_vol_surface(p, spot, rate, strikes, maturities):
    """Black-Scholes implied vols of Heston call prices, shape (len(maturities), len(strikes))."""
    out = np.empty((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            out[i, j] = bs.implied_vol(heston_price(p, spot, K, rate, T), spot, K, rate, T, "call")
    return out


# ---------------------------------------------------------------- calibration
def params_from_x(x):
    return HestonParams(math.exp(x[0]), math.exp(x[1]), math.exp(x[2]), math.tanh(x[3]), math.exp(x[4]))


def x_from_params(p):
    return np.array([math.log(p.kappa), math.log(p.theta), math.log(p.sigma), math.atanh(p.rho), math.log(p.v0)])


@dataclass
class Quote:
    strike: float
    maturity: float
    price: float     # market call price
    vega: float      # Black-Scholes vega at the market implied vol (weights the errors)


def make_quotes(spot, rate, strikes, maturities, implied_vols):
    quotes = []
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            iv = implied_vols[i][j]
            quotes.append(Quote(K, T, bs.price(spot, K, rate, iv, T, "call"), bs.vega(spot, K, rate, iv, T)))
    return quotes


def calibration_error(p, spot, rate, quotes):
    """Root-mean-square vega-weighted price error, approximately the implied-vol RMSE."""
    errs = [(heston_price(p, spot, q.strike, rate, q.maturity) - q.price) / q.vega for q in quotes]
    return math.sqrt(float(np.mean(np.square(errs))))


DEFAULT_STARTS = (HestonParams(2.0, 0.04, 0.5, -0.5, 0.04),
                  HestonParams(0.5, 0.09, 0.3, -0.7, 0.06),
                  HestonParams(5.0, 0.02, 0.9, -0.2, 0.02))


def calibrate(spot, rate, quotes, starts=DEFAULT_STARTS, maxiter=3000):
    """Nelder-Mead from several starts; returns (best params, rmse)."""
    def obj(x):
        try:
            return calibration_error(params_from_x(x), spot, rate, quotes)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 1e6
    best = None
    for s in starts:
        res = minimize(obj, x_from_params(s), method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": maxiter, "maxfev": 2 * maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return params_from_x(best.x), float(best.fun)
