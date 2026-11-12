"""Phase 12 reference: Monte Carlo Greeks -- bumping, pathwise derivatives and the likelihood ratio.

Price V(theta) = E[f(S_T(theta))] (discounted). Three ways to get dV/dtheta from simulation:

Bump and revalue: (V(theta + h) - V(theta - h)) / 2h with the same random numbers (common random
numbers) at both points. Biased by O(h^2), costs two extra pricings per input, and its variance blows up
as h -> 0 when the payoff is discontinuous.

Pathwise: differentiate inside the expectation, dV/dtheta = E[f'(S_T) dS_T/dtheta]. Unbiased when the
payoff is continuous (a kink is fine), usually low variance. Fails for discontinuous payoffs: for a digital
1{S_T > K}, f' = 0 almost everywhere and the estimate is exactly 0 whatever the true delta.
For GBM, S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z), so
    dS_T/dS0 = S_T / S0,     dS_T/dsigma = S_T (sqrt(T) Z - sigma T),     dS_T/dr = T S_T.

Likelihood ratio: differentiate the density instead, dV/dtheta = E[f(S_T) d ln p(S_T; theta)/dtheta].
Needs no derivative of the payoff, so it works for digitals, but the weights add variance:
    delta weight  Z / (S0 sigma sqrt(T)),       vega weight  (Z^2 - 1) / sigma - Z sqrt(T).

Adjoint algorithmic differentiation (qre.autodiff.aad) computes the pathwise estimator mechanically, for
every input at once, at a small constant multiple of the cost of the price.

Basket call on N correlated GBM assets (one step to T): B = sum w_i S_i(T), payoff e^{-rT} max(B - K, 0),
with x = L z (L the Cholesky factor of the correlation). Pathwise, with I = 1{B > K}:
    dV/dS_i0 = e^{-rT} I w_i S_i / S_i0,      dV/dsigma_i = e^{-rT} I w_i S_i (sqrt(T) x_i - sigma_i T)
    dV/dr = -T V_path + e^{-rT} I T B,        dV/dK = -e^{-rT} I
"""
import math

import numpy as np

from qre.pricing import black_scholes as bs


def _terminal(spot, rate, vol, time, z):
    return spot * np.exp((rate - 0.5 * vol * vol) * time + vol * math.sqrt(time) * np.asarray(z, dtype=float))


def _mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def european_call_pathwise(spot, strike, rate, vol, time, z):
    """{'price','delta','vega','rho'} -> (estimate, standard error)."""
    st = _terminal(spot, rate, vol, time, z)
    df = math.exp(-rate * time)
    itm = st > strike
    payoff = df * np.maximum(st - strike, 0.0)
    return {
        "price": _mean_se(payoff),
        "delta": _mean_se(df * itm * st / spot),
        "vega": _mean_se(df * itm * st * (math.sqrt(time) * np.asarray(z) - vol * time)),
        "rho": _mean_se(-time * payoff + df * itm * st * time),
    }


def european_call_lr(spot, strike, rate, vol, time, z):
    z = np.asarray(z, dtype=float)
    st = _terminal(spot, rate, vol, time, z)
    payoff = math.exp(-rate * time) * np.maximum(st - strike, 0.0)
    return {
        "price": _mean_se(payoff),
        "delta": _mean_se(payoff * z / (spot * vol * math.sqrt(time))),
        "vega": _mean_se(payoff * ((z * z - 1.0) / vol - z * math.sqrt(time))),
    }


def digital_call_price(spot, strike, rate, vol, time):
    _, d2 = bs.d1_d2(spot, strike, rate, vol, time)
    return math.exp(-rate * time) * bs._norm_cdf(d2)


def digital_call_delta(spot, strike, rate, vol, time):
    _, d2 = bs.d1_d2(spot, strike, rate, vol, time)
    return math.exp(-rate * time) * bs._norm_pdf(d2) / (spot * vol * math.sqrt(time))


def digital_call_delta_pathwise(spot, strike, rate, vol, time, z):
    """The payoff is flat except at the jump, so the pathwise estimate is identically zero."""
    return _mean_se(np.zeros(np.asarray(z).size))


def digital_call_delta_lr(spot, strike, rate, vol, time, z):
    z = np.asarray(z, dtype=float)
    payoff = math.exp(-rate * time) * (_terminal(spot, rate, vol, time, z) > strike)
    return _mean_se(payoff * z / (spot * vol * math.sqrt(time)))


def digital_call_delta_bump(spot, strike, rate, vol, time, z, h):
    """Central difference with common random numbers; h is the absolute spot bump."""
    z = np.asarray(z, dtype=float)
    df = math.exp(-rate * time)
    up = df * (_terminal(spot + h, rate, vol, time, z) > strike)
    dn = df * (_terminal(spot - h, rate, vol, time, z) > strike)
    return _mean_se((up - dn) / (2.0 * h))


def basket_call_pathwise(spots, vols, corr, weights, strike, rate, time, z):
    """Price and pathwise gradient of a basket call. z: (n, N) standard normals.
    Returns (price, {'spots': (N,), 'vols': (N,), 'rate': float, 'strike': float})."""
    spots, vols, weights = (np.asarray(a, dtype=float) for a in (spots, vols, weights))
    z = np.atleast_2d(np.asarray(z, dtype=float))
    if not (spots.shape == vols.shape == weights.shape) or z.shape[1] != spots.size:
        raise ValueError("spots, vols, weights and the columns of z must all have length N")
    if np.any(spots <= 0) or np.any(vols <= 0) or strike <= 0 or time <= 0:
        raise ValueError("spots, vols, strike and time must be positive")
    x = z @ np.linalg.cholesky(np.asarray(corr, dtype=float)).T
    st = spots * np.exp((rate - 0.5 * vols * vols) * time + vols * math.sqrt(time) * x)
    b = st @ weights
    df = math.exp(-rate * time)
    itm = (b > strike).astype(float)
    payoff = df * np.maximum(b - strike, 0.0)
    ws = st * weights
    grad = {
        "spots": df * (itm[:, None] * ws).mean(axis=0) / spots,
        "vols": df * (itm[:, None] * ws * (math.sqrt(time) * x - vols * time)).mean(axis=0),
        "rate": float((-time * payoff + df * itm * time * b).mean()),
        "strike": float(-df * itm.mean()),
    }
    return float(payoff.mean()), grad
