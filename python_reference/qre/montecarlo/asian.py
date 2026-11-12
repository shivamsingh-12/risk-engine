"""Phase 12 reference: Asian options, the test bed for quasi-Monte Carlo.

An Asian call pays max(A - K, 0) at T, where A averages the price over m monitoring dates t_k = k T / m.
Each path needs m normals, so the integration dimension is m: a natural test of how QMC copes with
dimension and of whether the Brownian bridge helps.

Under GBM, S(t_k) = S0 exp((r - sigma^2/2) t_k + sigma W(t_k)).
The geometric average G = (prod S(t_k))^(1/m) is lognormal, which gives an exact price to measure error against:
    ln G ~ N(mu, v),  mu = ln S0 + (r - sigma^2/2) T (m + 1) / (2m),   v = sigma^2 T (m + 1)(2m + 1) / (6 m^2)
    price = e^{-rT} (e^{mu + v/2} N(d1) - K N(d2)),   d2 = (mu - ln K) / sqrt(v),   d1 = d2 + sqrt(v)
(the variance is sigma^2 / m^2 times the sum of min(t_i, t_j) over all pairs of dates).
The arithmetic average, the contract actually traded, has no closed form.
"""
import math

import numpy as np

from qre.montecarlo import qmc
from qre.pricing import black_scholes as bs


def _validate(spot, strike, vol, time, steps):
    if spot <= 0.0 or strike <= 0.0 or vol <= 0.0 or time <= 0.0:
        raise ValueError("spot, strike, vol and time must be positive")
    if steps < 1:
        raise ValueError("need at least one monitoring date")


def geometric_asian_call(spot, strike, rate, vol, time, steps):
    _validate(spot, strike, vol, time, steps)
    m = steps
    mu = math.log(spot) + (rate - 0.5 * vol * vol) * time * (m + 1) / (2 * m)
    v = vol * vol * time * (m + 1) * (2 * m + 1) / (6 * m * m)
    d2 = (mu - math.log(strike)) / math.sqrt(v)
    d1 = d2 + math.sqrt(v)
    return math.exp(-rate * time) * (math.exp(mu + 0.5 * v) * bs._norm_cdf(d1) - strike * bs._norm_cdf(d2))


def asian_call_payoffs(spot, strike, rate, vol, time, w, average="arithmetic"):
    """Discounted payoffs from Brownian paths w of shape (n, m) at the monitoring dates."""
    w = np.atleast_2d(w)
    m = w.shape[1]
    _validate(spot, strike, vol, time, m)
    t = time * np.arange(1, m + 1) / m
    log_s = math.log(spot) + (rate - 0.5 * vol * vol) * t + vol * w
    if average == "arithmetic":
        a = np.exp(log_s).mean(axis=1)
    elif average == "geometric":
        a = np.exp(log_s.mean(axis=1))
    else:
        raise ValueError("average must be arithmetic or geometric")
    return math.exp(-rate * time) * np.maximum(a - strike, 0.0)


def asian_call_estimate(spot, strike, rate, vol, time, steps, n_paths, method="mc", average="arithmetic", seed=42):
    """One price estimate from n_paths paths.
    method: "mc" (pseudo-random, incremental), "sobol" (randomly shifted Sobol, incremental),
            "sobol_bridge" (randomly shifted Sobol, Brownian bridge).
    Repeat with different seeds for a standard error; each estimate is unbiased."""
    if method == "mc":
        z = np.random.default_rng(seed).standard_normal((n_paths, steps))
    elif method in ("sobol", "sobol_bridge"):
        z = qmc.sobol_normals(n_paths, steps, qmc.random_shift(steps, seed))
    else:
        raise ValueError("method must be mc, sobol or sobol_bridge")
    w = qmc.brownian_bridge(z, time) if method == "sobol_bridge" else qmc.incremental_path(z, time)
    return float(asian_call_payoffs(spot, strike, rate, vol, time, w, average).mean())
