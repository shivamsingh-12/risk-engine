"""Phase 6 reference: Geometric Brownian Motion simulation and Monte Carlo option pricing.

Risk-neutral dynamics: dS = r S dt + sigma S dW, simulated with the exact lognormal step
    S(t+dt) = S(t) * exp((r - sigma^2/2) dt + sigma sqrt(dt) Z),   Z ~ N(0, 1)
so there is no time-discretization error, only sampling error that shrinks like 1/sqrt(N).

Variance reduction (both optional, combinable):
    antithetic     : pair every draw Z with -Z and average the pair
    control_variate: use the discounted terminal price, whose expectation is exactly S0
"""
import math

import numpy as np


def _validate(spot, vol, time, n_paths):
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if vol <= 0.0:
        raise ValueError("vol must be positive")
    if time <= 0.0:
        raise ValueError("time must be positive")
    if n_paths < 2:
        raise ValueError("need at least 2 paths")


def _normals(rng, shape, antithetic):
    """Standard normal draws; with antithetic the second half mirrors the first (axis 0 is paths)."""
    n_paths = shape[0]
    if not antithetic:
        return rng.standard_normal(shape)
    if n_paths % 2 != 0:
        raise ValueError("antithetic requires an even number of paths")
    half = rng.standard_normal((n_paths // 2,) + tuple(shape[1:]))
    return np.concatenate([half, -half], axis=0)


def simulate_gbm_paths(spot, rate, vol, time, steps, n_paths, seed=42, antithetic=False):
    """Returns array of shape (n_paths, steps + 1); column 0 is the spot."""
    _validate(spot, vol, time, n_paths)
    if steps < 1:
        raise ValueError("need at least 1 step")
    rng = np.random.default_rng(seed)
    dt = time / steps
    z = _normals(rng, (n_paths, steps), antithetic)
    log_inc = (rate - 0.5 * vol * vol) * dt + vol * math.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return spot * np.exp(log_paths)


def simulate_correlated_gbm(spots, rate, vols, corr, time, steps, n_paths, seed=42, antithetic=False):
    """Multi-asset GBM with correlated shocks via Cholesky. Returns (n_assets, n_paths, steps + 1)."""
    spots = np.asarray(spots, dtype=float)
    vols = np.asarray(vols, dtype=float)
    corr = np.asarray(corr, dtype=float)
    n_assets = spots.shape[0]
    if vols.shape != (n_assets,) or corr.shape != (n_assets, n_assets):
        raise ValueError("spots, vols and corr have incompatible dimensions")
    if np.any(spots <= 0.0) or np.any(vols <= 0.0) or time <= 0.0 or steps < 1 or n_paths < 2:
        raise ValueError("spots, vols, time must be positive; need steps >= 1 and n_paths >= 2")
    if not np.allclose(corr, corr.T):
        raise ValueError("correlation matrix is not symmetric")
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation matrix is not positive-definite") from exc

    rng = np.random.default_rng(seed)
    dt = time / steps
    z = _normals(rng, (n_paths, steps, n_assets), antithetic)   # independent
    corr_z = z @ chol.T                                         # correlated across the asset axis
    drift = (rate - 0.5 * vols**2) * dt                         # (n_assets,)
    log_inc = drift + vols * math.sqrt(dt) * corr_z             # (n_paths, steps, n_assets)
    log_paths = np.cumsum(log_inc, axis=1)
    log_paths = np.concatenate([np.zeros((n_paths, 1, n_assets)), log_paths], axis=1)
    out = spots * np.exp(log_paths)                             # (n_paths, steps+1, n_assets)
    return np.transpose(out, (2, 0, 1))


def mc_european_price(spot, strike, rate, vol, time, n_paths, kind="call", seed=42,
                      antithetic=False, control_variate=False):
    """Monte Carlo European option price. Returns (price, standard_error).

    The standard error is computed over independent samples: single draws normally, or
    pair-averages under antithetic sampling (the two halves of a pair are not independent).
    """
    _validate(spot, vol, time, n_paths)
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")

    rng = np.random.default_rng(seed)
    z = _normals(rng, (n_paths,), antithetic)
    s_t = spot * np.exp((rate - 0.5 * vol * vol) * time + vol * math.sqrt(time) * z)
    disc = math.exp(-rate * time)
    payoff = disc * (np.maximum(s_t - strike, 0.0) if kind == "call" else np.maximum(strike - s_t, 0.0))
    control = disc * s_t                                        # E[control] = spot exactly

    if antithetic:
        half = n_paths // 2
        payoff = 0.5 * (payoff[:half] + payoff[half:])
        control = 0.5 * (control[:half] + control[half:])

    if control_variate:
        cov = np.cov(payoff, control, ddof=1)
        beta = cov[0, 1] / cov[1, 1]
        payoff = payoff - beta * (control - spot)

    m = payoff.shape[0]
    return float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(m))
