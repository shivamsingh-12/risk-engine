"""Phase 10 reference: covariance estimation when the number of assets N is not small relative to T.

The sample covariance S has N(N+1)/2 free entries estimated from T observations. When N/T is not small,
its eigenvalues spread out: the largest are biased up, the smallest biased down (to exactly zero once
N >= T), so S becomes ill-conditioned or singular and anything that inverts it (minimum-variance weights,
Mahalanobis distances, Cholesky for simulation) amplifies the noise.

Ledoit-Wolf (2004) linear shrinkage towards a scaled identity:
    S = X'X / T (X demeaned, T x N; the 1/T convention used in the derivation)
    mu = tr(S) / N,  target F = mu I,  <A, B> = tr(A B') / N
    d2 = ||S - F||^2,  b2_bar = (1 / T^2) sum_t ||x_t x_t' - S||^2,  b2 = min(b2_bar, d2)
    delta = b2 / d2  in [0, 1],   Sigma_LW = delta F + (1 - delta) S
delta is estimated from the data: more noise (b2) relative to dispersion (d2) means more shrinkage.
Every eigenvalue moves towards mu, so Sigma_LW is positive definite whenever delta > 0.

Random matrix theory (Marchenko-Pastur) eigenvalue clipping, on the correlation matrix:
    if returns were pure noise with N/T = q, the eigenvalues of the sample correlation matrix would fall
    in [(1 - sqrt q)^2, (1 + sqrt q)^2]. Eigenvalues above lambda_plus = (1 + sqrt q)^2 are treated as signal
    and kept; the rest are replaced by their average (which preserves the trace), the matrix is rebuilt
    and rescaled to unit diagonal, then converted back to a covariance with the sample volatilities.
"""
import math

import numpy as np


def _demean(returns):
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("returns must be a T x N array with T >= 2 and N >= 1")
    return x - x.mean(axis=0)


def ledoit_wolf(returns):
    """Returns (shrunk covariance, shrinkage intensity delta)."""
    x = _demean(returns)
    t, n = x.shape
    s = x.T @ x / t
    mu = np.trace(s) / n
    f = mu * np.eye(n)
    d2 = np.sum((s - f) ** 2) / n
    b2_bar = sum(np.sum((np.outer(row, row) - s) ** 2) / n for row in x) / t ** 2
    b2 = min(b2_bar, d2)
    delta = 0.0 if d2 == 0.0 else b2 / d2
    return delta * f + (1.0 - delta) * s, float(delta)


def cov_to_corr(cov):
    cov = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0.0):
        raise ValueError("variances must be positive")
    return cov / np.outer(sd, sd), sd


def marchenko_pastur_upper(n, t):
    """Largest eigenvalue of a pure-noise sample correlation matrix in the large-N limit."""
    return (1.0 + math.sqrt(n / t)) ** 2


def rmt_clip(returns):
    """Returns (cleaned covariance, lambda_plus, number of eigenvalues kept as signal)."""
    x = _demean(returns)
    t, n = x.shape
    s = x.T @ x / (t - 1)
    corr, sd = cov_to_corr(s)
    vals, vecs = np.linalg.eigh(corr)
    lam_plus = marchenko_pastur_upper(n, t)
    noise = vals <= lam_plus
    if noise.any():
        vals = vals.copy()
        vals[noise] = vals[noise].mean()
    cleaned = vecs @ np.diag(vals) @ vecs.T
    c_sd = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(c_sd, c_sd)
    return cleaned * np.outer(sd, sd), float(lam_plus), int((~noise).sum())


def min_variance_weights(cov):
    """Fully invested minimum-variance weights w = C^-1 1 / (1' C^-1 1); requires C positive definite."""
    cov = np.asarray(cov, dtype=float)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance is not positive definite") from exc
    ones = np.ones(cov.shape[0])
    y = np.linalg.solve(chol.T, np.linalg.solve(chol, ones))
    return y / y.sum()
