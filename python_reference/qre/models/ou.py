"""Phase 8 reference: the Ornstein-Uhlenbeck (Vasicek) mean-reverting process.

dX = kappa (theta - X) dt + sigma dW,  kappa > 0 the speed of mean reversion, theta the long-run mean.
Exact transition over a step dt (no discretization error):
    X_{t+dt} = theta + (X_t - theta) e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) Z
Stationary distribution: Normal(theta, sigma^2 / (2 kappa)); half-life of a deviation: ln 2 / kappa.

Maximum likelihood: the exact transition is an AR(1), X_{t+1} = a + b X_t + e with b = e^{-kappa dt},
so the MLE comes from the least-squares regression of X_{t+1} on X_t:
    kappa = -ln(b) / dt,   theta = a / (1 - b),   sigma = sd(e) sqrt(2 kappa / (1 - b^2))
(sd(e) with the MLE divisor n). Mean reversion is only identifiable when 0 < b < 1.
Caveat: on a random walk (no mean reversion) the least-squares slope is biased below 1, so the fit
usually returns a small spurious kappa rather than an error. A fitted kappa is not proof of mean
reversion, and kappa is the hardest parameter to pin down (it needs many half-lives of data).
"""
import math
from dataclasses import dataclass

import numpy as np


def _check(kappa, sigma, dt):
    if kappa <= 0.0 or sigma <= 0.0 or dt <= 0.0:
        raise ValueError("need kappa > 0, sigma > 0 and dt > 0")


def simulate_ou(x0, kappa, theta, sigma, dt, steps, n_paths=1, seed=42):
    """Exact simulation; returns (n_paths, steps + 1)."""
    _check(kappa, sigma, dt)
    if steps < 1 or n_paths < 1:
        raise ValueError("need steps >= 1 and n_paths >= 1")
    rng = np.random.default_rng(seed)
    decay = math.exp(-kappa * dt)
    sd = sigma * math.sqrt((1.0 - decay * decay) / (2.0 * kappa))
    x = np.empty((n_paths, steps + 1))
    x[:, 0] = x0
    z = rng.standard_normal((n_paths, steps))
    for k in range(steps):
        x[:, k + 1] = theta + (x[:, k] - theta) * decay + sd * z[:, k]
    return x


@dataclass
class OuFit:
    kappa: float
    theta: float
    sigma: float

    @property
    def half_life(self):
        return math.log(2.0) / self.kappa

    @property
    def stationary_sd(self):
        return self.sigma / math.sqrt(2.0 * self.kappa)


def fit_ou(x, dt):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 3 or dt <= 0.0:
        raise ValueError("need a 1-D series of at least 3 observations and dt > 0")
    x0, x1 = x[:-1], x[1:]
    mx0, mx1 = x0.mean(), x1.mean()
    sxx = np.sum((x0 - mx0) ** 2)
    if sxx <= 0.0:
        raise ValueError("series is constant")
    b = float(np.sum((x0 - mx0) * (x1 - mx1)) / sxx)
    a = mx1 - b * mx0
    if not (0.0 < b < 1.0):
        raise ValueError("no mean reversion detected (AR(1) slope outside (0, 1))")
    resid = x1 - a - b * x0
    s2 = float(np.mean(resid ** 2))
    kappa = -math.log(b) / dt
    return OuFit(kappa, a / (1.0 - b), math.sqrt(s2 * 2.0 * kappa / (1.0 - b * b)))
