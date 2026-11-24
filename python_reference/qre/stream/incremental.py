"""Incremental portfolio risk reference (Phase 22): update VaR on every price tick without recomputing it.

A portfolio holds q_i shares of each asset. Its dollar exposures are x_i = q_i p_i. Two risk measures are kept:

Parametric (normal)
    sigma^2 = x' S x, with S the covariance of daily returns. Recomputing it costs O(N^2). When asset i ticks and its
    exposure changes by d = q_i (p_new - p_old), with m = S x kept alongside:
        sigma^2 += 2 d m_i + d^2 S_ii        m += d S[:, i]                  O(N)
Historical
    Scenario losses L_s = -sum_i x_i R_si over the last T days of returns R (T x N). Recomputing costs O(T N); after a
    tick in asset i:
        L -= d R[:, i]                                                       O(T)
    VaR is a quantile of L, found by selection in O(T) (numpy.partition here, std::nth_element in C++), and ES the
    mean of the losses at or beyond it, matching var_es.var_es_from_losses.

So a tick costs O(N + T) instead of O(N^2 + N T). The updates accumulate rounding error, which resync() clears by
recomputing from scratch; Experiment 27 measures how fast it grows.

Matches cpp/include/qre/stream/incremental.hpp.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Snapshot:
    ticks: int
    value: float
    pnl: float                          # value now minus value at the start
    sigma: float                        # 1-day dollar standard deviation
    var_parametric: float               # 1-day dollar VaR and ES at the confidence
    es_parametric: float
    var_historical: float
    es_historical: float
    component_var: np.ndarray           # parametric, sums to var_parametric


def quantile_linear(x, q):
    """numpy.quantile's default (linear interpolation between order statistics) by selection, O(n)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = (n - 1) * q
    lo = int(np.floor(h))
    hi = min(lo + 1, n - 1)
    part = np.partition(x, (lo, hi))
    return float(part[lo] + (h - lo) * (part[hi] - part[lo]))


class IncrementalRisk:
    def __init__(self, shares, prices, cov, scenarios, confidence=0.99):
        self.q = np.asarray(shares, dtype=float).copy()
        self.p = np.asarray(prices, dtype=float).copy()
        self.cov = np.asarray(cov, dtype=float)
        self.r = np.asarray(scenarios, dtype=float)
        n = self.q.size
        if self.p.shape != (n,) or self.cov.shape != (n, n) or self.r.ndim != 2 or self.r.shape[1] != n:
            raise ValueError("shares, prices, covariance (N x N) and scenarios (T x N) must agree on N")
        if self.r.shape[0] < 20:
            raise ValueError("need at least 20 return scenarios")
        if np.any(self.p <= 0) or not np.all(np.isfinite(self.p)):
            raise ValueError("prices must be finite and positive")
        if not 0.5 <= confidence < 1.0:
            raise ValueError("confidence must be in [0.5, 1)")
        self.confidence = confidence
        self.z = float(norm.ppf(confidence))
        self.k_es = float(norm.pdf(self.z)) / (1.0 - confidence)
        self.ticks = 0
        self.resync()
        self.value0 = self.value

    def resync(self):
        """Recompute every running quantity from scratch."""
        self.x = self.q * self.p
        self.value = float(self.x.sum())
        self.m = self.cov @ self.x
        self.var2 = float(self.x @ self.m)
        self.loss = -(self.r @ self.x)

    def on_tick(self, i, price):
        if not price > 0:
            raise ValueError("price must be positive")
        d = self.q[i] * (price - self.p[i])
        self.p[i] = price
        self.x[i] += d
        self.value += d
        self.var2 += 2.0 * d * self.m[i] + d * d * self.cov[i, i]
        self.m += d * self.cov[:, i]
        self.loss -= d * self.r[:, i]
        self.ticks += 1

    def snapshot(self):
        sig = float(np.sqrt(max(self.var2, 0.0)))
        var_h = quantile_linear(self.loss, self.confidence)
        es_h = float(self.loss[self.loss >= var_h].mean())
        comp = self.z * self.x * self.m / sig if sig > 0 else np.zeros_like(self.x)
        return Snapshot(self.ticks, self.value, self.value - self.value0, sig, self.z * sig, self.k_es * sig,
                        var_h, es_h, comp)

    def drift(self):
        """Largest relative difference between the running values and a full recomputation."""
        x = self.q * self.p
        m = self.cov @ x
        v2 = float(x @ m)
        loss = -(self.r @ x)
        scale = max(np.abs(loss).max(), 1e-300)
        return {"variance": abs(self.var2 - v2) / v2, "exposure": float(np.abs(self.x - x).max() / np.abs(x).max()),
                "scenario_loss": float(np.abs(self.loss - loss).max() / scale)}
