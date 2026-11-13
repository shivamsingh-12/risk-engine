import math

import numpy as np
import pytest

from qre.hedging import deep_hedging as dh

T = 30 / 252


def deterministic_policy(n_in, hidden):
    """Same weights as the C++ test: W[i, j] = 0.3 sin(1 + i + 2 j) and friends."""
    w = lambda a, b, s: np.array([[0.3 * math.sin(s + i + 2 * j) for j in range(b)] for i in range(a)])
    return dh.Policy(w(n_in, hidden, 1.0), 0.1 * np.cos(np.arange(hidden)), w(hidden, hidden, 2.0),
                     0.05 * np.sin(np.arange(hidden)), w(hidden, 1, 3.0), np.array([0.5]), 1.5)


def deterministic_paths(n_paths=6, n_steps=5, with_v=False):
    i, k = np.meshgrid(np.arange(n_paths), np.arange(n_steps + 1), indexing="ij")
    S = 100 * np.exp(0.03 * np.sin(0.7 * i + 1.3 * k) * np.sqrt(k / n_steps) + 0.01 * k * (i % 3 - 1))
    v = 0.04 + 0.01 * np.cos(i + k) if with_v else None
    return S, v


def test_gradient_matches_finite_differences():
    for with_v, cost in ((False, 0.0), (False, 0.003), (True, 0.002)):
        S, v = deterministic_paths(40, 6, with_v)
        S = S * np.exp(0.02 * np.random.default_rng(1).standard_normal(S.shape))
        p = deterministic_policy(4 if with_v else 3, 7)
        J, g, _ = dh.objective_and_grad(p, S, 100.0, T, cost, 0.8, v)
        for name in dh.Policy.NAMES:
            a = getattr(p, name)
            for flat in range(0, a.size, max(1, a.size // 5)):
                idx = np.unravel_index(flat, a.shape)
                h = 1e-6
                up, dn = p.copy(), p.copy()
                getattr(up, name)[idx] += h
                getattr(dn, name)[idx] -= h
                fd = (dh.objective_and_grad(up, S, 100.0, T, cost, 0.8, v)[0]
                      - dh.objective_and_grad(dn, S, 100.0, T, cost, 0.8, v)[0]) / (2 * h)
                assert g[name][idx] == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_cross_language_reference_values():
    """Printed values are hard-coded in cpp/tests/test_phase17.cpp."""
    S, v = deterministic_paths(6, 5, True)
    p = deterministic_policy(4, 5)
    J, g, L = dh.objective_and_grad(p, S, 100.0, T, 0.002, 0.7, v)
    assert np.isfinite(J) and L.shape == (6,)
    assert dh.policy_deltas(p, S, 100.0, T, v).shape == (6, 5)


def test_hedge_loss_by_hand():
    S = np.array([[100.0, 102.0, 101.0, 105.0]])
    d = np.array([[0.5, 0.6, 0.2]])
    L = dh.hedge_loss(S, d, 100.0, 0.01)
    gains = 0.5 * 2 + 0.6 * -1 + 0.2 * 4
    costs = 0.01 * (100 * 0.5 + 102 * 0.1 + 101 * 0.4 + 105 * 0.2)
    assert L[0] == pytest.approx(5.0 - gains + costs)


def test_black_scholes_delta_hedge_replicates_with_frequent_trading():
    S = dh.simulate_gbm(20000, 200, 100, 0.0, 0.2, T, seed=3)
    L = dh.hedge_loss(S, dh.bs_hedge_deltas(S, 100.0, T, 0.2), 100.0, 0.0)
    assert np.mean(L) == pytest.approx(dh.bs_call(100, 100, 0.2, T), abs=0.01)
    S30 = dh.simulate_gbm(20000, 30, 100, 0.0, 0.2, T, seed=4)
    L30 = dh.hedge_loss(S30, dh.bs_hedge_deltas(S30, 100.0, T, 0.2), 100.0, 0.0)
    assert np.std(L) < 0.5 * np.std(L30)                           # hedging error shrinks like 1/sqrt(n)


def test_cvar_and_quantile():
    L = np.arange(1.0, 101.0)
    assert dh.cvar(L, 0.9) == pytest.approx(np.mean(L[90:]) + 0.0, rel=1e-12)     # mean of the worst 10
    p = deterministic_policy(3, 4)
    S, _ = deterministic_paths(50, 5)
    w = dh.var_threshold(p, S, 100.0, T, 0.0, 0.9)
    assert w == pytest.approx(np.quantile(dh.hedge_loss(S, dh.policy_deltas(p, S, 100.0, T), 100.0, 0.0), 0.9,
                                          method="inverted_cdf"))


def test_heston_simulation_shapes_and_martingale():
    S, v = dh.simulate_heston(40000, 10, 100, 0.0, 0.04, 2.0, 0.04, 0.5, -0.7, 0.5, seed=4)
    assert S.shape == v.shape == (40000, 11) and np.all(v >= 0)
    assert np.mean(S[:, -1]) == pytest.approx(100, abs=0.3)
    assert np.mean(v[:, -1]) == pytest.approx(0.04, abs=0.003)


def test_training_reduces_cvar_quickly():
    samp = lambda n, rng: (dh.simulate_gbm(n, 10, 100, 0.0, 0.2, T, seed=int(rng.integers(1 << 31))), None)
    p0 = dh.Policy.init(3, 8, seed=0)
    p1, log = dh.train(p0, samp, 100.0, T, 0.0, 0.9, iterations=150, batch=1024, lr=1e-2, seed=1)
    S = dh.simulate_gbm(20000, 10, 100, 0.0, 0.2, T, seed=9)
    before = dh.cvar(dh.hedge_loss(S, dh.policy_deltas(p0, S, 100.0, T), 100.0, 0.0), 0.9)
    after = dh.cvar(dh.hedge_loss(S, dh.policy_deltas(p1, S, 100.0, T), 100.0, 0.0), 0.9)
    bs = dh.cvar(dh.hedge_loss(S, dh.bs_hedge_deltas(S, 100.0, T, 0.2), 100.0, 0.0), 0.9)
    assert after < 0.6 * before and after < 1.15 * bs


def test_validation():
    S, _ = deterministic_paths()
    with pytest.raises(ValueError):
        dh.objective_and_grad(deterministic_policy(4, 3), S, 100.0, T, 0.0, 0.9)     # 4 inputs need variance
