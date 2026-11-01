import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from qre.models import merton as mj
from qre.models import ou
from qre.montecarlo import sde
from qre.pricing import black_scholes as bs

MU, SIG = 0.05, 0.4
a_gbm = lambda x, t: MU * x
b_gbm = lambda x, t: SIG * x
bx_gbm = lambda x, t: SIG * np.ones_like(x)


def _gbm_exact(x0, dW, T):
    return x0 * np.exp((MU - 0.5 * SIG ** 2) * T + SIG * dW.sum(axis=1))


# ---------------------------------------------------------------- SDE schemes
def test_increments_have_right_variance_and_coarsen_preserves_path():
    dW = sde.brownian_increments(20000, 64, 2.0, seed=1)
    assert dW.var() == pytest.approx(2.0 / 64, rel=0.02)
    c = sde.coarsen(dW, 8)
    assert c.shape == (20000, 8)
    assert c.sum(axis=1) == pytest.approx(dW.sum(axis=1))
    with pytest.raises(ValueError):
        sde.coarsen(dW, 5)


def test_schemes_with_zero_diffusion_reduce_to_ode_solvers():
    dW = np.zeros((1, 1000))
    x = sde.euler_maruyama(1.0, lambda x, t: -x, lambda x, t: 0.0 * x, 1.0, dW)
    assert x[0, -1] == pytest.approx(math.exp(-1.0), rel=1e-3)


def test_milstein_is_exact_for_additive_noise_term():
    """b' = 0 (additive noise): Milstein and Euler-Maruyama coincide."""
    dW = sde.brownian_increments(10, 50, 1.0, seed=2)
    a = lambda x, t: -2.0 * x
    b = lambda x, t: 0.3 + 0.0 * x
    e = sde.euler_maruyama(0.5, a, b, 1.0, dW)
    m = sde.milstein(0.5, a, b, lambda x, t: 0.0 * x, 1.0, dW)
    assert np.allclose(e, m)


def test_strong_convergence_orders():
    """Same Brownian paths at every step size; error against the exact GBM solution."""
    T, x0 = 1.0, 1.0
    fine = sde.brownian_increments(20000, 256, T, seed=3)
    exact = _gbm_exact(x0, fine, T)
    hs, s_em, s_mil = [], [], []
    for steps in (8, 16, 32, 64, 128):
        dW = sde.coarsen(fine, 256 // steps)
        hs.append(T / steps)
        s_em.append(np.mean(np.abs(sde.euler_maruyama(x0, a_gbm, b_gbm, T, dW)[:, -1] - exact)))
        s_mil.append(np.mean(np.abs(sde.milstein(x0, a_gbm, b_gbm, bx_gbm, T, dW)[:, -1] - exact)))
    slope = lambda e: np.polyfit(np.log(hs), np.log(e), 1)[0]
    assert slope(s_em) == pytest.approx(0.5, abs=0.12)
    assert slope(s_mil) == pytest.approx(1.0, abs=0.12)


def test_weak_convergence_order_on_second_moment():
    """Weak error of E[X_T^2]. (The bias in E[X_T] itself is smaller than Monte Carlo noise for GBM.)
    For Euler-Maruyama on GBM the exact weak error is known: ((1+mu h)^2 + sigma^2 h)^n - e^{(2 mu + sigma^2) T}."""
    T, x0 = 1.0, 1.0
    fine = sde.brownian_increments(200000, 64, T, seed=3)
    exact2 = (_gbm_exact(x0, fine, T) ** 2).mean()
    hs, w_em, w_mil, theory = [], [], [], []
    for steps in (4, 8, 16, 32):
        h = T / steps
        dW = sde.coarsen(fine, 64 // steps)
        hs.append(h)
        w_em.append(abs((sde.euler_maruyama(x0, a_gbm, b_gbm, T, dW)[:, -1] ** 2).mean() - exact2))
        w_mil.append(abs((sde.milstein(x0, a_gbm, b_gbm, bx_gbm, T, dW)[:, -1] ** 2).mean() - exact2))
        theory.append(abs(((1 + MU * h) ** 2 + SIG ** 2 * h) ** steps - math.exp((2 * MU + SIG ** 2) * T)))
    slope = lambda e: np.polyfit(np.log(hs), np.log(e), 1)[0]
    assert slope(w_em) == pytest.approx(1.0, abs=0.2)
    assert slope(w_mil) == pytest.approx(1.0, abs=0.2)
    assert w_em == pytest.approx(theory, rel=0.25)


# ---------------------------------------------------------------- Ito calculus
def test_stratonovich_telescopes_exactly():
    dW = sde.brownian_increments(100, 500, 1.0, seed=4)
    W_T = dW.sum(axis=1)
    assert sde.stratonovich_integral_w_dw(dW) == pytest.approx(W_T ** 2 / 2.0, rel=1e-12)


def test_ito_minus_stratonovich_is_half_quadratic_variation():
    dW = sde.brownian_increments(100, 500, 1.0, seed=5)
    gap = sde.stratonovich_integral_w_dw(dW) - sde.ito_integral_w_dw(dW)
    assert gap == pytest.approx(sde.quadratic_variation(dW) / 2.0, rel=1e-10)


def test_quadratic_variation_converges_to_T():
    T = 2.0
    for steps, tol in [(100, 0.6), (10000, 0.06)]:
        qv = sde.quadratic_variation(sde.brownian_increments(2000, steps, T, seed=6))
        assert qv.mean() == pytest.approx(T, rel=0.01)
        assert qv.std() == pytest.approx(T * math.sqrt(2.0 / steps), rel=0.1)   # Var = 2 T^2 / n
        assert qv.std() < tol


def test_ito_integral_has_zero_mean_and_variance_t_squared_over_two():
    dW = sde.brownian_increments(40000, 400, 1.0, seed=7)
    ito = sde.ito_integral_w_dw(dW)
    assert abs(ito.mean()) < 4 * 0.5 / math.sqrt(40000) * math.sqrt(2)
    assert ito.var() == pytest.approx(0.5, rel=0.05)


# ---------------------------------------------------------------- Ornstein-Uhlenbeck
def test_ou_exact_simulation_matches_stationary_distribution():
    x = ou.simulate_ou(0.04, 3.0, 0.04, 0.05, 1 / 252, 2520, n_paths=4000, seed=8)[:, -1]
    assert x.mean() == pytest.approx(0.04, abs=0.002)
    assert x.std() == pytest.approx(0.05 / math.sqrt(6.0), rel=0.05)


def test_ou_conditional_mean_decays_exponentially():
    x = ou.simulate_ou(0.10, 2.0, 0.04, 0.02, 0.5, 1, n_paths=200000, seed=9)[:, -1]
    assert x.mean() == pytest.approx(0.04 + 0.06 * math.exp(-1.0), abs=5e-4)


def test_ou_mle_recovers_parameters_on_long_sample():
    x = ou.simulate_ou(0.04, 3.0, 0.04, 0.05, 1 / 252, 252 * 50, seed=10)[0]
    f = ou.fit_ou(x, 1 / 252)
    assert f.kappa == pytest.approx(3.0, rel=0.3)
    assert f.theta == pytest.approx(0.04, abs=0.005)
    assert f.sigma == pytest.approx(0.05, rel=0.03)
    assert f.half_life == pytest.approx(math.log(2) / f.kappa)


def test_ou_fit_rejects_non_mean_reverting_data():
    with pytest.raises(ValueError):
        ou.fit_ou(1.01 ** np.arange(200), 1.0)                 # explosive: slope > 1
    with pytest.raises(ValueError):
        ou.fit_ou(np.array([1.0, -1.0] * 50) + 5.0, 1.0)       # alternating: slope < 0
    with pytest.raises(ValueError):
        ou.fit_ou(np.ones(10), 1.0)


def test_random_walk_looks_mildly_mean_reverting():
    """Least squares on a unit root is biased below 1 (Dickey-Fuller), so a random walk usually
    yields a small spurious kappa rather than an error: the fit cannot prove mean reversion."""
    rng = np.random.default_rng(11)
    f = ou.fit_ou(np.cumsum(rng.standard_normal(500)), 1.0)
    assert 0.0 < f.kappa < 0.05


# ---------------------------------------------------------------- Merton
BASE = mj.MertonParams(100.0, 100.0, 0.05, 0.2, 0.5, 1.0, -0.1, 0.15)


def test_merton_without_jumps_is_black_scholes():
    p = mj.MertonParams(100.0, 95.0, 0.03, 0.25, 1.0, 0.0, -0.1, 0.15)
    assert mj.merton_price(p) == pytest.approx(bs.price(100.0, 95.0, 0.03, 0.25, 1.0, "call"), rel=1e-14)


def test_merton_put_call_parity():
    c = mj.merton_price(BASE)
    p = mj.merton_price(mj.MertonParams(**{**BASE.__dict__, "kind": "put"}))
    assert c - p == pytest.approx(100.0 - 100.0 * math.exp(-0.05 * 0.5), abs=1e-10)


def test_merton_closed_form_matches_direct_integration():
    """Condition on the number of jumps and integrate the lognormal payoff numerically."""
    p = BASE
    total = 0.0
    for n in range(40):
        w = math.exp(-p.lam * p.time) * (p.lam * p.time) ** n / math.factorial(n)
        m = math.log(p.spot) + (p.rate - p.lam * p.k - 0.5 * p.vol ** 2) * p.time + n * p.mu_j
        s = math.sqrt(p.vol ** 2 * p.time + n * p.delta ** 2)
        integrand = lambda y: max(math.exp(y) - p.strike, 0.0) * norm.pdf(y, m, s)
        total += w * quad(integrand, math.log(p.strike), m + 12 * s)[0]
    assert mj.merton_price(p) == pytest.approx(math.exp(-p.rate * p.time) * total, rel=1e-8)


def test_merton_martingale_and_monte_carlo():
    s_t = mj.simulate_terminal(BASE, 400000, seed=12)
    assert s_t.mean() * math.exp(-0.05 * 0.5) == pytest.approx(100.0, rel=0.003)
    for strike in (80.0, 100.0, 120.0):
        p = mj.MertonParams(**{**BASE.__dict__, "strike": strike})
        price, se = mj.merton_mc_price(p, 400000, seed=13)
        assert abs(price - mj.merton_price(p)) < 4 * se


def test_negative_jumps_create_a_downside_skew():
    """Implied vols from Merton prices: low strikes carry more vol than high strikes."""
    ivs = []
    for strike in (80.0, 100.0, 120.0):
        p = mj.MertonParams(**{**BASE.__dict__, "strike": strike})
        ivs.append(bs.implied_vol(mj.merton_price(p), 100.0, strike, 0.05, 0.5, "call"))
    assert ivs[0] > ivs[1] > ivs[2]
    assert ivs[1] > 0.2                                  # jumps add variance on top of the diffusion


def test_merton_invalid_inputs():
    with pytest.raises(ValueError):
        mj.merton_price(mj.MertonParams(100, 100, 0.05, 0.2, 0.5, -1.0, 0.0, 0.1))
    with pytest.raises(ValueError):
        mj.merton_price(mj.MertonParams(100, 100, 0.05, 0.0, 0.5, 1.0, 0.0, 0.1))
