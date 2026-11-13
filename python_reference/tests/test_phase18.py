import math

import numpy as np
import pytest

from qre.execution import almgren_chriss as ac


@pytest.mark.parametrize("lam", [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
def test_closed_form_matches_the_numerical_optimum(lam):
    p = ac.paper_example()
    x, xn = ac.optimal_holdings(p, lam), ac.solve_numerically(p, lam)
    assert x[0] == pytest.approx(p.X, rel=1e-14) and x[-1] == pytest.approx(0.0, abs=1e-6)
    assert x == pytest.approx(xn, rel=1e-10, abs=1e-6)
    assert np.all(np.diff(x) <= 0)                                     # a pure sell program


def test_optimum_beats_perturbations():
    p = ac.Params(X=5e5, T=10.0, N=20, sigma=0.8, eta=3e-6, gamma=2e-7, eps=0.05)
    lam = 2e-6
    x = ac.optimal_holdings(p, lam)
    u = lambda y: ac.expected_cost(p, y) + lam * ac.cost_variance(p, y)
    rng = np.random.default_rng(1)
    for _ in range(50):
        y = x.copy()
        y[1:-1] *= 1 + 0.05 * rng.standard_normal(p.N - 1)
        y[1:-1] = np.minimum.accumulate(np.clip(y[1:-1], 0, p.X))      # keep it a monotone sell program
        assert u(x) <= u(y) + 1e-6


def test_twap_and_limits():
    p = ac.paper_example()
    assert ac.optimal_holdings(p, 0.0) == pytest.approx(ac.twap_holdings(p))
    # TWAP cost: gamma X^2 / 2 + eps X + eta~ X^2 / T
    assert ac.expected_cost(p, ac.twap_holdings(p)) == pytest.approx(0.5 * p.gamma * p.X ** 2 + p.eps * p.X + p.eta_tilde * p.X ** 2 / p.T)
    # sell everything in the first period: no risk beyond period one, maximal impact
    first = np.r_[p.X, np.zeros(p.N)]
    assert ac.cost_variance(p, first) == 0.0
    assert ac.expected_cost(p, first) == pytest.approx(0.5 * p.gamma * p.X ** 2 + p.eps * p.X + p.eta_tilde / p.tau * p.X ** 2)
    huge = ac.optimal_holdings(p, 1e3)
    assert huge[1] < 1e-3 * p.X and np.all(np.isfinite(huge))      # extreme risk aversion -> sell almost at once


def test_kappa_continuous_limit_and_half_life():
    p = ac.Params(X=1e6, T=5.0, N=5000, sigma=0.95, eta=2.5e-6, gamma=2.5e-7)
    lam = 1e-6
    assert ac.kappa(p, lam) == pytest.approx(math.sqrt(lam * p.sigma ** 2 / p.eta), rel=1e-4)   # tau -> 0


def test_frontier_is_monotone():
    p = ac.paper_example()
    f = ac.efficient_frontier(p, np.logspace(-8, -4, 20))
    E = [e for _, e, _ in f]
    sd = [s for _, _, s in f]
    assert np.all(np.diff(E) > 0) and np.all(np.diff(sd) < 0)          # more risk aversion: dearer and safer


def test_monte_carlo_matches_mean_and_variance():
    p = ac.paper_example()
    for lam in (0.0, 1e-6):
        x = ac.optimal_holdings(p, lam)
        L = ac.simulate_shortfall(p, x, 200_000, seed=3)
        sd = math.sqrt(ac.cost_variance(p, x))
        assert L.mean() == pytest.approx(ac.expected_cost(p, x), abs=4 * sd / math.sqrt(L.size))
        assert L.std() == pytest.approx(sd, rel=0.01)


def test_validation():
    with pytest.raises(ValueError):
        ac.Params(X=1e6, T=1.0, N=1, sigma=0.5, eta=1e-7, gamma=1e-6)  # eta~ <= 0
    with pytest.raises(ValueError):
        ac.kappa(ac.paper_example(), -1.0)
