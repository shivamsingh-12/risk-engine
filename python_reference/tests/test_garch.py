import math

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import norm

from qre.risk import backtest as bt
from qre.risk import ewma
from qre.risk import garch

TRUE = dict(omega=2e-6, alpha=0.08, beta=0.90)          # long-run daily vol 1%


def test_reparameterization_round_trip_and_constraints():
    for a, b in [(0.05, 0.90), (0.2, 0.5), (0.001, 0.998), (0.3, 0.0001)]:
        a2, b2 = garch.params_from_theta(garch.theta_from_params(a, b))
        assert (a2, b2) == pytest.approx((a, b), rel=1e-9)
    rng = np.random.default_rng(0)
    for theta in rng.normal(0, 20, size=(500, 2)):
        a, b = garch.params_from_theta(theta)
        assert a >= 0 and b >= 0 and a + b < 1


def test_variance_path_recursion():
    r = np.array([0.01, -0.02, 0.005])
    h = garch.variance_path(r, 1e-6, 0.1, 0.85, 1e-4)
    expected = [1e-4]
    for x in r:
        expected.append(1e-6 + 0.1 * x * x + 0.85 * expected[-1])
    assert h == pytest.approx(expected, rel=1e-15)


def test_ewma_is_garch_with_zero_omega_and_unit_persistence():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(300) * 0.01
    lam = 0.94
    h = garch.variance_path(r, 0.0, 1 - lam, lam, float(np.mean(r * r)))
    assert np.sqrt(h) == pytest.approx(ewma.ewma_vol_path(r, lam), rel=1e-12)


def test_recovers_true_parameters_on_long_simulation():
    r = garch.simulate_garch(8000, **TRUE, seed=2)
    fit = garch.fit_garch(r)
    assert fit.alpha == pytest.approx(0.08, abs=0.025)
    assert fit.beta == pytest.approx(0.90, abs=0.03)
    assert fit.long_run_var == pytest.approx(TRUE["omega"] / 0.02, rel=0.15)


def test_agrees_with_arch_package():
    arch = pytest.importorskip("arch")
    r = garch.simulate_garch(6000, **TRUE, seed=3)
    ours = garch.fit_garch(r)
    am = arch.arch_model(r * 100.0, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")
    assert ours.alpha == pytest.approx(res.params["alpha[1]"], abs=0.02)
    assert ours.beta == pytest.approx(res.params["beta[1]"], abs=0.02)


def test_nelder_mead_reaches_the_optimum():
    """A different optimizer started at our answer must not find a meaningfully better likelihood."""
    r = garch.simulate_garch(2000, **TRUE, seed=4)
    fit = garch.fit_garch(r)
    v = fit.long_run_var
    theta = garch.theta_from_params(fit.alpha, fit.beta)
    other = minimize(garch.neg_loglik, theta, args=(r, v), method="BFGS")
    assert -other.fun <= fit.loglik + 1e-6


def test_forecast_reacts_to_shock():
    r = garch.simulate_garch(1000, **TRUE, seed=5)
    calm = garch.fit_garch(r)
    shocked = r.copy()
    shocked[-1] = -0.06
    v_calm = garch.forecast_variance(calm, r)
    v_shock = garch.forecast_variance(calm, shocked)
    assert v_shock > 2.0 * v_calm


def test_parametric_var_uses_forecast_vol():
    r = garch.simulate_garch(1000, **TRUE, seed=6)
    fit = garch.fit_garch(r)
    sigma = math.sqrt(garch.forecast_variance(fit, r))
    var, es = garch.garch_var_es(r, 0.99)
    assert var == pytest.approx(sigma * norm.ppf(0.99), rel=1e-12)
    assert es == pytest.approx(sigma * norm.pdf(norm.ppf(0.99)) / 0.01, rel=1e-12)
    fv, fe = garch.garch_var_es(r, 0.99, filtered=True)
    assert fe >= fv > 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        garch.fit_garch(np.zeros(50))
    with pytest.raises(ValueError):
        garch.fit_garch(np.ones(5) * 0.01)


@pytest.mark.parametrize("method", ["garch", "fhs_garch"])
def test_garch_backtest_has_no_look_ahead(method):
    rng = np.random.default_rng(7)
    rets = rng.standard_normal((220, 2)) * 0.01
    p = 100.0 * np.vstack([np.ones(2), np.cumprod(1.0 + rets, axis=0)])
    w = np.array([0.6, 0.4])
    s = bt.rolling_backtest(p, w, window=150, method=method)
    future = p.copy()
    future[181:] *= np.linspace(0.5, 3.0, future.shape[0] - 181)[:, None]
    s2 = bt.rolling_backtest(future, w, window=150, method=method)
    i = int(np.flatnonzero(s.rows == 181)[0])
    assert np.array_equal(s.var[: i + 1], s2.var[: i + 1])
