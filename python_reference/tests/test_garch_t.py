import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, t as student_t

from qre.risk import backtest as bt
from qre.risk import garch

TRUE = dict(omega=2e-6, alpha=0.08, beta=0.90)


def _std_t_pdf(x, nu):
    s = math.sqrt((nu - 2.0) / nu)
    return student_t.pdf(x / s, nu) / s


@pytest.mark.parametrize("nu", [3.0, 5.0, 10.0])
@pytest.mark.parametrize("c", [0.95, 0.99, 0.999])
def test_std_t_var_es_match_numerical_integration(nu, c):
    var, es = garch.std_t_var_es(1.0, nu, c)
    tail_prob = quad(lambda x: _std_t_pdf(x, nu), var, np.inf)[0]
    assert tail_prob == pytest.approx(1.0 - c, rel=1e-8)
    tail_mean = quad(lambda x: x * _std_t_pdf(x, nu), var, np.inf)[0] / (1.0 - c)
    assert es == pytest.approx(tail_mean, rel=1e-7)


def test_std_t_has_unit_variance_and_scales_with_sigma():
    nu = 6.0
    var2 = quad(lambda x: x * x * _std_t_pdf(x, nu), -np.inf, np.inf)[0]
    assert var2 == pytest.approx(1.0, rel=1e-8)
    a = garch.std_t_var_es(1.0, nu, 0.99)
    b = garch.std_t_var_es(0.02, nu, 0.99)
    assert b == pytest.approx((0.02 * a[0], 0.02 * a[1]))


def test_large_nu_recovers_normal():
    var, es = garch.std_t_var_es(1.0, 1e7, 0.99)
    z = norm.ppf(0.99)
    assert var == pytest.approx(z, rel=1e-5)
    assert es == pytest.approx(norm.pdf(z) / 0.01, rel=1e-5)


def test_fat_tails_cross_the_normal():
    """Same variance: the t puts less mass in the shoulders and more in the far tail."""
    for c, below in [(0.95, True), (0.999, False)]:
        tv, _ = garch.std_t_var_es(1.0, 4.0, c)
        nv = norm.ppf(c)
        assert (tv < nv) if below else (tv > nv)


def test_t_loglik_matches_scipy_logpdf():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(300) * 0.01
    v = float(np.mean(r * r))
    theta = np.append(garch.theta_from_params(0.07, 0.9), math.log(6.0 - 2.0))
    nu = 6.0
    h = garch.variance_path(r, v * (1 - 0.97), 0.07, 0.9, v)[:-1]
    scale = np.sqrt(h * (nu - 2.0) / nu)
    expected = -np.sum(student_t.logpdf(r / scale, nu) - np.log(scale))
    assert garch.neg_loglik_t(theta, r, v) == pytest.approx(expected, rel=1e-12)


def test_recovers_fat_tails_on_simulated_t_garch():
    r = garch.simulate_garch(8000, **TRUE, seed=2, nu=5.0)
    fit = garch.fit_garch(r, dist="t")
    assert fit.nu == pytest.approx(5.0, abs=1.5)
    assert fit.alpha == pytest.approx(0.08, abs=0.03)
    assert fit.beta == pytest.approx(0.90, abs=0.03)


def test_normal_data_gives_large_nu():
    r = garch.simulate_garch(8000, **TRUE, seed=3)
    assert garch.fit_garch(r, dist="t").nu > 20.0


def test_t_likelihood_beats_normal_on_fat_tailed_data():
    r = garch.simulate_garch(4000, **TRUE, seed=4, nu=4.0)
    assert garch.fit_garch(r, dist="t").loglik > garch.fit_garch(r).loglik + 20.0


def test_agrees_with_arch_package_t():
    arch = pytest.importorskip("arch")
    r = garch.simulate_garch(6000, **TRUE, seed=5, nu=6.0)
    ours = garch.fit_garch(r, dist="t")
    res = arch.arch_model(r * 100.0, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    assert ours.nu == pytest.approx(res.params["nu"], abs=1.0)
    assert ours.alpha == pytest.approx(res.params["alpha[1]"], abs=0.02)
    assert ours.beta == pytest.approx(res.params["beta[1]"], abs=0.02)


def test_garch_t_var_es_uses_fitted_nu_and_forecast():
    r = garch.simulate_garch(1000, **TRUE, seed=6, nu=5.0)
    fit = garch.fit_garch(r, dist="t")
    sigma = math.sqrt(garch.forecast_variance(fit, r))
    assert garch.garch_var_es(r, 0.99, dist="t") == pytest.approx(garch.std_t_var_es(sigma, fit.nu, 0.99), rel=1e-12)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        garch.std_t_var_es(1.0, 2.0, 0.99)
    with pytest.raises(ValueError):
        garch.std_t_var_es(0.0, 5.0, 0.99)
    with pytest.raises(ValueError):
        garch.std_t_var_es(1.0, 5.0, 1.0)
    with pytest.raises(ValueError):
        garch.fit_garch(np.ones(50) * 0.01, dist="cauchy")


def test_garch_t_backtest_has_no_look_ahead():
    rng = np.random.default_rng(7)
    rets = rng.standard_normal((200, 2)) * 0.01
    p = 100.0 * np.vstack([np.ones(2), np.cumprod(1.0 + rets, axis=0)])
    w = np.array([0.6, 0.4])
    s = bt.rolling_backtest(p, w, window=150, method="garch_t")
    future = p.copy()
    future[181:] *= np.linspace(0.5, 3.0, future.shape[0] - 181)[:, None]
    s2 = bt.rolling_backtest(future, w, window=150, method="garch_t")
    i = int(np.flatnonzero(s.rows == 181)[0])
    assert np.array_equal(s.var[: i + 1], s2.var[: i + 1])
