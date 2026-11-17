import math

import numpy as np
import pytest
from scipy.stats import norm

from qre.risk import adaptive, scoring


def _normal_var_es(sigma, c):
    z = norm.ppf(c)
    return sigma * z, sigma * norm.pdf(z) / (1 - c)


# ---------------------------------------------------------------- scores
def test_quantile_and_fz0_values_by_hand():
    assert scoring.quantile_loss(3.0, 2.0, 0.99) == pytest.approx(0.99)          # breach: c (L - v)
    assert scoring.quantile_loss(1.0, 2.0, 0.99) == pytest.approx(0.01)          # (1 - c)(v - L)
    v, e = 2.0, 2.5
    assert scoring.fz0_loss(3.0, v, e, 0.99) == pytest.approx(1.0 / (0.01 * 2.5) + 0.8 + math.log(2.5) - 1.0)
    assert scoring.fz0_loss(1.0, v, e, 0.99) == pytest.approx(0.8 + math.log(2.5) - 1.0)
    with pytest.raises(ValueError):
        scoring.fz0_loss(1.0, 1.0, -1.0, 0.99)
    with pytest.raises(ValueError):
        scoring.quantile_loss(1.0, 1.0, 1.0)


@pytest.mark.parametrize("c", [0.975, 0.99])
def test_scores_are_minimized_by_the_true_forecast(c):
    """Strict consistency: the expected score is lowest at the true VaR (and ES), checked on a big sample."""
    loss = np.random.default_rng(1).standard_normal(2_000_000)
    v0, e0 = _normal_var_es(1.0, c)
    ql = lambda v: scoring.quantile_loss(loss, v, c).mean()
    fz = lambda v, e: scoring.fz0_loss(loss, v, e, c).mean()
    for f in (0.9, 0.97, 1.03, 1.1):
        assert ql(v0) < ql(f * v0)
        assert fz(v0, e0) < fz(f * v0, e0)
        assert fz(v0, e0) < fz(v0, f * e0)
        assert fz(v0, e0) < fz(f * v0, f * e0)


def test_fz0_differences_depend_only_on_forecast_ratios():
    """Scaling losses and forecasts by k shifts every FZ0 score by ln k, so differences between models are unchanged."""
    rng = np.random.default_rng(2)
    loss, v, e = rng.standard_normal(50), 2.0 + rng.random(50), 2.8 + rng.random(50)
    a = scoring.fz0_loss(loss, v, e, 0.99) - scoring.fz0_loss(loss, 1.1 * v, 0.9 * e, 0.99)
    b = scoring.fz0_loss(7 * loss, 7 * v, 7 * e, 0.99) - scoring.fz0_loss(7 * loss, 7.7 * v, 6.3 * e, 0.99)
    assert a == pytest.approx(b)


# ---------------------------------------------------------------- Diebold-Mariano
def test_diebold_mariano_by_hand_and_size():
    a = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 0.2, 1.1, 0.9, 2.2, 1.4, 0.7, 1.8])
    b = np.array([1.2, 1.5, 0.9, 1.0, 2.0, 0.4, 1.0, 1.3, 1.7, 1.2, 0.5, 1.6])
    stat, p, md, lag = scoring.diebold_mariano(a, b, lag=2)
    d = a - b
    dc = d - d.mean()
    lrv = dc @ dc / 12 + 2 * (2 / 3) * (dc[1:] @ dc[:-1]) / 12 + 2 * (1 / 3) * (dc[2:] @ dc[:-2]) / 12
    assert stat == pytest.approx(d.mean() / math.sqrt(lrv / 12))
    assert p == pytest.approx(2 * norm.sf(abs(stat)))
    assert md == pytest.approx(d.mean())
    rng = np.random.default_rng(3)
    pvals = [scoring.diebold_mariano(rng.standard_normal(300), rng.standard_normal(300))[1] for _ in range(500)]
    assert 0.02 < np.mean(np.array(pvals) < 0.05) < 0.09                     # about 5% false rejections
    x = rng.standard_normal(300)
    assert scoring.diebold_mariano(x - 0.3, x + 0.3 * rng.standard_normal(300))[1] < 1e-6
    assert scoring.newey_west_lag(502) == 5
    with pytest.raises(ValueError):
        scoring.diebold_mariano(np.ones(5), np.ones(5))


# ---------------------------------------------------------------- combination
def _two_models(n=600, seed=4):
    """Model 0 is the truth for N(0, 1) losses; model 1 understates risk by 30%."""
    loss = np.random.default_rng(seed).standard_normal(n)
    v, e = _normal_var_es(1.0, 0.975)
    var = np.column_stack([np.full(n, v), np.full(n, 0.7 * v)])
    es = np.column_stack([np.full(n, e), np.full(n, 0.7 * e)])
    return var, es, loss


def test_select_learns_the_better_model_and_exp_interpolates():
    var, es, loss = _two_models()
    _, _, w_sel = adaptive.combine(var, es, loss, 0.975, "select", lookback=250, burn=250)
    assert w_sel[:, 0].mean() > 0.95
    _, _, w0 = adaptive.combine(var, es, loss, 0.975, "exp", lookback=250, eta=0.0, burn=250)
    _, _, w5 = adaptive.combine(var, es, loss, 0.975, "exp", lookback=250, eta=5.0, burn=250)
    assert w0 == pytest.approx(np.full_like(w0, 0.5))
    assert w5[:, 0].mean() > 0.95
    v_eq, e_eq, _ = adaptive.combine(var, es, loss, 0.975, "equal", burn=250)
    assert v_eq == pytest.approx(var[250:].mean(axis=1))
    assert np.all(e_eq >= v_eq)


def test_combination_has_no_look_ahead():
    var, es, loss = _two_models()
    base = adaptive.combine(var, es, loss, 0.975, "exp", lookback=100, eta=1.0, burn=100)
    loss2 = loss.copy()
    loss2[400:] *= 10.0
    var2, es2 = var.copy(), es.copy()
    var2[400:] *= 3.0
    es2[400:] *= 3.0
    alt = adaptive.combine(var2, es2, loss2, 0.975, "exp", lookback=100, eta=1.0, burn=100)
    assert np.array_equal(base[2][:301], alt[2][:301])          # weights through day 400 use days < 400 only
    assert np.array_equal(base[0][:300], alt[0][:300])


def test_combination_validation():
    var, es, loss = _two_models(50)
    with pytest.raises(ValueError):
        adaptive.combine(var, es, loss, 0.975, "median")
    with pytest.raises(ValueError):
        adaptive.combine(var, es, loss, 0.975, "equal", burn=50)
    with pytest.raises(ValueError):
        adaptive.combine(var, es[:, :1], loss, 0.975)


def test_run_models_aligns_forecasts():
    rng = np.random.default_rng(5)
    p = 100.0 * np.cumprod(1.0 + 0.01 * rng.standard_normal((140, 2)), axis=0)
    var, es, loss, rows = adaptive.run_models(p, np.array([0.5, 0.5]), ["historical", "ewma"], window=100)
    assert var.shape == es.shape == (39, 2)
    assert loss.shape == rows.shape == (39,)
    assert np.all(es >= var)


# ---------------------------------------------------------------- regression: GARCH-t on normal data
def test_garch_t_on_normal_returns_stays_finite():
    """Found by the Phase 13 synthetic experiment: on calm normal data nu ran off to ~1e16 and the C++ VaR/ES broke
    (negative VaR, ES ~ 1e32). nu is now bounded, and reaching the bound gives the normal answer."""
    from qre.risk import garch
    r = 0.008 * np.random.default_rng(1313).standard_normal(250)
    fit = garch.fit_garch(r, "t")
    assert 2.0 < fit.nu <= garch.NU_MAX
    var, es = garch.garch_var_es(r, 0.975, dist="t")
    nvar, nes = garch.garch_var_es(r, 0.975)
    assert var == pytest.approx(nvar, rel=0.05)
    assert es == pytest.approx(nes, rel=0.05)
