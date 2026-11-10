import math

import numpy as np
import pytest

from qre.linalg import shrinkage as sh
from qre.stats import pca as pc


def _equicorr(n, rho):
    c = np.full((n, n), rho)
    np.fill_diagonal(c, 1.0)
    return c


def _draw(cov, t, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t, cov.shape[0])) @ np.linalg.cholesky(cov).T


# ---------------------------------------------------------------- Ledoit-Wolf
@pytest.mark.parametrize("n,t", [(5, 50), (40, 50), (80, 50), (10, 1000)])
def test_ledoit_wolf_matches_sklearn(n, t):
    sk = pytest.importorskip("sklearn.covariance")
    x = _draw(_equicorr(n, 0.3), t, seed=n + t)
    ours, delta = sh.ledoit_wolf(x)
    theirs, their_delta = sk.ledoit_wolf(x)
    assert delta == pytest.approx(their_delta, rel=1e-10, abs=1e-14)
    assert ours == pytest.approx(theirs, rel=1e-10, abs=1e-14)


def test_ledoit_wolf_is_positive_definite_even_when_n_exceeds_t():
    x = _draw(_equicorr(80, 0.3), 50, seed=1)
    s = np.cov(x, rowvar=False)
    assert np.linalg.eigvalsh(s).min() < 1e-10                   # sample covariance is singular
    lw, delta = sh.ledoit_wolf(x)
    assert 0.0 < delta <= 1.0
    assert np.linalg.eigvalsh(lw).min() > 0.0


def test_ledoit_wolf_shrinks_less_with_more_data():
    deltas = [sh.ledoit_wolf(_draw(_equicorr(20, 0.3), t, seed=t))[1] for t in (40, 400, 4000)]
    assert deltas[0] > deltas[1] > deltas[2]


def test_ledoit_wolf_keeps_the_average_variance():
    x = _draw(_equicorr(30, 0.3), 60, seed=2)
    lw, _ = sh.ledoit_wolf(x)
    s = np.cov(x, rowvar=False, ddof=0)
    assert np.trace(lw) == pytest.approx(np.trace(s), rel=1e-12)


# ---------------------------------------------------------------- RMT clipping
def test_marchenko_pastur_edge_matches_pure_noise():
    x = np.random.default_rng(3).standard_normal((2000, 500))
    corr = np.corrcoef(x, rowvar=False)
    assert np.linalg.eigvalsh(corr).max() == pytest.approx(sh.marchenko_pastur_upper(500, 2000), rel=0.03)


def test_rmt_finds_no_signal_in_noise_and_returns_identity_correlation():
    x = np.random.default_rng(4).standard_normal((400, 100))
    cleaned, lam_plus, k = sh.rmt_clip(x)
    assert k == 0
    corr, _ = sh.cov_to_corr(cleaned)
    assert corr == pytest.approx(np.eye(100), abs=1e-10)


def test_rmt_keeps_a_market_factor():
    x = _draw(_equicorr(50, 0.3), 100, seed=5)
    cleaned, lam_plus, k = sh.rmt_clip(x)
    assert k >= 1
    corr, sd = sh.cov_to_corr(cleaned)
    assert np.diag(corr) == pytest.approx(np.ones(50))
    assert sd == pytest.approx(np.sqrt(np.diag(np.cov(x, rowvar=False))))   # volatilities untouched
    assert np.linalg.eigvalsh(cleaned).min() > 0.0


# ---------------------------------------------------------------- PCA
def test_pca_on_equicorrelation():
    n, rho = 10, 0.3
    r = pc.pca(_equicorr(n, rho))
    assert r.eigenvalues[0] == pytest.approx(1 + (n - 1) * rho)
    assert r.eigenvalues[1:] == pytest.approx(np.full(n - 1, 1 - rho))
    assert r.components[:, 0] == pytest.approx(np.full(n, 1 / math.sqrt(n)))     # the market portfolio
    assert r.explained.sum() == pytest.approx(1.0)


def test_pca_reconstructs_and_orders():
    a = np.random.default_rng(6).standard_normal((6, 6))
    c = a @ a.T
    r = pc.pca(c)
    assert np.all(np.diff(r.eigenvalues) <= 0)
    assert r.components @ np.diag(r.eigenvalues) @ r.components.T == pytest.approx(c)
    assert np.all(r.components.sum(axis=0) >= 0)


def test_pca_invalid():
    with pytest.raises(ValueError):
        pc.pca(np.array([[1.0, 0.5], [0.4, 1.0]]))
    with pytest.raises(ValueError):
        pc.pca(np.ones((2, 3)))


# ---------------------------------------------------------------- minimum variance
def test_min_variance_weights():
    w = sh.min_variance_weights(np.diag([0.01, 0.04, 0.09]))
    inv = np.array([100.0, 25.0, 100.0 / 9.0])
    assert w == pytest.approx(inv / inv.sum())
    w2 = sh.min_variance_weights(_equicorr(5, 0.3))
    assert w2 == pytest.approx(np.full(5, 0.2))
    with pytest.raises(ValueError):
        sh.min_variance_weights(np.ones((3, 3)))


def test_shrinkage_beats_sample_out_of_sample_when_n_near_t():
    true = _equicorr(40, 0.3)
    opt = 1.0 / np.sum(np.linalg.inv(true))
    x = _draw(true, 50, seed=7)
    s = np.cov(x, rowvar=False)
    lw, _ = sh.ledoit_wolf(x)
    rv = lambda w: w @ true @ w
    assert rv(sh.min_variance_weights(lw)) < rv(sh.min_variance_weights(s))
    assert rv(sh.min_variance_weights(lw)) >= opt
