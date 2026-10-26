"""Phase 7 reference: Value-at-Risk and Expected Shortfall, three ways, plus risk contributions.

Conventions (match the C++ implementation):
    - VaR and ES are reported as POSITIVE LOSSES, as a fraction of portfolio value.
    - confidence c in (0, 1), e.g. 0.99: VaR is the loss exceeded with probability 1 - c.
    - ES (expected shortfall) is the average loss given the loss is at least VaR, so ES >= VaR.
    - weights are fractions of portfolio value (buy and hold over the horizon).
    - horizon h is in trading days. Historical and parametric scale the 1-day figure by sqrt(h)
      (an approximation that assumes i.i.d. returns); Monte Carlo simulates the horizon directly.
"""
import math

import numpy as np
from scipy.stats import norm

from qre.montecarlo.gbm import simulate_correlated_gbm


def _check_confidence(confidence):
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")


def _check_horizon(horizon_days):
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")


def var_es_from_losses(losses, confidence):
    """Empirical VaR (linear-interpolated quantile) and ES (mean of losses >= VaR) of a loss sample."""
    _check_confidence(confidence)
    losses = np.asarray(losses, dtype=float)
    if losses.size < 2:
        raise ValueError("need at least 2 observations")
    var = float(np.quantile(losses, confidence))
    es = float(losses[losses >= var].mean())
    return var, es


def historical_var_es(returns, confidence, horizon_days=1):
    """Historical simulation on a series of portfolio returns (1-day), sqrt-time scaled to the horizon."""
    _check_horizon(horizon_days)
    var, es = var_es_from_losses(-np.asarray(returns, dtype=float), confidence)
    scale = math.sqrt(horizon_days)
    return var * scale, es * scale


def parametric_var_es(mu, sigma, confidence, horizon_days=1):
    """Normal (variance-covariance) VaR/ES from 1-day portfolio mean and volatility."""
    _check_confidence(confidence)
    _check_horizon(horizon_days)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    mu_h = mu * horizon_days
    sigma_h = sigma * math.sqrt(horizon_days)
    z = norm.ppf(confidence)
    var = -mu_h + sigma_h * z
    es = -mu_h + sigma_h * norm.pdf(z) / (1.0 - confidence)
    return float(var), float(es)


def portfolio_returns(returns, weights):
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if returns.ndim != 2 or returns.shape[1] != weights.shape[0]:
        raise ValueError("returns columns must match number of weights")
    return returns @ weights


def portfolio_vol(weights, cov):
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (weights.shape[0],) * 2:
        raise ValueError("cov must be square and match the number of weights")
    var = float(weights @ cov @ weights)
    if var <= 0.0:
        raise ValueError("portfolio variance must be positive")
    return math.sqrt(var)


def risk_contributions(weights, cov):
    """Euler decomposition: RC_i = w_i (cov w)_i / sigma_p, so that sum(RC) == sigma_p."""
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    sigma_p = portfolio_vol(weights, cov)
    return weights * (cov @ weights) / sigma_p


def monte_carlo_var_es(weights, cov, confidence, horizon_days=1, n_paths=100_000, seed=42,
                       trading_days=252, antithetic=True):
    """Monte Carlo VaR/ES from correlated GBM (Phase 6). Zero expected simple return over the horizon."""
    _check_confidence(confidence)
    _check_horizon(horizon_days)
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n_assets = weights.shape[0]
    if cov.shape != (n_assets, n_assets):
        raise ValueError("cov must be square and match the number of weights")
    daily_vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(daily_vol, daily_vol)
    ann_vol = daily_vol * math.sqrt(trading_days)
    paths = simulate_correlated_gbm(np.ones(n_assets), 0.0, ann_vol, corr, horizon_days / trading_days,
                                    1, n_paths, seed=seed, antithetic=antithetic)
    asset_returns = paths[:, :, -1] - 1.0                  # (n_assets, n_paths)
    return var_es_from_losses(-(weights @ asset_returns), confidence)
