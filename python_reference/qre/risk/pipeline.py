"""Phase 7 glue: prices -> returns -> covariance -> portfolio -> simulation -> VaR/ES in one call."""
from dataclasses import dataclass

import numpy as np

from qre.risk import var_es as ve


@dataclass
class RiskReport:
    historical: tuple          # (VaR, ES)
    parametric: tuple
    monte_carlo: tuple
    portfolio_vol: float       # 1-day
    risk_contributions: np.ndarray   # sums to portfolio_vol
    n_returns: int


def forward_fill(prices, max_consec_days):
    """Fill NaN gaps with the last price for at most max_consec_days in a row; longer gaps stay NaN."""
    out = np.array(prices, dtype=float, copy=True)
    for c in range(out.shape[1]):
        last, gap = np.nan, 0
        for r in range(out.shape[0]):
            if np.isnan(out[r, c]):
                if np.isnan(last):
                    continue
                if gap < max_consec_days:
                    out[r, c] = last
                    gap += 1
            else:
                last, gap = out[r, c], 0
    return out


def run_pipeline(prices, weights, confidence=0.99, horizon_days=1, n_paths=100_000, seed=42,
                 max_fill_days=5, trading_days=252):
    prices = np.asarray(prices, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if prices.ndim != 2 or prices.shape[0] < 3:
        raise ValueError("need a 2-D price matrix with at least 3 rows")
    if prices.shape[1] != weights.shape[0]:
        raise ValueError("prices columns must match number of weights")
    prices = forward_fill(prices, max_fill_days)
    if np.isnan(prices).any():
        raise ValueError("prices still contain NaN after forward fill; drop or repair those assets")
    if np.any(prices <= 0.0):
        raise ValueError("prices must be positive")

    returns = prices[1:] / prices[:-1] - 1.0                # simple returns aggregate across assets
    cov = np.cov(returns, rowvar=False, ddof=1)
    port = ve.portfolio_returns(returns, weights)
    sigma_p = ve.portfolio_vol(weights, cov)
    return RiskReport(
        historical=ve.historical_var_es(port, confidence, horizon_days),
        parametric=ve.parametric_var_es(0.0, sigma_p, confidence, horizon_days),
        monte_carlo=ve.monte_carlo_var_es(weights, cov, confidence, horizon_days, n_paths, seed,
                                          trading_days),
        portfolio_vol=sigma_p,
        risk_contributions=ve.risk_contributions(weights, cov),
        n_returns=returns.shape[0],
    )
