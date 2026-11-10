"""Phase 10 reference: principal component analysis of a covariance (or correlation) matrix.

C = V diag(lambda) V', eigenvalues sorted from largest to smallest. Column k of V is the k-th principal
component (a portfolio of the assets), lambda_k its variance, and lambda_k / sum(lambda) the share of total
variance it explains. For equities the first component is usually close to an equal-weighted "market"
portfolio. Eigenvector signs are arbitrary; they are fixed here so each component's loadings sum to >= 0.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PcaResult:
    eigenvalues: np.ndarray        # descending
    components: np.ndarray         # columns are eigenvectors, same order
    explained: np.ndarray          # share of total variance per component


def pca(cov):
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 1:
        raise ValueError("need a square matrix")
    if not np.allclose(cov, cov.T):
        raise ValueError("matrix is not symmetric")
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    signs = np.where(vecs.sum(axis=0) < 0.0, -1.0, 1.0)
    vecs = vecs * signs
    total = vals.sum()
    if total <= 0.0:
        raise ValueError("total variance must be positive")
    return PcaResult(vals, vecs, vals / total)
