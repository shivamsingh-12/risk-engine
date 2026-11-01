"""Phase 8 reference: discretizing stochastic differential equations, and the Ito integral.

SDE  dX = a(X, t) dt + b(X, t) dW  on [0, T] with `steps` equal steps of size h = T / steps.
    Euler-Maruyama:  X_{k+1} = X_k + a h + b dW_k
    Milstein:        X_{k+1} = X_k + a h + b dW_k + 0.5 b b' (dW_k^2 - h),   b' = db/dx
Milstein's extra term comes from Ito's lemma applied to b(X): it is what lifts the strong (pathwise)
convergence order from 1/2 to 1. Both schemes have weak (distributional) order 1.

Drift/diffusion functions take (x, t) with x an array of the current values on every path.
Brownian increments are passed in explicitly so different schemes and step sizes can share one path:
`coarsen` sums fine increments into coarser ones, which is how strong error is measured.

Ito vs Stratonovich for the integral of W against itself:
    Ito (left point)        sum W_k dW_k                 -> (W_T^2 - T) / 2
    Stratonovich (average)  sum (W_k + W_{k+1})/2 dW_k   =  W_T^2 / 2   exactly (it telescopes)
    The gap is half the quadratic variation sum dW_k^2, which converges to T: the dt term in Ito's lemma.
"""
import numpy as np


def brownian_increments(n_paths, steps, T, seed=42):
    if n_paths < 1 or steps < 1 or T <= 0.0:
        raise ValueError("need n_paths >= 1, steps >= 1 and T > 0")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_paths, steps)) * np.sqrt(T / steps)


def coarsen(dW, factor):
    """Sum consecutive groups of `factor` increments: the same Brownian path on a coarser grid."""
    dW = np.asarray(dW, dtype=float)
    if factor < 1 or dW.shape[1] % factor != 0:
        raise ValueError("factor must divide the number of steps")
    return dW.reshape(dW.shape[0], dW.shape[1] // factor, factor).sum(axis=2)


def euler_maruyama(x0, drift, diffusion, T, dW):
    dW = np.asarray(dW, dtype=float)
    n_paths, steps = dW.shape
    h = T / steps
    x = np.empty((n_paths, steps + 1))
    x[:, 0] = x0
    for k in range(steps):
        t = k * h
        xk = x[:, k]
        x[:, k + 1] = xk + drift(xk, t) * h + diffusion(xk, t) * dW[:, k]
    return x


def milstein(x0, drift, diffusion, diffusion_dx, T, dW):
    dW = np.asarray(dW, dtype=float)
    n_paths, steps = dW.shape
    h = T / steps
    x = np.empty((n_paths, steps + 1))
    x[:, 0] = x0
    for k in range(steps):
        t = k * h
        xk = x[:, k]
        b = diffusion(xk, t)
        x[:, k + 1] = (xk + drift(xk, t) * h + b * dW[:, k]
                       + 0.5 * b * diffusion_dx(xk, t) * (dW[:, k] ** 2 - h))
    return x


def ito_integral_w_dw(dW):
    """Left-point sum of W dW per path."""
    dW = np.asarray(dW, dtype=float)
    W_left = np.cumsum(dW, axis=1) - dW
    return np.sum(W_left * dW, axis=1)


def stratonovich_integral_w_dw(dW):
    """Midpoint (average) sum of W dW per path."""
    dW = np.asarray(dW, dtype=float)
    W_right = np.cumsum(dW, axis=1)
    return np.sum((W_right - dW + W_right) / 2.0 * dW, axis=1)


def quadratic_variation(dW):
    return np.sum(np.asarray(dW, dtype=float) ** 2, axis=1)
