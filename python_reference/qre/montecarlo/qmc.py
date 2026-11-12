"""Phase 12 reference: quasi-Monte Carlo with Sobol sequences, and the Brownian bridge.

Monte Carlo error falls like N^(-1/2) because pseudo-random points clump and leave gaps by chance.
A low-discrepancy sequence fills [0,1)^d evenly by construction; for smooth integrands the error of
quasi-Monte Carlo (QMC) is close to O((log N)^d / N), so close to N^(-1) in practice when the effective
dimension is low.

Sobol sequence (Joe and Kuo 2008 direction numbers, the same table scipy.stats.qmc.Sobol uses).
For each dimension, a primitive polynomial over GF(2) of degree s with coefficients a_1..a_{s-1}, and
initial odd integers m_1..m_s (m_k < 2^k), define
    m_k = 2^s m_{k-s}  XOR  m_{k-s}  XOR  sum_{i=1}^{s-1} 2^i a_i m_{k-i}       (k > s, XOR-sum)
    V_k = m_k * 2^(32-k)                                                         (direction numbers)
and with Gray code g(i) = i XOR (i >> 1) the i-th point, as a 32-bit integer, is
    x_i = XOR over the set bits k of g(i) of V_{k+1},      u_i = x_i / 2^32.
The first dimension is the van der Corput sequence (all m_k = 1). The table below covers 64 dimensions.

Randomized QMC: XOR every point with one random 32-bit integer per dimension (a random digital shift).
Each shifted point is uniform on [0,1)^d, so the estimator stays unbiased, the evenness is kept, and
independent shifts give an honest standard error. Points are taken at the cell midpoint (x + 0.5) / 2^32
so that no coordinate is exactly 0 before the inverse normal cdf.

Brownian bridge. QMC is much better in its first few coordinates than in later ones, so the path
construction should put the most important randomness first. The incremental construction
W(t_k) = W(t_{k-1}) + sqrt(dt) Z_k spreads importance evenly over all coordinates. The bridge uses
Z_1 for W(T), Z_2 for the midpoint given both ends, then the quarter points, and so on:
    W(t_l) | W(t_j), W(t_k) ~ N( ((t_k - t_l) W(t_j) + (t_l - t_j) W(t_k)) / (t_k - t_j),
                                (t_l - t_j)(t_k - t_l) / (t_k - t_j) )
Both constructions give exactly the same distribution of paths; only which coordinate drives what changes.
(Construction order follows Jaeckel, "Monte Carlo Methods in Finance", 2002.)
"""
import math

import numpy as np
from scipy.special import ndtri


MAX_BITS = 32

# (primitive polynomial with leading and trailing bits, initial m_1..m_s) for dimensions 2..64
DIRECTION_TABLE = (
    (3, (1,)),
    (7, (1, 3)),
    (11, (1, 3, 1)),
    (13, (1, 1, 1)),
    (19, (1, 1, 3, 3)),
    (25, (1, 3, 5, 13)),
    (37, (1, 1, 5, 5, 17)),
    (41, (1, 1, 5, 5, 5)),
    (47, (1, 1, 7, 11, 19)),
    (55, (1, 1, 5, 1, 1)),
    (59, (1, 1, 1, 3, 11)),
    (61, (1, 3, 5, 5, 31)),
    (67, (1, 3, 3, 9, 7, 49)),
    (91, (1, 1, 1, 15, 21, 21)),
    (97, (1, 3, 1, 13, 27, 49)),
    (103, (1, 1, 1, 15, 7, 5)),
    (109, (1, 3, 1, 15, 13, 25)),
    (115, (1, 1, 5, 5, 19, 61)),
    (131, (1, 3, 7, 11, 23, 15, 103)),
    (137, (1, 3, 7, 13, 13, 15, 69)),
    (143, (1, 1, 3, 13, 7, 35, 63)),
    (145, (1, 3, 5, 9, 1, 25, 53)),
    (157, (1, 3, 1, 13, 9, 35, 107)),
    (167, (1, 3, 1, 5, 27, 61, 31)),
    (171, (1, 1, 5, 11, 19, 41, 61)),
    (185, (1, 3, 5, 3, 3, 13, 69)),
    (191, (1, 1, 7, 13, 1, 19, 1)),
    (193, (1, 3, 7, 5, 13, 19, 59)),
    (203, (1, 1, 3, 9, 25, 29, 41)),
    (211, (1, 3, 5, 13, 23, 1, 55)),
    (213, (1, 3, 7, 3, 13, 59, 17)),
    (229, (1, 3, 1, 3, 5, 53, 69)),
    (239, (1, 1, 5, 5, 23, 33, 13)),
    (241, (1, 1, 7, 7, 1, 61, 123)),
    (247, (1, 1, 7, 9, 13, 61, 49)),
    (253, (1, 3, 3, 5, 3, 55, 33)),
    (285, (1, 3, 1, 15, 31, 13, 49, 245)),
    (299, (1, 3, 5, 15, 31, 59, 63, 97)),
    (301, (1, 3, 1, 11, 11, 11, 77, 249)),
    (333, (1, 3, 1, 11, 27, 43, 71, 9)),
    (351, (1, 1, 7, 15, 21, 11, 81, 45)),
    (355, (1, 3, 7, 3, 25, 31, 65, 79)),
    (357, (1, 3, 1, 1, 19, 11, 3, 205)),
    (361, (1, 1, 5, 9, 19, 21, 29, 157)),
    (369, (1, 3, 7, 11, 1, 33, 89, 185)),
    (391, (1, 3, 3, 3, 15, 9, 79, 71)),
    (397, (1, 3, 7, 11, 15, 39, 119, 27)),
    (425, (1, 1, 3, 1, 11, 31, 97, 225)),
    (451, (1, 1, 1, 3, 23, 43, 57, 177)),
    (463, (1, 3, 7, 7, 17, 17, 37, 71)),
    (487, (1, 3, 1, 5, 27, 63, 123, 213)),
    (501, (1, 1, 3, 5, 11, 43, 53, 133)),
    (529, (1, 3, 5, 5, 29, 17, 47, 173, 479)),
    (539, (1, 3, 3, 11, 3, 1, 109, 9, 69)),
    (545, (1, 1, 1, 5, 17, 39, 23, 5, 343)),
    (557, (1, 3, 1, 5, 25, 15, 31, 103, 499)),
    (563, (1, 1, 1, 11, 11, 17, 63, 105, 183)),
    (601, (1, 1, 5, 11, 9, 29, 97, 231, 363)),
    (607, (1, 1, 5, 15, 19, 45, 41, 7, 383)),
    (617, (1, 3, 7, 7, 31, 19, 83, 137, 221)),
    (623, (1, 1, 1, 3, 23, 15, 111, 223, 83)),
    (631, (1, 1, 5, 13, 31, 15, 55, 25, 161)),
    (637, (1, 1, 3, 13, 25, 47, 39, 87, 257))
)
MAX_DIM = len(DIRECTION_TABLE) + 1


def direction_numbers(dim):
    """(dim, 32) array of uint64 direction numbers V_k, k = 1..32."""
    if not (1 <= dim <= MAX_DIM):
        raise ValueError(f"dim must be in 1..{MAX_DIM}")
    v = np.zeros((dim, MAX_BITS), dtype=np.uint64)
    v[0] = [1 << (MAX_BITS - k) for k in range(1, MAX_BITS + 1)]
    for j in range(1, dim):
        poly, m_init = DIRECTION_TABLE[j - 1]
        s = poly.bit_length() - 1
        a = (poly >> 1) & ((1 << (s - 1)) - 1)
        m = list(m_init)
        for k in range(s, MAX_BITS):                         # 0-based: computing m_{k+1}
            new = m[k - s] ^ (m[k - s] << s)
            for i in range(1, s):
                if (a >> (s - 1 - i)) & 1:
                    new ^= m[k - i] << i
            m.append(new)
        v[j] = [m[k] << (MAX_BITS - 1 - k) for k in range(MAX_BITS)]
    return v


def sobol_ints(n, dim):
    """First n Sobol points (including the origin) as 32-bit integers, shape (n, dim)."""
    if n < 1 or n > 2 ** MAX_BITS:
        raise ValueError("need 1 <= n <= 2^32")
    v = direction_numbers(dim)
    i = np.arange(n, dtype=np.uint64)
    gray = i ^ (i >> np.uint64(1))
    x = np.zeros((n, dim), dtype=np.uint64)
    for k in range(MAX_BITS):
        bit = ((gray >> np.uint64(k)) & np.uint64(1)).astype(bool)
        x[bit] ^= v[:, k]
    return x


def sobol_points(n, dim):
    """Unscrambled Sobol points in [0, 1), identical to scipy.stats.qmc.Sobol(dim, scramble=False)."""
    return sobol_ints(n, dim).astype(float) / 2.0 ** MAX_BITS


def random_shift(dim, seed=42):
    return np.random.default_rng(seed).integers(0, 2 ** MAX_BITS, size=dim, dtype=np.uint64)


def sobol_normals(n, dim, shift=None):
    """Standard normals from (digitally shifted) Sobol points at cell midpoints, shape (n, dim)."""
    x = sobol_ints(n, dim)
    if shift is not None:
        shift = np.asarray(shift, dtype=np.uint64)
        if shift.shape != (dim,):
            raise ValueError("shift must have one entry per dimension")
        x = x ^ shift
    return ndtri((x.astype(float) + 0.5) / 2.0 ** MAX_BITS)


def incremental_path(z, time):
    """W at t_k = k T / m from normals z (n, m) by summing increments."""
    z = np.atleast_2d(z)
    return np.cumsum(z, axis=1) * math.sqrt(time / z.shape[1])


def _bridge_plan(m, time):
    t = time * np.arange(1, m + 1) / m
    populated = np.zeros(m, dtype=bool)
    populated[m - 1] = True
    plan = [(m - 1, -1, -1, 0.0, 0.0, math.sqrt(t[m - 1]))]       # (target, left, right, wl, wr, sd)
    j = 0
    for _ in range(1, m):
        while populated[j]:
            j += 1
        k = j
        while not populated[k]:
            k += 1
        l = j + ((k - 1 - j) >> 1)
        populated[l] = True
        t_left = t[j - 1] if j > 0 else 0.0
        span = t[k] - t_left
        plan.append((l, j - 1, k, (t[k] - t[l]) / span, (t[l] - t_left) / span,
                     math.sqrt((t[l] - t_left) * (t[k] - t[l]) / span)))
        j = k + 1
        if j >= m:
            j = 0
    return plan


def brownian_bridge(z, time):
    """W at t_k = k T / m from normals z (n, m): column 0 sets W(T), column 1 the midpoint, and so on."""
    z = np.atleast_2d(z)
    n, m = z.shape
    w = np.empty((n, m))
    for i, (l, left, right, wl, wr, sd) in enumerate(_bridge_plan(m, time)):
        w[:, l] = sd * z[:, i]
        if right >= 0:
            w[:, l] += wr * w[:, right]
        if left >= 0:
            w[:, l] += wl * w[:, left]
    return w
