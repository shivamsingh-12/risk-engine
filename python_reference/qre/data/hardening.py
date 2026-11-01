"""Data-layer hardening reference: loading, corporate actions, quality checks, point-in-time windows.

Conventions (match the C++ implementation in cpp/include/qre/data/):
    - prices is a (T x N) array, rows = trading days in increasing date order, cols = assets.
      Missing values are NaN.
    - A corporate action's `row` is its ex-date: the first row that trades on the post-action basis.
    - Split value = shares out per share in (2.0 for a 2-for-1 split, 0.5 for a 1-for-2 reverse split).
      Dividend value = cash per share.
    - Back-adjustment multiplies every row BEFORE the ex-date, so the latest prices stay as traded.
      Split factor: 1 / ratio. Dividend factor: 1 - D / P(ex-1), using the raw close before the ex-date.
    - Point in time: the forecast for the return of row t (P_t / P_{t-1} - 1) may only use rows <= t-1.

Philosophy: detection is automatic, correction of anything ambiguous is not. A -50% day looks
exactly like an unadjusted 2-for-1 split, so split candidates are flagged for a human to confirm.
"""
import math
import re
from dataclasses import dataclass, field

import numpy as np

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SPLIT_RATIOS = (10.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1 / 1.5, 0.5, 1 / 3, 0.25, 0.2, 0.1)


# --------------------------------------------------------------------------- loading
@dataclass
class PriceTable:
    dates: list
    tickers: list
    prices: np.ndarray


def load_price_csv(path):
    """Header 'Date,T1,T2,...'; one row per date. Empty cells become NaN; malformed input raises."""
    with open(path) as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    if not lines:
        raise ValueError("empty file")
    header = lines[0].split(",")
    tickers = header[1:]
    if not tickers or any(t == "" for t in tickers) or len(set(tickers)) != len(tickers):
        raise ValueError("header must list unique, non-empty tickers")
    dates, rows = [], []
    for lineno, line in enumerate(lines[1:], start=2):
        if line == "":
            continue
        cells = line.split(",")
        if len(cells) != len(header):
            raise ValueError(f"line {lineno}: expected {len(header)} cells, got {len(cells)}")
        if not _DATE.match(cells[0]):
            raise ValueError(f"line {lineno}: date must be YYYY-MM-DD")
        if dates and cells[0] <= dates[-1]:
            raise ValueError(f"line {lineno}: dates must be strictly increasing")
        try:
            rows.append([math.nan if c == "" else float(c) for c in cells[1:]])
        except ValueError as exc:
            raise ValueError(f"line {lineno}: non-numeric price") from exc
        dates.append(cells[0])
    if not rows:
        raise ValueError("no data rows")
    return PriceTable(dates, tickers, np.array(rows, dtype=float))


# --------------------------------------------------------------------------- corporate actions
@dataclass(frozen=True)
class CorporateAction:
    asset: int
    row: int
    kind: str      # "split" or "dividend"
    value: float


def _action_factor(prices, a):
    t, n = prices.shape
    if not (0 <= a.asset < n) or not (1 <= a.row < t):
        raise ValueError("corporate action asset/row out of range")
    if a.kind == "split":
        if a.value <= 0.0:
            raise ValueError("split ratio must be positive")
        return 1.0 / a.value
    if a.kind == "dividend":
        prev = prices[a.row - 1, a.asset]
        if not np.isfinite(prev) or a.value <= 0.0 or a.value >= prev:
            raise ValueError("dividend must be positive and below the prior close")
        return 1.0 - a.value / prev
    raise ValueError("kind must be 'split' or 'dividend'")


def back_adjust(prices, actions):
    """Total-return back-adjustment. Returns from the result are split-free and include dividends."""
    prices = np.asarray(prices, dtype=float)
    factor = np.ones_like(prices)
    for a in actions:
        f = _action_factor(prices, a)      # validate before indexing with a.asset / a.row
        factor[: a.row, a.asset] *= f
    return prices * factor


def adjust_as_of(prices, actions, t):
    """Point-in-time adjustment: only actions whose ex-date is at or before row t-1 are applied."""
    return back_adjust(prices, [a for a in actions if a.row <= t - 1])


@dataclass(frozen=True)
class SplitCandidate:
    asset: int
    row: int
    ratio: float


def detect_split_candidates(prices, min_move=0.25, tol=0.03):
    """Days whose price ratio P(t-1)/P(t) is within tol of a common split ratio and moved >= min_move."""
    prices = np.asarray(prices, dtype=float)
    out = []
    for c in range(prices.shape[1]):
        for r in range(1, prices.shape[0]):
            p0, p1 = prices[r - 1, c], prices[r, c]
            if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0.0 or p1 <= 0.0:
                continue
            if abs(p1 / p0 - 1.0) < min_move:
                continue
            ratio = p0 / p1
            for s in _SPLIT_RATIOS:
                if abs(ratio / s - 1.0) <= tol:
                    out.append(SplitCandidate(c, r, s))
                    break
    return out


# --------------------------------------------------------------------------- quality checks
@dataclass(frozen=True)
class StaleRun:
    asset: int
    start: int     # first row that repeats the previous price
    length: int    # number of consecutive repeated rows


def stale_runs(prices, min_run=5):
    prices = np.asarray(prices, dtype=float)
    out = []
    for c in range(prices.shape[1]):
        start, length = -1, 0
        for r in range(1, prices.shape[0] + 1):
            same = (r < prices.shape[0] and np.isfinite(prices[r, c]) and np.isfinite(prices[r - 1, c])
                    and prices[r, c] == prices[r - 1, c])
            if same:
                if length == 0:
                    start = r
                length += 1
            else:
                if length >= min_run:
                    out.append(StaleRun(c, start, length))
                length = 0
    return out


def bad_ticks(prices, threshold=0.15):
    """Spike-and-revert: a big log move that reverses the next day back to near the prior level."""
    prices = np.asarray(prices, dtype=float)
    out = []
    for c in range(prices.shape[1]):
        for r in range(1, prices.shape[0] - 1):
            p0, p1, p2 = prices[r - 1, c], prices[r, c], prices[r + 1, c]
            if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(p2)) or min(p0, p1, p2) <= 0.0:
                continue
            a, b, back = math.log(p1 / p0), math.log(p2 / p1), math.log(p2 / p0)
            if abs(a) > threshold and abs(b) > threshold and a * b < 0.0 and abs(back) < threshold / 2.0:
                out.append((c, r))
    return out


def lifetimes(prices):
    """(first_valid_row, last_valid_row) per asset, or None if the asset has no data."""
    prices = np.asarray(prices, dtype=float)
    out = []
    for c in range(prices.shape[1]):
        idx = np.flatnonzero(np.isfinite(prices[:, c]))
        out.append(None if idx.size == 0 else (int(idx[0]), int(idx[-1])))
    return out


def forward_fill_within_life(prices, max_consec_days):
    """Fill gaps of at most max_consec_days, only between an asset's first and last valid price.
    Never fills before listing or after delisting, so dead stocks do not get fabricated flat prices."""
    out = np.array(prices, dtype=float, copy=True)
    for c, life in enumerate(lifetimes(out)):
        if life is None:
            continue
        first, last = life
        r = first + 1
        while r <= last:
            if np.isnan(out[r, c]):
                end = r
                while np.isnan(out[end, c]):
                    end += 1
                if end - r <= max_consec_days:
                    out[r:end, c] = out[r - 1, c]
                r = end
            else:
                r += 1
    return out


@dataclass
class QualityReport:
    non_positive: list = field(default_factory=list)       # (asset, row)
    leading_missing: list = field(default_factory=list)    # per asset: rows before listing
    trailing_missing: list = field(default_factory=list)   # per asset: rows after delisting
    internal_missing: list = field(default_factory=list)   # per asset: gaps inside the listed life (raw)
    remaining_internal_missing: list = field(default_factory=list)  # after hardening
    stale: list = field(default_factory=list)
    bad_ticks: list = field(default_factory=list)
    split_candidates: list = field(default_factory=list)


def _missing_counts(prices):
    lead, trail, internal = [], [], []
    for c, life in enumerate(lifetimes(prices)):
        if life is None:
            lead.append(prices.shape[0]); trail.append(0); internal.append(0)
            continue
        first, last = life
        lead.append(first)
        trail.append(prices.shape[0] - 1 - last)
        internal.append(int(np.isnan(prices[first:last + 1, c]).sum()))
    return lead, trail, internal


def harden(prices, max_fill_days=5, min_stale_run=5, bad_tick_threshold=0.15,
           split_min_move=0.25, split_tol=0.03):
    """Detect everything, repair only what is unambiguous. Returns (clean_prices, QualityReport)."""
    raw = np.asarray(prices, dtype=float)
    rep = QualityReport()
    rep.non_positive = [(int(c), int(r)) for r, c in zip(*np.nonzero(raw <= 0.0))]
    rep.non_positive.sort()
    clean = raw.copy()
    clean[clean <= 0.0] = np.nan
    rep.split_candidates = detect_split_candidates(clean, split_min_move, split_tol)
    rep.stale = stale_runs(clean, min_stale_run)
    rep.bad_ticks = bad_ticks(clean, bad_tick_threshold)
    for c, r in rep.bad_ticks:
        clean[r, c] = np.nan
    rep.leading_missing, rep.trailing_missing, rep.internal_missing = _missing_counts(clean)
    clean = forward_fill_within_life(clean, max_fill_days)
    rep.remaining_internal_missing = _missing_counts(clean)[2]
    return clean, rep


# --------------------------------------------------------------------------- point in time
def pit_window(prices, t, window):
    """Price rows usable to forecast the return of row t: rows [t-1-window, t-1] (window returns).
    t may equal len(prices), i.e. forecasting the next, not yet observed day."""
    prices = np.asarray(prices)
    if window < 1 or t - 1 - window < 0 or t > prices.shape[0]:
        raise ValueError("window does not fit before row t")
    return prices[t - 1 - window: t]


def universe_at(prices, t, window):
    """Assets with a complete price history in the point-in-time window: membership known at t-1."""
    w = pit_window(prices, t, window)
    return [c for c in range(w.shape[1]) if np.all(np.isfinite(w[:, c]))]
