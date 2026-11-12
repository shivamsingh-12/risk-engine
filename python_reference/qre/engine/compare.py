"""python -m qre.engine.compare DIR_A DIR_B [--rtol R] [--atol A]   (Phase 14)

Compares two report directories written by qre_run (C++) and qre.engine.run (Python), cell by cell, and prints the
largest relative difference per file and column. Text cells (dates, model names, traffic lights) must match exactly.
manifest.txt must match except for the output line. Exit code 1 if anything differs by more than the tolerance:
used in CI to keep the two engines from drifting apart.
"""
import argparse
import csv
import os
import sys

REPORT_CSVS = ("next_day.csv", "backtest_summary.csv")


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def compare_csv(path_a, path_b, columns=None):
    """{column: (max relative difference, row of the worst cell or None)}; raises on structural mismatch."""
    a, b = _rows(path_a), _rows(path_b)
    if a[0] != b[0] or len(a) != len(b):
        raise ValueError(f"{os.path.basename(path_a)}: header or row count differs")
    worst = {}
    for i, (ra, rb) in enumerate(zip(a[1:], b[1:]), start=1):
        for h, x, y in zip(a[0], ra, rb):
            if columns is not None and h not in columns:
                continue
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                if x != y:
                    raise ValueError(f"{os.path.basename(path_a)} row {i} column {h}: '{x}' vs '{y}'") from None
                continue
            d = abs(fx - fy) / max(abs(fx), abs(fy), 1e-300) if fx != fy else 0.0
            if d >= worst.get(h, (-1.0, None))[0]:
                worst[h] = (d, i if d > 0 else None)
    return worst


def report_files(directory):
    return sorted(f for f in os.listdir(directory) if f.startswith("forecasts_")) + list(REPORT_CSVS)


def manifests_match(dir_a, dir_b):
    def body(d):
        with open(os.path.join(d, "manifest.txt")) as fh:
            return [ln for ln in fh if not ln.startswith("output = ")]
    return body(dir_a) == body(dir_b)


def main(argv=None):
    p = argparse.ArgumentParser(prog="qre.engine.compare")
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--rtol", type=float, default=1e-9)
    a = p.parse_args(argv)
    ok = manifests_match(a.dir_a, a.dir_b)
    print(f"manifest.txt: {'same' if ok else 'DIFFERENT'}")
    for f in report_files(a.dir_a):
        worst = compare_csv(os.path.join(a.dir_a, f), os.path.join(a.dir_b, f))
        top = max(worst.items(), key=lambda kv: kv[1][0]) if worst else ("-", (0.0, None))
        flag = top[1][0] <= a.rtol
        ok &= flag
        print(f"{f}: max relative difference {top[1][0]:.2e} ({top[0]}){'' if flag else '  <-- above tolerance'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
