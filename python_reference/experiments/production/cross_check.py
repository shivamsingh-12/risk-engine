"""Phase 14 experiment, cross-language part: do the C++ and Python engines produce the same report?

Run from the repo root after building:
    python python_reference/experiments/production/cross_check.py --qre-run build/apps/qre_run --workers 2

1. Runs configs/daily_risk.conf through both engines (C++ qre_run and python -m qre.engine.run) and times them.
2. Compares every forecast series per confidence and model: largest relative difference, days differing by more than
   1e-6, and whether the backtest verdicts (breaches, traffic light) are the same.
3. GARCH(1,1) optimizer check: refits each backtest day's portfolio returns in Python and compares with the C++ fits
   written by production_experiment (cpp_garch_fits.csv): where the two optimizers disagree, which found the higher
   likelihood, and by how much.
Writes cross_language.csv, verdicts.csv, garch_optima.csv (deterministic) and runtime.csv (machine dependent) to
cpp/experiments/production/results/.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from qre.engine import config as cf          # noqa: E402
from qre.engine import engine as en          # noqa: E402
from qre.engine import report as rp          # noqa: E402
from qre.data import hardening as hd         # noqa: E402
from qre.risk import garch                   # noqa: E402
from qre.risk import var_es as ve            # noqa: E402

OUT = "cpp/experiments/production/results"


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qre-run", default="build/apps/qre_run")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = cf.load_config("configs/daily_risk.conf")
    cpp_dir, py_dir = "runs/xcheck_cpp", "runs/xcheck_py"

    t0 = time.perf_counter()
    subprocess.run([a.qre_run, "--config", "configs/daily_risk.conf", "--output", cpp_dir], check=True,
                   stdout=subprocess.DEVNULL)
    t_cpp = time.perf_counter() - t0
    t0 = time.perf_counter()
    cfg.output = py_dir
    r = en.run_engine(cfg, a.workers)
    rp.write_report(r, py_dir)
    t_py = time.perf_counter() - t0

    with open(os.path.join(OUT, "cross_language.csv"), "w") as out:
        out.write("confidence,series,max_rel_diff_var,max_rel_diff_es,days_over_1e-6,worst_date\n")
        for c in cfg.confidence:
            tag = "%g" % (100 * c)
            fa, fb = read(os.path.join(cpp_dir, f"forecasts_{tag}.csv")), read(os.path.join(py_dir, f"forecasts_{tag}.csv"))
            for m in cfg.models + ["combined"]:
                rows = [(x, y) for x, y in zip(fa, fb) if x[f"{m}_var"] != ""]
                dv = np.array([abs(float(x[f"{m}_var"]) / float(y[f"{m}_var"]) - 1) for x, y in rows])
                de = np.array([abs(float(x[f"{m}_es"]) / float(y[f"{m}_es"]) - 1) for x, y in rows])
                d = np.maximum(dv, de)
                out.write(f"{c:g},{m},{dv.max():.3e},{de.max():.3e},{int((d > 1e-6).sum())},{rows[int(d.argmax())][0]['date']}\n")

    sa, sb = read(os.path.join(cpp_dir, "backtest_summary.csv")), read(os.path.join(py_dir, "backtest_summary.csv"))
    na, nb = read(os.path.join(cpp_dir, "next_day.csv")), read(os.path.join(py_dir, "next_day.csv"))
    with open(os.path.join(OUT, "verdicts.csv"), "w") as out:
        out.write("confidence,model,breaches_cpp,breaches_py,traffic_cpp,traffic_py,kupiec_p_rel_diff,"
                  "mean_fz0_rel_diff,next_var_cpp,next_var_py\n")
        for x, y, u, v in zip(sa, sb, na, nb):
            out.write(f"{x['confidence']},{x['model']},{x['breaches']},{y['breaches']},{x['traffic_light']},"
                      f"{y['traffic_light']},{abs(float(x['kupiec_p']) / float(y['kupiec_p']) - 1):.2e},"
                      f"{abs(float(x['mean_fz0']) / float(y['mean_fz0']) - 1):.2e},{float(u['var']):.6g},{float(v['var']):.6g}\n")

    # GARCH optimizer check
    fits = read(os.path.join(OUT, "cpp_garch_fits.csv"))
    tab = hd.load_price_csv(cfg.data)
    prices, _ = hd.harden(tab.prices)
    w = en.resolve_weights(cfg, list(tab.tickers))
    with open(os.path.join(OUT, "garch_optima.csv"), "w") as out:
        out.write("date,alpha_cpp,beta_cpp,alpha_py,beta_py,loglik_cpp,loglik_py,loglik_py_minus_cpp\n")
        n_same, diffs = 0, []
        for f in fits:
            t = len(tab.dates) if f["date"] == "next" else tab.dates.index(f["date"])
            win = prices[t - 1 - cfg.window:t]
            p = ve.portfolio_returns(win[1:] / win[:-1] - 1.0, w)
            g = garch.fit_garch(p)
            dll = g.loglik - float(f["loglik"])
            if abs(g.alpha - float(f["alpha"])) < 1e-4 and abs(g.beta - float(f["beta"])) < 1e-4:
                n_same += 1
                diffs.append(abs(dll))
                continue
            out.write(f"{f['date']},{float(f['alpha']):.6f},{float(f['beta']):.6f},{g.alpha:.6f},{g.beta:.6f},"
                      f"{float(f['loglik']):.6f},{g.loglik:.6f},{dll:.3e}\n")
        out.write(f"#fits,{len(fits)},same_optimum_within_1e-4,{n_same},max_abs_loglik_diff_when_same,{max(diffs):.3e}\n")

    with open(os.path.join(OUT, "runtime.csv"), "w") as out:
        out.write("engine,parallelism,wall_s\n")
        out.write(f"cpp,all hardware threads ({os.cpu_count()}),{t_cpp:.3f}\n")
        out.write(f"python,{a.workers} worker process(es),{t_py:.3f}\n")
    print(f"C++ {t_cpp:.1f} s, Python {t_py:.1f} s; results in {OUT}")


if __name__ == "__main__":
    main()
