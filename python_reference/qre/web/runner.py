"""Phase 15: run the Phase 14 engine on a portfolio and read its report back as plain data.

The C++ qre_run is used when it is built (seconds for all eight models); otherwise the Python engine runs in-process
(minutes for the GARCH family, so the app defaults to the fast models in that case). Either way the engine writes the
standard report files into a scratch directory, and this module parses them, so the app shows exactly what qre_run
would write to disk.
"""
import csv
import os
import shutil
import subprocess
import tempfile

from qre.engine import config as cf
from qre.engine import engine as en
from qre.engine import report as rp
from qre.web.data import REPO_ROOT, write_price_csv

ALL_MODELS = ["historical", "parametric", "ewma", "fhs", "garch", "fhs_garch", "garch_t", "garch_evt"]
FAST_MODELS = ["historical", "parametric", "ewma", "fhs"]
DEFAULT_QRE_RUN = os.path.join(REPO_ROOT, "build", "apps", "qre_run")


def find_qre_run(path=None):
    for p in (path, os.environ.get("QRE_RUN"), DEFAULT_QRE_RUN):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    return None if x in ("", None) else float(x)


def run_risk(dates, tickers, prices, weights, confidence=(0.99, 0.975), models=None, combination="equal",
             window=250, burn=100, qre_run=None, keep_dir=None):
    """Returns a dict with next_day, backtest and forecast series, plus which engine ran."""
    models = list(models or ALL_MODELS)
    work = keep_dir or tempfile.mkdtemp(prefix="qre_web_")
    os.makedirs(work, exist_ok=True)
    try:
        data = os.path.join(work, "prices.csv")
        write_price_csv(data, dates, tickers, prices)
        weights_txt = ", ".join(f"{t}={w!r}" for t, w in zip(tickers, weights))
        out = os.path.join(work, "report")
        text = (f"data = {data}\nweights = {weights_txt}\nconfidence = {', '.join(repr(c) for c in confidence)}\n"
                f"window = {window}\nmodels = {', '.join(models)}\ncombination = {combination}\n"
                f"burn = {burn}\noutput = {out}\n")
        conf_path = os.path.join(work, "run.conf")
        with open(conf_path, "w") as fh:
            fh.write(text)
        exe = find_qre_run(qre_run)
        if exe:
            proc = subprocess.run([exe, "--config", conf_path], capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "qre_run failed")
            engine = "C++ (qre_run)"
        else:
            cfg = cf.parse_config(text)
            rp.write_report(en.run_engine(cfg, 1), out)
            engine = "Python reference"
        return {"engine": engine, "models": models, **read_report(out, confidence)}
    finally:
        if keep_dir is None:
            shutil.rmtree(work, ignore_errors=True)


def read_report(out, confidence):
    nxt = _rows(os.path.join(out, "next_day.csv"))
    summ = _rows(os.path.join(out, "backtest_summary.csv"))
    res = {"next_day": [], "backtest": [], "forecasts": {}}
    for r in nxt:
        res["next_day"].append({"confidence": float(r["confidence"]), "model": r["model"], "var": float(r["var"]),
                                "es": float(r["es"]), "weight": float(r["weight"])})
    for r in summ:
        res["backtest"].append({"confidence": float(r["confidence"]), "model": r["model"], "n_obs": int(r["n_obs"]),
                                "breaches": int(r["breaches"]), "expected": float(r["expected"]),
                                "kupiec_p": float(r["kupiec_p"]), "christoffersen_p": float(r["christoffersen_p"]),
                                "traffic_light": r["traffic_light"], "z2": float(r["z2"]),
                                "mean_fz0": float(r["mean_fz0"])})
    for c in confidence:
        tag = "%g" % (100 * c)
        rows = _rows(os.path.join(out, f"forecasts_{tag}.csv"))
        res["forecasts"][tag] = {"dates": [r["date"] for r in rows], "loss": [float(r["loss"]) for r in rows],
                                 "combined_var": [_f(r["combined_var"]) for r in rows],
                                 "combined_es": [_f(r["combined_es"]) for r in rows]}
    return res
