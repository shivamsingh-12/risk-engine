"""Phase 15 experiment (Experiment 20): what the portfolio app says about six example portfolios.

Runs the app's full analysis (the same code path as POST /api/analyze: data preparation, the C++ engine with all
eight models and equal-weight combination, and the analytics) on portfolios built from the bundled six stocks, so it
is reproducible offline. Run from the repo root after building qre_run:
    python python_reference/experiments/portfolio-app/example_portfolios.py
Writes results/portfolios.csv and results/risk_contributions.csv.
"""
import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from qre.data import hardening as hd            # noqa: E402
from qre.linalg import shrinkage as sh          # noqa: E402
from qre.web import data, runner, server        # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")
TICKERS = ["AAPL", "CAT", "JNJ", "JPM", "PG", "XOM"]


def long_only_min_variance(prices):
    """Ledoit-Wolf minimum-variance weights on the first 250 days, negatives set to 0 and renormalized
    (so the weights use no information from the period being evaluated)."""
    r = prices[1:251] / prices[:250] - 1.0
    cov, _ = sh.ledoit_wolf(r)
    w = np.maximum(sh.min_variance_weights(cov), 0.0)
    return w / w.sum()


def main():
    if runner.find_qre_run() is None:
        sys.exit("build qre_run first (cmake --build build)")
    table = hd.load_price_csv(data.BUNDLED)
    mv = long_only_min_variance(table.prices)
    portfolios = {
        "Equal weight, 6 stocks": dict(zip(TICKERS, [1] * 6)),
        "Single stock (AAPL)": {"AAPL": 1},
        "Defensive (PG, JNJ)": {"PG": 1, "JNJ": 1},
        "Cyclical (XOM, CAT)": {"XOM": 1, "CAT": 1},
        "Tech and banks (AAPL, JPM)": {"AAPL": 1, "JPM": 1},
        "Min-variance (long-only, first year)": {t: float(w) for t, w in zip(TICKERS, mv) if w > 1e-6},
    }
    app = server.App(tempfile.mkdtemp(prefix="qre_exp20_"))
    rows, contrib = [], []
    for name, hold in portfolios.items():
        r = app.analyze({"holdings": [{"ticker": t, "weight": w} for t, w in hold.items()], "source": "bundled",
                         "years": 3, "value": 1_000_000, "confidence": [0.99, 0.975], "models": "all"})
        a, nd = r["analytics"], r["risk"]["next_day"]
        comb = {x["confidence"]: x for x in nd if x["model"] == "combined"}
        hist = {x["confidence"]: x for x in nd if x["model"] == "historical"}
        bt = {x["confidence"]: x for x in r["risk"]["backtest"] if x["model"] == "combined"}
        top = max(a["assets"], key=lambda x: x["risk_contribution"])
        rows.append({
            "portfolio": name, "holdings": len(hold),
            "vol_full": a["volatility"]["realized_full"], "vol_250": a["volatility"]["realized_250"],
            "garch_next_day_vol": a["volatility"]["garch_next_day"], "garch_persistence": a["volatility"]["garch_persistence"],
            "var99_combined": comb[0.99]["var"], "es99_combined": comb[0.99]["es"],
            "var99_historical": hist[0.99]["var"], "var975_combined": comb[0.975]["var"],
            "max_drawdown": a["returns"]["max_drawdown"], "annualized_return": a["returns"]["annualized_return"],
            "diversification_ratio": a["diversification_ratio"],
            "top_risk_asset": top["ticker"], "top_risk_weight": top["weight"], "top_risk_share": top["risk_contribution"],
            "breaches99": bt[0.99]["breaches"], "expected99": bt[0.99]["expected"], "kupiec99": bt[0.99]["kupiec_p"],
            "christoffersen99": bt[0.99]["christoffersen_p"],
            "breaches975": bt[0.975]["breaches"], "expected975": bt[0.975]["expected"], "kupiec975": bt[0.975]["kupiec_p"],
            "christoffersen975": bt[0.975]["christoffersen_p"],
        })
        for x in a["assets"]:
            contrib.append({"portfolio": name, "ticker": x["ticker"], "weight": x["weight"], "vol": x["vol"],
                            "risk_contribution": x["risk_contribution"]})
        print(f"{name}: 99% VaR {100 * comb[0.99]['var']:.2f}%, vol {100 * a['volatility']['realized_250']:.1f}%")
    os.makedirs(OUT, exist_ok=True)
    for fname, data_rows in (("portfolios.csv", rows), ("risk_contributions.csv", contrib)):
        with open(os.path.join(OUT, fname), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data_rows[0]))
            w.writeheader()
            for row in data_rows:
                w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
