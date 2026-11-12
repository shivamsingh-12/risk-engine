"""Phase 14 reference: writes the same files as cpp/include/qre/engine/report.hpp (same columns, same %.15g numbers)."""
import os

from qre.engine.config import canonical
from qre.engine.engine import VERSION, is_clean


def _g(x):
    return "%.15g" % x


def _tag(c):
    return "%g" % (100.0 * c)


def write_report(r, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg, k, burn = r.config, len(r.config.models), r.config.burn
    with open(os.path.join(out_dir, "backtest_summary.csv"), "w") as summ, \
            open(os.path.join(out_dir, "next_day.csv"), "w") as nxt:
        summ.write("confidence,model,n_obs,breaches,expected,breach_rate,kupiec_p,christoffersen_p,conditional_p,"
                   "traffic_light,z2,mean_fz0\n")
        nxt.write("confidence,model,var,es,weight\n")
        for cr in r.results:
            with open(os.path.join(out_dir, f"forecasts_{_tag(cr.confidence)}.csv"), "w") as fc:
                fc.write("date,loss" + "".join(f",{m}_var,{m}_es" for m in cfg.models) + ",combined_var,combined_es\n")
                for d, row in enumerate(r.rows):
                    line = f"{r.dates[row]},{_g(r.loss[d])}" + "".join(
                        f",{_g(cr.var[d, j])},{_g(cr.es[d, j])}" for j in range(k))
                    line += f",{_g(cr.combined_var[d - burn])},{_g(cr.combined_es[d - burn])}" if d >= burn else ",,"
                    fc.write(line + "\n")
            for j, s in enumerate(cr.summaries):
                name = cfg.models[j] if j < k else "combined"
                summ.write(f"{_g(cr.confidence)},{name},{s.n_obs},{s.n_breaches},{_g((1 - cr.confidence) * s.n_obs)},"
                           f"{_g(s.breach_rate)},{_g(s.kupiec_p)},{_g(s.christoffersen_p)},{_g(s.cc_p)},"
                           f"{s.traffic_light},{_g(s.z2)},{_g(cr.mean_fz0[j])}\n")
            for j in range(k):
                nxt.write(f"{_g(cr.confidence)},{cfg.models[j]},{_g(cr.next_var[j])},{_g(cr.next_es[j])},"
                          f"{_g(cr.next_weights[j])}\n")
            nxt.write(f"{_g(cr.confidence)},combined,{_g(cr.next_combined[0])},{_g(cr.next_combined[1])},1\n")
    filled = int(sum(r.quality.internal_missing))
    with open(os.path.join(out_dir, "manifest.txt"), "w") as man:
        man.write(f"qre_version = {VERSION}\n[config]\n{canonical(cfg)}[data]\nfnv1a64 = {r.data_hash:016x}\n"
                  f"bytes = {r.data_bytes}\ndates = {r.dates[0]} .. {r.dates[-1]} ({len(r.dates)} rows)\n"
                  f"tickers = {', '.join(r.tickers)}\n"
                  f"weights = {', '.join(f'{t}={_g(w)}' for t, w in zip(r.tickers, r.weights))}\n"
                  f"quality = {'clean' if is_clean(r.quality) else 'dirty'} ({filled} internal gaps filled)\n"
                  f"[run]\nbacktest_days = {r.rows.size}\nevaluation_days = {r.rows.size - burn} "
                  f"(first {burn} days only build scores)\nfirst_evaluated = {r.dates[r.rows[burn]]}\n")
    with open(os.path.join(out_dir, "run_log.txt"), "w") as log:
        log.write("".join(f"{key} = {_g(v) if isinstance(v, float) else v}\n" for key, v in r.timings.items()))
