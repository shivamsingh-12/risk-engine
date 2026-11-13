"""python -m qre.engine.run --config FILE [--workers N] [--output DIR]   (Phase 14, mirrors cpp/apps/qre_run.cpp)"""
import argparse
import sys

from qre.engine.config import load_config
from qre.engine.engine import run_engine
from qre.engine.report import write_report


def main(argv=None):
    p = argparse.ArgumentParser(prog="qre.engine.run")
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=1, help="processes; results do not depend on it")
    p.add_argument("--output")
    a = p.parse_args(argv)
    try:
        cfg = load_config(a.config)
        if a.output:
            cfg.output = a.output
        r = run_engine(cfg, max(1, a.workers))
        write_report(r, cfg.output)
    except (ValueError, OSError) as e:
        print(f"qre.engine.run: {e}", file=sys.stderr)
        return 1
    for cr in r.results:
        print(f"  next-day {100 * cr.confidence:g}% VaR {100 * cr.next_combined[0]:.4f}%, "
              f"ES {100 * cr.next_combined[1]:.4f}% (combined)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
