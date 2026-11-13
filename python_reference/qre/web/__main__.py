"""python -m qre.web [--port 8000] [--open] [--qre-run build/apps/qre_run] [--cache-dir runs/cache]"""
import argparse

from qre.web.server import serve


def main():
    p = argparse.ArgumentParser(prog="qre.web", description="Interactive portfolio risk app (Phase 15)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--qre-run", help="path to the qre_run executable (default: build/apps/qre_run)")
    p.add_argument("--cache-dir", help="price cache directory (default: runs/cache)")
    p.add_argument("--open", action="store_true", help="open the app in the browser")
    a = p.parse_args()
    serve(a.host, a.port, a.cache_dir, a.qre_run, a.open)


if __name__ == "__main__":
    main()
