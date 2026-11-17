"""Download the market data used by Phases 20 to 27 into data/.

Run from the repo root, with yfinance installed:
    python3 tools/fetch_data.py

Writes (all CSV, overwriting previous downloads):
    data/market_data/long_history_close.csv     adjusted closes, 2005 onwards, stocks + ETFs + VIX
    data/market_data/long_history_volume.csv    daily share volumes, same tickers
    data/options/<TICKER>_chain.csv             today's option chain (all expiries up to ~2.5 years)
    data/factors/ff5_mom_daily.csv              Fama-French 5 factors + momentum, daily, in percent
    data/rates/treasury_par_yields.csv          US Treasury constant-maturity yields, daily, in percent
    data/rates/credit_spreads.csv               ICE BofA US IG and HY option-adjusted spreads, in percent
    data/fundamentals/balance_sheets.csv        latest debt, liabilities, shares and market cap
    data/FETCHED.md                             what was fetched and when

Each part is independent: a failure in one is reported and the rest still run.
"""
import csv
import datetime as dt
import io
import os
import re
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
START = "2005-01-01"

STOCKS = ["AAPL", "JPM", "XOM", "JNJ", "PG", "CAT",            # the bundled six
          "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BAC", "GS", "KO", "WMT"]
ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "USO", "VNQ",
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
INDEX = ["^VIX", "^GSPC"]
CREDIT_NAMES = ["F", "AAL", "CCL", "PARA", "WBD", "M", "T", "VZ", "BA", "INTC"]   # more levered firms for Merton
OPTION_TICKERS = ["SPY", "AAPL", "QQQ"]
FRED_YIELDS = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
FRED_SPREADS = ["BAMLC0A0CM", "BAMLH0A0HYM2"]
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
UA = {"User-Agent": "Mozilla/5.0 (qre research; fetch_data.py)"}

log = []


def report(part, ok, msg):
    line = f"[{'ok' if ok else 'FAILED'}] {part}: {msg}"
    print(line, flush=True)
    log.append(line)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def ensure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def prices():
    import yfinance as yf
    tickers = STOCKS + ETFS + INDEX + CREDIT_NAMES
    df = yf.download(tickers, start=START, auto_adjust=True, progress=False, group_by="column", threads=True)
    close = df["Close"].dropna(how="all")
    vol = df["Volume"].reindex(close.index)
    close.index.name = vol.index.name = "date"
    close.round(6).to_csv(ensure(os.path.join(DATA, "market_data", "long_history_close.csv")))
    vol.to_csv(ensure(os.path.join(DATA, "market_data", "long_history_volume.csv")))
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    report("prices", True, f"{close.shape[1]} tickers, {close.index[0].date()} to {close.index[-1].date()}"
           + (f"; missing {missing}" if missing else ""))


def options():
    import yfinance as yf
    today = dt.date.today()
    for sym in OPTION_TICKERS:
        try:
            t = yf.Ticker(sym)
            spot = float(t.history(period="5d")["Close"].iloc[-1])
            rows = []
            for exp in t.options:
                if (dt.date.fromisoformat(exp) - today).days > 920:
                    continue
                oc = t.option_chain(exp)
                for kind, frame in (("call", oc.calls), ("put", oc.puts)):
                    for _, r in frame.iterrows():
                        rows.append([today.isoformat(), spot, exp, kind, r["strike"], r.get("bid"), r.get("ask"),
                                     r.get("lastPrice"), r.get("volume"), r.get("openInterest"),
                                     r.get("impliedVolatility")])
            path = ensure(os.path.join(DATA, "options", f"{sym}_chain.csv"))
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["asof", "spot", "expiry", "type", "strike", "bid", "ask", "last", "volume",
                            "open_interest", "yahoo_iv"])
                w.writerows(rows)
            report(f"options {sym}", True, f"{len(rows)} quotes, spot {spot:.2f}")
        except Exception as e:  # noqa: BLE001
            report(f"options {sym}", False, repr(e))


def _french(name):
    z = zipfile.ZipFile(io.BytesIO(get(FRENCH + name)))
    text = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    header, rows = None, {}
    for i, line in enumerate(text):
        if re.match(r"^\s*\d{8}\s*,", line):
            if header is None:
                header = [c.strip() for c in text[i - 1].split(",")][1:]
            parts = [c.strip() for c in line.split(",")]
            rows[parts[0]] = parts[1:]
        elif header is not None and rows and not line.strip():
            break
    return header, rows


def factors():
    h5, r5 = _french("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    hm, rm = _french("F-F_Momentum_Factor_daily_CSV.zip")
    path = ensure(os.path.join(DATA, "factors", "ff5_mom_daily.csv"))
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + h5 + ["MOM"])
        for d in sorted(r5):
            if d in rm and d >= START.replace("-", ""):
                w.writerow([f"{d[:4]}-{d[4:6]}-{d[6:]}"] + r5[d] + rm[d][:1])
                n += 1
    report("factors", True, f"{n} days, columns {h5 + ['MOM']}")


def _fred(ids, fname, part):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(ids) + f"&cosd={START}"
    text = get(url).decode()
    path = ensure(os.path.join(DATA, "rates", fname))
    with open(path, "w") as f:
        f.write(text)
    lines = text.strip().splitlines()
    report(part, True, f"{len(lines) - 1} days, {lines[1].split(',')[0]} to {lines[-1].split(',')[0]}")


def rates():
    _fred(FRED_YIELDS, "treasury_par_yields.csv", "treasury yields")


def spreads():
    _fred(FRED_SPREADS, "credit_spreads.csv", "credit spreads")


def fundamentals():
    import yfinance as yf
    fields = ["Total Debt", "Current Debt", "Long Term Debt", "Total Liabilities Net Minority Interest",
              "Ordinary Shares Number"]
    path = ensure(os.path.join(DATA, "fundamentals", "balance_sheets.csv"))
    ok = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "balance_sheet_date", "total_debt", "current_debt", "long_term_debt",
                    "total_liabilities", "shares", "market_cap", "price"])
        for sym in STOCKS + CREDIT_NAMES:
            try:
                t = yf.Ticker(sym)
                bs = t.quarterly_balance_sheet
                if bs is None or bs.empty:
                    bs = t.balance_sheet
                col = bs.columns[0]
                vals = [bs.at[k, col] if k in bs.index else "" for k in fields]
                fi = t.fast_info
                w.writerow([sym, str(col.date())] + vals + [fi.get("market_cap"), fi.get("last_price")])
                ok += 1
            except Exception as e:  # noqa: BLE001
                report(f"fundamentals {sym}", False, repr(e))
    report("fundamentals", ok > 0, f"{ok} companies")


def main():
    try:
        import yfinance  # noqa: F401
    except ImportError:
        sys.exit("yfinance is not installed: pip install yfinance")
    for part in (prices, options, factors, rates, spreads, fundamentals):
        try:
            part()
        except Exception as e:  # noqa: BLE001
            report(part.__name__, False, repr(e))
    with open(os.path.join(DATA, "FETCHED.md"), "w") as f:
        f.write(f"Fetched {dt.datetime.now().isoformat(timespec='seconds')} by tools/fetch_data.py\n\n")
        f.write("\n".join(log) + "\n")
    print("\nDone. Summary written to data/FETCHED.md")


if __name__ == "__main__":
    main()
