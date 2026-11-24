"""Phase 22: run qre_stream for the app and feed it.

StreamManager starts one qre_stream process at a time for the portfolio on screen. It writes the holdings' recent
prices to a scratch CSV, launches qre_stream on a free port (port 0: the OS chooses; qre_stream prints the port
it got), and stops it on request or when the app exits. The browser then connects to the stream directly over
WebSocket.

Two sources:
    simulated   qre_stream's synthetic feed: a market with the holdings' own covariance, sped up so a trading day
                passes in about a minute (speed 0.02 days per second)
    live        qre_stream reads TICKER,PRICE lines on stdin; a thread here polls Yahoo Finance for the latest
                1-minute prices every `poll_s` seconds and writes them in. Only useful while the market is open.
"""
import atexit
import os
import subprocess
import tempfile
import threading
import time

from qre.web import data
from qre.web.data import REPO_ROOT, write_price_csv

DEFAULT_QRE_STREAM = os.path.join(REPO_ROOT, "build", "apps", "qre_stream")


def find_qre_stream(path=None):
    for p in (path, os.environ.get("QRE_STREAM"), DEFAULT_QRE_STREAM):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


class StreamManager:
    def __init__(self, exe=None, quotes=None, poll_s=5.0):
        self.exe = exe
        self.quotes = quotes or data.yfinance_quotes
        self.poll_s = poll_s
        self.lock = threading.Lock()
        self.proc = None
        self.info = None
        self.tmp = None
        self._feeder = None
        self._stop_feed = threading.Event()
        atexit.register(self.stop)

    def status(self):
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            return {"available": find_qre_stream(self.exe) is not None, "running": running,
                    **(self.info or {} if running else {})}

    def start(self, dates, tickers, prices, weights, value, source="simulated", rate=1000.0, speed=0.02):
        exe = find_qre_stream(self.exe)
        if exe is None:
            raise data.DataError("qre_stream is not built: run  cmake --build build  first")
        if source not in ("simulated", "live"):
            raise data.DataError("stream source must be simulated or live")
        if not (0 < rate <= 1_000_000) or not (0 < speed <= 10):
            raise data.DataError("rate must be in (0, 1,000,000] ticks per second and speed in (0, 10] days per second")
        self.stop()
        tmp = tempfile.mkdtemp(prefix="qre_stream_")
        csv_path = os.path.join(tmp, "prices.csv")
        write_price_csv(csv_path, dates, tickers, prices)
        holdings = ",".join(f"{t}={w!r}" for t, w in zip(tickers, weights))
        cmd = [exe, "--data", csv_path, "--holdings", holdings, "--weights", "--value", repr(float(value)),
               "--port", "0", "--window", "250", "--source", "synthetic" if source == "simulated" else "stdin",
               "--rate", repr(float(rate)), "--speed", repr(float(speed))]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if source == "live" else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        line = proc.stdout.readline()                       # "qre_stream: listening on 127.0.0.1:PORT (...)"
        if "listening on 127.0.0.1:" not in line:
            err = proc.stderr.read() if proc.poll() is not None else ""
            proc.kill()
            raise data.DataError(f"qre_stream did not start: {(err or line).strip()}")
        port = int(line.split("127.0.0.1:")[1].split()[0])
        # drain stdout so the process never blocks on a full pipe
        threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
        with self.lock:
            self.proc, self.tmp = proc, tmp
            self.info = {"port": port, "source": source, "tickers": list(tickers), "started": time.time(),
                         "url": f"ws://127.0.0.1:{port}/"}
        if source == "live":
            self._stop_feed.clear()
            self._feeder = threading.Thread(target=self._feed_live, args=(proc, list(tickers)), daemon=True)
            self._feeder.start()
        return self.status()

    def _feed_live(self, proc, tickers):
        while not self._stop_feed.is_set() and proc.poll() is None:
            try:
                q = self.quotes(tickers)
                lines = "".join(f"{t},{p!r}\n" for t, (p, _) in q.items() if t in tickers)
                if lines:
                    proc.stdin.write(lines)
                    proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                return
            except Exception:                               # a failed poll is retried on the next round
                pass
            self._stop_feed.wait(self.poll_s)

    def stop(self):
        self._stop_feed.set()
        with self.lock:
            proc, tmp = self.proc, self.tmp
            self.proc = self.info = self.tmp = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if tmp:
            for name in os.listdir(tmp):
                os.remove(os.path.join(tmp, name))
            os.rmdir(tmp)
        return {"running": False}
