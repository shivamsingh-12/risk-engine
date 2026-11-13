"""Phase 14 reference: the run configuration shared with the C++ engine (cpp/include/qre/engine/config.hpp).

Plain "key = value" lines, '#' starts a comment; unknown keys, repeated keys and malformed values are errors.
Keys: data, weights (equal | TICKER=w, ...), confidence (list), window, models (list), combination (equal | select |
exp), lookback, eta, burn, lambda, mc_paths, seed, output. The thread count is not part of the config, because it
must not change the results.
"""
import math
from dataclasses import dataclass, field

from qre.risk import backtest as bt

MODELS = ("historical", "parametric", "monte_carlo", "ewma", "fhs", "garch", "fhs_garch", "garch_t", "garch_evt")
COMBINATIONS = {"equal": "equal", "select": "select", "exp": "exp"}


@dataclass
class RunConfig:
    data: str = ""
    weights: object = "equal"                 # "equal" or list of (ticker, weight)
    confidence: list = field(default_factory=lambda: [0.99])
    window: int = 250
    models: list = field(default_factory=lambda: ["historical", "parametric", "ewma", "fhs", "garch", "fhs_garch",
                                                  "garch_t", "garch_evt"])
    combination: str = "equal"
    lookback: int = 100
    eta: float = 1.0
    burn: int = 100
    lam: float = 0.94
    mc_paths: int = 20000
    seed: int = 42
    output: str = "runs/latest"


def _num(key, v):
    try:
        x = float(v)
    except ValueError:
        x = math.nan
    if not math.isfinite(x):
        raise ValueError(f"'{key}' needs a number, got '{v}'")
    return x


def _int(key, v):
    x = _num(key, v)
    if x != math.floor(x):
        raise ValueError(f"'{key}' needs an integer, got '{v}'")
    return int(x)


def _items(v):
    items = [s.strip() for s in v.split(",")]
    if any(not s for s in items):
        raise ValueError(f"empty item in list '{v}'")
    return items


def parse_config(text):
    c = RunConfig()
    seen = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"line {lineno}: expected key = value")
        key, v = (s.strip() for s in line.split("=", 1))
        if not v:
            raise ValueError(f"line {lineno}: '{key}' has no value")
        if key in seen:
            raise ValueError(f"line {lineno}: '{key}' given twice")
        seen.add(key)
        try:
            if key == "data":
                c.data = v
            elif key == "output":
                c.output = v
            elif key == "weights":
                if v == "equal":
                    c.weights = "equal"
                else:
                    pairs = []
                    for item in _items(v):
                        if "=" not in item:
                            raise ValueError("weights must be 'equal' or TICKER=w pairs")
                        t, w = item.split("=", 1)
                        pairs.append((t.strip(), _num(key, w.strip())))
                    c.weights = pairs
            elif key == "confidence":
                c.confidence = [_num(key, s) for s in _items(v)]
                if any(not (0.5 < x < 1.0) for x in c.confidence):
                    raise ValueError("confidence must be in (0.5, 1)")
            elif key == "window":
                c.window = _int(key, v)
            elif key == "models":
                c.models = _items(v)
                for m in c.models:
                    if m not in MODELS:
                        raise ValueError(f"unknown model '{m}'")
            elif key == "combination":
                if v not in COMBINATIONS:
                    raise ValueError("combination must be equal, select or exp")
                c.combination = v
            elif key == "lookback":
                c.lookback = _int(key, v)
            elif key == "eta":
                c.eta = _num(key, v)
            elif key == "burn":
                c.burn = _int(key, v)
            elif key == "lambda":
                c.lam = _num(key, v)
            elif key == "mc_paths":
                c.mc_paths = _int(key, v)
            elif key == "seed":
                c.seed = _int(key, v)
            else:
                raise ValueError(f"unknown key '{key}'")
        except ValueError as e:
            msg = str(e)
            raise ValueError(msg if msg.startswith("line ") else f"line {lineno}: {msg}") from None
    if not c.data:
        raise ValueError("config needs 'data'")
    if c.window < 2 or c.lookback < 1 or c.burn < 1 or c.eta < 0 or c.mc_paths < 100 or c.seed < 0 or not (0 < c.lam < 1):
        raise ValueError("need window >= 2, lookback >= 1, burn >= 1, eta >= 0, mc_paths >= 100, seed >= 0, 0 < lambda < 1")
    if not c.models:
        raise ValueError("need at least one model")
    if len(set(c.models)) != len(c.models):
        raise ValueError("a model is listed twice")
    return c


def load_config(path):
    with open(path) as fh:
        return parse_config(fh.read())


def _g(x):
    return "%.15g" % x


def canonical(c):
    w = "equal" if c.weights == "equal" else ", ".join(f"{t}={_g(x)}" for t, x in c.weights)
    return (f"data = {c.data}\nweights = {w}\nconfidence = {', '.join(_g(x) for x in c.confidence)}\n"
            f"window = {c.window}\nmodels = {', '.join(c.models)}\ncombination = {c.combination}\n"
            f"lookback = {c.lookback}\neta = {_g(c.eta)}\nburn = {c.burn}\nlambda = {_g(c.lam)}\n"
            f"mc_paths = {c.mc_paths}\nseed = {c.seed}\noutput = {c.output}\n")
