// Phase 15: front end of the portfolio risk app. React 18 + htm (JSX-like templates without a build step).
// Talks to the local server: GET /api/status, GET /api/examples, POST /api/analyze.
const { useState, useEffect, useMemo, useRef } = React;
const html = htm.bind(React.createElement);

const pct = (x, d = 2) => (x == null || !isFinite(x) ? "–" : (100 * x).toFixed(d) + "%");
const num = (x, d = 2) => (x == null || !isFinite(x) ? "–" : x.toFixed(d));
const money = (x) => (x == null || !isFinite(x) ? "–" : x.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 }));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const MODEL_LABEL = { historical: "Historical", parametric: "Parametric (normal)", ewma: "EWMA", fhs: "Filtered historical",
  garch: "GARCH", fhs_garch: "FHS-GARCH", garch_t: "GARCH-t", garch_evt: "Conditional EVT", combined: "Combined" };

// ------------------------------------------------------------------ charts (plain SVG)
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(v);
  return out;
}

function LineChart({ dates, series, yFormat = (v) => v.toFixed(0), height = 240, width = 760, bars = null, markers = null }) {
  const W = width, H = height, L = 52, R = 12, T = 10, B = 26;
  const [hover, setHover] = useState(null);
  const ref = useRef(null);
  const n = dates.length;
  const vals = series.flatMap((s) => s.values.filter((v) => v != null && isFinite(v)));
  if (bars) vals.push(...bars.values.filter((v) => v != null && v > 0));   // bars show losses only
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (bars) lo = Math.min(lo, 0);
  const pad = (hi - lo) * 0.05 || 1; lo -= bars ? 0 : pad; hi += pad;
  const x = (i) => L + (i / Math.max(n - 1, 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const ticks = niceTicks(lo, hi);
  const xt = [0, 0.2, 0.4, 0.6, 0.8, 1].map((f) => Math.round(f * (n - 1)));
  const path = (vs) => {
    let d = "", pen = false;
    vs.forEach((v, i) => { if (v == null || !isFinite(v)) { pen = false; return; } d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); pen = true; });
    return d;
  };
  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.round(((px - L) / (W - L - R)) * (n - 1));
    setHover(i >= 0 && i < n ? i : null);
  };
  const bw = Math.max(1, (W - L - R) / n - 0.5);
  return html`<div>
    <div class="tip">${hover == null ? " " : html`<b>${dates[hover]}</b>${bars ? html` · ${bars.name} ${yFormat(bars.values[hover])}` : ""}${series.map((s) => html` · <span style=${{ color: s.color }}>${s.name}</span> ${s.values[hover] == null ? "–" : yFormat(s.values[hover])}`)}`}</div>
    <svg ref=${ref} class="chart" viewBox=${`0 0 ${W} ${H}`} onMouseMove=${onMove} onMouseLeave=${() => setHover(null)}>
      ${ticks.map((t) => html`<g key=${t}><line class="grid" x1=${L} x2=${W - R} y1=${y(t)} y2=${y(t)} /><text x=${L - 6} y=${y(t) + 4} text-anchor="end">${yFormat(t)}</text></g>`)}
      ${xt.map((i) => html`<text key=${"x" + i} x=${x(i)} y=${H - 6} text-anchor="middle">${dates[i] ? dates[i].slice(0, 7) : ""}</text>`)}
      ${bars ? bars.values.map((v, i) => v == null || v <= 0 ? null : html`<rect key=${"b" + i} x=${x(i) - bw / 2} y=${y(v)} width=${bw} height=${Math.max(0.5, y(0) - y(v))} fill=${bars.colorAt ? bars.colorAt(i) : bars.color} />`) : null}
      ${series.map((s) => html`<path key=${s.name} d=${path(s.values)} fill="none" stroke=${s.color} stroke-width=${s.width || 1.6} stroke-dasharray=${s.dash || ""} />`)}
      ${markers ? markers.map((m) => html`<circle key=${"m" + m.i} cx=${x(m.i)} cy=${y(m.v)} r="3" fill=${m.color} />`) : null}
      ${hover != null ? html`<line x1=${x(hover)} x2=${x(hover)} y1=${T} y2=${H - B} stroke=${css("--muted")} stroke-dasharray="3 3" />` : null}
    </svg>
    <div class="legend">
      ${bars ? html`<span><i style=${{ background: bars.color }}></i>${bars.name}</span>` : null}
      ${series.map((s) => html`<span key=${s.name}><i style=${{ background: s.color }}></i>${s.name}</span>`)}
      ${markers && markers.length ? html`<span><i style=${{ background: css("--bad"), height: "8px", width: "8px", borderRadius: "50%" }}></i>breach</span>` : null}
    </div>
  </div>`;
}

function ContributionBars({ assets }) {
  const max = Math.max(...assets.flatMap((a) => [a.weight, a.risk_contribution]), 0.01);
  return html`<div>
    ${assets.map((a) => html`<div key=${a.ticker} style=${{ display: "grid", gridTemplateColumns: "64px 1fr 120px", alignItems: "center", gap: "8px", margin: "6px 0" }}>
      <b>${a.ticker}</b>
      <div>
        <div style=${{ height: "8px", width: (100 * a.weight / max) + "%", background: css("--s1"), borderRadius: "3px", marginBottom: "3px" }}></div>
        <div style=${{ height: "8px", width: (100 * Math.max(a.risk_contribution, 0) / max) + "%", background: css("--s2"), borderRadius: "3px" }}></div>
      </div>
      <span style=${{ fontSize: "12px", color: css("--muted"), textAlign: "right" }}>${pct(a.weight, 1)} → ${pct(a.risk_contribution, 1)}</span>
    </div>`)}
    <div class="legend"><span><i style=${{ background: css("--s1") }}></i>share of capital</span><span><i style=${{ background: css("--s2") }}></i>share of risk (volatility)</span></div>
  </div>`;
}

function Heatmap({ tickers, corr }) {
  const n = tickers.length;
  const color = (c) => {
    const a = Math.min(Math.abs(c), 1);
    const base = c >= 0 ? css("--s1") : css("--s2");
    return `color-mix(in srgb, ${base} ${Math.round(a * 85)}%, transparent)`;
  };
  return html`<div class="scroll"><div class="heat" style=${{ gridTemplateColumns: `56px repeat(${n}, minmax(44px, 1fr))` }}>
    <div class="lab"></div>${tickers.map((t) => html`<div class="lab" key=${"h" + t}>${t}</div>`)}
    ${tickers.map((t, i) => html`<${React.Fragment} key=${"r" + t}><div class="lab" style=${{ textAlign: "left" }}>${t}</div>
      ${corr[i].map((c, j) => html`<div key=${j} style=${{ background: i === j ? css("--soft") : color(c), color: Math.abs(c) > 0.55 && i !== j ? "#fff" : css("--ink") }}>${c.toFixed(2)}</div>`)}<//>`)}
  </div></div>`;
}

// ------------------------------------------------------------------ inputs
function PortfolioForm({ status, examples, onRun, busy }) {
  const [holdings, setHoldings] = useState([{ ticker: "AAPL", weight: 1 }, { ticker: "JPM", weight: 1 }, { ticker: "XOM", weight: 1 }]);
  const [source, setSource] = useState("bundled");
  const [years, setYears] = useState(3);
  const [value, setValue] = useState(100000);
  const [conf, setConf] = useState({ 0.99: true, 0.975: true });
  const [models, setModels] = useState(null);
  const [combination, setCombination] = useState("equal");
  const total = holdings.reduce((s, h) => s + (parseFloat(h.weight) || 0), 0);
  const modelSet = models || (status ? status.default_models : "fast");

  const set = (i, k, v) => setHoldings(holdings.map((h, j) => (j === i ? { ...h, [k]: v } : h)));
  const loadExample = (name) => {
    const ex = examples.find((e) => e.name === name);
    if (ex) { setHoldings(ex.holdings.map((h) => ({ ...h }))); setSource(ex.source); }
  };
  const submit = (e) => {
    e.preventDefault();
    const confidence = Object.keys(conf).filter((k) => conf[k]).map(Number).sort((a, b) => b - a);
    onRun({ holdings: holdings.filter((h) => h.ticker.trim()).map((h) => ({ ticker: h.ticker.trim(), weight: parseFloat(h.weight) || 0 })),
      source, years: Number(years), value: Number(value), confidence, models: modelSet, combination });
  };
  return html`<form class="panel side" onSubmit=${submit}>
    <h2>Portfolio</h2>
    <label>Start from an example</label>
    <select onChange=${(e) => loadExample(e.target.value)} value="">
      <option value="">Choose a preset…</option>
      ${examples.map((e) => html`<option key=${e.name} value=${e.name}>${e.name}</option>`)}
    </select>
    <label>Holdings (weights are relative; they are scaled to sum to 100%)</label>
    <div class="holdings">
      <span class="hdr">Ticker</span><span class="hdr">Weight</span><span class="hdr" style=${{ textAlign: "right" }}>%</span><span></span>
      ${holdings.map((h, i) => html`<${React.Fragment} key=${i}>
        <input value=${h.ticker} placeholder="e.g. MSFT" onInput=${(e) => set(i, "ticker", e.target.value.toUpperCase())} aria-label="ticker" />
        <input type="number" min="0" step="any" value=${h.weight} onInput=${(e) => set(i, "weight", e.target.value)} aria-label="weight" />
        <span class="pct">${total > 0 ? ((100 * (parseFloat(h.weight) || 0)) / total).toFixed(1) : "–"}</span>
        <button type="button" class="icon" title="Remove" onClick=${() => setHoldings(holdings.filter((_, j) => j !== i))}>✕</button>
      <//>`)}
    </div>
    <button type="button" class="link" onClick=${() => setHoldings([...holdings, { ticker: "", weight: 1 }])}>+ Add holding</button>

    <h2 style=${{ marginTop: "14px" }}>Settings</h2>
    <div class="row2">
      <div><label>Data source</label>
        <select value=${source} onChange=${(e) => setSource(e.target.value)}>
          <option value="yahoo">Yahoo Finance${status && !status.yfinance ? " (needs yfinance)" : ""}</option>
          <option value="bundled">Bundled 6 stocks (offline)</option>
        </select></div>
      <div><label>History (years)</label><input type="number" min="2" max="15" value=${years} onInput=${(e) => setYears(e.target.value)} /></div>
    </div>
    <label>Portfolio value (USD)</label>
    <input type="number" min="1" step="any" value=${value} onInput=${(e) => setValue(e.target.value)} />
    <label>Confidence levels</label>
    <div class="checks">
      ${[0.99, 0.975, 0.95].map((c) => html`<label key=${c}><input type="checkbox" checked=${!!conf[c]} onChange=${(e) => setConf({ ...conf, [c]: e.target.checked })} />${(100 * c).toFixed(1).replace(".0", "")}%</label>`)}
    </div>
    <div class="row2">
      <div><label>Models</label>
        <select value=${modelSet} onChange=${(e) => setModels(e.target.value)}>
          <option value="all">All 8 (incl. GARCH family)</option><option value="fast">Fast 4 (no GARCH)</option>
        </select></div>
      <div><label>Combine models by</label>
        <select value=${combination} onChange=${(e) => setCombination(e.target.value)}>
          <option value="equal">Equal weights</option><option value="select">Best recent score</option><option value="exp">Exponential weights</option>
        </select></div>
    </div>
    <button class="run" disabled=${busy}>${busy ? html`<span class="spinner"></span>Analyzing…` : "Analyze risk"}</button>
    <p class="hint">${status ? `Engine: ${status.engine}.` : ""} ${modelSet === "all" ? "All 8 models take about 5 to 30 seconds (GARCH-t refits every day)." : "Fast models take about a second."}</p>
  </form>`;
}

// ------------------------------------------------------------------ results
function Results({ r }) {
  const levels = Object.keys(r.risk.forecasts);
  const [level, setLevel] = useState(levels[0]);
  const a = r.analytics, v = a.volatility, value = r.input.value;
  const next = r.risk.next_day;
  const top = Math.max(...r.input.confidence);
  const comb = next.find((x) => x.model === "combined" && x.confidence === top);
  const fc = r.risk.forecasts[level];
  const breaches = useMemo(() => fc.loss.map((l, i) => (fc.combined_var[i] != null && l > fc.combined_var[i] ? { i, v: 100 * l, color: css("--bad") } : null)).filter(Boolean), [fc]);
  const bt = r.risk.backtest.filter((b) => (100 * b.confidence).toString() === level || (100 * b.confidence).toFixed(1) === level);
  const tl = (t) => html`<span class=${"badge " + t}>${t}</span>`;
  return html`<div class="main">
    ${r.data.notes.length ? html`<div class="notes">${r.data.notes.map((n) => html`<p key=${n}>${n}</p>`)}</div>` : null}
    <div class="kpis">
      <div class="kpi"><div class="k">1-day ${(100 * top).toFixed(1).replace(".0", "")}% VaR (combined)</div><div class="v">${money(comb.var * value)}</div><div class="d">${pct(comb.var)} of the portfolio</div></div>
      <div class="kpi"><div class="k">1-day ${(100 * top).toFixed(1).replace(".0", "")}% Expected Shortfall</div><div class="v">${money(comb.es * value)}</div><div class="d">average loss on the worst days, ${pct(comb.es)}</div></div>
      <div class="kpi"><div class="k">Volatility, annualized</div><div class="v">${pct(v.realized_250, 1)}</div><div class="d">last year · GARCH next day ${pct(v.garch_next_day, 1)}</div></div>
      <div class="kpi"><div class="k">Max drawdown</div><div class="v">${pct(-a.returns.max_drawdown, 1)}</div><div class="d">${a.returns.max_drawdown_peak} → ${a.returns.max_drawdown_trough}</div></div>
      <div class="kpi"><div class="k">Diversification ratio</div><div class="v">${num(a.diversification_ratio)}×</div><div class="d">1.00× means no diversification benefit</div></div>
    </div>

    <div class="panel">
      <h3>Next-day risk by model</h3>
      <p class="caption">Each model forecasts tomorrow's loss from the last 250 days. VaR: the loss exceeded on only ${(100 * (1 - top)).toFixed(1)}% of days. ES: the average loss on those days. Data ${r.data.start} to ${r.data.end} (${r.data.rows} days), ${r.risk.engine} engine.</p>
      <div class="scroll"><table>
        <thead><tr><th>Model</th>${r.input.confidence.map((c) => html`<th key=${"v" + c}>VaR ${(100 * c).toFixed(1).replace(".0", "")}%</th><th key=${"e" + c}>ES ${(100 * c).toFixed(1).replace(".0", "")}%</th>`)}<th>Weight</th></tr></thead>
        <tbody>${[...r.risk.models, "combined"].map((m) => html`<tr key=${m} class=${m === "combined" ? "total" : ""}><td>${MODEL_LABEL[m] || m}</td>
          ${r.input.confidence.map((c) => { const x = next.find((y) => y.model === m && y.confidence === c); return html`<td key=${"v" + c}>${pct(x.var)} <span class="caption">${money(x.var * value)}</span></td><td key=${"e" + c}>${pct(x.es)}</td>`; })}
          <td>${m === "combined" ? "" : pct(next.find((y) => y.model === m && y.confidence === top).weight, 0)}</td></tr>`)}</tbody>
      </table></div>
    </div>

    <div class="grid2">
      <div class="panel"><h3>Portfolio value</h3><p class="caption">Growth of 100 with daily rebalancing to the target weights.</p>
        <${LineChart} dates=${a.series.dates} series=${[{ name: "value", color: css("--s1"), values: a.series.value }]} yFormat=${(t) => t.toFixed(0)} /></div>
      <div class="panel"><h3>Volatility over time</h3><p class="caption">Annualized: rolling 20 and 60 days, and the EWMA estimate (reacts fastest).</p>
        <${LineChart} dates=${a.series.return_dates} series=${[{ name: "20-day", color: css("--s2"), values: a.series.vol20, width: 1.2 }, { name: "60-day", color: css("--s1"), values: a.series.vol60 }, { name: "EWMA", color: css("--s3"), values: a.series.ewma, dash: "4 3" }]} yFormat=${(t) => (100 * t).toFixed(0) + "%"} /></div>
    </div>

    <div class="panel">
      <h3>Did the forecasts hold? (backtest)</h3>
      <p class="caption">Bars: each day's realized loss. Line: the combined VaR forecast made the day before. Red dots: breaches, days the loss exceeded the forecast.</p>
      ${levels.length > 1 ? html`<div class="tabs">${levels.map((l) => html`<button type="button" key=${l} class=${l === level ? "on" : ""} onClick=${() => setLevel(l)}>${l}%</button>`)}</div>` : null}
      <${LineChart} dates=${fc.dates} width=${1100} height=${260} bars=${{ name: "daily loss", color: css("--line"), values: fc.loss.map((l) => 100 * l) }}
        series=${[{ name: "combined VaR", color: css("--s1"), values: fc.combined_var.map((x) => (x == null ? null : 100 * x)) }]} markers=${breaches} yFormat=${(t) => t.toFixed(1) + "%"} />
      <div class="scroll" style=${{ marginTop: "12px" }}><table>
        <thead><tr><th>Model</th><th>Breaches</th><th>Expected</th><th>Kupiec p</th><th>Clustering p</th><th>Basel light</th><th>ES test Z2</th></tr></thead>
        <tbody>${bt.map((b) => html`<tr key=${b.model} class=${b.model === "combined" ? "total" : ""}><td>${MODEL_LABEL[b.model] || b.model}</td><td>${b.breaches}</td><td>${num(b.expected, 1)}</td>
          <td class=${b.kupiec_p < 0.05 ? "flag" : ""}>${num(b.kupiec_p, 3)}</td><td class=${b.christoffersen_p < 0.05 ? "flag" : ""}>${num(b.christoffersen_p, 3)}</td><td>${tl(b.traffic_light)}</td><td>${num(b.z2, 2)}</td></tr>`)}</tbody>
      </table></div>
      <p class="caption" style=${{ marginTop: "8px" }}>p below 0.05 (red) flags a model: too many or too few breaches (Kupiec) or breaches that cluster (Christoffersen). Z2 above 0 means ES was too small.</p>
    </div>

    <div class="grid2">
      <div class="panel"><h3>Where the risk comes from</h3><p class="caption">Each holding's share of capital vs its share of portfolio volatility (last 250 days). Shares of risk sum to 100%.</p>
        <${ContributionBars} assets=${a.assets} />
        <div class="scroll" style=${{ marginTop: "10px" }}><table><thead><tr><th>Ticker</th><th>Weight</th><th>Own vol</th><th>Risk share</th></tr></thead>
          <tbody>${a.assets.map((x) => html`<tr key=${x.ticker}><td>${x.ticker}</td><td>${pct(x.weight, 1)}</td><td>${pct(x.vol, 1)}</td><td>${pct(x.risk_contribution, 1)}</td></tr>`)}
          <tr class="total"><td>Portfolio</td><td>100%</td><td>${pct(a.portfolio_vol_250, 1)}</td><td>100%</td></tr></tbody></table></div></div>
      <div class="panel"><h3>Correlations</h3><p class="caption">Daily returns, last 250 days. Low or negative correlation is what diversifies.</p>
        ${a.assets.length > 1 ? html`<${Heatmap} tickers=${a.assets.map((x) => x.ticker)} corr=${a.correlation} />` : html`<p class="caption">Add a second holding to see correlations.</p>`}</div>
    </div>

    <div class="grid2">
      <div class="panel"><h3>Volatility models</h3><div class="facts">
        <span>Realized, whole period</span><span>${pct(v.realized_full, 1)}</span>
        <span>Realized, last 250 / 60 / 20 days</span><span>${pct(v.realized_250, 1)} / ${pct(v.realized_60, 1)} / ${pct(v.realized_20, 1)}</span>
        <span>EWMA today (λ = 0.94)</span><span>${pct(v.ewma_now, 1)}</span>
        <span>GARCH(1,1) next day</span><span>${pct(v.garch_next_day, 1)}</span>
        <span>GARCH long-run level</span><span>${pct(v.garch_long_run, 1)}</span>
        <span>GARCH persistence α + β</span><span>${num(v.garch_persistence, 3)}</span>
        <span>Half-life of a volatility shock</span><span>${v.garch_half_life_days == null ? "–" : num(v.garch_half_life_days, 0) + " days"}</span>
      </div></div>
      <div class="panel"><h3>Tail facts</h3><div class="facts">
        <span>Annualized return</span><span>${pct(a.returns.annualized_return, 1)}</span>
        <span>Skewness / excess kurtosis</span><span>${num(a.returns.skew)} / ${num(a.returns.excess_kurtosis)}</span>
        <span>Worst 5-day stretch (from ${a.returns.worst_week.start})</span><span>${pct(a.returns.worst_week.return)} · ${money(a.returns.worst_week.return * value)}</span>
        ${a.returns.worst_days.map((d) => html`<${React.Fragment} key=${d.date}><span>Worst day ${d.date}</span><span>${pct(d.return)} · ${money(d.return * value)}</span><//>`)}
        <span>Current drawdown</span><span>${pct(-a.returns.current_drawdown, 1)}</span>
      </div></div>
    </div>
  </div>`;
}

function App() {
  const [status, setStatus] = useState(null);
  const [examples, setExamples] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    fetch("/api/status").then((r) => r.json()).then(setStatus).catch(() => setError("Cannot reach the qre server."));
    fetch("/api/examples").then((r) => r.json()).then(setExamples).catch(() => {});
  }, []);
  const run = async (req) => {
    setBusy(true); setError(null);
    try {
      const resp = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body.error || "analysis failed");
      setResult(body);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  return html`<div>
    <header><h1>qre · Portfolio risk</h1><span class="sub">VaR, Expected Shortfall and volatility for any portfolio</span>
      <span class="pill">${status ? status.engine : "connecting…"}</span></header>
    <div class="layout">
      <${PortfolioForm} status=${status} examples=${examples} onRun=${run} busy=${busy} />
      <div class="main">
        ${error ? html`<div class="error"><b>Could not analyze:</b> ${error}</div>` : null}
        ${result ? html`<${Results} key=${JSON.stringify(result.input)} r=${result} />` : html`<div class="panel empty">Enter holdings and weights, then <b>Analyze risk</b>. The bundled six stocks work offline; any other ticker downloads from Yahoo Finance.</div>`}
      </div>
    </div>
  </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
