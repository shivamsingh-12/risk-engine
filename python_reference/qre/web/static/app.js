// Phase 15: front end of the portfolio risk app. React 18 + htm (JSX-like templates without a build step).
// Talks to the local server: GET /api/status, GET /api/examples, POST /api/analyze, POST /api/stress (Phase 20).
// Phase 20 adds the Refresh button, live mode (re-run every minute while the market is open) and stress tests.
const { useState, useEffect, useMemo, useRef } = React;
const html = htm.bind(React.createElement);

const pct = (x, d = 2) => (x == null || !isFinite(x) ? "–" : (100 * x).toFixed(d) + "%");
const num = (x, d = 2) => (x == null || !isFinite(x) ? "–" : x.toFixed(d));
const signed = (x, d = 2) => (x == null || !isFinite(x) ? "–" : (x > 0 ? "+" : "") + (100 * x).toFixed(d) + "%");
const tone = (x) => (x == null || !isFinite(x) || x === 0 ? "" : x > 0 ? "pos" : "neg");
const LIVE_EVERY_MS = 60000;
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
function PortfolioForm({ status, examples, onRun, busy, live, setLive }) {
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
  const request = () => {
    const confidence = Object.keys(conf).filter((k) => conf[k]).map(Number).sort((a, b) => b - a);
    return { holdings: holdings.filter((h) => h.ticker.trim()).map((h) => ({ ticker: h.ticker.trim(), weight: parseFloat(h.weight) || 0 })),
      source, years: Number(years), value: Number(value), confidence, models: modelSet, combination };
  };
  const submit = (e) => { e.preventDefault(); onRun(request()); };
  const canLive = source === "yahoo";
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
    <label class="checks" style=${{ marginTop: "12px" }}><input type="checkbox" checked=${live && canLive} disabled=${!canLive}
      onChange=${(e) => setLive(e.target.checked)} />Live: re-run every minute while the market is open${canLive ? "" : " (Yahoo Finance only)"}</label>
    <div class="btnrow">
      <button class="run" disabled=${busy}>${busy ? html`<span class="spinner"></span>Analyzing…` : "Analyze risk"}</button>
      <button type="button" class="secondary" disabled=${busy} title="Download prices again instead of using the copy saved in the last 12 hours"
        onClick=${() => onRun({ ...request(), refresh: true })}>Refresh data</button>
    </div>
    <p class="hint">${status ? `Engine: ${status.engine}.` : ""} ${modelSet === "all" ? "All 8 models take about 5 to 30 seconds (GARCH-t refits every day)." : "Fast models take about a second."}</p>
  </form>`;
}

// ------------------------------------------------------------------ results
function Results({ r, liveRunning }) {
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
  const lt = r.latest;
  return html`<div class="main">
    ${r.live ? html`<${LiveBar} live=${r.live} running=${liveRunning} />` : null}
    ${r.data.notes.length ? html`<div class="notes">${r.data.notes.map((n) => html`<p key=${n}>${n}</p>`)}</div>` : null}
    <div class="kpis">
      <div class="kpi"><div class="k">${r.live && r.live.provisional ? "Today so far (provisional)" : "Latest day"}</div>
        <div class=${"v " + tone(lt.pnl)}>${lt.pnl > 0 ? "+" : ""}${money(lt.pnl)}</div><div class="d">${signed(lt.return)} · ${lt.date} vs ${lt.previous}</div></div>
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

    <${StressPanel} input=${r.input} />

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

// ------------------------------------------------------------------ Phase 20: live mode and stress tests
function LiveBar({ live, running }) {
  const state = live.market_open
    ? (live.provisional ? `today's prices are live (last quote ${live.asof ? live.asof.slice(11) : "–"} New York time) and provisional until the close`
      : "market open, but no intraday quotes for every holding yet")
    : "market closed: showing the last close; updates resume when the market opens";
  return html`<div class="livebar"><span><span class=${"dot" + (live.market_open ? " on" : "")}></span><b>Live</b></span>
    <span>${state}</span><span class="caption" style=${{ margin: 0 }}>checked ${live.checked.slice(11)} · re-runs every minute${running ? " · updating…" : ""}</span>
    ${live.error ? html`<span class="neg">quotes failed: ${live.error}</span>` : null}</div>`;
}

function ScenarioDetail({ s, value }) {
  return html`<div style=${{ marginTop: "12px" }}>
    <div class="subhead">${s.name}: ${s.from} to ${s.to}, ${s.days} trading days</div>
    <${LineChart} dates=${s.path_dates} series=${[{ name: "portfolio value (start = 100)", color: css("--s1"), values: s.value.map((v) => 100 * v) }]} yFormat=${(t) => t.toFixed(0)} height=${200} />
    <div class="scroll"><table><thead><tr><th>Holding</th><th>Return</th><th>Contribution</th><th>P&L</th><th>Note</th></tr></thead>
      <tbody>${s.holdings.map((h) => html`<tr key=${h.ticker}><td>${h.ticker}</td><td class=${tone(h.return)}>${signed(h.return, 1)}</td>
        <td>${signed(h.contribution, 1)}</td><td>${money(h.contribution * value)}</td>
        <td class="caption">${h.proxied_days ? `proxied by beta × SPY on ${h.proxied_days} of ${s.days} days` : ""}</td></tr>`)}</tbody></table></div>
  </div>`;
}

function StressPanel({ input }) {
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [sel, setSel] = useState("covid");
  const [shocks, setShocks] = useState([{ ticker: "SPY", shock: -20 }]);
  const [horizon, setHorizon] = useState(10);
  const [reverse, setReverse] = useState(20);
  const value = input.value;
  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const body = { holdings: input.tickers.map((t, i) => ({ ticker: t, weight: input.weights[i] })), value, source: input.source,
        shocks: shocks.filter((x) => x.ticker.trim() && x.shock !== "").map((x) => ({ ticker: x.ticker.trim(), shock: Number(x.shock) / 100 })),
        horizon_days: Number(horizon), reverse_loss: Number(reverse) / 100, confidence: 0.99 };
      const resp = await fetch("/api/stress", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const out = await resp.json();
      if (!resp.ok) throw new Error(out.error || "stress test failed");
      setRes(out);
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  useEffect(() => { run(); }, []);
  const setShock = (i, k, v) => setShocks(shocks.map((x, j) => (j === i ? { ...x, [k]: v } : x)));
  const avail = res ? res.scenarios.filter((s) => s.available) : [];
  const chosen = avail.find((s) => s.key === sel) || avail[0];
  const sh = res && res.shock, rv = res && res.reverse;
  return html`<div class="panel">
    <h3>Stress tests</h3>
    <p class="caption">What past crises and hypothetical shocks would do to this portfolio today, held without trading. These are far larger than VaR, which describes an ordinary bad day (Experiment 25).</p>
    ${err ? html`<div class="error"><b>Stress test failed:</b> ${err}</div>` : null}
    ${!res ? html`<p class="caption">${busy ? html`<span class="spinner"></span>Loading prices since 2007…` : ""}</p>` : html`
      <div class="scroll"><table>
        <thead><tr><th>Scenario</th><th>Dates</th><th>Portfolio</th><th>P&L</th><th>Max drawdown</th><th>Worst day</th><th>S&P 500</th><th></th></tr></thead>
        <tbody>${res.scenarios.map((s) => s.available ? html`<tr key=${s.key} class=${"click" + (chosen && s.key === chosen.key ? " sel" : "")} onClick=${() => setSel(s.key)}>
            <td>${s.name}</td><td class="caption">${s.from.slice(0, 7)} → ${s.to.slice(0, 7)}</td><td class=${tone(s.total_return)}>${signed(s.total_return, 1)}</td>
            <td>${money(s.pnl)}</td><td>${pct(-s.max_drawdown, 1)}</td><td>${signed(s.worst_day, 1)}</td><td>${signed(s.market_return, 1)}</td>
            <td class="caption">${s.proxied.length ? "proxied: " + s.proxied.join(", ") : ""}</td></tr>`
          : html`<tr key=${s.key}><td>${s.name}</td><td colspan="7" class="caption" style=${{ textAlign: "left" }}>${s.reason}</td></tr>`)}</tbody>
      </table></div>
      <p class="caption" style=${{ marginTop: "6px" }}>Click a scenario for its path and each holding's part. A holding that did not trade then is replaced by its beta to SPY times SPY's daily moves; that keeps the market's part of the move and misses the holding's own (off by about 6 points on average in Experiment 25).</p>
      ${chosen ? html`<${ScenarioDetail} s=${chosen} value=${value} />` : null}

      <div class="grid2" style=${{ marginTop: "8px" }}>
        <div>
          <div class="subhead">What if…</div>
          <p class="caption">Shock one or more tickers (held or not, e.g. SPY, TLT, an oil ETF). Every holding moves by what its correlations with them over the last ${res.data.cov_days} days imply.</p>
          <div class="shocks">
            <span class="caption">Ticker</span><span class="caption">Move %</span><span></span>
            ${shocks.map((x, i) => html`<${React.Fragment} key=${i}>
              <input value=${x.ticker} onInput=${(e) => setShock(i, "ticker", e.target.value.toUpperCase())} aria-label="shocked ticker" />
              <input type="number" step="any" value=${x.shock} onInput=${(e) => setShock(i, "shock", e.target.value)} aria-label="shock in percent" />
              <button type="button" class="icon" title="Remove" onClick=${() => setShocks(shocks.filter((_, j) => j !== i))}>✕</button><//>`)}
          </div>
          <button type="button" class="link" onClick=${() => setShocks([...shocks, { ticker: "", shock: -10 }])}>+ Add shock</button>
          <div class="inline" style=${{ marginTop: "6px" }}><span class="caption" style=${{ margin: 0 }}>over</span>
            <input type="number" min="1" max="250" value=${horizon} onInput=${(e) => setHorizon(e.target.value)} /><span class="caption" style=${{ margin: 0 }}>trading days</span></div>
          <div><button type="button" class="secondary" style=${{ marginTop: "10px" }} disabled=${busy} onClick=${run}>${busy ? "Running…" : "Apply"}</button></div>
          ${sh ? html`<div class="facts" style=${{ marginTop: "12px" }}>
            <span>Expected P&L</span><span class=${tone(sh.expected_return)}>${money(sh.pnl)} (${signed(sh.expected_return, 1)})</span>
            <span>Bad case at 99% (allowing for the spread around the expected move)</span><span class="neg">${money(sh.pnl_at_confidence)} (${signed(-sh.loss_at_confidence, 1)})</span>
            ${sh.holdings.map((h) => html`<${React.Fragment} key=${h.ticker}><span>${h.ticker} expected move</span><span class=${tone(h.expected_return)}>${signed(h.expected_return, 1)}</span><//>`)}
          </div>` : null}
        </div>
        <div>
          <div class="subhead">Reverse stress test</div>
          <p class="caption">How far the market (SPY) would have to fall for this portfolio to lose a given amount, on average.</p>
          <div class="inline"><span class="caption" style=${{ margin: 0 }}>Loss of</span>
            <input type="number" min="1" max="99" value=${reverse} onInput=${(e) => setReverse(e.target.value)} /><span class="caption" style=${{ margin: 0 }}>%</span>
            <button type="button" class="link" onClick=${run}>update</button></div>
          ${rv ? html`<div class="facts" style=${{ marginTop: "12px" }}>
            <span>Portfolio beta to SPY</span><span>${num(res.portfolio_beta)}</span>
            <span>Market move that loses ${pct(rv.target_loss, 0)} (${money(-rv.target_loss * value)})</span>
            <span class="neg">${rv.market_return == null ? "none: the portfolio gains when the market falls" : rv.market_return <= -1 ? "more than the market can fall" : signed(rv.market_return, 1)}</span>
            ${Object.entries(res.betas).map(([t, b]) => html`<${React.Fragment} key=${t}><span>${t} beta</span><span>${num(b)}</span><//>`)}
          </div>` : null}
          <p class="caption" style=${{ marginTop: "8px" }}>Correlations shift in a crisis: in 2022 bonds and energy moved against what every earlier crisis implied. Shock several tickers to test such a break.</p>
        </div>
      </div>
      <p class="caption">Prices ${res.data.start} to ${res.data.end}. Correlations: ${res.data.cov_from} to ${res.data.cov_to}.</p>`}
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
  const [live, setLive] = useState(false);
  const [liveRunning, setLiveRunning] = useState(false);
  const lastReq = useRef(null);
  const inFlight = useRef(false);
  const post = async (req) => {
    const resp = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.error || "analysis failed");
    return body;
  };
  const run = async (req) => {
    const full = { ...req, live: live && req.source === "yahoo" };
    lastReq.current = full;
    setBusy(true); setError(null); inFlight.current = true;
    try { setResult(await post(full)); } catch (e) { setError(e.message); } finally { setBusy(false); inFlight.current = false; }
  };
  // Live mode: re-run the last analysis every minute in the background; the server only adds intraday prices
  // while the market is open, so outside trading hours this just keeps the last close
  useEffect(() => {
    if (!live) return undefined;
    const tick = async () => {
      const req = lastReq.current;
      if (!req || req.source !== "yahoo" || inFlight.current) return;
      inFlight.current = true; setLiveRunning(true);
      try { setResult(await post({ ...req, live: true })); setError(null); }
      catch (e) { setError("Live update failed: " + e.message); }
      finally { inFlight.current = false; setLiveRunning(false); }
    };
    tick();
    const id = setInterval(tick, LIVE_EVERY_MS);
    return () => clearInterval(id);
  }, [live]);
  return html`<div>
    <header><h1>qre · Portfolio risk</h1><span class="sub">VaR, Expected Shortfall and volatility for any portfolio</span>
      <span class="pill">${status ? status.engine : "connecting…"}</span></header>
    <div class="layout">
      <${PortfolioForm} status=${status} examples=${examples} onRun=${run} busy=${busy} live=${live} setLive=${setLive} />
      <div class="main">
        ${error ? html`<div class="error"><b>Could not analyze:</b> ${error}</div>` : null}
        ${result ? html`<${Results} key=${JSON.stringify({ ...result.input, live: undefined })} r=${result} liveRunning=${liveRunning} />` : html`<div class="panel empty">Enter holdings and weights, then <b>Analyze risk</b>. The bundled six stocks work offline; any other ticker downloads from Yahoo Finance.</div>`}
      </div>
    </div>
  </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
