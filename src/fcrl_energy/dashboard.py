import json
from pathlib import Path

import numpy as np

from .pipeline import FCRL, FORECAST, HINDSIGHT, STANDARD

COUNTRIES = {"EL": "Greece", "BG": "Bulgaria", "ES": "Spain", "PT": "Portugal", "IT": "Italy", "CY": "Cyprus"}
KEYS = {"a_per_resident": "pr", "a_per_poor_resident": "std", "a_last_year": "ly", "a_forecast": "fc", "a_fcrl": "fcrl", "a_hindsight": "hind"}
LABELS = {"pr": "Equal amount per resident", "std": "Equal amount per resident at risk of poverty", "ly": "Proportional to last year's need", "fc": "Proportional to forecast need", "fcrl": "FCRL-energy", "hind": "Hindsight benchmark (not implementable)"}
STRATEGY_KEYS = {"equal amount per resident": "pr", STANDARD: "std", "proportional to last year's need": "ly", FORECAST: "fc", FCRL: "fcrl", HINDSIGHT: "hind"}


def _num(value):
    value = float(value)
    return None if np.isnan(value) or np.isinf(value) else round(value, 5)


def _cells(detail, extra=()):
    out = []
    keys = ["region", "season_label", *extra]
    for group, block in detail.groupby(keys, sort=True):
        first = block.iloc[0]
        need = float(first["need"])
        record = {"r": first["region"], "n": first["name"], "c": first["country"], "s": first["season_label"], "need": _num(need), "months": _num(first["months"]), "v": {}, "met": {}}
        for row in block.itertuples():
            key = KEYS[row.key]
            record["v"][key] = _num(row.covered / need) if need > 0 else None
            record["met"][key] = _num(row.met)
        if extra:
            record["sc"] = first["scenario"]
        out.append(record)
    return out


def payload(performance, detail, stress_detail, statements, meta):
    perf = {}
    for row in performance.to_dict("records"):
        key = STRATEGY_KEYS.get(row["strategy"])
        if key:
            perf[key] = {"lo": _num(row["worst_off_coverage_ci_low"]), "hi": _num(row["worst_off_coverage_ci_high"]), "leak": _num(row["leakage_share"])}
    notes = []
    for row in statements.to_dict("records"):
        notes.append({"r": row["region"], "s": row["season"], "t": row["statement"]})
    regions = _cells(detail)
    countries = sorted({row["c"] for row in regions})
    return {
        "meta": meta,
        "labels": LABELS,
        "perf": perf,
        "regions": regions,
        "stress": _cells(stress_detail, ("scenario",)),
        "scenarios": list(dict.fromkeys(stress_detail["scenario"])),
        "statements": notes,
        "countries": [{"code": code, "name": COUNTRIES.get(code, code)} for code in countries],
    }


def meta_from_result(result):
    plan = result.selected
    return {
        "title": "FCRL-energy",
        "source": result.config["data"]["source"],
        "regions": int(result.panel["region"].nunique()),
        "seasons": sorted(result.panel.loc[result.panel["split"] == "test", "season_label"].unique().tolist()),
        "plan": {season: {"level": plan[season]["level"], "reserve": plan[season]["reserve"]} for season in plan},
        "miss": _num(result.notes["guarantee_test_miss_rate"]),
        "budget": result.notes["budget_share"],
    }


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    path.write_text(html, encoding="utf-8")
    return path


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FCRL-energy: fair allocation of energy support</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3f0;
  --surface-1: #fcfcfb;
  --hairline: #e4e3df;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #7a7974;
  --muted-mark: #c7c6c1;
  --fcrl: #2a78d6;
  --std: #eb6834;
  --fc: #1baf7a;
  --hind: #9c9b96;
  --accent-wash: rgba(42, 120, 214, 0.08);
  --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.05);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211;
    --surface-1: #1a1a19;
    --hairline: #2e2e2c;
    --text-primary: #f0efec;
    --text-secondary: #c3c2b7;
    --text-muted: #8f8e88;
    --muted-mark: #4a4a47;
    --fcrl: #3987e5;
    --std: #d95926;
    --fc: #199e70;
    --hind: #6f6e69;
    --accent-wash: rgba(57, 135, 229, 0.14);
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.25);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211;
  --surface-1: #1a1a19;
  --hairline: #2e2e2c;
  --text-primary: #f0efec;
  --text-secondary: #c3c2b7;
  --text-muted: #8f8e88;
  --muted-mark: #4a4a47;
  --fcrl: #3987e5;
  --std: #d95926;
  --fc: #199e70;
  --hind: #6f6e69;
  --accent-wash: rgba(57, 135, 229, 0.14);
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.25);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary); font: 15px/1.5 Inter, "Segoe UI", system-ui, -apple-system, Roboto, sans-serif; -webkit-font-smoothing: antialiased; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 40px 24px 64px; }
header { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; margin-bottom: 28px; }
.eyebrow { font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); margin: 0 0 8px; }
h1 { font-size: 34px; line-height: 1.15; margin: 0 0 10px; letter-spacing: -0.02em; }
.lede { color: var(--text-secondary); max-width: 760px; margin: 0; font-size: 16px; }
.theme { border: 1px solid var(--hairline); background: var(--surface-1); color: var(--text-secondary); border-radius: 999px; padding: 8px 14px; font: inherit; font-size: 13px; cursor: pointer; white-space: nowrap; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 0 0 28px; }
.tile { background: var(--surface-1); border-radius: 14px; padding: 18px 20px; box-shadow: var(--shadow); }
.tile .label { font-size: 13px; color: var(--text-secondary); margin: 0 0 6px; }
.tile .value { font-size: 30px; font-weight: 650; letter-spacing: -0.02em; margin: 0; }
.tile .delta { font-size: 13px; color: var(--text-secondary); margin: 4px 0 0; }
.tile .delta b { color: var(--text-primary); font-weight: 600; }
.filters { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 0 0 20px; }
.filters .caption { font-size: 13px; color: var(--text-secondary); margin-right: 4px; }
.chip { border: 1px solid var(--hairline); background: var(--surface-1); color: var(--text-primary); border-radius: 999px; padding: 6px 14px; font: inherit; font-size: 13px; cursor: pointer; }
.chip[aria-pressed="true"] { background: var(--text-primary); color: var(--surface-1); border-color: var(--text-primary); }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.card { background: var(--surface-1); border-radius: 16px; padding: 22px 24px 18px; box-shadow: var(--shadow); min-width: 0; }
.card.wide { grid-column: 1 / -1; }
.card h2 { font-size: 18px; margin: 0 0 4px; letter-spacing: -0.01em; }
.card .sub { font-size: 13.5px; color: var(--text-secondary); margin: 0 0 14px; }
.card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.seg { display: inline-flex; border: 1px solid var(--hairline); border-radius: 10px; overflow: hidden; flex-shrink: 0; }
.seg button { border: 0; background: transparent; color: var(--text-secondary); padding: 6px 10px; font: inherit; font-size: 12.5px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--accent-wash); color: var(--text-primary); font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 12.5px; color: var(--text-secondary); margin: 0 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { font-family: inherit; }
.axis text { fill: var(--text-muted); font-size: 11px; }
.axis line { stroke: var(--hairline); stroke-width: 1; }
.lab { fill: var(--text-primary); font-size: 12.5px; }
.lab-2 { fill: var(--text-secondary); font-size: 12px; }
.val { fill: var(--text-primary); font-size: 12px; font-variant-numeric: tabular-nums; }
.mark { transition: opacity 120ms ease; }
.dim .mark { opacity: 0.25; }
.dim .mark.on { opacity: 1; }
details { margin-top: 10px; font-size: 13px; color: var(--text-secondary); }
summary { cursor: pointer; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--hairline); font-size: 12.5px; }
th { color: var(--text-secondary); font-weight: 600; }
td.n, th.n { text-align: right; }
select { font: inherit; font-size: 14px; padding: 8px 12px; border-radius: 10px; border: 1px solid var(--hairline); background: var(--surface-1); color: var(--text-primary); min-width: 280px; }
.explain { display: grid; grid-template-columns: 1.1fr 1fr; gap: 24px; align-items: start; }
.statement { background: var(--surface-0); border-radius: 12px; padding: 14px 16px; font-size: 14px; line-height: 1.55; margin: 0 0 10px; }
.statement b { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 4px; }
#tip { position: fixed; pointer-events: none; background: var(--surface-1); color: var(--text-primary); border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,0.18); padding: 10px 12px; font-size: 12.5px; opacity: 0; transition: opacity 80ms ease; max-width: 280px; z-index: 10; border: 1px solid var(--hairline); }
#tip .t-title { color: var(--text-secondary); margin-bottom: 4px; }
#tip .t-row { display: flex; align-items: center; gap: 8px; }
#tip .t-row strong { font-size: 14px; font-variant-numeric: tabular-nums; }
#tip .key { width: 12px; height: 2px; border-radius: 1px; display: inline-block; }
footer { margin-top: 28px; font-size: 12.5px; color: var(--text-muted); }
footer a { color: inherit; }
@media (max-width: 860px) { .grid, .explain { grid-template-columns: 1fr; } .tiles { grid-template-columns: 1fr 1fr; } h1 { font-size: 28px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <p class="eyebrow">Energy support under price shocks</p>
      <h1>Who stays warm, who stays cool, when the budget is fixed?</h1>
      <p class="lede" id="lede"></p>
    </div>
    <button class="theme" id="theme" type="button">Dark mode</button>
  </header>
  <section class="tiles" id="tiles"></section>
  <div class="filters" role="group" aria-label="Country filter"><span class="caption">Country</span><span id="chips"></span></div>
  <div class="grid">
    <section class="card">
      <h2>The worst-off regions</h2>
      <p class="sub" id="s-worst"></p>
      <div id="c-worst"></div>
      <details><summary>Show table</summary><div id="t-worst"></div></details>
    </section>
    <section class="card">
      <h2>Under price and budget shocks</h2>
      <p class="sub" id="s-stress"></p>
      <div class="seg" id="seg-stress" role="group" aria-label="Measure" style="margin-bottom:12px"></div>
      <div class="legend" id="l-stress"></div>
      <div id="c-stress"></div>
      <details><summary>Show table</summary><div id="t-stress"></div></details>
    </section>
    <section class="card wide">
      <h2>Every region, every season</h2>
      <p class="sub" id="s-strip"></p>
      <div id="c-strip"></div>
    </section>
    <section class="card wide">
      <h2>The least protected region of each country</h2>
      <p class="sub">Lowest coverage of any region of the country in any test season, before and after.</p>
      <div class="legend" id="l-country"></div>
      <div id="c-country"></div>
      <details><summary>Show table</summary><div id="t-country"></div></details>
    </section>
    <section class="card wide">
      <div class="card-head">
        <div><h2>Find a region</h2><p class="sub">Coverage of the energy need of residents at risk of poverty, season by season, and why the region received what it did.</p></div>
        <select id="region" aria-label="Region"></select>
      </div>
      <div class="explain">
        <div><div class="legend" id="l-region"></div><div id="c-region"></div></div>
        <div id="statements"></div>
      </div>
    </section>
  </div>
  <footer id="foot"></footer>
</div>
<div id="tip" role="tooltip"></div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const KEYS = [["fcrl", "FCRL-energy"], ["std", "Equal amount per resident at risk of poverty"], ["fc", "Proportional to forecast need"]];
const NAME = Object.fromEntries(KEYS);
const pct = v => v == null ? "n/a" : Math.round(v * 100) + "%";
const pct1 = v => v == null ? "n/a" : (v * 100).toFixed(1) + "%";
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
let country = "all";
let selected = null;
let highlight = null;

function el(tag, attrs, parent, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}
function sv(tag, attrs, parent, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}
function svg(host, w, h) {
  host.replaceChildren();
  return sv("svg", {viewBox: `0 0 ${w} ${h}`, role: "img"}, host);
}
const tip = document.getElementById("tip");
function showTip(event, title, rows) {
  tip.replaceChildren();
  el("div", {class: "t-title"}, tip, title);
  for (const [color, value, label] of rows) {
    const row = el("div", {class: "t-row"}, tip);
    if (color) { const k = el("span", {class: "key"}, row); k.style.background = color; }
    el("strong", {}, row, value);
    el("span", {}, row, label);
  }
  const x = Math.min(event.clientX + 14, window.innerWidth - 300);
  const y = Math.min(event.clientY + 14, window.innerHeight - 140);
  tip.style.left = x + "px";
  tip.style.top = y + "px";
  tip.style.opacity = 1;
}
function hideTip() { tip.style.opacity = 0; }
function bindTip(node, fn) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointermove", e => fn(e));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => { const r = node.getBoundingClientRect(); fn({clientX: r.right, clientY: r.top}); });
  node.addEventListener("blur", hideTip);
}
function axis(g, x, y0, y1, ticks, fmt) {
  const a = sv("g", {class: "axis"}, g);
  for (const t of ticks) {
    sv("line", {x1: x(t), x2: x(t), y1: y0, y2: y1}, a);
    sv("text", {x: x(t), y: y1 + 16, "text-anchor": "middle"}, a, fmt(t));
  }
}
function table(host, head, rows) {
  host.replaceChildren();
  const t = el("table", {}, host);
  const tr = el("tr", {}, el("thead", {}, t));
  head.forEach((h, i) => el("th", {class: i ? "n" : ""}, tr, h));
  const body = el("tbody", {}, t);
  for (const r of rows) { const row = el("tr", {}, body); r.forEach((c, i) => el("td", {class: i ? "n" : ""}, row, c)); }
}
function legend(host, keys) {
  host.replaceChildren();
  for (const k of keys) { const s = el("span", {}, host); const i = el("i", {}, s); i.style.background = css("--" + k); s.appendChild(document.createTextNode(NAME[k])); }
}
function regionsInScope() { return DATA.regions.filter(d => country === "all" || d.c === country); }
function scopeName() { return country === "all" ? "all five countries" : (DATA.countries.find(c => c.code === country) || {name: country}).name; }
function agg(rows, k) {
  const seasons = {};
  let need = 0, covered = 0, met = 0, months = 0, lowest = Infinity;
  for (const d of rows) {
    const v = d.v[k];
    if (v == null) continue;
    (seasons[d.s] = seasons[d.s] || []).push(v);
    need += d.need; covered += v * d.need; met += d.met[k] || 0; months += d.months || 0;
    lowest = Math.min(lowest, v);
  }
  const tails = Object.values(seasons).map(list => { list.sort((a, b) => a - b); const n = Math.max(1, Math.ceil(0.1 * list.length)); return list.slice(0, n).reduce((a, b) => a + b, 0) / n; });
  return {worst: tails.length ? tails.reduce((a, b) => a + b, 0) / tails.length : null, lowest: isFinite(lowest) ? lowest : null, overall: need ? covered / need : null, attain: months ? met / months : null};
}
const ORDER = ["pr", "std", "ly", "fc", "fcrl", "hind"];

function tiles() {
  const rows = regionsInScope();
  const f = agg(rows, "fcrl"), s = agg(rows, "std");
  const host = document.getElementById("tiles");
  host.replaceChildren();
  const items = [
    ["Worst-off tenth of regions", pct(f.worst), `from <b>${pct(s.worst)}</b> under an equal amount per poor resident`],
    ["Least protected region", pct(f.lowest), `from <b>${pct(s.lowest)}</b>`],
    ["Guaranteed minimum met", pct1(f.attain), `winter ${pct(DATA.meta.plan.winter.level)}, summer ${pct(DATA.meta.plan.summer.level)} of need`],
    ["Total need covered", pct1(f.overall), `against <b>${pct1(s.overall)}</b>, same budget`],
  ];
  for (const [label, value, delta] of items) {
    const t = el("div", {class: "tile"}, host);
    el("p", {class: "label"}, t, label + (country === "all" ? "" : ", " + scopeName()));
    el("p", {class: "value"}, t, value);
    const d = el("p", {class: "delta"}, t);
    delta.split(/<b>|<\/b>/).forEach((p, i) => i % 2 ? el("b", {}, d, p) : d.appendChild(document.createTextNode(p)));
  }
  document.getElementById("lede").textContent = `A fixed energy allowance is shared among ${DATA.meta.regions} regions before each month begins, with a budget of ${pct(DATA.meta.budget)} of forecast need. FCRL-energy guarantees every region a minimum, shares the rest to protect the worst-off first, and keeps a ${pct(DATA.meta.plan.winter.reserve)} reserve for the regions left furthest behind. Test seasons: ${DATA.meta.seasons.join(", ")}.`;
}

function worst() {
  const scoped = regionsInScope();
  const rows = ORDER.map(k => ({k, label: DATA.labels[k], ...agg(scoped, k), ...(country === "all" && DATA.perf[k] ? {lo: DATA.perf[k].lo, hi: DATA.perf[k].hi} : {})}));
  document.getElementById("s-worst").textContent = `Need covered in the worst-covered tenth of regions, ${scopeName()}.` + (country === "all" ? " Lines show 95% intervals." : "");
  const W = 520, rowH = 40, top = 6, left = 196, right = 50, H = top + rows.length * rowH + 26;
  const s = svg(document.getElementById("c-worst"), W, H);
  const x = v => left + Math.min(v, 1) * (W - left - right);
  axis(s, x, top, H - 26, [0, 0.25, 0.5, 0.75, 1], pct);
  rows.forEach((d, i) => {
    const y = top + i * rowH + rowH / 2;
    const fc = d.k === "fcrl";
    const g = sv("g", {class: "mark"}, s);
    const lines = d.label.length > 30 ? [d.label.slice(0, d.label.lastIndexOf(" ", 30)), d.label.slice(d.label.lastIndexOf(" ", 30) + 1)] : [d.label];
    lines.forEach((t, j) => sv("text", {x: left - 10, y: y + 4 + (j - (lines.length - 1) / 2) * 14, "text-anchor": "end", class: fc ? "lab" : "lab-2", "font-weight": fc ? 600 : 400}, g, t));
    const w = Math.max(0, x(d.worst) - left);
    sv("path", {d: `M${left},${y - 9} h${Math.max(w - 4, 0)} a4,4 0 0 1 4,4 v10 a4,4 0 0 1 -4,4 h${-Math.max(w - 4, 0)} z`, fill: fc ? css("--fcrl") : css("--muted-mark")}, g);
    if (d.lo != null) sv("line", {x1: x(d.lo), x2: x(d.hi), y1: y, y2: y, stroke: css("--text-secondary"), "stroke-width": 1.5}, g);
    sv("text", {x: x(Math.max(d.worst, d.hi || 0)) + 8, y: y + 4, class: "val", "font-weight": fc ? 700 : 400}, g, pct(d.worst));
    const hit = sv("rect", {x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent"}, g);
    const tipRows = [[fc ? css("--fcrl") : css("--muted-mark"), pct1(d.worst), "worst-off tenth"]];
    if (d.lo != null) tipRows.push([null, `${pct1(d.lo)} to ${pct1(d.hi)}`, "95% interval"]);
    tipRows.push([null, pct1(d.lowest), "least protected region"], [null, pct1(d.overall), "total need covered"]);
    bindTip(hit, e => showTip(e, d.label, tipRows));
  });
  table(document.getElementById("t-worst"), ["Rule", "Worst-off tenth", "Least protected region", "Total need covered"], rows.map(d => [d.label, pct1(d.worst), pct1(d.lowest), pct1(d.overall)]));
}

let stressMetric = "lowest";
function stress() {
  const metrics = [["lowest", "Least protected region"], ["worst", "Worst-off tenth"], ["attain", "Guarantee met"]];
  const seg = document.getElementById("seg-stress");
  seg.replaceChildren();
  for (const [k, label] of metrics) {
    const b = el("button", {type: "button", "aria-pressed": String(k === stressMetric)}, seg, label);
    b.addEventListener("click", () => { stressMetric = k; stress(); });
  }
  document.getElementById("s-stress").textContent = `Same budget, three rules, ${scopeName()}.`;
  legend(document.getElementById("l-stress"), ["fcrl", "std", "fc"]);
  const short = {"as observed": "As observed", "unexpected 30% rise in energy prices": "Prices +30%", "20% cut in the support budget": "Budget −20%", "price rise and budget cut together": "Both"};
  const values = {};
  for (const sc of DATA.scenarios) {
    const rows = DATA.stress.filter(d => d.sc === sc && (country === "all" || d.c === country));
    values[sc] = Object.fromEntries(["fcrl", "std", "fc"].map(k => [k, agg(rows, k)]));
  }
  const W = 520, groupH = 74, top = 4, left = 104, right = 46, H = top + DATA.scenarios.length * groupH + 24;
  const s = svg(document.getElementById("c-stress"), W, H);
  const x = v => left + Math.min(v, 1) * (W - left - right);
  axis(s, x, top, H - 24, [0, 0.25, 0.5, 0.75, 1], pct);
  DATA.scenarios.forEach((sc, i) => {
    const y0 = top + i * groupH + 8;
    sv("text", {x: left - 10, y: y0 + 30, "text-anchor": "end", class: "lab"}, s, short[sc] || sc);
    ["fcrl", "std", "fc"].forEach((k, j) => {
      const d = values[sc][k];
      const v = d[stressMetric];
      if (v == null) return;
      const y = y0 + j * 20;
      const w = Math.max(0, x(v) - left);
      const g = sv("g", {class: "mark"}, s);
      sv("path", {d: `M${left},${y} h${Math.max(w - 4, 0)} a4,4 0 0 1 4,4 v8 a4,4 0 0 1 -4,4 h${-Math.max(w - 4, 0)} z`, fill: css("--" + k)}, g);
      sv("text", {x: x(v) + 6, y: y + 12, class: "val", "font-weight": k === "fcrl" ? 700 : 400}, g, pct(v));
      const hit = sv("rect", {x: left, y: y - 2, width: W - left, height: 20, fill: "transparent"}, g);
      bindTip(hit, e => showTip(e, short[sc] || sc, [[css("--" + k), pct1(v), NAME[k]], [null, pct1(d.overall), "total need covered"]]));
    });
  });
  table(document.getElementById("t-stress"), ["Scenario", "FCRL-energy", "Equal per poor resident", "Forecast need"], DATA.scenarios.map(sc => [short[sc] || sc, ...["fcrl", "std", "fc"].map(k => pct1(values[sc][k][stressMetric]))]));
}

function strip() {
  const data = regionsInScope();
  const rows = [["std", "Equal amount per resident at risk of poverty"], ["fc", "Proportional to forecast need"], ["fcrl", "FCRL-energy"]];
  document.getElementById("s-strip").textContent = `One dot per region and test season, ${scopeName()}. The black mark is the lowest coverage. Hover a dot to see the region; click to open it below.`;
  const W = 1060, rowH = 92, top = 18, left = 290, right = 20, H = top + rows.length * rowH + 26;
  const host = document.getElementById("c-strip");
  const s = svg(host, W, H);
  const x = v => left + Math.min(v, 1) * (W - left - right);
  axis(s, x, top - 10, H - 26, [0, 0.2, 0.4, 0.6, 0.8, 1], pct);
  const layer = sv("g", {class: highlight ? "dim" : ""}, s);
  const jitter = id => { let h = 0; for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) % 997; return (h / 997 - 0.5) * 44; };
  rows.forEach(([k, label], i) => {
    const cy = top + i * rowH + rowH / 2;
    sv("text", {x: left - 14, y: cy + 4, "text-anchor": "end", class: k === "fcrl" ? "lab" : "lab-2", "font-weight": k === "fcrl" ? 600 : 400}, s, label);
    const values = data.map(d => d.v[k]).filter(v => v != null);
    if (!values.length) return;
    const low = Math.min(...values);
    for (const d of data) {
      const v = d.v[k];
      if (v == null) continue;
      const on = highlight && d.r === highlight;
      const g = sv("g", {class: "mark" + (on ? " on" : "")}, layer);
      sv("circle", {cx: x(v), cy: cy + jitter(d.r + d.s), r: on ? 6.5 : 4.5, fill: css("--" + k), "fill-opacity": on ? 1 : 0.7, stroke: css("--surface-1"), "stroke-width": 1.5}, g);
      const hit = sv("circle", {cx: x(v), cy: cy + jitter(d.r + d.s), r: 12, fill: "transparent", style: "cursor:pointer"}, g);
      bindTip(hit, e => showTip(e, `${d.n} (${d.c}), ${d.s}`, rows.map(([kk, ll]) => [css("--" + kk), pct1(d.v[kk]), kk === "std" ? "equal per poor resident" : kk === "fc" ? "forecast need" : "FCRL-energy"])));
      hit.addEventListener("click", () => { selected = d.r; highlight = d.r; document.getElementById("region").value = d.r; strip(); region(); document.getElementById("region").scrollIntoView({behavior: "smooth", block: "center"}); });
    }
    sv("line", {x1: x(low), x2: x(low), y1: cy - 30, y2: cy + 30, stroke: css("--text-primary"), "stroke-width": 2.5}, s);
    sv("text", {x: x(low) - 6, y: cy - 34, "text-anchor": "end", class: "val"}, s, "lowest " + pct(low));
  });
}

function countryChart() {
  legend(document.getElementById("l-country"), ["std", "fcrl"]);
  const groups = {};
  for (const c of DATA.countries) { const rows = DATA.regions.filter(d => d.c === c.code); groups[c.code] = {std: agg(rows, "std").lowest, fcrl: agg(rows, "fcrl").lowest}; }
  const names = Object.fromEntries(DATA.countries.map(c => [c.code, c.name]));
  const list = Object.entries(groups).sort((a, b) => b[1].fcrl - a[1].fcrl);
  const W = 1060, rowH = 44, top = 6, left = 130, right = 30, H = top + list.length * rowH + 26;
  const s = svg(document.getElementById("c-country"), W, H);
  const x = v => left + v * (W - left - right) / 0.8;
  axis(s, x, top, H - 26, [0, 0.2, 0.4, 0.6, 0.8], pct);
  list.forEach(([code, g], i) => {
    const y = top + i * rowH + rowH / 2;
    const on = country === "all" || country === code;
    const grp = sv("g", {opacity: on ? 1 : 0.3}, s);
    sv("text", {x: left - 14, y: y + 4, "text-anchor": "end", class: "lab"}, grp, names[code] || code);
    sv("line", {x1: x(g.std), x2: x(g.fcrl), y1: y, y2: y, stroke: css("--muted-mark"), "stroke-width": 3, "stroke-linecap": "round"}, grp);
    sv("circle", {cx: x(g.std), cy: y, r: 6, fill: css("--std"), stroke: css("--surface-1"), "stroke-width": 2}, grp);
    sv("circle", {cx: x(g.fcrl), cy: y, r: 6, fill: css("--fcrl"), stroke: css("--surface-1"), "stroke-width": 2}, grp);
    const stdLeft = g.std <= g.fcrl;
    sv("text", {x: x(g.std) + (stdLeft ? -12 : 12), y: y + 4, "text-anchor": stdLeft ? "end" : "start", class: "lab-2"}, grp, pct(g.std));
    sv("text", {x: x(g.fcrl) + (stdLeft ? 12 : -12), y: y + 4, "text-anchor": stdLeft ? "start" : "end", class: "val", "font-weight": 700}, grp, pct(g.fcrl));
    const hit = sv("rect", {x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent"}, grp);
    bindTip(hit, e => showTip(e, names[code] || code, [[css("--fcrl"), pct1(g.fcrl), "FCRL-energy"], [css("--std"), pct1(g.std), "equal per poor resident"]]));
  });
  table(document.getElementById("t-country"), ["Country", "Equal per poor resident", "FCRL-energy"], list.map(([code, g]) => [names[code] || code, pct1(g.std), pct1(g.fcrl)]));
}

function regionList() {
  const select = document.getElementById("region");
  const seen = new Map();
  for (const d of regionsInScope()) if (!seen.has(d.r)) seen.set(d.r, d);
  const items = [...seen.values()].sort((a, b) => (a.c + a.n).localeCompare(b.c + b.n));
  select.replaceChildren();
  for (const d of items) el("option", {value: d.r}, select, `${d.n} (${d.c})`);
  if (!items.find(d => d.r === selected)) { const preferred = items.find(d => d.r === "EL30"); selected = preferred ? preferred.r : (items.length ? items[0].r : null); }
  if (highlight && !items.find(d => d.r === highlight)) highlight = null;
  if (selected) select.value = selected;
}

function region() {
  legend(document.getElementById("l-region"), ["fcrl", "std", "fc"]);
  const rows = DATA.regions.filter(d => d.r === selected).sort((a, b) => a.s.localeCompare(b.s));
  const W = 560, groupH = 66, top = 4, left = 120, right = 44, H = top + rows.length * groupH + 24;
  const s = svg(document.getElementById("c-region"), W, H);
  const x = v => left + Math.min(v, 1) * (W - left - right);
  axis(s, x, top, H - 24, [0, 0.25, 0.5, 0.75, 1], pct);
  rows.forEach((d, i) => {
    const y0 = top + i * groupH + 6;
    sv("text", {x: left - 10, y: y0 + 26, "text-anchor": "end", class: "lab"}, s, d.s);
    ["fcrl", "std", "fc"].forEach((k, j) => {
      const v = d.v[k];
      const y = y0 + j * 18;
      const w = Math.max(0, x(v) - left);
      const g = sv("g", {}, s);
      sv("path", {d: `M${left},${y} h${Math.max(w - 4, 0)} a4,4 0 0 1 4,4 v6 a4,4 0 0 1 -4,4 h${-Math.max(w - 4, 0)} z`, fill: css("--" + k)}, g);
      sv("text", {x: x(v) + 6, y: y + 11, class: "val", "font-weight": k === "fcrl" ? 700 : 400}, g, pct(v));
      const hit = sv("rect", {x: left, y: y - 2, width: W - left, height: 18, fill: "transparent"}, g);
      bindTip(hit, e => showTip(e, `${d.n}, ${d.s}`, [[css("--" + k), pct1(v), NAME[k]]]));
    });
  });
  const host = document.getElementById("statements");
  host.replaceChildren();
  const notes = DATA.statements.filter(n => n.r === selected);
  if (!notes.length) el("p", {class: "statement"}, host, "No allocation statement for this region.");
  for (const n of notes) { const p = el("div", {class: "statement"}, host); el("b", {}, p, n.s); p.appendChild(document.createTextNode(n.t)); }
}

function chips() {
  const host = document.getElementById("chips");
  host.replaceChildren();
  const all = [{code: "all", name: "All"}, ...DATA.countries];
  for (const c of all) {
    const b = el("button", {class: "chip", type: "button", "aria-pressed": String(country === c.code)}, host, c.name);
    b.style.marginRight = "6px";
    b.addEventListener("click", () => { country = c.code; chips(); tiles(); worst(); stress(); regionList(); strip(); countryChart(); region(); });
  }
}

function render() { tiles(); chips(); worst(); stress(); regionList(); strip(); countryChart(); region(); }
document.getElementById("region").addEventListener("change", e => { selected = e.target.value; highlight = e.target.value; strip(); region(); });
const themeButton = document.getElementById("theme");
function themeLabel() { const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; themeButton.textContent = dark ? "Light mode" : "Dark mode"; }
themeButton.addEventListener("click", () => { const dark = themeButton.textContent === "Dark mode"; document.documentElement.dataset.theme = dark ? "dark" : "light"; themeLabel(); render(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { themeLabel(); render(); });
const foot = document.getElementById("foot");
foot.appendChild(document.createTextNode((DATA.meta.source === "synthetic" ? "Synthetic demonstration data. " : "Data: Eurostat (degree days, population, poverty, energy prices). ") + "Coverage is support divided by the estimated energy need of residents at risk of poverty. "));
el("a", {href: "https://github.com/uomoathens/fcrl-energy"}, foot, "Code, method and limitations on GitHub");
themeLabel();
render();
</script>
</body>
</html>
"""
