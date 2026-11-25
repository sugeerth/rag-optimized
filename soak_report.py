"""Turn a soak run (summary + time series) into report.md and a self-contained report.html.

    python soak_report.py reports/soak-1h     # re-render from summary.json + timeseries.jsonl
"""

import json
import sys
from pathlib import Path

CHARTS = [
    ("Throughput", "requests per second", [("rps", "req/s")], ""),
    ("Latency, client side", "milliseconds, per interval", [("p50", "p50"), ("p99", "p99")], " ms"),
    ("Server memory", "resident set size, MB", [("rss_mb", "RSS")], " MB"),
    ("Cache hit rate", "share of queries served from cache", [("cache_hit_rate", "hit rate")], "%"),
    ("Retrieval quality", "checked against ground truth", [("recall_at_5", "recall@5"), ("answer_accuracy", "answer accuracy")], "%"),
    ("Index tombstones", "dead chunks awaiting compaction", [("tombstones", "tombstones")], ""),
]

PAGE = """<title>Soak Test Report</title>
<style>
:root{color-scheme:light;--bg:#f6f6f4;--surface:#ffffff;--text:#191919;--text2:#666666;--grid:#e6e6e3;--border:#dededa;
--s1:#257de0;--s2:#f47230;--good:#008a17;--bad:#c62828}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#111111;--surface:#191919;
--text:#efefef;--text2:#9c9c9c;--grid:#2c2c2c;--border:#333333;--s1:#338ced;--s2:#e76624;--good:#0aac23;--bad:#eb6263}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111111;--surface:#191919;--text:#efefef;--text2:#9c9c9c;--grid:#2c2c2c;
--border:#333333;--s1:#338ced;--s2:#e76624;--good:#0aac23;--bad:#eb6263}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:17px;margin:36px 0 12px}
.sub{color:var(--text2);margin:0 0 24px}
.verdict{display:inline-flex;gap:8px;align-items:center;font-weight:600;padding:6px 12px;border-radius:999px;border:1px solid var(--border);background:var(--surface);margin-bottom:20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile b{display:block;font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.tile span{color:var(--text2);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:12px}
.chart{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px 8px;position:relative}
.chart h3{margin:0;font-size:14px}.chart p{margin:0 0 6px;color:var(--text2);font-size:12px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--text2);margin-bottom:2px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{fill:var(--text2);font-size:11px;font-variant-numeric:tabular-nums}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.14);white-space:nowrap;display:none;z-index:2}
table{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;font-size:14px}
th,td{text-align:right;padding:8px 12px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--text2);font-weight:500;font-size:12px}tr:last-child td{border-bottom:0}
.wrap{overflow-x:auto}code{font-size:13px}
</style>
<main>
<h1>Soak test: __TITLE__</h1>
<p class="sub">__SUB__</p>
<div class="verdict"><span style="color:var(--__VCOLOR__)">__VICON__</span> __VERDICT__</div>
<div class="tiles">__TILES__</div>
<h2>Over the run</h2>
<div class="grid" id="charts"></div>
<h2>Did anything drift?</h2>
<p class="sub">Average of the first 10% of the run against the last 10%. A leak or a slowly degrading index shows up here.</p>
<div class="wrap"><table><tr><th>Metric</th><th>First 10%</th><th>Last 10%</th><th>Change</th></tr>__DRIFT__</table></div>
<h2>Workload</h2>
<p class="sub">__WORKLOAD__</p>
</main>
<script>
const ROWS = __ROWS__, CHARTS = __CHARTS__;
const fmt = (v, unit) => v == null ? "–" : (unit === "%" ? (v * 100).toFixed(1) + "%" : (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : (+v.toFixed(2)).toString()) + unit);
const mins = t => (t / 60).toFixed(1) + " min";
function draw(host, [title, sub, series, unit]) {
  const W = 520, H = 190, L = 44, R = 12, T = 8, B = 22, cols = ["var(--s1)", "var(--s2)"];
  const xs = ROWS.map(r => r.t), tmax = xs[xs.length - 1] || 1;
  let ymax = Math.max(...series.flatMap(([k]) => ROWS.map(r => r[k] ?? 0)));
  ymax = unit === "%" ? 1 : (ymax || 1) * 1.12;
  const X = t => L + (t / tmax) * (W - L - R), Y = v => T + (1 - v / ymax) * (H - T - B);
  let g = "";
  for (let i = 0; i <= 4; i++) { const v = ymax * i / 4, y = Y(v);
    g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="var(--grid)" stroke-width="1"/><text x="${L - 6}" y="${y + 4}" text-anchor="end">${fmt(v, unit === "%" ? "%" : "").replace(".0%", "%")}</text>`; }
  for (let i = 0; i <= 4; i++) { const t = tmax * i / 4;
    g += `<text x="${X(t)}" y="${H - 5}" text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}">${Math.round(t / 60)} min</text>`; }
  series.forEach(([k], si) => { let d = "", pen = false;
    ROWS.forEach(r => { if (r[k] == null) { pen = false; return; } d += (pen ? "L" : "M") + X(r.t).toFixed(1) + " " + Y(r[k]).toFixed(1); pen = true; });
    g += `<path d="${d}" fill="none" stroke="${cols[si]}" stroke-width="2" stroke-linejoin="round"/>`; });
  const legend = series.length > 1 ? `<div class="legend">${series.map(([, n], i) => `<span><i style="background:${cols[i]}"></i>${n}</span>`).join("")}</div>` : "";
  const el = document.createElement("div"); el.className = "chart";
  el.innerHTML = `<h3>${title}</h3><p>${sub}</p>${legend}<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">${g}<line class="x" y1="${T}" y2="${H - B}" stroke="var(--text2)" stroke-width="1" visibility="hidden"/><rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/></svg><div class="tip"></div>`;
  host.appendChild(el);
  const svg = el.querySelector("svg"), cross = el.querySelector(".x"), tip = el.querySelector(".tip");
  svg.addEventListener("pointermove", e => { const b = svg.getBoundingClientRect(), t = ((e.clientX - b.left) / b.width * W - L) / (W - L - R) * tmax;
    let r = ROWS[0]; for (const row of ROWS) if (Math.abs(row.t - t) < Math.abs(r.t - t)) r = row;
    cross.setAttribute("x1", X(r.t)); cross.setAttribute("x2", X(r.t)); cross.setAttribute("visibility", "visible");
    tip.innerHTML = `<b>${mins(r.t)}</b><br>` + series.map(([k, n], i) => `<i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${cols[i]};margin-right:5px"></i>${n}: <b>${fmt(r[k], unit)}</b>`).join("<br>");
    tip.style.display = "block"; const px = (e.clientX - b.left) + 14; tip.style.left = Math.min(px, b.width - tip.offsetWidth) + "px"; tip.style.top = "56px"; });
  svg.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.style.display = "none"; });
}
CHARTS.forEach(c => draw(document.getElementById("charts"), c));
</script>
"""


def _dur(s: float) -> str:
    return f"{s / 3600:.2f} hours" if s >= 3600 else f"{s / 60:.1f} minutes"


def _drift_rows(summary: dict) -> list[tuple[str, float, float, str]]:
    names = {"rps": "Throughput (req/s)", "p50_ms": "p50 latency (ms)", "p99_ms": "p99 latency (ms)",
             "rss_mb": "Server memory (MB)", "recall_at_5": "Recall@5"}
    out = []
    for k, (a, b) in summary["drift_first_vs_last_10pct"].items():
        change = f"{(b - a) / a * 100:+.1f}%" if a else "–"
        out.append((names.get(k, k), a, b, change))
    return out


def _verdict(s: dict) -> tuple[bool, str]:
    ok = s["errors"] == 0 and s["wrong_answers_to_safety_checks"] == 0
    if ok:
        return True, f"Passed: 0 errors and 0 failed safety checks across {s['requests']:,} requests"
    return False, (f"Failed: {s['errors']:,} request errors and {s['wrong_answers_to_safety_checks']:,} "
                   f"failed safety checks across {s['requests']:,} requests")


def write(out: Path, summary: dict, rows: list[dict]) -> None:
    s = summary
    ok, verdict = _verdict(s)
    lat = s["latency_ms"]
    tiles = [(f"{s['requests']:,}", "requests served"), (f"{s['avg_rps']:,.0f}", "requests / second, average"),
             (f"{lat['p50']:.2f} ms", "p50 latency"), (f"{lat['p99']:.2f} ms", "p99 latency"),
             (f"{s['errors']:,}", "errors"), (f"{s['recall_at_5'] * 100:.1f}%", "recall@5"),
             (f"{s['answer_accuracy'] * 100:.1f}%", "answer accuracy"), (f"{s['cache_hit_rate'] * 100:.1f}%", "cache hit rate")]
    workload = (f"{s['workers']} closed-loop query clients against one uvicorn worker on a {s['host']['cpus']}-CPU box, "
                f"over {s['corpus_docs']:,} documents ({s['corpus_chunks']:,} chunks). {s['quality_checked_queries']:,} answers "
                f"were checked against ground truth. Alongside: {s['reingests_during_run']:,} document re-ingests "
                f"(each one invalidates the cache; {s['index_compactions']} index compactions), "
                f"{s['lora_finetunes_during_run']:,} LoRA fine-tunes, {s['injection_or_pii_blocked']:,} prompt-injection or "
                f"PII queries blocked, and {s['offtopic_declined']:,} off-topic questions declined instead of answered. "
                "Latency is measured at the client, including HTTP.")
    drift = _drift_rows(s)
    html = (PAGE.replace("__TITLE__", _dur(s["duration_s"]))
            .replace("__SUB__", f"Started {s['started_utc']} UTC. Pure-NumPy hybrid RAG engine, no GPU, no vector database.")
            .replace("__VCOLOR__", "good" if ok else "bad").replace("__VICON__", "&#10003;" if ok else "&#10007;")
            .replace("__VERDICT__", verdict)
            .replace("__TILES__", "".join(f'<div class="tile"><b>{v}</b><span>{k}</span></div>' for v, k in tiles))
            .replace("__DRIFT__", "".join(f"<tr><td>{n}</td><td>{a:,}</td><td>{b:,}</td><td>{c}</td></tr>" for n, a, b, c in drift))
            .replace("__WORKLOAD__", workload)
            .replace("__ROWS__", json.dumps(rows, separators=(",", ":")))
            .replace("__CHARTS__", json.dumps(CHARTS)))
    (out / "report.html").write_text("<!doctype html>\n<meta charset=\"utf-8\">\n"
                                     "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n" + html)

    md = [f"# Soak test: {_dur(s['duration_s'])}", "", f"**{verdict}.**", "",
          f"Started {s['started_utc']} UTC. {workload}", "",
          "| Metric | Value |", "| --- | --- |",
          f"| Requests served | {s['requests']:,} |", f"| Average throughput | {s['avg_rps']:,} req/s |",
          f"| Errors | {s['errors']:,} |", f"| Failed safety checks | {s['wrong_answers_to_safety_checks']:,} |",
          f"| Latency p50 / p95 / p99 / p99.9 / max | {lat['p50']} / {lat['p95']} / {lat['p99']} / {lat['p999']} / {lat['max']} ms |",
          f"| Recall@5 | {s['recall_at_5'] * 100:.1f}% |", f"| Answer accuracy | {s['answer_accuracy'] * 100:.1f}% |",
          f"| Cache hit rate | {s['cache_hit_rate'] * 100:.1f}% |",
          f"| Server memory, min to max | {s['rss_mb_min_max'][0]} to {s['rss_mb_min_max'][1]} MB |", "",
          "## Drift: first 10% of the run vs last 10%", "", "| Metric | First 10% | Last 10% | Change |", "| --- | --- | --- | --- |"]
    md += [f"| {n} | {a:,} | {b:,} | {c} |" for n, a, b, c in drift]
    if s["sample_failures"]:
        md += ["", "## Sample failures", ""] + [f"- `{f}`" for f in s["sample_failures"]]
    md += ["", "Charts: open `report.html`. Raw data: `timeseries.jsonl`, `summary.json`.", ""]
    (out / "report.md").write_text("\n".join(md))


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/soak")
    write(d, json.loads((d / "summary.json").read_text()),
          [json.loads(line) for line in (d / "timeseries.jsonl").read_text().splitlines() if line])
    print(f"wrote {d}/report.html and {d}/report.md")
