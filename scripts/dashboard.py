"""
Local training dashboard: a wandb-free way to watch runs.

Reads the JSONL metric files written by nanochat.common.FileLogger
(<NANOCHAT_BASE_DIR>/metrics/<project>/<run>.jsonl) and serves a single
page of live-updating charts. No dependencies beyond the standard library;
Chart.js is loaded by the browser from a CDN.

On the training node:
    python -m scripts.dashboard            # serves on http://localhost:8080

From your laptop, forward the port over SSH and open the URL in a browser:
    ssh -N -L 8080:localhost:8080 ubuntu@<lambda-ip>
    open http://localhost:8080

Options:
    --port 8080     port to listen on
    --host 127.0.0.1  bind address (keep it local and use the SSH tunnel)
    --metrics-dir   override the metrics directory
"""

import os
import json
import glob
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Data loading

X_KEYS = ["step", "total_training_time", "total_training_flops"]

def load_runs(metrics_dir):
    """
    Returns {"<project>/<run>": {"config": {...}, "rows": [{...}, ...]}}.
    Only scalar numeric metrics are kept in rows; strings and nested dicts are dropped.
    """
    runs = {}
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*", "*.jsonl"))):
        project = os.path.basename(os.path.dirname(path))
        run = os.path.splitext(os.path.basename(path))[0]
        config, rows = {}, []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue # a row that is still being written
                if "_config" in obj:
                    config = obj["_config"]
                    continue
                row = {k: v for k, v in obj.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                if row:
                    rows.append(row)
        runs[f"{project}/{run}"] = {"config": config, "rows": rows, "mtime": os.path.getmtime(path)}
    return runs

# -----------------------------------------------------------------------------
# Web page

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>nanochat dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px; background: #fafafa; color: #222; }
  @media (prefers-color-scheme: dark) { body { background: #111; color: #ddd; } .card { background: #1c1c1c !important; border-color: #333 !important; } }
  header { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 12px; }
  h1 { font-size: 18px; margin: 0; }
  label { font-size: 13px; }
  select, input { font-size: 13px; }
  #runs { display: flex; flex-wrap: wrap; gap: 8px; font-size: 13px; }
  #runs label { display: flex; gap: 4px; align-items: center; padding: 2px 8px; border-radius: 12px; border: 1px solid #8884; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 8px; }
  .card h2 { font-size: 13px; font-weight: 600; margin: 0 0 4px 4px; font-family: ui-monospace, monospace; }
  canvas { width: 100% !important; height: 220px !important; }
  #status { font-size: 12px; opacity: .7; }
</style>
</head>
<body>
<header>
  <h1>nanochat dashboard</h1>
  <label>x-axis
    <select id="xkey">
      <option value="step">step</option>
      <option value="total_training_time">total_training_time (s)</option>
      <option value="total_training_flops">total_training_flops</option>
    </select>
  </label>
  <label><input type="checkbox" id="logy"> log y</label>
  <label>refresh <input type="number" id="interval" value="5" min="1" style="width:4em"> s</label>
  <span id="status"></span>
</header>
<div id="runs"></div>
<div class="grid" id="grid"></div>
<script>
const COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
const X_KEYS = new Set(["step","total_training_time","total_training_flops","_time"]);
let charts = {};       // metric -> Chart
let hidden = new Set(); // run names the user has unchecked
let runOrder = [];      // stable colour assignment

function fmt(v) {
  if (Math.abs(v) >= 1e15 || (Math.abs(v) < 1e-3 && v !== 0)) return v.toExponential(2);
  return Number.isInteger(v) ? v.toString() : v.toPrecision(5);
}

function buildRunToggles(runs) {
  const el = document.getElementById("runs");
  el.innerHTML = "";
  for (const name of Object.keys(runs)) {
    if (!runOrder.includes(name)) runOrder.push(name);
    const color = COLORS[runOrder.indexOf(name) % COLORS.length];
    const last = runs[name].rows.at(-1) || {};
    const lab = document.createElement("label");
    lab.style.borderColor = color;
    lab.innerHTML = `<input type="checkbox" ${hidden.has(name) ? "" : "checked"}> <span style="color:${color}">●</span> ${name} <span style="opacity:.6">step ${last.step ?? "-"}</span>`;
    lab.querySelector("input").onchange = e => { e.target.checked ? hidden.delete(name) : hidden.add(name); render(); };
    el.appendChild(lab);
  }
}

function metricsOf(runs) {
  const keys = new Set();
  for (const r of Object.values(runs)) for (const row of r.rows) for (const k of Object.keys(row)) if (!X_KEYS.has(k)) keys.add(k);
  // put the headline metrics first
  const pri = ["val/bpb","core_metric","train/loss","train/mfu","train/tok_per_sec","chatcore_metric","reward"];
  return [...keys].sort((a,b) => (pri.indexOf(a)+1 || 99) - (pri.indexOf(b)+1 || 99) || a.localeCompare(b));
}

function series(rows, xkey, ykey) {
  // rows that lack xkey (e.g. an eval row without total_training_time) borrow it from the
  // most recent row at the same step that has it, so every metric works on every x-axis
  const pts = [];
  const xAtStep = new Map();
  for (const row of rows) {
    if (xkey in row && "step" in row) xAtStep.set(row.step, row[xkey]);
    if (!(ykey in row)) continue;
    const x = xkey in row ? row[xkey] : xAtStep.get(row.step);
    if (x !== undefined) pts.push({x, y: row[ykey]});
  }
  return pts;
}

let latest = {};
function render() {
  const runs = latest;
  const xkey = document.getElementById("xkey").value;
  const logy = document.getElementById("logy").checked;
  const grid = document.getElementById("grid");
  const wanted = metricsOf(runs);
  // drop charts for metrics that vanished
  for (const m of Object.keys(charts)) if (!wanted.includes(m)) { charts[m].destroy(); delete charts[m]; document.getElementById("card-"+CSS.escape(m))?.remove(); }
  for (const m of wanted) {
    const datasets = [];
    for (const name of Object.keys(runs)) {
      if (hidden.has(name)) continue;
      const pts = series(runs[name].rows, xkey, m);
      if (!pts.length) continue;
      const color = COLORS[runOrder.indexOf(name) % COLORS.length];
      datasets.push({label: name, data: pts, borderColor: color, backgroundColor: color, borderWidth: 1.5, pointRadius: pts.length > 200 ? 0 : 2, tension: 0});
    }
    const id = "card-" + m;
    let card = document.getElementById(id);
    if (!card) {
      card = document.createElement("div"); card.className = "card"; card.id = id;
      card.innerHTML = `<h2></h2><canvas></canvas>`;
      grid.appendChild(card);
    }
    const lastVals = datasets.map(d => `${d.label.split("/").pop()}=${fmt(d.data.at(-1).y)}`).join("  ");
    card.querySelector("h2").textContent = `${m}   ${lastVals}`;
    if (!charts[m]) {
      charts[m] = new Chart(card.querySelector("canvas"), {
        type: "line",
        data: {datasets},
        options: {
          animation: false, responsive: true, maintainAspectRatio: false, parsing: false, normalized: true,
          interaction: {mode: "nearest", intersect: false},
          plugins: {legend: {display: false}},
          scales: {x: {type: "linear", title: {display: true, text: xkey}}, y: {type: logy ? "logarithmic" : "linear"}},
        },
      });
    } else {
      const c = charts[m];
      c.data.datasets = datasets;
      c.options.scales.x.title.text = xkey;
      c.options.scales.y.type = logy ? "logarithmic" : "linear";
      c.update("none");
    }
  }
}

async function refresh() {
  try {
    const res = await fetch("/api/runs", {cache: "no-store"});
    latest = await res.json();
    buildRunToggles(latest);
    render();
    document.getElementById("status").textContent = `${Object.keys(latest).length} run(s) · updated ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    document.getElementById("status").textContent = "fetch failed: " + e;
  }
  setTimeout(refresh, 1000 * Math.max(1, +document.getElementById("interval").value || 5));
}
document.getElementById("xkey").onchange = render;
document.getElementById("logy").onchange = render;
refresh();
</script>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# Server

def make_handler(metrics_dir):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/runs"):
                body = json.dumps(load_runs(metrics_dir)).encode()
                ctype = "application/json"
            elif self.path == "/" or self.path.startswith("/index"):
                body = PAGE.encode()
                ctype = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass # keep the terminal quiet; the training log is what matters

    return Handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a local dashboard of nanochat training metrics")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on")
    parser.add_argument("--metrics-dir", type=str, default=None, help="directory of metrics (default: <base_dir>/metrics)")
    args = parser.parse_args()
    metrics_dir = args.metrics_dir or os.path.join(get_base_dir(), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(metrics_dir))
    print(f"Serving metrics from {metrics_dir}")
    print(f"Dashboard at http://{args.host}:{args.port}")
    print(f"From your laptop: ssh -N -L {args.port}:localhost:{args.port} <user>@<host>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
