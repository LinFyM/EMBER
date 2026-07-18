"""Self-contained local HTML report for the canonical LIBERO-90 manifest."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_report(
    manifest: Mapping[str, Any], quality_report: Mapping[str, Any]
) -> str:
    """Render a dependency-free report that works from a local file or HTTP server."""

    summary = manifest["summary"]
    tasks = manifest["tasks"]
    scenes = sorted({str(task["scene"]) for task in tasks})
    status = html.escape(str(quality_report.get("status", "unknown")))
    reseal = manifest.get("split_reseal", {})
    split_audit = reseal.get("selection_diagnostics", {}).get("audit", {})
    task_json = _json_for_script(tasks)
    scene_options = "".join(
        f'<option value="{html.escape(scene)}">{html.escape(scene)}</option>'
        for scene in scenes
    )
    card_values = [
        ("Tasks", summary["tasks"]),
        ("Source", summary["source"]),
        ("Validation", summary["validation"]),
        ("Held-out", summary["held_out"]),
        ("Demos", summary.get("demonstrations", sum(task["demonstrations"]["count"] for task in tasks))),
        ("Frames", summary.get("frames", sum(task["demonstrations"]["steps"] for task in tasks))),
    ]
    if split_audit:
        card_values.extend(
            [
                ("Novel eval compositions", split_audit["novel_full_composition_count"]),
                ("Same-scene hard negatives", split_audit["same_scene_hard_negative_count"]),
                (
                    "Min source role count",
                    split_audit["minimum_observed_source_count_for_evaluation_roles"],
                ),
            ]
        )
    cards = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{int(value)}</strong></div>'
        for label, value in card_values
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMBER · LIBERO-90 authority</title>
<style>
:root {{ color-scheme: dark; --bg:#0b0f14; --panel:#131a22; --line:#273241;
  --text:#e8edf3; --muted:#96a3b3; --accent:#5ad1a6; --warn:#ffc266; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif }}
main {{ max-width:1500px; margin:auto; padding:28px }} h1 {{ margin:0 0 4px; font-size:26px }}
.sub {{ color:var(--muted); margin-bottom:20px }} .cards {{ display:grid;
  grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:16px 0 }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px }}
.card span {{ color:var(--muted); display:block }} .card strong {{ font-size:22px }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:8px; margin:16px 0 }} input,select,a.button {{
  background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:7px;
  padding:8px 10px; text-decoration:none }} input {{ min-width:280px; flex:1 }}
.badge {{ border:1px solid var(--line); border-radius:99px; padding:3px 9px; color:var(--accent) }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:10px }}
table {{ width:100%; border-collapse:collapse; min-width:1450px }} th,td {{ text-align:left;
  border-bottom:1px solid var(--line); padding:9px 10px; vertical-align:top }}
th {{ position:sticky; top:0; background:#19212b; color:var(--muted); font-weight:600 }}
tr:hover {{ background:#111923 }} .muted {{ color:var(--muted) }} .num {{ font-variant-numeric:tabular-nums }}
.source {{ color:#5ad1a6 }} .validation {{ color:#70b7ff }} .held_out {{ color:#d7a7ff }}
footer {{ color:var(--muted); margin-top:14px }}
</style>
</head>
<body><main>
<h1>EMBER · Canonical LIBERO-90 authority</h1>
<div class="sub">Pinned data, task, BDDL, init-state, camera/controller and split audit.
  Quality: <span class="badge">{status}</span></div>
<section class="cards">{cards}</section>
<div class="toolbar">
  <input id="query" type="search" placeholder="Search task or instruction">
  <select id="split-filter"><option value="">All splits</option><option>source</option>
    <option>validation</option><option value="held_out">held-out</option></select>
  <select id="scene"><option value="">All scenes</option>{scene_options}</select>
  <a class="button" href="manifest.json">manifest.json</a>
  <a class="button" href="normalization_source_only.json">normalization</a>
  <a class="button" href="quality_report.json">quality report</a>
</div>
<div class="table-wrap"><table><thead><tr><th>#</th><th>Split</th><th>Scene</th><th>Task</th>
  <th>Instruction</th><th>Order</th><th>Role atoms</th><th>Demos</th><th>Frames</th><th>Audit</th></tr></thead><tbody id="rows"></tbody></table></div>
<footer><span id="visible"></span> · Values for validation and held-out HDF5 files were not read;
only schema and metadata were audited. Source normalization is isolated in its own artifact.</footer>
</main>
<script>
const tasks={task_json};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
const q=document.querySelector('#query'), split=document.querySelector('#split-filter'), scene=document.querySelector('#scene');
function draw() {{
  const needle=q.value.trim().toLowerCase();
  const shown=tasks.filter(t=>{{const f=t.specification_factors||{{primitive_role_atoms:[]}}; return (!split.value||t.split===split.value)&&(!scene.value||t.scene===scene.value)&&
    (!needle||(t.task_name+' '+t.language+' '+f.primitive_role_atoms.join(' ')).toLowerCase().includes(needle));}});
  document.querySelector('#rows').innerHTML=shown.map(t=>{{const f=t.specification_factors||{{order_signature:'—',primitive_role_atoms:[]}}; return `<tr><td class="num">${{t.task_index}}</td>
    <td class="${{esc(t.split)}}">${{esc(t.split)}}</td><td>${{esc(t.scene)}}</td>
    <td>${{esc(t.task_name)}}</td><td>${{esc(t.language)}}</td>
    <td>${{esc(f.order_signature)}}</td><td class="muted">${{esc(f.primitive_role_atoms.join(' · '))}}</td>
    <td class="num">${{t.demonstrations.count}}</td><td class="num">${{t.demonstrations.steps.toLocaleString()}}</td>
    <td>${{esc(t.quality.status)}}${{t.quality.warning_count?' · '+t.quality.warning_count+' note':''}}</td></tr>`;}}).join('');
  document.querySelector('#visible').textContent=`${{shown.length}} / ${{tasks.length}} tasks`;
}}
[q,split,scene].forEach(x=>x.addEventListener('input',draw)); draw();
</script></body></html>\n"""
