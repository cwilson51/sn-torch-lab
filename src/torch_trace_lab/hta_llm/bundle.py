"""Assemble the analysis bundle (spec §8) from allowlisted fields only (§8a).

``build`` produces ``bundle.json`` from a fixed schema; ``render_markdown`` renders
``bundle.md`` from that JSON and nothing else, so both carry exactly the same data.
"""

import re
from dataclasses import asdict

from torch_trace_lab.hta_llm import tables
from torch_trace_lab.hta_llm.context import RunContext
from torch_trace_lab.hta_llm.heuristics import Flag
from torch_trace_lab.hta_llm.normalize import (CATEGORIES, RunMetrics, category_comparison,
                                               kernel_comparison, step_time_differences)

SCHEMA = "torch-lab-hta-bundle/2"
PATHLIKE = re.compile(r"(?:[A-Za-z]:)?(?:/[^/\s]+){2,}/")

METRIC_FIELDS = [
    "steps", "step_time_ms", "samples_per_s", "gpu_idle_pct", "gpu_compute_pct",
    "gpu_non_compute_pct", "idle_ms_per_step", "host_wait_pct_of_idle",
    "host_wait_ms_per_step", "kernel_wait_pct_of_idle",
    "other_idle_pct_of_idle", "kernel_ms_per_step", "top_kernel", "top_kernel_pct", "memcpy_pct",
    "gemm_conv_precision_pct", "categories", "category_pct", "kernels_per_step",
    "median_kernel_us", "median_launch_cpu_us",
    "median_launch_delay_us", "short_kernel_pct",
]

DESCRIPTIONS = {
    "temporal": "How the GPU timeline splits into idle, compute kernels, and non-compute "
                "(memory copy/set) kernels.",
    "idle": "Why the GPU was idle: host_wait = waiting for the CPU to launch work; "
            "kernel_wait = short gaps between back-to-back kernels; other = anything else.",
    "kernels": "GPU kernel time per step, first by category, then the top kernels by name "
               "(the rest are collapsed into '(remaining kernels)'). Categories line up across "
               "eager and torch.compile runs, whose kernel names differ; 'uncategorized' holds "
               "kernels that fit no category. Category and top-kernel tables are separate "
               "views of the same time: don't add them together.",
    "launch": "CPU-side kernel launches: how long kernels run vs. how long launches take. "
              "Launch delay is the time from a launch call to the kernel starting on the GPU; "
              "a large delay means the CPU is running ahead of a busy GPU (queued work), "
              "not that launches are slow.",
    "derived": "Computed by torch-lab, not by HTA. Throughput needs the batch size. "
               "Deltas are against the baseline run, in ms and %.",
    "differences": "Each cell is the column run's step time minus the row run's step time; "
                   "the % is relative to the row run. Negative means the column run is faster. "
                   "Quote these instead of subtracting step times.",
    "shared": "Kernels from any run's top list that occur in at least two runs. Values are "
              "each run's full ms/step for that kernel, even when it is outside that run's top "
              "list; '–' means the kernel does not occur in that run at all.",
}


def _strip_path(name: str) -> str:
    return PATHLIKE.sub("", name) if PATHLIKE.search(name) else name


def build(*, scenario: str, metrics: list[RunMetrics], contexts: list[RunContext],
          flags: dict[str, list[Flag]], baseline: int, config_diff: list[dict],
          distinguishable: bool, data_quality: list[str], cluster: str | None,
          kernel_names: dict[str, str]) -> dict:
    runs = []
    for m, c in zip(metrics, contexts):
        runs.append({
            "label": m.label,
            "facts": dict(c.facts),
            "config": {k: dict(v) for k, v in c.config.items()},
            "notes": c.notes,
            "metrics": {f: getattr(m, f) for f in METRIC_FIELDS},
            "kernels": [asdict(k) for k in m.kernels],
            "flags": [f.to_dict() for f in flags.get(m.label, [])],
            "unavailable": dict(m.unavailable),
        })
    used = {k["short"] for r in runs for k in r["kernels"]}
    names = {short: _strip_path(full)[:300] for full, short in kernel_names.items()
             if short in used}
    base = metrics[baseline]
    derived = []
    for m in metrics:
        row = {"run": m.label, "step_time_ms": m.step_time_ms, "samples_per_s": m.samples_per_s}
        if len(metrics) > 1:
            if m.step_time_ms and base.step_time_ms:
                row["step_time_vs_baseline_ms"] = round(m.step_time_ms - base.step_time_ms, 2)
                row["step_time_vs_baseline_pct"] = round(
                    100 * (m.step_time_ms - base.step_time_ms) / base.step_time_ms, 1)
                row["speedup_vs_baseline"] = round(base.step_time_ms / m.step_time_ms, 2)
            else:
                row["step_time_vs_baseline_ms"] = row["step_time_vs_baseline_pct"] = None
                row["speedup_vs_baseline"] = None
        derived.append(row)
    return {
        "schema": SCHEMA,
        "scenario": scenario,
        "cluster": cluster,
        "baseline": base.label if len(metrics) > 1 else None,
        "runs": runs,
        "config_diff": config_diff if len(metrics) > 1 else [],
        "configs_distinguishable": distinguishable if len(metrics) > 1 else None,
        "kernel_comparison": kernel_comparison(metrics, baseline) if len(metrics) > 1 else [],
        "category_comparison": category_comparison(metrics, baseline) if len(metrics) > 1 else [],
        "derived": derived,
        "step_time_differences": step_time_differences(metrics) if len(metrics) > 1 else [],
        "data_quality": list(data_quality),
        "kernel_names": names,
    }


def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return tables._cell(value)


def _config_cell(entry: dict) -> str:
    source = entry["source"]
    if source == "inferred" and entry.get("confidence"):
        source = f"inferred, {entry['confidence']}"
    return f"{_fmt(entry['value'])} ({source})"


def _diff_cell(row: dict, label: str) -> str:
    value = _fmt(row["values"][label])
    source = row.get("sources", {}).get(label)
    if row["source"] == "mixed" and source not in (None, "unknown"):
        return f"{value} ({source})"
    return value


def render_markdown(b: dict) -> str:
    runs = b["runs"]
    labels = [r["label"] for r in runs]
    out = ["# Trace analysis bundle", ""]

    out += ["## 1. Context", "",
            f"- Scenario: {b['scenario']} "
            + ("(single run, single GPU)" if b["scenario"] == "S1"
               else f"(comparison of {len(runs)} single-GPU runs; baseline: {b['baseline']})")]
    if b["cluster"]:
        out.append(f"- Cluster: {b['cluster']}")
    out.append("")
    fields = []
    for r in runs:
        for k in [*r["facts"], *r["config"]]:
            if k not in fields:
                fields.append(k)
    rows = []
    for k in fields:
        row = {"field": k}
        for r in runs:
            if k in r["config"]:
                row[r["label"]] = _config_cell(r["config"][k])
            elif k in r["facts"]:
                row[r["label"]] = f"{_fmt(r['facts'][k])} (trace)"
            else:
                row[r["label"]] = "unknown"
        rows.append(row)
    out += [tables.render(rows, ["field", *labels]), ""]
    out += ["Sources: trace = trace metadata; embedded = the trace's torch_lab config; "
            "user = the user's --meta file; inferred = derived by torch-lab from the trace "
            "contents, with a confidence. Inferred values are evidence, not fact.", ""]
    evidence = [(r["label"], k, e) for r in runs for k, e in r["config"].items()
                if e["source"] == "inferred" and e.get("evidence")]
    if evidence:
        out += ["### Evidence for inferred values", ""]
        out += [f"- {label}: {k} = {_fmt(e['value'])}: {e['evidence']}."
                for label, k, e in evidence]
        out.append("")
    for r in runs:
        if r["notes"]:
            out.append(f"- Notes for {r['label']} (from the user): {r['notes']}")
    if any(r["notes"] for r in runs):
        out.append("")

    if len(runs) > 1:
        out += ["### Config diff", ""]
        if b["config_diff"]:
            diff_rows = [{"field": d["field"], "source": d["source"],
                          **{lbl: _diff_cell(d, lbl) for lbl in labels}}
                         for d in b["config_diff"]]
            out.append(tables.render(diff_rows, ["field", "source", *labels]))
        else:
            out.append("No config differences found in the available metadata.")
        if not b["configs_distinguishable"]:
            out.append("")
            out.append("**Configs not distinguishable from available data.** Explain the "
                       "differences from the trace evidence alone.")
        out.append("")

    out += ["## 2. Data-quality notes", ""]
    notes = list(b["data_quality"])
    for r in runs:
        steps = r["metrics"]["steps"]
        notes.append(f"{r['label']}: {steps} profiler steps analyzed." if steps
                     else f"{r['label']}: no profiler steps found; values are trace totals.")
        for name, reason in r["unavailable"].items():
            notes.append(f"{r['label']}: {name} analysis unavailable: {reason}")
    out += [f"- {n}" for n in notes] + [""]

    out += ["## 3. Rule-based flags", ""]
    for r in runs:
        if len(runs) > 1:
            out.append(f"### {r['label']}")
        if r["flags"]:
            out.append(tables.render(r["flags"], ["id", "severity", "evidence", "threshold"]))
        else:
            out.append("No flags raised.")
        out.append("")

    def metric_table(cols: list[tuple[str, str]]) -> str:
        return tables.render([{"run": r["label"], **{c: r["metrics"][c] for c, _ in cols}}
                              for r in runs], ["run", *[c for c, _ in cols]],
                             {c: h for c, h in cols})

    out += ["## 4. Temporal breakdown", "", DESCRIPTIONS["temporal"], "",
            metric_table([("step_time_ms", "step time (ms)"), ("gpu_idle_pct", "GPU idle %"),
                          ("gpu_compute_pct", "compute %"), ("gpu_non_compute_pct", "non-compute %"),
                          ("idle_ms_per_step", "idle (ms/step)"),
                          ("kernel_ms_per_step", "kernel time (ms/step)")]), ""]
    out += ["## 5. Idle time breakdown", "", DESCRIPTIONS["idle"], "",
            metric_table([("host_wait_ms_per_step", "host wait (ms/step)"),
                          ("host_wait_pct_of_idle", "host wait % of idle"),
                          ("kernel_wait_pct_of_idle", "kernel wait % of idle"),
                          ("other_idle_pct_of_idle", "other % of idle")]), ""]

    out += ["## 6. GPU kernel breakdown", "", DESCRIPTIONS["kernels"], ""]
    if b["category_comparison"]:
        out += [f"### Kernel time by category across runs (ms/step; Δ vs. {b['baseline']})", "",
                tables.render(b["category_comparison"]), ""]
    elif runs and runs[0]["metrics"]["categories"]:
        r = runs[0]
        out += ["### Kernel time by category", "",
                tables.render([{"category": c, "ms/step": r["metrics"]["categories"][c],
                                "% GPU time": r["metrics"]["category_pct"].get(c)}
                               for c in CATEGORIES if c in r["metrics"]["categories"]]), ""]
    if b["kernel_comparison"]:
        out += [f"### Kernels shared by several runs (ms/step; Δ vs. {b['baseline']})", "",
                DESCRIPTIONS["shared"], "", tables.render(b["kernel_comparison"]), ""]
    kcols = ["short", "kernel_type", "ms_per_step", "pct_gpu", "calls_per_step", "mean_us",
             "precision"]
    kheads = {"short": "kernel", "kernel_type": "type", "ms_per_step": "ms/step",
              "pct_gpu": "% GPU time", "calls_per_step": "calls/step", "mean_us": "mean (µs)"}
    for r in runs:
        if len(runs) > 1:
            out.append(f"### {r['label']}")
        out += [tables.render(r["kernels"], kcols, kheads), ""]
        if r["metrics"]["gemm_conv_precision_pct"]:
            out += [f"GEMM/conv time by precision: "
                    f"{_fmt(r['metrics']['gemm_conv_precision_pct'])} (%).", ""]

    out += ["## 7. CUDA kernel launch stats", "", DESCRIPTIONS["launch"], "",
            metric_table([("kernels_per_step", "kernels/step"),
                          ("median_kernel_us", "median kernel (µs)"),
                          ("median_launch_cpu_us", "median launch, CPU (µs)"),
                          ("median_launch_delay_us", "median launch delay (µs)"),
                          ("short_kernel_pct", "% kernels < 10 µs"),
                          ("memcpy_pct", "H2D+D2H % of GPU time")]), ""]

    out += ["## 8. Derived metrics", "", DESCRIPTIONS["derived"], "",
            tables.render(b["derived"]), ""]
    if b["step_time_differences"]:
        diffs = {(d["from"], d["to"]): d for d in b["step_time_differences"]}
        matrix = []
        for a in labels:
            row = {"from \\ to": a}
            for c in labels:
                d = diffs.get((a, c))
                row[c] = "–" if d is None else f"{d['delta_ms']:+.2f} ms ({d['delta_pct']:+.1f}%)"
            matrix.append(row)
        out += ["### Step-time difference between any two runs", "",
                DESCRIPTIONS["differences"], "", tables.render(matrix), ""]

    if b["kernel_names"]:
        out += ["## Appendix: kernel names", "",
                tables.render([{"short": s, "full": f} for s, f in b["kernel_names"].items()
                               if s != f]) if any(s != f for s, f in b["kernel_names"].items())
                else "Short names are the full names.", ""]
    return "\n".join(out)
