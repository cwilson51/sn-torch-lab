"""Assemble the analysis bundle (spec §8) from allowlisted fields only (§8a).

``build`` produces ``bundle.json`` from a fixed schema; ``render_markdown`` renders
``bundle.md`` from that JSON and nothing else, so both carry exactly the same data.
"""

import re
from dataclasses import asdict

from torch_trace_lab.hta_llm import tables
from torch_trace_lab.hta_llm.context import RunContext
from torch_trace_lab.hta_llm.heuristics import Flag
from torch_trace_lab.hta_llm.normalize import RunMetrics, kernel_comparison

SCHEMA = "torch-lab-hta-bundle/1"
PATHLIKE = re.compile(r"(?:[A-Za-z]:)?(?:/[^/\s]+){2,}/")

METRIC_FIELDS = [
    "steps", "step_time_ms", "samples_per_s", "gpu_idle_pct", "gpu_compute_pct",
    "gpu_non_compute_pct", "idle_ms_per_step", "host_wait_pct_of_idle", "kernel_wait_pct_of_idle",
    "other_idle_pct_of_idle", "kernel_ms_per_step", "top_kernel", "top_kernel_pct", "memcpy_pct",
    "gemm_conv_precision_pct", "kernels_per_step", "median_kernel_us", "median_launch_cpu_us",
    "median_launch_delay_us", "short_kernel_pct",
]

DESCRIPTIONS = {
    "temporal": "How the GPU timeline splits into idle, compute kernels, and non-compute "
                "(memory copy/set) kernels.",
    "idle": "Why the GPU was idle: host_wait = waiting for the CPU to launch work; "
            "kernel_wait = short gaps between back-to-back kernels; other = anything else.",
    "kernels": "GPU kernels by time per step; the rest are collapsed into 'other'.",
    "launch": "CPU-side kernel launches: how long kernels run vs. how long launches take.",
    "derived": "Computed by torch-lab, not by HTA. Throughput needs the batch size.",
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
                row["step_time_vs_baseline_pct"] = round(
                    100 * (m.step_time_ms - base.step_time_ms) / base.step_time_ms, 1)
                row["speedup_vs_baseline"] = round(base.step_time_ms / m.step_time_ms, 2)
            else:
                row["step_time_vs_baseline_pct"] = row["speedup_vs_baseline"] = None
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
        "derived": derived,
        "data_quality": list(data_quality),
        "kernel_names": names,
    }


def _fmt(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return tables._cell(value)


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
                row[r["label"]] = f"{_fmt(r['config'][k]['value'])} ({r['config'][k]['source']})"
            elif k in r["facts"]:
                row[r["label"]] = f"{_fmt(r['facts'][k])} (trace)"
            else:
                row[r["label"]] = "unknown"
        rows.append(row)
    out += [tables.render(rows, ["field", *labels]), ""]
    for r in runs:
        if r["notes"]:
            out.append(f"- Notes for {r['label']} (from the user): {r['notes']}")
    if any(r["notes"] for r in runs):
        out.append("")

    if len(runs) > 1:
        out += ["### Config diff", ""]
        if b["config_diff"]:
            diff_rows = [{"field": d["field"], "source": d["source"],
                          **{lbl: _fmt(d["values"][lbl]) for lbl in labels}}
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
            metric_table([("host_wait_pct_of_idle", "host wait % of idle"),
                          ("kernel_wait_pct_of_idle", "kernel wait % of idle"),
                          ("other_idle_pct_of_idle", "other % of idle")]), ""]

    out += ["## 6. GPU kernel breakdown", "", DESCRIPTIONS["kernels"], ""]
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
    if b["kernel_comparison"]:
        out += [f"### Kernel time across runs (ms/step; Δ vs. {b['baseline']})", "",
                tables.render(b["kernel_comparison"]), ""]

    out += ["## 7. CUDA kernel launch stats", "", DESCRIPTIONS["launch"], "",
            metric_table([("kernels_per_step", "kernels/step"),
                          ("median_kernel_us", "median kernel (µs)"),
                          ("median_launch_cpu_us", "median launch, CPU (µs)"),
                          ("median_launch_delay_us", "median launch delay (µs)"),
                          ("short_kernel_pct", "% kernels < 10 µs"),
                          ("memcpy_pct", "H2D+D2H % of GPU time")]), ""]

    out += ["## 8. Derived metrics", "", DESCRIPTIONS["derived"], "",
            tables.render(b["derived"]), ""]

    if b["kernel_names"]:
        out += ["## Appendix: kernel names", "",
                tables.render([{"short": s, "full": f} for s, f in b["kernel_names"].items()
                               if s != f]) if any(s != f for s, f in b["kernel_names"].items())
                else "Short names are the full names.", ""]
    return "\n".join(out)
