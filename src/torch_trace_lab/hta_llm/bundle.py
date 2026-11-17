"""Assemble the analysis bundle (spec §8) from allowlisted fields only (§8a).

``build`` produces ``bundle.json`` from a fixed schema; ``render_markdown`` renders
``bundle.md`` from that JSON and nothing else, so both carry exactly the same data.
"""

from dataclasses import asdict
from itertools import count

from torch_trace_lab.hta_llm import settings, tables
from torch_trace_lab.hta_llm.context import RunContext
from torch_trace_lab.hta_llm.heuristics import Flag
from torch_trace_lab.hta_llm.frameworks import COLLECTIVES
from torch_trace_lab.hta_llm.normalize import (CATEGORIES, DIFF_CHANGES, RunMetrics,
                                               aggregate_ranks, category_comparison,
                                               collective_table, kernel_comparison,
                                               step_time_differences, strip_path)

SCHEMA = "torch-lab-hta-bundle/5"

METRIC_FIELDS = [
    "steps", "step_time_ms", "samples_per_s", "gpu_idle_pct", "gpu_compute_pct",
    "gpu_non_compute_pct", "idle_ms_per_step", "host_wait_pct_of_idle",
    "host_wait_ms_per_step", "kernel_wait_pct_of_idle",
    "other_idle_pct_of_idle", "kernel_ms_per_step", "top_kernel", "top_kernel_pct", "memcpy_pct",
    "gemm_conv_precision_pct", "categories", "category_pct", "kernels_per_step",
    "median_kernel_us", "median_launch_cpu_us",
    "median_launch_delay_us", "short_kernel_pct", "comm_ms_per_step", "comm_pct",
    "exposed_comm_ms_per_step", "overlap_pct", "busy_ms_per_step",
    "comm_by_collective_ms_per_step", "allreduce_ms_per_step", "exposed_allreduce_ms_per_step",
]

# Columns of the every-rank table, and labels for rank-comparison rows and columns.
RANK_COLUMNS = ["rank", "step_time_ms", "busy_ms_per_step", "gpu_idle_pct",
                "host_wait_ms_per_step", "comm_ms_per_step", "comm_pct",
                "exposed_comm_ms_per_step", "overlap_pct", "exposed_allreduce_ms_per_step"]
METRIC_LABELS = {
    "step_time_ms": "step time (ms)", "busy_ms_per_step": "busy GPU time (ms/step)",
    "kernel_ms_per_step": "GPU kernel time (ms/step)", "gpu_idle_pct": "GPU idle %",
    "idle_ms_per_step": "idle (ms/step)", "host_wait_ms_per_step": "host wait (ms/step)",
    "comm_ms_per_step": "communication (ms/step)", "comm_pct": "communication % of GPU time",
    "exposed_comm_ms_per_step": "exposed communication (ms/step)",
    "overlap_pct": "comm overlapped with compute %",
    "allreduce_ms_per_step": "all-reduce (ms/step)",
    "exposed_allreduce_ms_per_step": "exposed all-reduce (ms/step)",
}

DERIVED_HEADERS = {"step_time_ms": "step time (ms)", "gpus": "GPUs",
                   "samples_per_s_per_gpu": "samples/s per GPU",
                   "samples_per_s_total": "samples/s, all GPUs",
                   "scaling_efficiency_pct": "scaling efficiency %",
                   "scaling_basis": "scaling efficiency from"}

DIFF_COLUMNS = ["op", "device", "baseline_ms_per_step", "run_ms_per_step", "delta_ms_per_step",
                "baseline_calls_per_step", "run_calls_per_step", "change"]
DIFF_HEADERS = {"op": "kernel / op", "baseline_ms_per_step": "baseline ms/step",
                "run_ms_per_step": "run ms/step", "delta_ms_per_step": "Δ ms/step",
                "baseline_calls_per_step": "baseline calls/step",
                "run_calls_per_step": "run calls/step"}

DESCRIPTIONS = {
    "temporal": "How the GPU timeline splits into idle, compute kernels, and non-compute "
                "(memory copy/set and communication) kernels.",
    "idle": "Why the GPU was idle: host_wait = waiting for the CPU to launch work; "
            "kernel_wait = short gaps between back-to-back kernels; other = anything else. "
            "Shares are measured per CUDA stream (streams that only run communication are "
            "left out); host wait in ms/step is its share of the GPU idle time.",
    "kernels": "GPU kernel time per step, first by category, then the top kernels by name "
               "(the rest are collapsed into '(remaining kernels)'). Categories line up across "
               "eager and torch.compile runs, whose kernel names differ; 'uncategorized' holds "
               "kernels that fit no category. Category and top-kernel tables are separate "
               "views of the same time: don't add them together.",
    "launch": "CPU-side kernel launches: how long kernels run vs. how long launches take. "
              "Launch delay is the time from a launch call to the kernel starting on the GPU; "
              "a large delay means the CPU is running ahead of a busy GPU (queued work), "
              "not that launches are slow.",
    "derived": "Computed by torch-lab, not by HTA. Throughput needs the batch size: "
               "samples/s per GPU is the batch size per GPU ÷ step time; for a multi-GPU run, "
               "the total is that × the number of GPUs. "
               "Deltas are against the baseline run, in ms and %. Scaling efficiency "
               "(runs on a different number of GPUs than the baseline only) is weak scaling: "
               "the run's samples/s per GPU ÷ the baseline's, or, without a batch size, the "
               "baseline's step time ÷ the run's, which assumes the same per-GPU batch size. "
               "100% means each added GPU is as productive as in the baseline.",
    "differences": "Each cell is the column run's step time minus the row run's step time; "
                   "the % is relative to the row run. Negative means the column run is faster. "
                   "Quote these instead of subtracting step times.",
    "ranks": "How the ranks of this run differ. For each metric: the lowest, median and "
             "highest value across ranks, which rank holds the lowest and highest, and highest "
             "÷ median. Busy GPU time is GPU kernel time minus communication: the work a rank "
             "does itself. With synchronous collectives every rank's step ends together, so "
             "a slow rank shows up as more busy time, and the other ranks as more "
             "communication time (they wait inside the collective). The slowest rank is the "
             "one with the most busy time.",
    "comm": "Communication = NCCL collective kernels. Exposed communication runs while no "
            "compute kernel runs, so it adds to the step; overlapped communication is hidden "
            "behind compute. Waiting for a slower rank inside a collective counts as "
            "communication.",
    "collectives": "Communication time per step by NCCL collective type, across ranks. "
                   "A rank that waits for a slower rank inside a collective shows the wait "
                   "as time in that collective, so the lowest value is the best estimate of "
                   "the collective's own cost, and an upper bound on it.",
    "trace_diff": "HTA TraceDiff of one rank of this run against one rank of the baseline, "
                  "over each run's complete profiler steps, per step. GPU rows are kernels and "
                  "memory copies/sets; CPU rows are PyTorch operators, whose time includes the "
                  "operators they call, so CPU rows overlap and must not be added up. Rows are "
                  "the largest changes in ms/step; 'added'/'removed' ops occur in only one "
                  "run, the other changes compare calls per step. The counts cover every op.",
    "rank_summary": "One row per run: how unevenly its ranks work (busy GPU time max ÷ "
                    "median, see RANK_STRAGGLER), the step-time range across ranks, and the "
                    "median rank's communication share, overlap and exposed communication. "
                    "Single-GPU runs have no rank or communication values. To compare "
                    "communication between runs, use these median-rank columns and the "
                    "communication-by-collective table, not the slowest rank's values.",
    "comm_s4": "Each run's row is its slowest rank. The slowest rank does the most work and "
               "arrives last at each collective, so it waits least: its communication is "
               "usually the lowest of its run's ranks, not typical of the run. Don't compare "
               "runs' communication with this table; use the median-rank columns of the rank "
               "comparison and the communication-by-collective table.",
    "collectives_by_run": "Communication time per step by NCCL collective type, aligned "
                          "across runs on the collective type (kernel names differ between "
                          "NCCL versions and algorithms). min is the lowest rank, the best "
                          "estimate of the collective's own cost; median is the median rank.",
    "shared": "Kernels from any run's top list that occur in at least two runs. Values are "
              "each run's full ms/step for that kernel, even when it is outside that run's top "
              "list; '–' means the kernel does not occur in that run at all.",
}


def build(*, scenario: str, metrics: list[RunMetrics], contexts: list[RunContext],
          flags: dict[str, list[Flag]], baseline: int, config_diff: list[dict],
          distinguishable: bool, data_quality: list[str], cluster: str | None,
          kernel_names: dict[str, str], budget: dict | None = None,
          ranks: dict[str, list[RunMetrics]] | None = None,
          scaling: dict[str, tuple[float, str]] | None = None,
          trace_diff: dict[str, dict] | None = None, diff_rank=None) -> dict:
    """``metrics`` holds each run's representative (slowest) rank; ``ranks`` every rank's
    metrics by run label. ``budget`` records how the bundle was shrunk to fit the prompt
    budget: top_n_used, top_n_requested and dropped_sections (names from
    settings.RANK_SECTIONS and settings.OPTIONAL_SECTIONS). ``scaling`` is each run's
    (scaling efficiency %, basis) against the baseline, ``trace_diff`` each non-baseline
    run's normalize.trace_diff_rows (plus ``rank``/``baseline_rank``, or ``unavailable``),
    and ``diff_rank`` the requested --diff-rank (comparisons only)."""
    budget = dict(budget or {})
    budget.setdefault("dropped_sections", [])
    dropped = set(budget["dropped_sections"])
    runs = []
    for m, c in zip(metrics, contexts):
        per_rank = (ranks or {}).get(m.label) or [m]
        multi = len(per_rank) > 1
        every_rank = (multi and len(per_rank) <= settings.MAX_RANKS_IN_TABLES
                      and "all-ranks tables" not in dropped)
        runs.append({
            "label": m.label,
            "ranks_analyzed": len(per_rank),
            "slowest_rank": m.rank if multi else None,
            "rank_aggregate": aggregate_ranks(per_rank) if multi else [],
            "ranks": ([{col: getattr(r, col) for col in RANK_COLUMNS} for r in per_rank]
                      if every_rank else []),
            "collectives": collective_table(per_rank, per_rank=every_rank) if multi else [],
            "facts": dict(c.facts),
            "config": {k: dict(v) for k, v in c.config.items()},
            "notes": c.notes,
            "metrics": {f: getattr(m, f) for f in METRIC_FIELDS},
            "kernels": [asdict(k) for k in m.kernels],
            "flags": [f.to_dict() for f in flags.get(m.label, [])],
            "unavailable": dict(m.unavailable),
        })
    trace_diff = trace_diff or {}
    used = {k["short"] for r in runs for k in r["kernels"]}
    used |= {row["op"] for d in trace_diff.values() for row in d.get("rows", [])
             if row["device"] == "GPU"}
    names = {short: strip_path(full)[:300] for full, short in kernel_names.items()
             if short in used and "kernel-name appendix" not in dropped}
    base = metrics[baseline]
    s4 = scenario == "S4"
    scaling = scaling or {}
    derived = []
    for m in metrics:
        n_ranks = len((ranks or {}).get(m.label) or [m])
        row = {"run": m.label, "step_time_ms": m.step_time_ms}
        if s4:
            row["gpus"] = n_ranks
        row["samples_per_s_per_gpu"] = m.samples_per_s
        if n_ranks > 1 or s4:  # S4: every row has the column, so tables show it
            row["samples_per_s_total"] = (round(m.samples_per_s * n_ranks, 1)
                                          if m.samples_per_s is not None else None)
        if len(metrics) > 1:
            if m.step_time_ms and base.step_time_ms:
                row["step_time_vs_baseline_ms"] = round(m.step_time_ms - base.step_time_ms, 2)
                row["step_time_vs_baseline_pct"] = round(
                    100 * (m.step_time_ms - base.step_time_ms) / base.step_time_ms, 1)
                row["speedup_vs_baseline"] = round(base.step_time_ms / m.step_time_ms, 2)
            else:
                row["step_time_vs_baseline_ms"] = row["step_time_vs_baseline_pct"] = None
                row["speedup_vs_baseline"] = None
        if scaling:
            eff = scaling.get(m.label)
            row["scaling_efficiency_pct"] = eff[0] if eff else None
            row["scaling_basis"] = eff[1] if eff else None
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
        "category_comparison": (category_comparison(metrics, baseline)
                                if len(metrics) > 1 and "category comparison" not in dropped
                                else []),
        "comparison_kind": comparison_kind(contexts, runs) if len(metrics) > 1 else None,
        "rank_summary": rank_summary(runs) if s4 else [],
        "collectives_by_run": collectives_by_run(runs) if s4 else [],
        "diff_rank": diff_rank,
        "trace_diff": trace_diff,
        "derived": derived,
        "step_time_differences": step_time_differences(metrics) if len(metrics) > 1 else [],
        "data_quality": list(data_quality),
        "kernel_names": names,
        "budget": budget,
    }


FRAMEWORK_NONE = {"", "none", "unknown"}


def comparison_kind(contexts: list[RunContext], runs: list[dict]) -> str | None:
    """``cross-framework`` when the runs use different distributed frameworks (resolved
    ``parallelism``), ``same framework`` when every multi-GPU run uses the same one, else
    None (not known). Single-GPU runs and unknown values never make a comparison
    cross-framework."""
    values = {}
    for c, r in zip(contexts, runs):
        v = str((c.config.get("parallelism") or {}).get("value") or "").strip()
        if v.lower() not in FRAMEWORK_NONE:
            values[c.label] = v.lower()
    if len(set(values.values())) > 1:
        return "cross-framework"
    multi = [r["label"] for r in runs if r["ranks_analyzed"] > 1]
    if values and multi and all(label in values for label in multi):
        return "same framework"
    return None


def _agg(run: dict, metric: str) -> dict:
    return next((row for row in run["rank_aggregate"] if row["metric"] == metric), {})


def rank_summary(runs: list[dict]) -> list[dict]:
    """S4: one row per run comparing how its ranks behave (from each run's rank aggregate)."""
    rows = []
    for r in runs:
        busy, step = _agg(r, "busy_ms_per_step"), _agg(r, "step_time_ms")
        rows.append({
            "run": r["label"], "ranks": r["ranks_analyzed"], "slowest_rank": r["slowest_rank"],
            "busy_max_over_median": busy.get("max_over_median"),
            "step_time_min_ms": step.get("min"), "step_time_max_ms": step.get("max"),
            "comm_pct_median": _agg(r, "comm_pct").get("median"),
            "overlap_pct_median": _agg(r, "overlap_pct").get("median"),
            "exposed_comm_ms_median": _agg(r, "exposed_comm_ms_per_step").get("median"),
        })
    return rows


RANK_SUMMARY_HEADERS = {"slowest_rank": "slowest rank",
                        "busy_max_over_median": "busy GPU time max ÷ median",
                        "step_time_min_ms": "step time min (ms)",
                        "step_time_max_ms": "step time max (ms)",
                        "comm_pct_median": "comm % (median rank)",
                        "overlap_pct_median": "overlap % (median rank)",
                        "exposed_comm_ms_median": "exposed comm ms/step (median rank)"}


def collectives_by_run(runs: list[dict]) -> list[dict]:
    """S4: each collective type's min and median ms/step across ranks, one column pair per
    multi-GPU run (spec §6a: align on collective type, not kernel name)."""
    by_run = {r["label"]: {c["collective"]: c for c in r["collectives"]}
              for r in runs if r["collectives"]}
    rows = []
    for c in COLLECTIVES:
        if not any(c in cols for cols in by_run.values()):
            continue
        row = {"collective": c}
        for label, cols in by_run.items():
            row[f"{label} min"] = cols.get(c, {}).get("min")
            row[f"{label} median"] = cols.get(c, {}).get("median")
        rows.append(row)
    return rows


def budget_note(budget: dict) -> str | None:
    """A data-quality line saying how the bundle was shrunk, or None if it wasn't."""
    parts = []
    used, requested = budget.get("top_n_used"), budget.get("top_n_requested")
    if used and requested and used != requested:
        parts.append(f"kernel tables show the top {used} kernels instead of {requested}")
    if budget.get("dropped_sections"):
        parts.append("these sections were left out: " + ", ".join(budget["dropped_sections"]))
    return "; ".join(parts) + "." if parts else None


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


def rank_aggregate_table(r: dict) -> str:
    return tables.render([{"metric": METRIC_LABELS[row["metric"]],
                           **{k: row[k] for k in ("min", "min_rank", "median", "max",
                                                  "max_rank", "max_over_median")}}
                          for row in r["rank_aggregate"]],
                         ["metric", "min", "min_rank", "median", "max", "max_rank",
                          "max_over_median"],
                         {"min_rank": "rank (min)", "max_rank": "rank (max)",
                          "max_over_median": "max ÷ median"})


def render_markdown(b: dict) -> str:
    runs = b["runs"]
    labels = [r["label"] for r in runs]
    out = ["# Trace analysis bundle", ""]

    s3, s4 = b["scenario"] == "S3", b["scenario"] == "S4"
    slowest = (f" (slowest rank: rank {runs[0]['slowest_rank']})" if s3
               else " (each multi-GPU run's slowest rank)" if s4 else "")
    gpu_list = ", ".join(f"{r['label']}: {r['ranks_analyzed']} GPU"
                         f"{'s' if r['ranks_analyzed'] != 1 else ''}" for r in runs)
    scenario = {
        "S1": "(single run, single GPU)",
        "S3": (f"(single run, {runs[0]['ranks_analyzed']} GPUs; the communication, temporal, "
               f"idle, kernel and launch sections show the slowest rank, rank "
               f"{runs[0]['slowest_rank']}; the rank comparison covers all ranks)"),
        "S4": (f"(comparison of {len(runs)} runs ({gpu_list}); baseline: {b['baseline']}; "
               "the communication, temporal, idle, kernel and launch sections show each "
               "multi-GPU run's slowest rank; the rank comparison covers all ranks)"),
    }.get(b["scenario"],
          f"(comparison of {len(runs)} single-GPU runs; baseline: {b['baseline']})")
    out += ["## 1. Context", "", f"- Scenario: {b['scenario']} {scenario}"]
    if b.get("comparison_kind") == "cross-framework":
        out.append("- **Cross-framework comparison:** the runs use different distributed "
                   "frameworks (see parallelism below). Communication is compared by "
                   "collective type; differences in it may come from the framework itself.")
    elif b.get("comparison_kind") == "same framework":
        out.append("- All multi-GPU runs use the same distributed framework.")
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
    budget = b.get("budget") or {}
    dropped = set(budget.get("dropped_sections", []))
    if note := budget_note(budget):
        notes.append(f"To fit the prompt budget, {note}")
    for r in runs:
        steps = r["metrics"]["steps"]
        who = r["label"] + (f" (rank {r['slowest_rank']})" if r["slowest_rank"] is not None
                            else "")
        notes.append(f"{who}: {steps} profiler steps analyzed." if steps
                     else f"{who}: no profiler steps found; values are trace totals.")
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

    section = count(4)

    def heading(title: str) -> str:
        return f"## {next(section)}. {title}"

    if s3:
        r = runs[0]
        out += [heading("Rank comparison"), "", DESCRIPTIONS["ranks"], "",
                rank_aggregate_table(r), ""]
        if r["ranks"]:
            out += ["### Every rank", "", tables.render(r["ranks"], RANK_COLUMNS, METRIC_LABELS),
                    ""]
        out += [heading("Communication" + slowest), "", DESCRIPTIONS["comm"], "",
                metric_table([("comm_ms_per_step", "communication (ms/step)"),
                              ("comm_pct", "communication % of GPU time"),
                              ("exposed_comm_ms_per_step", "exposed (ms/step)"),
                              ("overlap_pct", "overlapped with compute %"),
                              ("exposed_allreduce_ms_per_step", "exposed all-reduce (ms/step)"),
                              ("busy_ms_per_step", "busy GPU time (ms/step)")]), ""]
        if r["collectives"]:
            out += ["### Communication by collective (ms/step, all ranks)", "",
                    DESCRIPTIONS["collectives"], "", tables.render(r["collectives"]), ""]
    if s4:
        out += [heading("Rank comparison"), "", DESCRIPTIONS["rank_summary"], "",
                tables.render(b["rank_summary"], headers=RANK_SUMMARY_HEADERS), "",
                DESCRIPTIONS["ranks"], ""]
        for r in runs:
            if not r["rank_aggregate"]:
                continue
            out += [f"### {r['label']} ({r['ranks_analyzed']} ranks)", "",
                    rank_aggregate_table(r), ""]
            if r["ranks"]:
                out += [f"#### {r['label']}: every rank", "",
                        tables.render(r["ranks"], RANK_COLUMNS, METRIC_LABELS), ""]
        out += [heading("Communication" + slowest), "", DESCRIPTIONS["comm"], "",
                DESCRIPTIONS["comm_s4"], "",
                metric_table([("comm_ms_per_step", "communication (ms/step)"),
                              ("comm_pct", "communication % of GPU time"),
                              ("exposed_comm_ms_per_step", "exposed (ms/step)"),
                              ("overlap_pct", "overlapped with compute %"),
                              ("exposed_allreduce_ms_per_step", "exposed all-reduce (ms/step)"),
                              ("busy_ms_per_step", "busy GPU time (ms/step)")]), ""]
        if b["collectives_by_run"]:
            out += ["### Communication by collective across runs (ms/step)", "",
                    DESCRIPTIONS["collectives_by_run"], "",
                    tables.render(b["collectives_by_run"]), ""]

    out += [heading("Temporal breakdown" + slowest), "", DESCRIPTIONS["temporal"], "",
            metric_table([("step_time_ms", "step time (ms)"), ("gpu_idle_pct", "GPU idle %"),
                          ("gpu_compute_pct", "compute %"), ("gpu_non_compute_pct", "non-compute %"),
                          ("idle_ms_per_step", "idle (ms/step)"),
                          ("kernel_ms_per_step", "kernel time (ms/step)")]), ""]
    out += [heading("Idle time breakdown" + slowest), "", DESCRIPTIONS["idle"], "",
            metric_table([("host_wait_ms_per_step", "host wait (ms/step)"),
                          ("host_wait_pct_of_idle", "host wait % of idle"),
                          ("kernel_wait_pct_of_idle", "kernel wait % of idle"),
                          ("other_idle_pct_of_idle", "other % of idle")]), ""]

    out += [heading("GPU kernel breakdown" + slowest), "", DESCRIPTIONS["kernels"], ""]
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

    if "launch stats" not in dropped:
        out += [heading("CUDA kernel launch stats" + slowest), "", DESCRIPTIONS["launch"],
                "",
                metric_table([("kernels_per_step", "kernels/step"),
                              ("median_kernel_us", "median kernel (µs)"),
                              ("median_launch_cpu_us", "median launch, CPU (µs)"),
                              ("median_launch_delay_us", "median launch delay (µs)"),
                              ("short_kernel_pct", "% kernels < 10 µs"),
                              ("memcpy_pct", "H2D+D2H % of GPU time")]), ""]

    if b.get("trace_diff"):
        rank_note = (f" (--diff-rank {b['diff_rank']})" if s4 else "")
        out += [heading(f"Trace diff vs. {b['baseline']}{rank_note}"), "",
                DESCRIPTIONS["trace_diff"], ""]
        for label, d in b["trace_diff"].items():
            out += [f"### {label} (rank {d['rank']}) vs. {b['baseline']} "
                    f"(rank {d['baseline_rank']})", ""]
            if "unavailable" in d:
                out += [f"Unavailable: {d['unavailable']}", ""]
                continue
            for device in ("GPU", "CPU"):
                counts = d["summary"][device]
                out.append(f"- {device}: " + ", ".join(f"{counts[c]} {c}" for c in DIFF_CHANGES))
            out += ["", tables.render(d["rows"], DIFF_COLUMNS, DIFF_HEADERS), ""]

    out += [heading("Derived metrics"), "", DESCRIPTIONS["derived"], "",
            tables.render(b["derived"], headers=DERIVED_HEADERS), ""]
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
