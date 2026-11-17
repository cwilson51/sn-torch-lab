"""Stages 1-6 of analyze-traces: ingest, analyze, normalize, flag, bundle, Brine, report (spec §4).

Stages 1-4 never need the LLM. On any Brine problem the bundle, prompt and a flags-only
report are still written and the run exits 1 (the one failure rule, §4).
"""

import json
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from torch_trace_lab.hta_llm import (analyses, brine, bundle, context, frameworks, heuristics,
                                     infer, ingest, meta, normalize, report, sanitize, settings)


@dataclass
class Options:
    paths: list[Path]
    output_dir: Path
    run_notes: list[str] = field(default_factory=list)
    baseline: str | None = None
    top_n: int = settings.TOP_N
    model: str | None = None
    no_llm: bool = False
    meta: Path | None = None
    diff_rank: int | str = 0   # a rank number, or "slowest"


class Log:
    """Messages for the terminal and run.log. Never given the API key."""

    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(msg)
        print(msg, flush=True)

    def warn(self, msg: str) -> None:
        self.lines.append(f"WARNING: {msg}")
        print(f"Warning: {msg}", file=sys.stderr, flush=True)

    def error(self, msg: str) -> None:
        self.lines.append(f"ERROR: {msg}")
        print(f"Error: {msg}", file=sys.stderr, flush=True)


# The user.md section for each scenario; the other sections are removed.
PROMPT_SECTIONS = {"S1": "single", "S2": "comparison", "S3": "distributed",
                   "S4": "multi-comparison"}


def build_prompt(b: dict, bundle_md: str) -> tuple[str, str]:
    system = report.prompt_file("system.md")
    user = report.prompt_file("user.md")
    keep = PROMPT_SECTIONS[b["scenario"]]
    for drop in sorted(set(PROMPT_SECTIONS.values()) - {keep}):
        user = _strip_section(user, drop)
    user = user.replace(f"<!-- {keep} -->\n", "").replace(f"<!-- /{keep} -->\n", "")
    user = (user.replace("{{N_RUNS}}", str(len(b["runs"])))
                .replace("{{N_RANKS}}", str(b["runs"][0]["ranks_analyzed"]))
                .replace("{{BASELINE}}", str(b["baseline"]))
                .replace("{{RUN_GPUS}}", run_gpus(b))
                .replace("{{META_KEYS}}", meta.key_list())
                .replace("{{FRAMEWORK}}", framework_section(b))
                .replace("{{REMEDIATION}}", remediation_for(b["scenario"]))
                .replace("{{BUNDLE}}", bundle_md))
    return system, user


def run_gpus(b: dict) -> str:
    """'a (4 GPUs), b (1 GPU)' for the prompt."""
    return ", ".join(f"{r['label']} ({r['ranks_analyzed']} GPU"
                     f"{'s' if r['ranks_analyzed'] != 1 else ''})" for r in b["runs"])


NO_FRAMEWORK = ("The bundle does not say which distributed framework (DDP, FSDP, DeepSpeed) "
                "the run uses unless the context table shows `parallelism`. Do not assume one; "
                "when a fix depends on the framework, say so.")


def framework_section(b: dict) -> str:
    """Guidance for each distributed framework the runs use (their resolved ``parallelism``,
    so a --meta value wins over the inferred one), once each, or the don't-assume note."""
    seen, parts = set(), []
    who = "the run uses" if len(b["runs"]) == 1 else "a run uses"
    for r in b["runs"]:
        parallelism = (r["config"].get("parallelism") or {}).get("value")
        key = str(parallelism or "").strip().lower()
        if not key or key in ("none", "unknown") or key in seen:
            continue
        seen.add(key)
        guidance = frameworks.framework_prompt(parallelism)
        parts.append(guidance or
                     f"The context table says {who} {parallelism}. There is no "
                     f"{parallelism}-specific guidance here: when a fix depends on the "
                     "framework's settings, say so rather than guessing them.")
    return "\n\n".join(parts) if parts else NO_FRAMEWORK


def remediation_for(scenario: str) -> str:
    """remediation.md, without the multi-GPU flags' sections unless a run is multi-GPU, and
    without SCALING_INEFFICIENT unless multi-GPU runs are compared (S4)."""
    sections = report.remediation_sections()
    if scenario not in ("S3", "S4"):
        sections = {k: v for k, v in sections.items() if k not in heuristics.RANK_FLAGS}
    if scenario != "S4":
        sections = {k: v for k, v in sections.items() if k not in heuristics.S4_FLAGS}
    return "\n\n".join(f"## {flag_id}\n{text}" for flag_id, text in sections.items())


def _strip_section(text: str, name: str) -> str:
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    i, j = text.index(start), text.index(end) + len(end)
    return text[:i] + text[j:].lstrip("\n")


def prompt_markdown(system: str, user: str) -> str:
    return f"# System prompt\n\n{system.strip()}\n\n# User prompt\n\n{user.strip()}\n"


@dataclass
class Collected:
    """Everything the expensive stages produce; assemble() can rebuild the bundle from it."""
    scenario: str
    baseline: int
    contexts: list[context.RunContext]
    # Per run, {rank: RunAnalysis}: plain DataFrames, independent of the staging dir.
    results: list[dict[int, analyses.RunAnalysis]]
    warnings: list[str]
    # Trace diff (comparisons only): the requested --diff-rank, each run's representative
    # rank, and per non-baseline run label the analyses.trace_diff table or why it failed.
    diff_rank: int | str = 0
    diff_ranks: list[int] = field(default_factory=list)
    diffs: dict = field(default_factory=dict)


def collect(opts: Options, log: Log) -> Collected:
    """Stages 1-2 plus run context. Raises IngestError / ValueError."""
    with tempfile.TemporaryDirectory(prefix="torch-lab-hta-") as stage:
        runs, warnings = ingest.ingest(opts.paths, Path(stage))
        labels = [r.label for r in runs]
        baseline = normalize.resolve_baseline(labels, opts.baseline)
        notes = context.parse_run_notes(opts.run_notes, labels)
        user = {}
        if opts.meta:
            meta_values, meta_warnings = meta.load_meta(opts.meta, labels)
            for w in meta_warnings:  # log only: they describe the user's file, not the runs
                log.warn(w)
            for label, values in meta_values.items():
                user[label], meta_notes = meta.as_config(values)
                if meta_notes:
                    notes[label] = f"{meta_notes}; {notes[label]}" if label in notes else meta_notes
        inferred = {r.label: {**infer.infer_config(r.rank0.evidence), **run_topology(r)}
                    for r in runs}
        contexts, conflicts = context.build_contexts(runs, notes, inferred=inferred, user=user)
        warnings += conflicts
        for w in warnings:
            log.warn(w)
        multi_rank = any(r.ranks > 1 for r in runs)
        scenario = (("S4" if multi_rank else "S2") if len(runs) > 1
                    else "S3" if multi_rank else "S1")
        if scenario == "S4":
            log.info(f"Scenario S4: {len(runs)} runs: "
                     + ", ".join(f"{r.label} ({r.ranks} rank{'s' if r.ranks != 1 else ''})"
                                 for r in runs))
        else:
            log.info(f"Scenario {scenario}: {len(runs)} run(s): {', '.join(labels)}"
                     + (f" ({runs[0].ranks} ranks)" if scenario == "S3" else ""))
        if len(runs) > 1:
            check_diff_rank(runs, opts.diff_rank)

        results = []
        for run in runs:
            start = time.monotonic()
            log.info(f"Analyzing {run.label} ...")
            result = analyses.run_ranks(run.staged_dir)
            for name, reason in result[min(result)].unavailable.items():
                log.warn(f"{run.label}: {name} analysis unavailable: {reason}")
            for note in (step_count_note(run.label, result), few_steps_note(run.label, result)):
                if note:
                    log.warn(note)
                    warnings.append(note)
            results.append(result)
            log.info(f"  {run.label}: done ({time.monotonic() - start:.1f} s)")
        diff_ranks, diffs = [], {}
        if len(runs) > 1:
            diff_ranks = [representative_rank(opts.diff_rank, r, ctx.batch_size)
                          for r, ctx in zip(results, contexts)]
            diffs = trace_diffs(runs, results, baseline, diff_ranks, Path(stage), log)
    return Collected(scenario, baseline, contexts, results, warnings, opts.diff_rank,
                     diff_ranks, diffs)


def check_diff_rank(runs: list[ingest.Run], choice: int | str) -> None:
    """--diff-rank N must exist in every multi-rank run (a single-rank run uses its only
    rank). Checked before the analyses, so a typo fails fast."""
    if choice == "slowest":
        return
    missing = [r for r in runs if r.ranks > 1 and not 0 <= choice < r.ranks]
    if missing:
        raise ValueError(f"--diff-rank {choice}: not every run has that rank ("
                         + ", ".join(f"{r.label} has ranks 0-{r.ranks - 1}" for r in missing)
                         + "). Use a rank every multi-GPU run has, or --diff-rank slowest.")


def representative_rank(choice: int | str, result: dict[int, analyses.RunAnalysis],
                        batch_size: int | None = None) -> int:
    """The rank of a run that the trace diff uses (spec §7.7): the requested rank, or with
    ``slowest`` the rank with the most busy GPU time (normalize.slowest_rank). A single-rank
    run always uses its only rank."""
    ranks = sorted(result)
    if len(ranks) == 1:
        return ranks[0]
    if choice == "slowest":
        per_rank = [normalize.normalize_run("", result[r], rank=r, batch_size=batch_size)
                    for r in ranks]
        return normalize.slowest_rank(per_rank).rank
    return int(choice)


def trace_diffs(runs: list[ingest.Run], results: list[dict[int, analyses.RunAnalysis]],
                baseline: int, ranks: list[int], stage: Path, log: Log) -> dict:
    """HTA TraceDiff of each run's representative rank against the baseline's.

    Each rank's trace is staged and loaded on its own, so at most two single-rank traces
    are in memory at once. A failure makes that run's diff unavailable (spec §6).
    Returns {label: DataFrame, or the reason it is unavailable}.
    """
    def load(i: int, label: str):
        position = sorted(results[i]).index(ranks[i])
        directory = ingest.stage_traces(runs[i].traces[position:position + 1],
                                        stage / f"diff-{i}")
        return analyses.load_rank_trace(directory, label)

    others = [i for i in range(len(runs)) if i != baseline]
    log.info(f"Trace diff against {runs[baseline].label} (rank {ranks[baseline]}) ...")
    try:
        control = load(baseline, "control")
    except Exception as e:  # noqa: BLE001 - HTA raises many types
        reason = f"the baseline trace could not be loaded: {type(e).__name__}: {e}"
        log.warn(f"Trace diff unavailable: {reason}")
        return {runs[i].label: reason for i in others}
    diffs = {}
    for i in others:
        label = runs[i].label
        try:
            test = load(i, "test")
            diffs[label] = analyses.trace_diff(
                control, test, results[baseline][ranks[baseline]].iterations,
                results[i][ranks[i]].iterations)
        except Exception as e:  # noqa: BLE001
            diffs[label] = f"{type(e).__name__}: {e}"
            log.warn(f"{label}: trace diff unavailable: {diffs[label]}")
        test = None  # free it before loading the next run
    return diffs


def run_topology(run: ingest.Run) -> dict:
    """Inferred num_nodes / gpus_per_node of a multi-rank run (see infer.infer_topology)."""
    if run.ranks < 2 or any(t.evidence is None for t in run.traces):
        return {}
    visible = []
    for t in run.traces:
        props = t.metadata.get("deviceProperties")
        visible.append(len(props) if isinstance(props, list) else 0)
    return infer.infer_topology([t.evidence.devices for t in run.traces], visible, run.ranks)


def step_count_note(label: str, ranks: dict[int, analyses.RunAnalysis]) -> str | None:
    """A data-quality note when ranks recorded different numbers of complete steps."""
    counts = [a.steps for a in ranks.values() if a.steps]
    if len(set(counts)) <= 1:
        return None
    return (f"{label}: ranks recorded different numbers of complete profiler steps "
            f"({min(counts)} to {max(counts)}); each rank's per-step values use its own "
            "step count.")


def few_steps_note(label: str, ranks: dict[int, analyses.RunAnalysis]) -> str | None:
    """A data-quality note when per-step values rest on only a few profiler steps."""
    counts = [a.steps for a in ranks.values() if a.steps]
    if not counts or min(counts) >= settings.FEW_STEPS:
        return None
    n = min(counts)
    return (f"{label}: only {n} complete profiler step{'s' if n != 1 else ''} analyzed; "
            "per-step values are means of so few steps and can be noisy. Profiling at least "
            f"{settings.FEW_STEPS} active steps gives steadier numbers.")


def assemble(c: Collected, top_n: int, drop: tuple[str, ...] = (),
             top_n_requested: int | None = None) -> tuple[dict, str]:
    """Stages 3-4: normalize, flag, bundle. Cheap, so the budget cascade can repeat it.

    Each run is normalized per rank. The slowest rank (normalize.slowest_rank) represents
    the run in the per-run tables and flags; the rank flags compare all ranks.
    """
    names: dict[str, str] = {}
    metrics, flags, ranks = [], {}, {}
    gpus = [len(result) for result in c.results]
    for ctx, result in zip(c.contexts, c.results):
        per_rank = [normalize.normalize_run(ctx.label, a, top_n=top_n, names=names,
                                            batch_size=ctx.batch_size, rank=r)
                    for r, a in sorted(result.items())]
        m = normalize.slowest_rank(per_rank)
        metrics.append(m)
        ranks[ctx.label] = per_rank
        run_flags = heuristics.flags_for(m, ctx.compute_capability)
        if len(per_rank) > 1:
            run_flags = [replace(f, evidence=f"rank {m.rank}: {f.evidence}") for f in run_flags]
        flags[ctx.label] = run_flags + heuristics.rank_flags(per_rank)
    base, base_gpus = metrics[c.baseline], gpus[c.baseline]
    scaling = {}
    base_batch = c.contexts[c.baseline].batch_size
    for m, n, ctx in zip(metrics, gpus, c.contexts):
        eff = None if m is base else normalize.scaling_efficiency(
            m, base, n, base_gpus, ctx.batch_size, base_batch)
        if eff is None:
            continue
        scaling[m.label] = eff
        if flag := heuristics.scaling_inefficient(*eff, n, base.label, base_gpus):
            flags[m.label].append(flag)
    trace_diff = {}
    if "trace diff" not in drop:
        base_analysis = (c.results[c.baseline][c.diff_ranks[c.baseline]]
                         if c.diff_ranks else None)
        for i, (ctx, result) in enumerate(zip(c.contexts, c.results)):
            if i == c.baseline or ctx.label not in c.diffs:
                continue
            d = c.diffs[ctx.label]
            entry = {"rank": c.diff_ranks[i], "baseline_rank": c.diff_ranks[c.baseline]}
            if isinstance(d, str):
                entry["unavailable"] = d
            else:
                entry.update(normalize.trace_diff_rows(
                    d, base_analysis.steps, result[c.diff_ranks[i]].steps, names, top_n))
            trace_diff[ctx.label] = entry
    diff, distinguishable = context.config_diff(c.contexts)
    budget = {"top_n_requested": top_n_requested or top_n, "top_n_used": top_n,
              "dropped_sections": list(drop)}
    b = bundle.build(scenario=c.scenario, metrics=metrics, contexts=c.contexts, flags=flags,
                     baseline=c.baseline, config_diff=diff, distinguishable=distinguishable,
                     data_quality=c.warnings, cluster=context.cluster_name(), kernel_names=names,
                     budget=budget, ranks=ranks, scaling=scaling, trace_diff=trace_diff,
                     diff_rank=c.diff_rank if len(c.contexts) > 1 else None)
    return b, bundle.render_markdown(b)


def budget_plan(top_n: int, multi_rank: bool = False) -> list[tuple[int, tuple[str, ...]]]:
    """Sizes to try, in order: as requested, smaller top-N, then fewer optional sections.

    Multi-rank runs drop the all-ranks tables first. The rank aggregation and the slowest
    rank's tables are never dropped (spec §7.9).
    """
    plan = [(top_n, ())]
    plan += [(n, ()) for n in settings.BUDGET_TOP_N_STEPS if n < top_n]
    smallest = plan[-1][0]
    sections = (settings.RANK_SECTIONS if multi_rank else ()) + settings.OPTIONAL_SECTIONS
    plan += [(smallest, sections[:i + 1]) for i in range(len(sections))]
    return plan


def estimate_tokens(system: str, user: str) -> int:
    return int(len(system + user) / settings.CHARS_PER_TOKEN)


def fit_budget(c: Collected, top_n: int, log: Log) -> tuple[dict, str, str, str, int]:
    """Assemble the bundle and prompt, shrinking them until the prompt fits (spec §7.9)."""
    multi_rank = any(len(r) > 1 for r in c.results)
    for n, drop in budget_plan(top_n, multi_rank):
        b, bundle_md = assemble(c, n, drop, top_n_requested=top_n)
        system, user = build_prompt(b, bundle_md)
        tokens = estimate_tokens(system, user)
        if tokens <= settings.MAX_PROMPT_TOKENS:
            break
        log.info(f"Prompt: about {tokens:,} tokens with top-n {n}"
                 + (f" and without {', '.join(drop)}" if drop else "")
                 + f"; over the {settings.MAX_PROMPT_TOKENS:,}-token budget, shrinking ...")
    else:
        log.warn(f"The prompt is still larger than {settings.MAX_PROMPT_TOKENS:,} tokens after "
                 "every reduction; sending it anyway. Try fewer runs.")
    if b["budget"]["top_n_used"] != top_n or b["budget"]["dropped_sections"]:
        log.warn("To fit the prompt budget: " + bundle.budget_note(b["budget"]))
    b["budget"]["estimated_tokens"] = tokens
    b["budget"]["max_prompt_tokens"] = settings.MAX_PROMPT_TOKENS
    return b, bundle_md, system, user, tokens


def run(opts: Options, log: Log | None = None) -> int:
    log = log or Log()
    out = opts.output_dir
    try:
        collected = collect(opts, log)
    except (ingest.IngestError, ValueError) as e:
        log.error(str(e))
        return 1

    b, bundle_md, system, user, tokens = fit_budget(collected, opts.top_n, log)
    for r in b["runs"]:
        log.info(f"  {r['label']}: {r['metrics']['steps'] or 'no'} steps, flags: "
                 f"{', '.join(f['id'] for f in r['flags']) or 'none'}")
    out.mkdir(parents=True, exist_ok=True)
    prompt_md = prompt_markdown(system, user)
    (out / "bundle.json").write_text(json.dumps(b, indent=2) + "\n")
    (out / "bundle.md").write_text(bundle_md)
    (out / "prompt.md").write_text(prompt_md)

    log.info(f"Prompt: about {tokens:,} tokens (estimate).")

    def finish(code: int, reason: str | None = None, text: str | None = None) -> int:
        (out / "report.md").write_text(text if text is not None else report.flags_only(b, reason))
        (out / "run.log").write_text("\n".join(log.lines) + "\n")
        log.info(f"Wrote report.md, bundle.md, bundle.json, prompt.md and run.log to {out}")
        return code

    if opts.no_llm:
        for leak in sanitize.find_leaks(prompt_md):
            log.warn(f"prompt.md contains {leak}; a Brine call would be blocked.")
        return finish(0, "run with --no-llm.")

    try:
        key = brine.load_key()
    except brine.BrineError as e:
        log.error(str(e))
        return finish(1, "no Brine API key.")

    leaks = sanitize.find_leaks(prompt_md, key=key)
    if leaks:
        log.error("Not sending the prompt to Brine: it contains " + "; ".join(leaks) + ". "
                  "Check --run-note text and run directory names, then see prompt.md.")
        return finish(1, "the leak check blocked the Brine call.")

    model = brine.model_name(opts.model)
    try:
        client = brine.Client(key)
        log.info(f"Checking model {model} on Brine ...")
        client.check_model(model)
        log.info("Asking Brine for the report (this can take a minute) ...")
        completion = client.chat(model, system, user)
    except brine.BrineError as e:
        log.error(str(e))
        return finish(1, f"the Brine call failed: {e}")
    log.info(f"Brine answered in {completion.latency_s:.1f} s; usage: "
             f"{json.dumps(completion.usage)}")
    return finish(0, text=report.llm_report(completion.content, completion.model))
