"""Stages 1-6 of analyze-traces: ingest, analyze, normalize, flag, bundle, Brine, report (spec §4).

Stages 1-4 never need the LLM. On any Brine problem the bundle, prompt and a flags-only
report are still written and the run exits 1 (the one failure rule, §4).
"""

import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from torch_trace_lab.hta_llm import (analyses, brine, bundle, context, heuristics, infer, ingest,
                                     meta, normalize, report, sanitize, settings)


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


def build_prompt(b: dict, bundle_md: str) -> tuple[str, str]:
    system = report.prompt_file("system.md")
    user = report.prompt_file("user.md")
    keep, drop = ("single", "comparison") if b["scenario"] == "S1" else ("comparison", "single")
    user = _strip_section(user, drop)
    user = user.replace(f"<!-- {keep} -->\n", "").replace(f"<!-- /{keep} -->\n", "")
    user = (user.replace("{{N_RUNS}}", str(len(b["runs"])))
                .replace("{{BASELINE}}", str(b["baseline"]))
                .replace("{{REMEDIATION}}", report.prompt_file("remediation.md").strip())
                .replace("{{BUNDLE}}", bundle_md))
    return system, user


def _strip_section(text: str, name: str) -> str:
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    i, j = text.index(start), text.index(end) + len(end)
    return text[:i] + text[j:].lstrip("\n")


def prompt_markdown(system: str, user: str) -> str:
    return f"# System prompt\n\n{system.strip()}\n\n# User prompt\n\n{user.strip()}\n"


def analyze(opts: Options, log: Log) -> tuple[dict, str]:
    """Stages 1-4. Returns (bundle dict, bundle markdown). Raises IngestError / ValueError."""
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
        inferred = {r.label: infer.infer_config(r.rank0.evidence) for r in runs}
        contexts, conflicts = context.build_contexts(runs, notes, inferred=inferred, user=user)
        warnings += conflicts
        for w in warnings:
            log.warn(w)
        scenario = "S1" if len(runs) == 1 else "S2"
        log.info(f"Scenario {scenario}: {len(runs)} run(s): {', '.join(labels)}")

        names: dict[str, str] = {}
        metrics, flags = [], {}
        for run, ctx in zip(runs, contexts):
            start = time.monotonic()
            log.info(f"Analyzing {run.label} ...")
            result = analyses.run_all(run.staged_dir)
            for name, reason in result.unavailable.items():
                log.warn(f"{run.label}: {name} analysis unavailable: {reason}")
            m = normalize.normalize_run(run.label, result, top_n=opts.top_n, names=names,
                                        batch_size=ctx.batch_size)
            metrics.append(m)
            flags[run.label] = heuristics.flags_for(m, ctx.compute_capability)
            log.info(f"  {run.label}: {m.steps or 'no'} steps, "
                     f"flags: {', '.join(f.id for f in flags[run.label]) or 'none'} "
                     f"({time.monotonic() - start:.1f} s)")

    diff, distinguishable = context.config_diff(contexts)
    b = bundle.build(scenario=scenario, metrics=metrics, contexts=contexts, flags=flags,
                     baseline=baseline, config_diff=diff, distinguishable=distinguishable,
                     data_quality=warnings, cluster=context.cluster_name(), kernel_names=names)
    return b, bundle.render_markdown(b)


def run(opts: Options, log: Log | None = None) -> int:
    log = log or Log()
    out = opts.output_dir
    try:
        b, bundle_md = analyze(opts, log)
    except (ingest.IngestError, ValueError) as e:
        log.error(str(e))
        return 1

    out.mkdir(parents=True, exist_ok=True)
    system, user = build_prompt(b, bundle_md)
    prompt_md = prompt_markdown(system, user)
    (out / "bundle.json").write_text(json.dumps(b, indent=2) + "\n")
    (out / "bundle.md").write_text(bundle_md)
    (out / "prompt.md").write_text(prompt_md)

    tokens = int(len(system + user) / settings.CHARS_PER_TOKEN)
    log.info(f"Prompt: about {tokens:,} tokens (estimate).")
    if tokens > settings.MAX_PROMPT_TOKENS:
        log.warn(f"The prompt is larger than {settings.MAX_PROMPT_TOKENS:,} tokens; "
                 "consider a smaller --top-n.")

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
