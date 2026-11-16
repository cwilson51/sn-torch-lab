"""Compare SHARCNET Brine chat models on analyze-traces prompts (spec §14, M2 model eval).

Manual tool, not part of the installed package or of pytest (it needs a Brine key and the
network). For each report directory written by ``torch-lab analyze-traces`` (it holds
``prompt.md`` and ``bundle.json``), every model gets exactly the prompt in ``prompt.md``.
Each answer is scored on three automatic checks:

* ungrounded numbers: numbers with a unit (ms, µs, %, x, samples/s) that match no number
  in the bundle;
* invented names: `backticked` identifiers, and kernel-like names, that appear nowhere in
  the prompt the model was given;
* flag mismatches: flag IDs cited that fired in no run, and fired flags never mentioned.

Fewer issues is better. Pick the default model in hta_llm/settings.py by hand, after
reading a few of the reports as well: the checks catch hallucination, not bad advice.

    uv run python scripts/eval_models.py hta-report/ other-report/ --models A,B --out eval

By default every chat model on Brine is tried (embedding, speech and proxy-alias models are
skipped). Reasoning models spend output tokens on thinking, and large models can be slow:
raise --max-tokens and --timeout so they get a fair run.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from torch_trace_lab.hta_llm import brine, sanitize, settings
from torch_trace_lab.hta_llm.heuristics import MEANINGS

NUMBER_WITH_UNIT = re.compile(
    r"(?<![\w.])([-+−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+−]?\d+(?:\.\d+)?)\s*"
    r"(ms|µs|us|%|×|x(?![a-z])|samples/s)", re.I)
NUMBER = re.compile(r"[-+−]?\d+(?:\.\d+)?")
BACKTICKED = re.compile(r"`([^`\n]{2,120})`")
KERNEL_LIKE = re.compile(r"\b[\w:<>,]*(?:gemm|xmma|cudnn|triton_|_kernel|Kernel)[\w:<>,]*", re.I)


# ---------------------------------------------------------------------------- scoring

def bundle_numbers(bundle) -> list[float]:
    """Every number in the bundle, including numbers inside strings (evidence, notes)."""
    found: list[float] = []

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            found.append(float(x))
        elif isinstance(x, str):
            found.extend(float(n.replace("−", "-")) for n in NUMBER.findall(x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(bundle)
    return found


def _grounded(text: str, numbers: list[float]) -> bool:
    value = abs(float(text.replace(",", "").replace("−", "-")))
    decimals = len(text.split(".")[1]) if "." in text else 0
    tolerance = 0.5 * 10 ** -decimals + 1e-9  # the report may round a bundle value
    return any(abs(value - abs(n)) <= tolerance for n in numbers)


def ungrounded_numbers(report: str, bundle: dict) -> tuple[list[str], int]:
    """(numbers with units not found in the bundle, how many numbers with units were checked)."""
    numbers = bundle_numbers(bundle)
    checked, bad = 0, []
    for m in NUMBER_WITH_UNIT.finditer(report):
        checked += 1
        if not _grounded(m.group(1), numbers):
            bad.append(m.group(0).strip())
    return bad, checked


def invented_names(report: str, prompt: str) -> list[str]:
    """Identifiers in the report that the prompt never mentions."""
    seen = []
    kernel_like = [n for n in KERNEL_LIKE.findall(report) if "_" in n or "::" in n]
    for name in BACKTICKED.findall(report) + kernel_like:
        name = name.strip()
        if name and name not in prompt and name not in seen:
            seen.append(name)
    return seen


def flag_mismatches(report: str, bundle: dict) -> tuple[list[str], list[str]]:
    """(flags cited that fired in no run, flags that fired but the report never mentions)."""
    fired = {f["id"] for r in bundle.get("runs", []) for f in r.get("flags", [])}
    cited = {flag for flag in MEANINGS if re.search(rf"\b{flag}\b", report)}
    return sorted(cited - fired), sorted(fired - cited)


@dataclass
class Score:
    model: str
    bundle: str
    numbers_checked: int = 0
    ungrounded: list[str] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    cited_not_fired: list[str] = field(default_factory=list)
    fired_not_mentioned: list[str] = field(default_factory=list)
    latency_s: float | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def issues(self) -> int:
        return (len(self.ungrounded) + len(self.invented) + len(self.cited_not_fired)
                + len(self.fired_not_mentioned))


def score(model: str, name: str, report: str, prompt: str, bundle: dict) -> Score:
    s = Score(model, name)
    s.ungrounded, s.numbers_checked = ungrounded_numbers(report, bundle)
    s.invented = invented_names(report, prompt)
    s.cited_not_fired, s.fired_not_mentioned = flag_mismatches(report, bundle)
    return s


def summary_table(scores: list[Score]) -> str:
    models = sorted({s.model for s in scores})
    rows = []
    for model in models:
        mine = [s for s in scores if s.model == model]
        ok = [s for s in mine if s.error is None]
        checked = sum(s.numbers_checked for s in ok)
        rows.append((
            sum(s.issues for s in ok) + 10 * (len(mine) - len(ok)), model,
            f"{sum(len(s.ungrounded) for s in ok)}/{checked}",
            sum(len(s.invented) for s in ok),
            sum(len(s.cited_not_fired) for s in ok),
            sum(len(s.fired_not_mentioned) for s in ok),
            len(mine) - len(ok),
            (f"{sum(s.latency_s for s in ok) / len(ok):.0f}" if ok else "–"),
        ))
    rows.sort()
    head = ("| model | issues | ungrounded numbers | invented names | flags cited, not fired "
            "| fired flags not mentioned | errors | mean latency (s) |\n"
            "|---|---|---|---|---|---|---|---|")
    body = "\n".join(f"| {m} | {issues} | {u} | {i} | {c} | {f} | {e} | {lat} |"
                     for issues, m, u, i, c, f, e, lat in rows)
    return f"{head}\n{body}\n\nAn error counts as 10 issues."


def details(scores: list[Score]) -> str:
    out = []
    for s in scores:
        out.append(f"### {s.model} on {s.bundle}")
        if s.error:
            out.append(f"- error: {s.error}")
        else:
            out.append(f"- finish_reason: {s.finish_reason}; usage: {json.dumps(s.usage)}; "
                       f"latency: {s.latency_s:.0f} s")
        for label, items in [("ungrounded numbers", s.ungrounded),
                             ("invented names", s.invented),
                             ("flags cited, not fired", s.cited_not_fired),
                             ("fired flags not mentioned", s.fired_not_mentioned)]:
            if items:
                out.append(f"- {label}: " + ", ".join(f"`{x}`" for x in items[:20]))
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------- running

def split_prompt(prompt_md: str) -> tuple[str, str]:
    """Undo pipeline.prompt_markdown: (system, user)."""
    head, sep, user = prompt_md.partition("\n\n# User prompt\n\n")
    if not sep or not head.startswith("# System prompt\n\n"):
        raise ValueError("not a prompt.md written by torch-lab analyze-traces")
    return head[len("# System prompt\n\n"):], user


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs="+", type=Path,
                        help="analyze-traces output directories (with prompt.md, bundle.json)")
    parser.add_argument("--models", help="Comma-separated models (default: every chat model "
                                         "on Brine).")
    parser.add_argument("--max-tokens", type=int, default=settings.MAX_TOKENS,
                        help=f"Output token limit per report (default: {settings.MAX_TOKENS}, "
                             "as analyze-traces uses). Raise it for reasoning models.")
    parser.add_argument("--timeout", type=float, default=settings.REQUEST_TIMEOUT_S,
                        help=f"Seconds to wait per request (default: "
                             f"{settings.REQUEST_TIMEOUT_S}). Each request is retried "
                             f"{settings.RETRIES} times.")
    parser.add_argument("--out", type=Path, default=Path("eval"),
                        help="Where to write reports and summary.md (default: ./eval).")
    args = parser.parse_args(argv)

    inputs = []
    for d in args.reports:
        prompt_md = (d / "prompt.md").read_text()
        system, user = split_prompt(prompt_md)
        bundle = json.loads((d / "bundle.json").read_text())
        inputs.append((d.resolve().name, prompt_md, system, user, bundle))

    try:
        key = brine.load_key()
        client = brine.Client(key, timeout=args.timeout)
        models = (args.models.split(",") if args.models
                  else [m for m in client.models() if brine.is_chat_model(m)])
    except brine.BrineError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    for name, prompt_md, *_ in inputs:
        if leaks := sanitize.find_leaks(prompt_md, key=key):
            print(f"Error: {name}/prompt.md contains {'; '.join(leaks)}; not sending it.",
                  file=sys.stderr)
            return 1

    scores = []
    for model in models:
        for name, prompt_md, system, user, bundle in inputs:
            print(f"{model} on {name} ...", flush=True)
            start = time.monotonic()
            try:
                completion = client.chat(model, system, user, max_tokens=args.max_tokens)
            except brine.BrineError as e:
                scores.append(Score(model, name, error=str(e)[:200]))
                continue
            report = completion.content
            target = args.out / model.replace("/", "_") / f"{name}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(report)
            s = score(model, name, report, prompt_md, bundle)
            s.latency_s = time.monotonic() - start
            s.finish_reason, s.usage = completion.finish_reason, completion.usage
            scores.append(s)

    table = summary_table(scores)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(f"# Brine model eval\n\n{table}\n\n## Details\n\n"
                                         f"{details(scores)}")
    print(table)
    print(f"\nWrote reports and summary.md to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
