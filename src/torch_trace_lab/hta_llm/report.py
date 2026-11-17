"""Render report.md: the LLM report, or a flags-only report without Brine."""

import re
from importlib.resources import files

from torch_trace_lab.hta_llm import tables

PERFETTO = ("Open the trace files in https://ui.perfetto.dev (or TensorBoard) to check the "
            "findings on the timeline.")


def prompt_file(name: str) -> str:
    return (files("torch_trace_lab") / "hta_llm" / "prompts" / name).read_text()


def remediation_sections() -> dict[str, str]:
    """Flag id -> its section of remediation.md."""
    text = prompt_file("remediation.md")
    parts = re.split(r"^## ", text, flags=re.M)
    return {p.split("\n", 1)[0].strip(): p.split("\n", 1)[1].strip() for p in parts if p.strip()}


def flags_only(bundle: dict, reason: str | None = None) -> str:
    fixes = remediation_sections()
    out = ["# Trace analysis report (flags only)", ""]
    if reason:
        out += [f"> No LLM report: {reason}", ""]
    out += ["This report lists the rule-based flags computed from the traces. Full tables are "
            "in `bundle.md`.", ""]
    out += ["## Summary", "", tables.render(bundle["derived"]), ""]
    seen = []
    for run in bundle["runs"]:
        out.append(f"## {run['label']}")
        out.append("")
        if run.get("slowest_rank") is not None:
            out += [f"{run['ranks_analyzed']} ranks analyzed. The slowest rank (most busy GPU "
                    f"time per step) is rank {run['slowest_rank']}; flags marked with a rank "
                    "come from its tables, the others compare all ranks.", ""]
        if not run["flags"]:
            out += ["No flags raised.", ""]
            continue
        for f in run["flags"]:
            out.append(f"- **{f['id']}** ({f['severity']}): {f['evidence']}. "
                       f"_Threshold: {f['threshold']}._")
            if f["id"] not in seen:
                seen.append(f["id"])
        out.append("")
    if seen:
        out += ["## Known fixes", ""]
        for flag_id in seen:
            out += [f"### {flag_id}", "", fixes.get(flag_id, "_No guide._"), ""]
    out += ["## Verify", "", PERFETTO, ""]
    return "\n".join(out)


def llm_report(content: str, model: str) -> str:
    return "\n".join([
        "# Trace analysis report", "",
        f"_Written by {model} on SHARCNET Brine from `bundle.md`. Numbers come from the bundle; "
        "check them there and in the trace viewer before acting on them._", "",
        content.strip(), "",
    ])
