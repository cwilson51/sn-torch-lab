"""Run context for the bundle: allowlisted trace facts, config, notes, config diff (§5.2).

Everything here is an allowlist. Trace metadata outside these fields (``traceName``,
``trace_id``, ``host_name``, ...) is never read.

Config values come from three sources, in increasing priority: ``inferred`` from trace
contents (infer.py), ``embedded`` in the trace's ``torch_lab`` object, and ``user``
(``--meta``). The highest-priority value is used; an inferred value that disagrees with
it becomes a data-quality note.
"""

import os
from dataclasses import dataclass, field

from torch_trace_lab.hta_llm.ingest import Run

# Allowlisted keys of the embedded `torch_lab` object, and the types each may have.
TORCH_LAB_KEYS = {
    "model": str,
    "batch_size_per_gpu": int,
    "dataloader_workers": int,
    "pin_memory": bool,
    "torch_compile": bool,
    "precision": str,
    "torch_version": str,
    "cuda_version": str,
    "profiler_schedule": dict,
}
SCHEDULE_KEYS = ("skip_first", "wait", "warmup", "active")
MAX_TEXT = 120


@dataclass
class RunContext:
    label: str
    facts: dict = field(default_factory=dict)    # from trace metadata (source: trace)
    # name -> {"value", "source"} (+ "confidence", "evidence" when inferred)
    config: dict = field(default_factory=dict)
    notes: str | None = None

    @property
    def batch_size(self) -> int | None:
        entry = self.config.get("batch_size_per_gpu")
        value = entry["value"] if entry else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def compute_capability(self) -> float | None:
        return self.facts.get("compute_capability")


def trace_facts(metadata: dict) -> dict:
    facts = {}
    devices = metadata.get("deviceProperties")
    if isinstance(devices, list) and devices and isinstance(devices[0], dict):
        d = devices[0]
        if isinstance(d.get("name"), str):
            facts["gpu"] = d["name"][:MAX_TEXT]
        major, minor = d.get("computeMajor"), d.get("computeMinor")
        if isinstance(major, int) and isinstance(minor, int):
            facts["compute_capability"] = float(f"{major}.{minor}")
        if isinstance(d.get("totalGlobalMem"), (int, float)):
            facts["gpu_memory_gb"] = round(d["totalGlobalMem"] / 1e9, 1)
        if isinstance(d.get("numSms"), int):
            facts["sm_count"] = d["numSms"]
    for key in ("cuda_runtime_version", "cuda_driver_version"):
        if isinstance(metadata.get(key), (int, str)):
            facts[key] = metadata[key]
    dist = metadata.get("distributedInfo")
    ws = dist.get("world_size") if isinstance(dist, dict) else None
    facts["world_size"] = ws if isinstance(ws, int) else 1
    return facts


def embedded_config(metadata: dict) -> dict:
    raw = metadata.get("torch_lab")
    if not isinstance(raw, dict):
        return {}
    config = {}
    for key, kind in TORCH_LAB_KEYS.items():
        value = raw.get(key)
        if value is None:
            continue
        if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
            continue
        if kind is not int and not isinstance(value, kind):
            continue
        if kind is str:
            value = value[:MAX_TEXT]
        if kind is dict:
            value = {k: value[k] for k in SCHEDULE_KEYS
                     if isinstance(value.get(k), int) and not isinstance(value.get(k), bool)}
            if not value:
                continue
        config[key] = {"value": value, "source": "embedded"}
    return config


def parse_run_notes(notes: list[str], labels: list[str]) -> dict[str, str]:
    """``--run-note [LABEL=]TEXT``. Without a label, the note needs a single run."""
    result: dict[str, str] = {}
    for note in notes:
        label, sep, text = note.partition("=")
        if sep and label in labels:
            target = label
        elif len(labels) == 1:
            target, text = labels[0], note
        else:
            raise ValueError(f"--run-note {note!r}: start it with one of the run labels "
                             f"({', '.join(labels)}), as LABEL=\"text\".")
        text = " ".join(text.split())[:500]
        result[target] = f"{result[target]}; {text}" if target in result else text
    return result


SOURCES = ("inferred", "embedded", "user")  # increasing priority
SOURCE_NAMES = {"inferred": "the trace contents", "embedded": "the trace's torch_lab metadata",
                "user": "--meta"}


def precision_family(value) -> str | None:
    """bf16 / fp16 / tf32 / fp32 for any precision label (``"fp32 (TF32 convs)"`` is tf32)."""
    v = str(value).lower()
    for family, pattern in (("bf16", "bf16"), ("fp16", "fp16"), ("fp16", "half"),
                            ("tf32", "tf32"), ("fp32", "fp32"), ("fp32", "float32")):
        if pattern in v:
            return family
    return None


def same_value(key: str, a, b) -> bool:
    """Whether two config values agree, allowing for how inference reports them."""
    if key == "precision":
        fa, fb = precision_family(a), precision_family(b)
        if fa and fb:
            return fa == fb
    if key == "dataloader_workers" and ">0" in (a, b):
        other = b if a == ">0" else a
        return other == ">0" or (isinstance(other, int) and other > 0)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def resolve_config(label: str, layers: dict[str, dict]) -> tuple[dict, list[str]]:
    """Merge ``{source: {key: entry}}``; the highest-priority source wins.

    Returns (config, conflict notes). A conflict is an inferred value that disagrees with
    the embedded or user value that was used instead.
    """
    config: dict = {}
    for source in SOURCES:
        for key, entry in (layers.get(source) or {}).items():
            config[key] = {**entry, "source": source}
    conflicts = []
    for key, entry in (layers.get("inferred") or {}).items():
        used = config[key]
        if used["source"] == "inferred" or same_value(key, used["value"], entry["value"]):
            continue
        conflicts.append(
            f"{label}: {key}: using {_fmt_value(used['value'])} from "
            f"{SOURCE_NAMES[used['source']]}, but the trace contents suggest "
            f"{_fmt_value(entry['value'])} (inferred, {entry['confidence']} confidence: "
            f"{entry['evidence']}).")
    return config, conflicts


def _fmt_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_contexts(runs: list[Run], notes: dict[str, str], *,
                   inferred: dict[str, dict] | None = None,
                   user: dict[str, dict] | None = None) -> tuple[list[RunContext], list[str]]:
    """Contexts per run, and conflict notes for the data-quality section.

    ``inferred`` and ``user`` map run label -> {key: entry}.
    """
    contexts, conflicts = [], []
    for run in runs:
        meta = run.rank0.metadata
        config, run_conflicts = resolve_config(run.label, {
            "inferred": (inferred or {}).get(run.label, {}),
            "embedded": embedded_config(meta),
            "user": (user or {}).get(run.label, {}),
        })
        conflicts += run_conflicts
        contexts.append(RunContext(run.label, trace_facts(meta), config, notes.get(run.label)))
    return contexts, conflicts


def cluster_name() -> str | None:
    name = os.environ.get("CC_CLUSTER", "").strip()
    return name[:40] or None


DIFF_FACTS = ("gpu", "compute_capability", "gpu_memory_gb", "world_size")


def _low_confidence(entry: dict) -> bool:
    return entry.get("source") == "inferred" and entry.get("confidence") == "low"


def config_diff(contexts: list[RunContext]) -> tuple[list[dict], bool]:
    """Fields whose values differ across runs. Returns (rows, distinguishable).

    Each row has the field, its per-run values and sources, and ``source``: the shared
    source, or ``"mixed"`` when runs got the value from different sources.
    """
    rows = []
    config_keys: list[str] = []
    for c in contexts:
        config_keys += [k for k in c.config if k not in config_keys]
    for key in config_keys:
        # Low-confidence inferred values never create a diff row on their own: they are
        # left out of the comparison and shown as unknown if the field differs anyway.
        low = {c.label for c in contexts if key in c.config and _low_confidence(c.config[key])}
        values = {c.label: c.config[key]["value"] if key in c.config and c.label not in low
                  else "unknown" for c in contexts}
        compared = {repr(v) for label, v in values.items() if label not in low}
        if len(compared) > 1:
            sources = {c.label: c.config[key]["source"]
                       if key in c.config and c.label not in low else "unknown"
                       for c in contexts}
            known = {s for s in sources.values() if s != "unknown"}
            rows.append({"field": key, "source": known.pop() if len(known) == 1 else "mixed",
                         "sources": sources, "values": values})
    for key in DIFF_FACTS:
        values = {c.label: c.facts.get(key, "unknown") for c in contexts}
        if len({repr(v) for v in values.values()}) > 1:
            rows.append({"field": key, "source": "trace",
                         "sources": {c.label: "trace" for c in contexts}, "values": values})
    noted = [c.label for c in contexts if c.notes]
    known_rows = [r for r in rows if not all(v == "unknown" for v in r["values"].values())]
    distinguishable = bool(known_rows) or len(noted) == len(contexts)
    return rows, distinguishable
