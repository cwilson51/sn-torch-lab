"""Run context for the bundle: allowlisted trace facts, embedded config, notes, config diff (§5.2).

Everything here is an allowlist. Trace metadata outside these fields (``traceName``,
``trace_id``, ``host_name``, ...) is never read.
"""

import os
from dataclasses import dataclass, field

from torch_trace_lab.hta_llm.ingest import Run

# Allowlisted keys of the embedded `torch_lab` object, and the types each may have.
TORCH_LAB_KEYS = {
    "model": str,
    "batch_size_per_gpu": int,
    "dataloader_workers": int,
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
    config: dict = field(default_factory=dict)   # name -> {"value", "source"}
    notes: str | None = None

    @property
    def batch_size(self) -> int | None:
        entry = self.config.get("batch_size_per_gpu")
        return entry["value"] if entry else None

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


def build_contexts(runs: list[Run], notes: dict[str, str]) -> list[RunContext]:
    contexts = []
    for run in runs:
        meta = run.rank0.metadata
        contexts.append(RunContext(run.label, trace_facts(meta), embedded_config(meta),
                                   notes.get(run.label)))
    return contexts


def cluster_name() -> str | None:
    name = os.environ.get("CC_CLUSTER", "").strip()
    return name[:40] or None


DIFF_FACTS = ("gpu", "compute_capability", "gpu_memory_gb", "world_size")


def config_diff(contexts: list[RunContext]) -> tuple[list[dict], bool]:
    """Fields whose values differ across runs. Returns (rows, distinguishable)."""
    rows = []
    config_keys = [k for k in TORCH_LAB_KEYS if any(k in c.config for c in contexts)]
    for key in config_keys:
        values = {c.label: c.config[key]["value"] if key in c.config else "unknown"
                  for c in contexts}
        if len({repr(v) for v in values.values()}) > 1:
            rows.append({"field": key, "source": "embedded", "values": values})
    for key in DIFF_FACTS:
        values = {c.label: c.facts.get(key, "unknown") for c in contexts}
        if len({repr(v) for v in values.values()}) > 1:
            rows.append({"field": key, "source": "trace", "values": values})
    noted = [c.label for c in contexts if c.notes]
    known = [r for r in rows if not all(v == "unknown" for v in r["values"].values())]
    distinguishable = bool(known) or len(noted) == len(contexts)
    return rows, distinguishable
