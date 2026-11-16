"""Find trace files, group them into runs, validate ranks, and stage each run for HTA (spec §5.1).

File names are never parsed. Structure comes from directories; rank identity comes from
each trace's own ``distributedInfo``.
"""

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from torch_trace_lab.hta_llm.sanitize import unsafe_label

LAYOUT_HELP = """\
Expected layout (all names are arbitrary):
  one run:       my-run/trace.json               (or a single trace file)
  several runs:  experiments/baseline/trace.json
                 experiments/bs128-amp/trace.json
Each run directory holds exactly one run; each trace file in it is one rank."""


class IngestError(Exception):
    """The inputs can't be analyzed; the message says why and what to do."""


@dataclass
class TraceInfo:
    path: Path
    metadata: dict          # top-level keys except traceEvents (not allowlisted yet)
    has_gpu_kernels: bool
    has_profiler_steps: bool

    @property
    def distributed(self) -> dict | None:
        info = self.metadata.get("distributedInfo")
        return info if isinstance(info, dict) else None


@dataclass
class Run:
    label: str
    traces: list[TraceInfo]  # sorted by rank
    staged_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def rank0(self) -> TraceInfo:
        return self.traces[0]


def _is_trace_name(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".json") or path.name.endswith(".json.gz"))


def read_trace(path: Path) -> dict | None:
    """Return the parsed trace, or None if the file is not a Kineto trace."""
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt") as f:
            data = json.load(f)
    except (OSError, ValueError, EOFError):
        return None
    if isinstance(data, dict) and "traceEvents" in data:
        return data
    return None


def inspect_trace(path: Path) -> TraceInfo | None:
    data = read_trace(path)
    if data is None:
        return None
    events = data.get("traceEvents") or []
    has_kernels = False
    has_steps = False
    for e in events:
        if not isinstance(e, dict):
            continue
        cat = e.get("cat")
        if cat == "kernel":
            has_kernels = True
        elif cat == "user_annotation" and str(e.get("name", "")).startswith("ProfilerStep#"):
            has_steps = True
        if has_kernels and has_steps:
            break
    metadata = {k: v for k, v in data.items() if k != "traceEvents"}
    return TraceInfo(path, metadata, has_kernels, has_steps)


def _trace_candidates(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if _is_trace_name(p))


def _traces_in(directory: Path) -> list[TraceInfo]:
    return [info for p in _trace_candidates(directory) if (info := inspect_trace(p))]


def discover(paths: list[Path]) -> list[tuple[str, list[TraceInfo]]]:
    """Turn the command-line paths into (label, traces) pairs, one per run."""
    paths = [Path(p) for p in paths]
    for p in paths:
        if not p.exists():
            raise IngestError(f"{p.name}: no such file or directory.")

    if len(paths) == 1 and paths[0].is_file():
        info = inspect_trace(paths[0])
        if info is None:
            raise IngestError(f"{paths[0].name} is not a PyTorch profiler trace "
                              "(a JSON object with a 'traceEvents' key).")
        return [("run-1", [info])]

    if len(paths) == 1:
        directory = paths[0]
        traces = _traces_in(directory)
        subdirs = sorted(d for d in directory.iterdir() if d.is_dir() and _traces_in(d))
        if traces and subdirs:
            raise IngestError(
                f"{directory.name}/ contains both trace files and run subdirectories "
                f"({', '.join(d.name for d in subdirs)}). Move the trace files into their "
                f"own run directory.\n{LAYOUT_HELP}")
        if traces:
            return [(directory.name, traces)]
        if subdirs:
            return [(d.name, _traces_in(d)) for d in subdirs]
        raise IngestError(f"No trace files found in {directory.name}/ or its subdirectories.\n"
                          f"{LAYOUT_HELP}")

    runs = []
    for p in paths:
        if p.is_file():
            raise IngestError(f"{p.name}: to compare runs, pass run directories, not trace files.\n"
                              f"{LAYOUT_HELP}")
        traces = _traces_in(p)
        if not traces:
            raise IngestError(f"No trace files found in {p.name}/.\n{LAYOUT_HELP}")
        runs.append((p.name, traces))
    return runs


def validate_ranks(label: str, traces: list[TraceInfo]) -> tuple[list[TraceInfo], list[str]]:
    """Check rank identity; return the traces sorted by rank, and any warnings."""
    if len(traces) == 1:
        info = traces[0].distributed or {}
        ws = info.get("world_size")
        if isinstance(ws, int) and ws > 1:
            return traces, [f"{label}: the trace is one rank of a {ws}-rank run; the other ranks "
                            "are missing, so this is a single-rank analysis."]
        return traces, []

    several_runs = (f"{label}/ looks like it holds several runs in one directory. Put each run "
                    f"in its own directory.\n{LAYOUT_HELP}")
    infos = [t.distributed for t in traces]
    if any(i is None or not isinstance(i.get("rank"), int) for i in infos):
        raise IngestError(f"{several_runs}\n(Some trace files have no distributedInfo rank.)")
    ranks = [i["rank"] for i in infos]
    if len(set(ranks)) != len(ranks):
        raise IngestError(f"{several_runs}\n(Duplicate ranks: {sorted(ranks)}.)")
    sizes = {i.get("world_size") for i in infos}
    if len(sizes) != 1 or not isinstance(next(iter(sizes)), int):
        raise IngestError(f"{label}/: trace files disagree on world_size ({sorted(map(str, sizes))}). "
                          f"{several_runs}")
    world_size = sizes.pop()
    missing = sorted(set(range(world_size)) - set(ranks))
    if missing:
        raise IngestError(f"{label}/: incomplete run, ranks {missing} of world_size {world_size} "
                          "are missing.")
    raise IngestError(f"{label}/: multi-GPU runs ({world_size} ranks) are not yet supported. "
                      "Analyze one rank's trace file on its own for now.")


def check_content(label: str, traces: list[TraceInfo]) -> list[str]:
    for t in traces:
        if not t.has_gpu_kernels:
            raise IngestError(
                f"{label}: {t.path.name} has no GPU kernel events. The analysis needs a trace "
                "recorded on a CUDA GPU. Note that the example training script falls back to "
                "CPU when no GPU is available (for example on a Mac).")
    if not all(t.has_profiler_steps for t in traces):
        return [f"{label}: no ProfilerStep annotations; times are trace totals, not per step. "
                "Call prof.step() once per training iteration."]
    return []


def clean_labels(labels: list[str], environ=None) -> tuple[list[str], list[str]]:
    """Replace labels that look sensitive with run-N, and make duplicates unique."""
    warnings, cleaned = [], []
    for i, label in enumerate(labels, start=1):
        if unsafe_label(label, environ=environ):
            warnings.append(f"Run {i}: its directory name looks like it contains a path, "
                            f"username, hostname or job ID, so it is labelled run-{i}.")
            label = f"run-{i}"
        cleaned.append(label)
    counts: dict[str, int] = {}
    for label in cleaned:
        counts[label] = counts.get(label, 0) + 1
    seen: dict[str, int] = {}
    result = []
    for label in cleaned:
        if counts[label] > 1:
            seen[label] = seen.get(label, 0) + 1
            label = f"{label}-{seen[label]}"
        result.append(label)
    return result, warnings


def stage(run: Run, root: Path) -> Path:
    """Symlink the run's validated trace files into their own directory for HTA."""
    directory = root / f"run{len(list(root.iterdir())) + 1}"
    directory.mkdir()
    for t in run.traces:
        target = directory / t.path.name
        try:
            os.symlink(t.path.resolve(), target)
        except OSError:
            import shutil
            shutil.copy2(t.path, target)
    run.staged_dir = directory
    return directory


def ingest(paths: list[Path], stage_root: Path, environ=None) -> tuple[list[Run], list[str]]:
    """Discover, validate and stage the runs. Returns (runs, data-quality warnings)."""
    found = discover(paths)
    labels, warnings = clean_labels([label for label, _ in found], environ=environ)
    runs = []
    for label, (_, traces) in zip(labels, found):
        traces, rank_warnings = validate_ranks(label, traces)
        warnings += rank_warnings
        warnings += check_content(label, traces)
        run = Run(label, traces)
        stage(run, stage_root)
        runs.append(run)
    return runs, warnings
