"""Thin wrappers over HTA's TraceAnalysis (traceinsight 0.6.1, import name ``hta``).

Each wrapper returns a plain DataFrame with stable column names and a ``rank`` column, so
HTA API churn stays here. ``run_ranks`` splits the results into one ``RunAnalysis`` per
rank. A failing analysis is recorded as unavailable and the others still run (spec §6).
"""

import contextlib
import io
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from torch_trace_lab.hta_llm import frameworks

ANALYSES = ("temporal", "idle", "kernels", "launch", "comm")


@dataclass
class RunAnalysis:
    """One rank's results."""
    steps: int | None = None            # ProfilerSteps kept (the last, partial one is dropped)
    iterations: list[int] | None = None  # the kept ProfilerStep numbers (for the trace diff)
    step_time_us: float | None = None   # mean ProfilerStep duration
    temporal: pd.DataFrame | None = None  # idle_us, compute_us, non_compute_us, kernel_us (+ pctg)
    idle: pd.DataFrame | None = None      # category, idle_us, ratio
    kernels: pd.DataFrame | None = None   # name, kernel_type, sum_us, mean_us, count
    launch: pd.DataFrame | None = None    # cpu_duration_us, gpu_duration_us, launch_delay_us
    comm: dict | None = None              # comm_us, overlapped_us, by_collective, allreduce_*
    unavailable: dict[str, str] = field(default_factory=dict)


@contextlib.contextmanager
def quiet():
    """Silence HTA's logging and prints; it is chatty on every parse."""
    hta_logger = logging.getLogger("hta")
    level = hta_logger.level
    hta_logger.setLevel(logging.ERROR)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        hta_logger.setLevel(level)


def load(trace_dir: Path):
    from hta.configs.config import logger as hta_config_logger
    from hta.trace_analysis import TraceAnalysis

    hta_config_logger.setLevel(logging.ERROR)
    with quiet():
        # HTA would drop every GPU kernel launched in the last ProfilerStep; keep them, and let
        # trim_to_complete_steps decide whether that step is complete.
        ta = TraceAnalysis(trace_dir=str(trace_dir), include_last_profiler_step=True)
    drop_gpu_annotations(ta.t)
    return ta


def drop_gpu_annotations(trace) -> None:
    """Remove ``gpu_user_annotation`` events (ProfilerStep#N, record_function ranges).

    CUDA traces mirror each annotation on the GPU timeline with a stream, so HTA would
    count them as kernels: they would appear in the kernel table, inflate GPU kernel time,
    and double the ProfilerStep count.
    """
    anno_id = trace.symbol_table.get_sym_id_map().get("gpu_user_annotation")
    if anno_id is None:
        return
    for rank, df in trace.traces.items():
        trace.traces[rank] = df[df["cat"] != anno_id]


def _ranks(ta) -> list[int]:
    return sorted(ta.t.traces)


def temporal(ta) -> pd.DataFrame:
    df = ta.get_temporal_breakdown(visualize=False)
    return df.rename(columns={
        "idle_time(us)": "idle_us", "compute_time(us)": "compute_us",
        "non_compute_time(us)": "non_compute_us", "kernel_time(us)": "kernel_us",
        "idle_time_pctg": "idle_pct", "compute_time_pctg": "compute_pct",
        "non_compute_time_pctg": "non_compute_pct",
    }).reset_index(drop=True)


def comm_only_streams(ta) -> set[tuple[int, int]]:
    """(rank, stream) pairs whose kernels are all communication (NCCL) kernels."""
    from hta.utils.utils import is_comm_kernel

    symbols = ta.t.symbol_table.get_sym_table()
    result = set()
    for rank, df in ta.t.traces.items():
        gpu = df[df["stream"].ne(-1)]
        for stream, names in gpu.groupby("stream")["name"]:
            if all(is_comm_kernel(str(symbols[n])) for n in names.unique()):
                result.add((rank, stream))
    return result


def idle(ta) -> pd.DataFrame:
    df, _ = ta.get_idle_time_breakdown(ranks=_ranks(ta), visualize=False, visualize_pctg=False)
    df = df.rename(columns={"idle_category": "category", "idle_time": "idle_us"})
    # A stream that only runs collectives is idle most of the step by design; its idle time
    # says nothing about the GPU waiting for work, so it is left out.
    skip = comm_only_streams(ta)
    if skip and "stream" in df.columns:
        df = df[[(r, s) not in skip for r, s in zip(df["rank"], df["stream"])]]
    # Sum over streams, so ratios are of each rank's total idle time.
    out = df.groupby(["rank", "category"], as_index=False)["idle_us"].sum()
    totals = out.groupby("rank")["idle_us"].transform("sum")
    out["ratio"] = (out["idle_us"] / totals.where(totals != 0)).fillna(0.0)
    return out


def kernels(ta) -> pd.DataFrame:
    _, df = ta.get_gpu_kernel_breakdown(visualize=False, duration_ratio=1.0,
                                        num_kernels=1_000_000, include_memory_kernels=True)
    df = df.rename(columns={"sum (us)": "sum_us", "mean (us)": "mean_us"})
    df = df[df["name"] != "others"].copy()
    # HTA has no count column. sum/mean gives it, except for kernels shorter than 1 µs,
    # which the trace records as 0 µs (0/0); count those in the trace itself.
    count = (df["sum_us"] / df["mean_us"].where(df["mean_us"] > 0)).round()
    if count.isna().any():
        counts = kernel_counts(ta)
        count = count.fillna(pd.Series([counts.get((r, n), 0) for r, n in
                                        zip(df["rank"], df["name"])], index=df.index))
    df["count"] = count.astype(int)
    return df[["rank", "name", "kernel_type", "sum_us", "mean_us", "count"]].reset_index(drop=True)


def kernel_counts(ta) -> dict[tuple[int, str], int]:
    """(rank, kernel name) -> number of GPU events with that name."""
    symbols = ta.t.symbol_table.get_sym_table()
    result = {}
    for rank, df in ta.t.traces.items():
        gpu = df[df["stream"].ne(-1)]
        for name, n in gpu["name"].value_counts().items():
            result[(rank, str(symbols[name]))] = int(n)
    return result


def launch(ta) -> pd.DataFrame:
    stats = ta.get_cuda_kernel_launch_stats(ranks=_ranks(ta), visualize=False)
    frames = [df.assign(rank=rank) for rank, df in stats.items()]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["rank"])
    return df.rename(columns={"cpu_duration": "cpu_duration_us", "gpu_duration": "gpu_duration_us",
                              "launch_delay": "launch_delay_us"})


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted, non-overlapping union of (start, end) intervals."""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def intersection_length(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total length where two merged interval lists overlap."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def comm_overlap(kernels: list[tuple[str, float, float]]) -> dict:
    """Communication time on one rank's GPU, and how much of it overlaps compute.

    ``kernels`` are (kernel_type, start, duration) with HTA's kernel types (COMMUNICATION,
    COMPUTATION, MEMORY, OTHER). Each kind's intervals are merged first, so time counts
    once however many streams are busy at once. Returns ``comm_us`` (time with any comm
    kernel running) and ``overlapped_us`` (the part of it with a compute kernel running too).
    This is HTA's ``get_comm_comp_overlap`` definition, without its NaN for a rank with no
    communication.
    """
    comm_iv = merge_intervals([(s, s + d) for t, s, d in kernels if t == "COMMUNICATION"])
    comp_iv = merge_intervals([(s, s + d) for t, s, d in kernels if t == "COMPUTATION"])
    return {"comm_us": float(sum(e - s for s, e in comm_iv)),
            "overlapped_us": float(intersection_length(comm_iv, comp_iv))}


def collective_breakdown(kernels: list[tuple[str, str | None, float, float]]) -> dict:
    """Comm time per collective type, and how much all-reduce time overlaps compute.

    ``kernels`` are (kernel_type, collective, start, duration); ``collective`` is
    frameworks.collective_type of the kernel name (None for non-NCCL kernels). Each
    collective's intervals are merged, as in comm_overlap. Returns ``by_collective``
    ({collective: µs}, only those that ran), ``allreduce_us`` and ``allreduce_overlapped_us``.
    """
    comp_iv = merge_intervals([(s, s + d) for t, _, s, d in kernels if t == "COMPUTATION"])
    by_collective = {}
    for c in frameworks.COLLECTIVES:
        iv = merge_intervals([(s, s + d) for _, k, s, d in kernels if k == c])
        if iv:
            by_collective[c] = float(sum(e - s for s, e in iv))
    ar_iv = merge_intervals([(s, s + d) for _, k, s, d in kernels if k == "all-reduce"])
    return {"by_collective": by_collective,
            "allreduce_us": by_collective.get("all-reduce", 0.0),
            "allreduce_overlapped_us": float(intersection_length(ar_iv, comp_iv))}


def comm(ta) -> pd.DataFrame:
    from hta.utils.utils import get_kernel_type

    symbols = ta.t.symbol_table.get_sym_table()
    types: dict[int, tuple[str, str | None]] = {}
    rows = []
    for rank in _ranks(ta):
        df = ta.t.traces[rank]
        gpu = df[df["stream"].ne(-1)]
        kinds = []
        for name, ts, dur in zip(gpu["name"], gpu["ts"], gpu["dur"]):
            if name not in types:
                full = str(symbols[name])
                kind = get_kernel_type(full)
                types[name] = (kind, frameworks.collective_type(full)
                               if kind == "COMMUNICATION" else None)
            kinds.append((*types[name], float(ts), float(dur)))
        rows.append({"rank": rank, **comm_overlap([(k, s, d) for k, _, s, d in kinds]),
                     **collective_breakdown(kinds)})
    return pd.DataFrame(rows)


# CUDA runtime/driver calls that put work on the GPU, each correlated with one GPU event
# (cudaLaunchKernel, cudaLaunchKernelExC_v11060, cuLaunchKernel, cudaMemcpyAsync, ...).
GPU_LAUNCH = re.compile(r"^cu(da)?(LaunchKernel|LaunchCooperativeKernel|Memcpy|Memset)")


def _last_step_complete(df: pd.DataFrame, iterations: list[int], launch_ids: list[int]) -> bool:
    """Whether the last step lost no GPU events at export.

    Counts, per step, the GPU launches without a GPU event in the trace. Some launches never
    get one in any step (e.g. zero-byte cudaMemsetAsync calls), so the last step is complete
    if it misses no more than the worst earlier step; a step cut off at export misses more.
    """
    launches = df[df["name"].isin(launch_ids) & (df["stream"] < 0)
                  & df["iteration"].isin(iterations) & (df["correlation"] > 0)]
    recorded = set(df.loc[df["stream"] >= 0, "correlation"])
    missing = (~launches["correlation"].isin(recorded)).groupby(launches["iteration"]).sum()
    missing = missing.reindex(iterations, fill_value=0)
    return int(missing.iloc[-1]) <= int(missing.iloc[:-1].max())


def trim_to_complete_steps(ta) -> dict[int, tuple[int, float, list[int]]]:
    """Keep only events from complete ProfilerSteps; return
    {rank: (steps, mean step µs, kept step numbers)}.

    The last recorded step is kept only if it is complete: no more of its kernel, memcpy and
    memset launches lack a GPU event than in earlier steps. A step cut off at export lacks more.
    Events outside any step are dropped. Every analysis then sees exactly the same steps, so
    totals divided by ``steps`` are true per-step means. Ranks without ProfilerStep
    annotations are left untrimmed and not returned.
    """
    trace = ta.t
    symbols = trace.symbol_table.get_sym_table()
    step_ids = [i for i, s in enumerate(symbols) if str(s).startswith("ProfilerStep#")]
    launch_ids = [i for i, s in enumerate(symbols) if GPU_LAUNCH.search(str(s))]
    result = {}
    for rank, df in trace.traces.items():
        steps = df[df["name"].isin(step_ids) & (df["stream"] < 0)]  # CPU-side annotations
        iterations = sorted(i for i in steps["iteration"].unique() if i >= 0)
        if not iterations:
            continue
        kept = iterations
        if len(iterations) > 1 and not _last_step_complete(df, iterations, launch_ids):
            kept = iterations[:-1]
        trace.traces[rank] = df[df["iteration"].isin(kept)]
        kept_steps = steps[steps["iteration"].isin(kept)]
        result[rank] = (len(kept), float(kept_steps["dur"].mean()), [int(i) for i in kept])
    return result


def _rank_slice(df: pd.DataFrame | None, rank: int) -> pd.DataFrame | None:
    if df is None or "rank" not in df.columns:
        return df
    return df[df["rank"] == rank].drop(columns="rank").reset_index(drop=True)


def run_ranks(trace_dir: Path) -> dict[int, RunAnalysis]:
    """Run every analysis on every rank in ``trace_dir``: {rank: RunAnalysis}, sorted by rank."""
    try:
        ta = load(trace_dir)
    except Exception as e:  # noqa: BLE001 - HTA raises many types
        reason = f"trace could not be loaded by HTA: {e}"
        return {0: RunAnalysis(unavailable={name: reason for name in ANALYSES})}
    results = {rank: RunAnalysis() for rank in _ranks(ta)}
    with quiet():
        try:
            for rank, (steps, step_us, kept) in trim_to_complete_steps(ta).items():
                r = results[rank]
                r.steps, r.step_time_us, r.iterations = steps, step_us, kept
        except Exception as e:  # noqa: BLE001
            for r in results.values():
                r.unavailable["steps"] = f"could not split the trace into steps: {e}"
        for name, fn in [("temporal", temporal), ("idle", idle), ("kernels", kernels),
                         ("launch", launch), ("comm", comm)]:
            try:
                df = fn(ta)
            except Exception as e:  # noqa: BLE001
                for r in results.values():
                    r.unavailable[name] = f"{type(e).__name__}: {e}"
                continue
            for rank, r in results.items():
                part = _rank_slice(df, rank)
                if name == "comm":
                    part = part.iloc[0].to_dict() if part is not None and not part.empty else None
                setattr(r, name, part)
    return results


# Event categories the trace diff compares, per HTA device type. Everything else (python_function
# events with path-like names, annotations, CUDA runtime calls) is left out (spec §8a).
DIFF_CATEGORIES = {"GPU": ("kernel", "gpu_memcpy", "gpu_memset"), "CPU": ("cpu_op",)}
DIFF_COLUMNS = ["name", "device", "control_count", "test_count", "control_us", "test_us"]


def load_rank_trace(trace_dir: Path, label: str):
    """One staged rank as an HTA ``LabeledTrace`` for ``TraceDiff``, loaded like ``load``
    (last profiler step's GPU events kept, GPU annotations dropped)."""
    from hta.common.trace import Trace
    from hta.configs.config import logger as hta_config_logger
    from hta.trace_diff import LabeledTrace

    hta_config_logger.setLevel(logging.ERROR)
    with quiet():
        trace = Trace(trace_dir=str(trace_dir))
        trace.load_traces(include_last_profiler_step=True)
        drop_gpu_annotations(trace)
        return LabeledTrace(label=label, t=trace)  # already parsed: not parsed again


def _names_in(lt, categories: tuple[str, ...]) -> set[str]:
    sym_ids = lt.t.symbol_table.get_sym_id_map()
    cat_ids = [sym_ids[c] for c in categories if c in sym_ids]
    symbols = lt.t.symbol_table.get_sym_table()
    names = set()
    for rank in lt.ranks():
        df = lt.t.get_trace(rank)
        names |= {str(symbols[n]) for n in df.loc[df["cat"].isin(cat_ids), "name"].unique()}
    return names


def trace_diff(control, test, control_iterations: list[int] | None,
               test_iterations: list[int] | None) -> pd.DataFrame:
    """HTA ``TraceDiff.compare_traces`` of two ``LabeledTrace``s (one rank each), over the
    given (complete) profiler steps, for GPU kernels/memcpy/memset and CPU ops only.

    Returns DIFF_COLUMNS: per name, the call count and total µs in each trace (totals over
    the steps, not per step). CPU op times include their child ops, as in the trace.
    """
    from hta.trace_diff import DeviceType, TraceDiff

    frames = []
    with quiet():
        for device, device_type in (("GPU", DeviceType.GPU), ("CPU", DeviceType.CPU)):
            df = TraceDiff.compare_traces(control, test, control_iteration=control_iterations,
                                          test_iteration=test_iterations,
                                          device_type=device_type)
            allowed = (_names_in(control, DIFF_CATEGORIES[device])
                       | _names_in(test, DIFF_CATEGORIES[device]))
            df = df[df.index.isin(allowed)]
            c, t = control.label, test.label
            frames.append(pd.DataFrame({
                "name": df.index.astype(str), "device": device,
                "control_count": df[f"{c}_counts"].astype(int).to_numpy(),
                "test_count": df[f"{t}_counts"].astype(int).to_numpy(),
                "control_us": df[f"{c}_total_duration"].astype(float).to_numpy(),
                "test_us": df[f"{t}_total_duration"].astype(float).to_numpy()}))
    return pd.concat(frames, ignore_index=True)[DIFF_COLUMNS]


def run_all(trace_dir: Path) -> RunAnalysis:
    """The lowest rank's results (a single-rank run has only one)."""
    ranks = run_ranks(trace_dir)
    return ranks[min(ranks)]
