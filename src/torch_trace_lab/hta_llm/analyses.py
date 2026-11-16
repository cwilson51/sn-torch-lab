"""Thin wrappers over HTA's TraceAnalysis (traceinsight 0.6.1, import name ``hta``).

Each wrapper returns a plain DataFrame with stable column names, so HTA API churn stays here.
A failing analysis is recorded as unavailable and the others still run (spec §6).
"""

import contextlib
import io
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ANALYSES = ("temporal", "idle", "kernels", "launch")


@dataclass
class RunAnalysis:
    steps: int | None = None            # ProfilerSteps HTA analyzed (it drops the last, partial one)
    step_time_us: float | None = None   # mean ProfilerStep duration
    temporal: pd.DataFrame | None = None  # idle_us, compute_us, non_compute_us, kernel_us (+ pctg)
    idle: pd.DataFrame | None = None      # category, idle_us, ratio
    kernels: pd.DataFrame | None = None   # name, kernel_type, sum_us, mean_us, count
    launch: pd.DataFrame | None = None    # cpu_duration_us, gpu_duration_us, launch_delay_us
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
        ta = TraceAnalysis(trace_dir=str(trace_dir))
    drop_gpu_annotations(ta)
    return ta


def drop_gpu_annotations(ta) -> None:
    """Remove ``gpu_user_annotation`` events (ProfilerStep#N, record_function ranges).

    CUDA traces mirror each annotation on the GPU timeline with a stream, so HTA would
    count them as kernels: they would appear in the kernel table, inflate GPU kernel time,
    and double the ProfilerStep count.
    """
    anno_id = ta.t.symbol_table.get_sym_id_map().get("gpu_user_annotation")
    if anno_id is None:
        return
    for rank, df in ta.t.traces.items():
        ta.t.traces[rank] = df[df["cat"] != anno_id]


def temporal(ta) -> pd.DataFrame:
    df = ta.get_temporal_breakdown(visualize=False)
    df = df.rename(columns={
        "idle_time(us)": "idle_us", "compute_time(us)": "compute_us",
        "non_compute_time(us)": "non_compute_us", "kernel_time(us)": "kernel_us",
        "idle_time_pctg": "idle_pct", "compute_time_pctg": "compute_pct",
        "non_compute_time_pctg": "non_compute_pct",
    })
    return df[df["rank"] == df["rank"].min()].reset_index(drop=True)


def idle(ta) -> pd.DataFrame:
    df, _ = ta.get_idle_time_breakdown(visualize=False, visualize_pctg=False)
    df = df.rename(columns={"idle_category": "category", "idle_time": "idle_us",
                            "idle_time_ratio": "ratio"})
    df = df[df["rank"] == df["rank"].min()]
    # Sum over streams, so ratios are of the rank's total idle time.
    out = df.groupby("category", as_index=False)["idle_us"].sum()
    total = out["idle_us"].sum()
    out["ratio"] = out["idle_us"] / total if total else 0.0
    return out


def kernels(ta) -> pd.DataFrame:
    _, df = ta.get_gpu_kernel_breakdown(visualize=False, duration_ratio=1.0,
                                        num_kernels=1_000_000, include_memory_kernels=True)
    df = df.rename(columns={"sum (us)": "sum_us", "mean (us)": "mean_us"})
    df = df[df["rank"] == df["rank"].min()]
    df = df[df["name"] != "others"]
    df["count"] = (df["sum_us"] / df["mean_us"]).round().astype(int)
    return df[["name", "kernel_type", "sum_us", "mean_us", "count"]].reset_index(drop=True)


def launch(ta) -> pd.DataFrame:
    stats = ta.get_cuda_kernel_launch_stats(visualize=False)
    df = stats[min(stats)]
    return df.rename(columns={"cpu_duration": "cpu_duration_us", "gpu_duration": "gpu_duration_us",
                              "launch_delay": "launch_delay_us"}).reset_index(drop=True)


def trim_to_complete_steps(ta) -> tuple[int | None, float | None]:
    """Keep only events from complete ProfilerSteps; return (steps, mean step time in µs).

    The last recorded step is dropped, since it may be cut off at export. Every analysis then
    sees exactly the same steps, so totals divided by ``steps`` are true per-step means.
    HTA does its own trimming, but its cutoff shifts with GPU-side annotations, so it is not
    relied on. Without ProfilerStep annotations nothing is trimmed and (None, None) is returned.
    """
    trace = ta.t
    symbols = trace.symbol_table.get_sym_table()
    step_ids = [i for i, s in enumerate(symbols) if str(s).startswith("ProfilerStep#")]
    result = (None, None)
    for rank, df in trace.traces.items():
        steps = df[df["name"].isin(step_ids) & (df["stream"] < 0)]  # CPU-side annotations
        iterations = sorted(i for i in steps["iteration"].unique() if i >= 0)
        if not iterations:
            continue
        kept = iterations[:-1] if len(iterations) > 1 else iterations
        trace.traces[rank] = df[df["iteration"].isin(kept)]
        if rank == min(trace.traces):
            kept_steps = steps[steps["iteration"].isin(kept)]
            result = (len(kept), float(kept_steps["dur"].mean()))
    return result


def run_all(trace_dir: Path) -> RunAnalysis:
    result = RunAnalysis()
    try:
        ta = load(trace_dir)
    except Exception as e:  # noqa: BLE001 - HTA raises many types
        result.unavailable = {name: f"trace could not be loaded by HTA: {e}" for name in ANALYSES}
        return result
    with quiet():
        try:
            result.steps, result.step_time_us = trim_to_complete_steps(ta)
        except Exception as e:  # noqa: BLE001
            result.unavailable["steps"] = f"could not split the trace into steps: {e}"
        for name, fn in [("temporal", temporal), ("idle", idle), ("kernels", kernels),
                         ("launch", launch)]:
            try:
                setattr(result, name, fn(ta))
            except Exception as e:  # noqa: BLE001
                result.unavailable[name] = f"{type(e).__name__}: {e}"
    return result
