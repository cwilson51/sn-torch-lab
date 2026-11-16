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
        return TraceAnalysis(trace_dir=str(trace_dir))


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


def profiler_steps(ta) -> tuple[int | None, float | None]:
    trace = ta.t
    rank = min(trace.traces)
    df = trace.get_trace(rank)
    symbols = trace.symbol_table.get_sym_table()
    step_ids = [i for i, s in enumerate(symbols) if str(s).startswith("ProfilerStep#")]
    steps = df[df["name"].isin(step_ids)]
    if steps.empty:
        return None, None
    return len(steps), float(steps["dur"].mean())


def run_all(trace_dir: Path) -> RunAnalysis:
    result = RunAnalysis()
    try:
        ta = load(trace_dir)
    except Exception as e:  # noqa: BLE001 - HTA raises many types
        result.unavailable = {name: f"trace could not be loaded by HTA: {e}" for name in ANALYSES}
        return result
    with quiet():
        try:
            result.steps, result.step_time_us = profiler_steps(ta)
        except Exception:  # noqa: BLE001
            pass
        for name, fn in [("temporal", temporal), ("idle", idle), ("kernels", kernels),
                         ("launch", launch)]:
            try:
                setattr(result, name, fn(ta))
            except Exception as e:  # noqa: BLE001
                result.unavailable[name] = f"{type(e).__name__}: {e}"
    return result
