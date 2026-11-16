"""Build tiny synthetic Kineto (PyTorch Profiler) traces for tests.

A trace has ``steps`` ProfilerStep annotations. Inside each step the CPU launches the
given kernels back to back; each launch is a ``cudaLaunchKernel`` runtime event that
correlates with a GPU ``kernel`` event. ``gap_us`` of GPU idle time follows each step,
during which the CPU is doing something else (host wait).
"""

import gzip
import json
from pathlib import Path

DEFAULT_KERNELS = [
    ("volta_sgemm_128x64_nn", 400),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::AddFunctor<float>>", 100),
]

DEVICE_PROPERTIES = [{
    "id": 0, "name": "NVIDIA H100 80GB HBM3", "totalGlobalMem": 85_000_000_000,
    "computeMajor": 9, "computeMinor": 0, "numSms": 132,
}]


def make_trace(
    *,
    steps: int = 3,
    kernels=DEFAULT_KERNELS,
    launch_us: int = 5,
    gap_us: int = 100,
    memcpy_us: int = 0,
    gpu: bool = True,
    profiler_steps: bool = True,
    rank: int | None = None,
    world_size: int | None = None,
    torch_lab: dict | None = None,
    extra_top: dict | None = None,
    python_functions: list[str] = (),
    device_properties=DEVICE_PROPERTIES,
) -> dict:
    events = []
    t = 1000
    corr = 1
    pid, tid = 100, 100
    for step in range(steps):
        step_start = t
        cpu_t = t
        gpu_t = t + launch_us
        if memcpy_us:
            events.append(_runtime("cudaMemcpyAsync", cpu_t, launch_us, corr, pid, tid))
            if gpu:
                events.append({"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD (Pageable -> Device)",
                               "pid": 0, "tid": 7, "ts": gpu_t, "dur": memcpy_us,
                               "args": {"device": 0, "stream": 7, "correlation": corr}})
            gpu_t += memcpy_us
            cpu_t += launch_us
            corr += 1
        for name, dur in kernels:
            events.append({"ph": "X", "cat": "cpu_op", "name": "aten::op", "pid": pid, "tid": tid,
                           "ts": cpu_t, "dur": launch_us, "args": {"External id": corr}})
            events.append(_runtime("cudaLaunchKernel", cpu_t, launch_us, corr, pid, tid))
            if gpu:
                start = max(gpu_t, cpu_t + launch_us)
                events.append({"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 7,
                               "ts": start, "dur": dur,
                               "args": {"device": 0, "stream": 7, "correlation": corr,
                                        "External id": corr}})
                gpu_t = start + dur
            cpu_t += launch_us
            corr += 1
        end = max(cpu_t, gpu_t) + gap_us
        if profiler_steps:
            events.append({"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{step}",
                           "pid": pid, "tid": tid, "ts": step_start, "dur": end - step_start,
                           "args": {}})
        t = end
    for i, name in enumerate(python_functions):
        events.append({"ph": "X", "cat": "python_function", "name": name, "pid": pid, "tid": tid,
                       "ts": 1000 + i, "dur": 1, "args": {}})

    trace = {
        "schemaVersion": 1,
        "deviceProperties": device_properties if gpu else [],
        "displayTimeUnit": "ms",
        "baseTimeNanoseconds": 1_700_000_000_000_000_000,
        "traceEvents": events,
    }
    if rank is not None or world_size is not None:
        trace["distributedInfo"] = {"backend": "nccl", "rank": rank or 0,
                                    "world_size": world_size or 1}
    if torch_lab is not None:
        trace["torch_lab"] = torch_lab
    trace.update(extra_top or {})
    return trace


def _runtime(name, ts, dur, corr, pid, tid):
    return {"ph": "X", "cat": "cuda_runtime", "name": name, "pid": pid, "tid": tid,
            "ts": ts, "dur": dur, "args": {"correlation": corr, "External id": corr}}


def write_trace(path: Path, trace: dict | None = None, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(trace if trace is not None else make_trace(**kwargs))
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as f:
            f.write(data)
    else:
        path.write_text(data)
    return path
