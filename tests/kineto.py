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
    gpu_annotations: bool = True,
    rank: int | None = None,
    world_size: int | None = None,
    torch_lab: dict | None = None,
    extra_top: dict | None = None,
    python_functions: list[str] = (),
    device_properties=DEVICE_PROPERTIES,
    extra_events: list[dict] = (),
    comm_kernels=(),
    comm_overlap: bool = False,
    device: int = 0,
    truncate_last_step: bool = False,
    eventless_launches: int = 0,
) -> dict:
    """``extra_events`` are partial events (at least ``cat`` and ``name``, optionally ``args``);
    they are placed on the CPU thread in the first step, e.g. DataLoader annotations,
    ``Torch-Compiled Region`` ops, ``cpu_op`` events with ``Input Dims``, ``cudaGraphLaunch``.

    ``comm_kernels`` are (name, µs) NCCL kernels run each step on their own stream (8):
    with ``comm_overlap`` they start with the step's first compute kernel (as gradient
    buckets do during the backward pass); otherwise they run after the compute kernels.

    ``device`` is the local GPU index in each GPU event's args (the rank's GPU on its node).
    ``truncate_last_step`` leaves out the GPU event of the last step's last compute kernel,
    as when a trace is exported before the GPU work of its final step is collected.
    ``eventless_launches`` adds that many ``cudaMemsetAsync`` calls per step that never get a
    GPU event, as zero-byte memsets in real compiled-model traces."""
    # Start with an unrelated CPU op, as real traces do: HTA links GPU events to their launch
    # by row index and treats index 0 as "no link".
    events = [{"ph": "X", "cat": "cpu_op", "name": "aten::empty", "pid": 100, "tid": 100,
               "ts": 990, "dur": 1, "args": {}}]
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
                               "args": {"device": device, "stream": 7, "correlation": corr}})
            gpu_t += memcpy_us
            cpu_t += launch_us
            corr += 1
        compute_start = gpu_t
        for k, (name, dur) in enumerate(kernels):
            events.append({"ph": "X", "cat": "cpu_op", "name": "aten::op", "pid": pid, "tid": tid,
                           "ts": cpu_t, "dur": launch_us, "args": {"External id": corr}})
            events.append(_runtime("cudaLaunchKernel", cpu_t, launch_us, corr, pid, tid))
            cut = truncate_last_step and step == steps - 1 and k == len(kernels) - 1
            if gpu and not cut:
                start = max(gpu_t, cpu_t + launch_us)
                events.append({"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 7,
                               "ts": start, "dur": dur,
                               "args": {"device": device, "stream": 7, "correlation": corr,
                                        "External id": corr}})
                gpu_t = start + dur
            cpu_t += launch_us
            corr += 1
        for _ in range(eventless_launches):
            events.append(_runtime("cudaMemsetAsync", cpu_t, launch_us, corr, pid, tid))
            cpu_t += launch_us
            corr += 1
        comm_t = compute_start + launch_us if comm_overlap else gpu_t
        for name, dur in comm_kernels:
            events.append(_runtime("cudaLaunchKernel", cpu_t, launch_us, corr, pid, tid))
            if gpu:
                events.append({"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 8,
                               "ts": comm_t, "dur": dur,
                               "args": {"device": device, "stream": 8, "correlation": corr,
                                        "External id": corr}})
                comm_t += dur
            cpu_t += launch_us
            corr += 1
        gpu_t = max(gpu_t, comm_t) if comm_kernels and gpu else gpu_t
        end = max(cpu_t, gpu_t) + gap_us
        if profiler_steps:
            events.append({"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{step}",
                           "pid": pid, "tid": tid, "ts": step_start, "dur": end - step_start,
                           "args": {}})
            if gpu and gpu_annotations:
                # Real CUDA traces mirror each step annotation on the GPU timeline,
                # spanning the step's kernels.
                first = step_start + launch_us
                events.append({"ph": "X", "cat": "gpu_user_annotation",
                               "name": f"ProfilerStep#{step}", "pid": 0, "tid": 7,
                               "ts": first, "dur": gpu_t - first,
                               "args": {"device": device, "stream": 7, "External id": corr}})
        t = end
    for i, partial in enumerate(extra_events):
        events.append({"ph": "X", "pid": pid, "tid": tid, "ts": 1001 + i, "dur": 1, "args": {},
                       **partial})
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


ALLREDUCE = "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
BROADCAST = "ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
# DDP markers, as recorded in real DDP traces (torch 2.x).
DDP_EVENTS = [
    {"cat": "cpu_op", "name": "torch::distributed::reducer::mul_out"},
    {"cat": "cpu_op", "name": "torch.distributed.ddp.reducer::copy_bucket_to_grad"},
    {"cat": "user_annotation", "name": "DistributedDataParallel.forward"},
]


def write_run(directory: Path, world_size: int, per_rank: dict[int, dict] | None = None,
              **common) -> Path:
    """One multi-rank run: a trace file per rank, named so the names mislead (rank ``r`` is
    in ``trace-<world_size - 1 - r>.json``). ``per_rank`` overrides ``common`` kwargs."""
    directory = Path(directory)
    for rank in range(world_size):
        kwargs = {**common, **(per_rank or {}).get(rank, {})}
        write_trace(directory / f"trace-{world_size - 1 - rank}.json", rank=rank,
                    world_size=world_size, **kwargs)
    return directory


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
