"""Infer a run's config from what its trace contains (spec §5.2, source 2). Pure functions.

``collect_evidence`` makes one pass over a trace's events and keeps only a few counts and
times. ``infer_config`` turns that evidence into config values, each tagged
``source = "inferred"`` with a confidence and a one-line evidence string. Values are
evidence, not fact: anything the trace says nothing about is left out (shown as unknown),
never guessed.

Evidence strings are built from fixed templates and numbers only, so no trace text
reaches the bundle through them.
"""

import re
from dataclasses import dataclass, field

from torch_trace_lab.hta_llm import frameworks, settings
from torch_trace_lab.hta_llm.normalize import is_gemm_or_conv, kernel_precision

# CPU ops whose first input is the activation: its leading dimension is the batch size.
BATCH_OPS = frozenset({"aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::convolution",
                       "aten::linear"})
SCALER_OPS = frozenset({"aten::_amp_foreach_non_finite_check_and_unscale_",
                        "aten::_amp_update_scale_"})
COMPILED = re.compile(r"^(Torch-Compiled Region|CompiledFunction|## Call CompiledFxGraph)")
GRAPH_LAUNCH = frozenset({"cudaGraphLaunch", "cuGraphLaunch"})
TRANSPOSE = re.compile(r"nchwToNhwc|nhwcToNchw")
NHWC_KERNEL = re.compile(r"nhwc", re.I)
DTYPES = {"float": "fp32", "c10::Half": "fp16", "c10::BFloat16": "bf16", "double": "fp64"}


@dataclass
class Evidence:
    cpu_ops: int = 0                      # 0 means the CPU side was not profiled
    gemm_conv_us: dict[str, float] = field(default_factory=dict)  # precision -> kernel time
    nhwc_gemm_conv_us: float = 0.0
    transpose_us: float = 0.0
    triton_kernels: int = 0
    scaler_ops: int = 0
    compiled_regions: int = 0
    graph_launches: int = 0
    single_process_loader: int = 0
    multi_process_loader: int = 0
    batch_op_dtypes: dict[str, int] = field(default_factory=dict)  # input dtype of conv/linear
    first_batch_dims: list | None = None  # Input Dims of the first conv/linear, by time
    shapes_recorded: bool = False
    ddp_reducer_ops: int = 0              # DDP gradient-reducer CPU ops
    ddp_forward: int = 0                  # DistributedDataParallel.forward annotations
    devices: set[int] = field(default_factory=set)  # local GPU index of the kernels

    @property
    def gemm_conv_total_us(self) -> float:
        return sum(self.gemm_conv_us.values())


def collect_evidence(events: list) -> Evidence:
    ev = Evidence()
    gemm_cache: dict[str, tuple[bool, str | None]] = {}
    first_ts = None
    for e in events:
        if not isinstance(e, dict):
            continue
        cat = e.get("cat")
        if cat == "python_function":
            continue
        name = e.get("name")
        if not isinstance(name, str):
            continue
        if cat == "kernel":
            dur = e.get("dur")
            dur = float(dur) if isinstance(dur, (int, float)) else 0.0
            if name not in gemm_cache:
                gc = is_gemm_or_conv(name)
                gemm_cache[name] = (gc, (kernel_precision(name) or "unknown") if gc else None)
            gc, precision = gemm_cache[name]
            if gc:
                ev.gemm_conv_us[precision] = ev.gemm_conv_us.get(precision, 0.0) + dur
                if NHWC_KERNEL.search(name):
                    ev.nhwc_gemm_conv_us += dur
            elif TRANSPOSE.search(name):
                ev.transpose_us += dur
            if name.startswith("triton_"):
                ev.triton_kernels += 1
            device = (e.get("args") or {}).get("device")
            if isinstance(device, int) and not isinstance(device, bool):
                ev.devices.add(device)
        elif cat == "cpu_op":
            ev.cpu_ops += 1
            if frameworks.is_ddp_cpu_op(name):
                ev.ddp_reducer_ops += 1
            elif name in SCALER_OPS:
                ev.scaler_ops += 1
            elif COMPILED.match(name):
                ev.compiled_regions += 1
            elif name in BATCH_OPS:
                args = e.get("args") or {}
                dims, types = args.get("Input Dims"), args.get("Input type")
                if isinstance(dims, list):
                    ev.shapes_recorded = True
                if isinstance(types, list) and types and isinstance(types[0], str):
                    dtype = DTYPES.get(types[0], "other")
                    ev.batch_op_dtypes[dtype] = ev.batch_op_dtypes.get(dtype, 0) + 1
                ts = e.get("ts")
                if (isinstance(dims, list) and dims and isinstance(dims[0], list) and dims[0]
                        and isinstance(ts, (int, float)) and (first_ts is None or ts < first_ts)):
                    first_ts, ev.first_batch_dims = ts, dims[0]
        elif cat == "user_annotation":
            if "_SingleProcessDataLoaderIter" in name:
                ev.single_process_loader += 1
            elif "_MultiProcessingDataLoaderIter" in name:
                ev.multi_process_loader += 1
            elif COMPILED.match(name):
                ev.compiled_regions += 1
            elif name == frameworks.DDP_ANNOTATION:
                ev.ddp_forward += 1
        elif cat in ("cuda_runtime", "cuda_driver") and name in GRAPH_LAUNCH:
            ev.graph_launches += 1
    return ev


def _entry(value, confidence: str, evidence: str) -> dict:
    return {"value": value, "source": "inferred", "confidence": confidence, "evidence": evidence}


def _pct(part: float, total: float) -> float:
    return round(100 * part / total, 1) if total else 0.0


def infer_precision(ev: Evidence) -> dict | None:
    known = {p: us for p, us in ev.gemm_conv_us.items() if p != "unknown" and us > 0}
    total = sum(known.values())
    if not total:
        return None
    dominant = max(known, key=known.get)
    share = _pct(known[dominant], total)
    if share >= settings.INFER_HIGH_SHARE_PCT:
        confidence = "high"
    elif share >= settings.INFER_MIN_SHARE_PCT:
        confidence = "medium"
    else:
        confidence = "low"
    evidence = f"{dominant} kernels are {share}% of GEMM/conv time with a known precision"
    value = dominant
    if dominant == "fp16":
        if ev.scaler_ops:
            value = "fp16-amp"
            evidence += "; GradScaler ops present"
        else:
            confidence = "medium" if confidence == "high" else confidence
            evidence += "; no GradScaler ops (autocast without a scaler, or a half model)"
    inputs = ev.batch_op_dtypes
    if inputs:
        top = max(inputs, key=inputs.get)
        evidence += f"; conv/linear inputs mostly {top}"
    return _entry(value, confidence, evidence)


def infer_torch_compile(ev: Evidence) -> dict | None:
    if ev.compiled_regions:
        return _entry(True, "high", f"{ev.compiled_regions} compiled-region events")
    if ev.triton_kernels:
        return _entry(True, "medium", f"{ev.triton_kernels} Triton kernel launches")
    if ev.cpu_ops:
        return _entry(False, "medium", "no compiled regions or Triton kernels")
    return None


def infer_dataloader_workers(ev: Evidence) -> dict | None:
    if ev.multi_process_loader and not ev.single_process_loader:
        return _entry(">0", "high", "multi-process DataLoader iterator (worker count unknown)")
    if ev.single_process_loader and not ev.multi_process_loader:
        return _entry(0, "high", "single-process DataLoader iterator")
    return None


def infer_batch_size(ev: Evidence) -> dict | None:
    dims = ev.first_batch_dims
    if not dims or not isinstance(dims[0], int) or isinstance(dims[0], bool) or dims[0] < 1:
        return None
    return _entry(dims[0], "medium", "leading input dimension of the first conv/linear op "
                                     "(record_shapes=True)")


def infer_cuda_graphs(ev: Evidence) -> dict | None:
    if ev.graph_launches:
        return _entry(True, "high", f"{ev.graph_launches} CUDA graph launches")
    if ev.cpu_ops:
        return _entry(False, "medium", "no CUDA graph launches")
    return None


def infer_memory_format(ev: Evidence) -> dict | None:
    total = ev.gemm_conv_total_us
    if not total:
        return None
    transpose = _pct(ev.transpose_us, total)
    if transpose >= settings.INFER_TRANSPOSE_PCT:
        return _entry("contiguous (NCHW)", "medium",
                      f"NCHW<->NHWC transpose kernels take {transpose}% as much time as "
                      "GEMM/conv kernels")
    nhwc = _pct(ev.nhwc_gemm_conv_us, total)
    if nhwc >= settings.INFER_MIN_SHARE_PCT:
        return _entry("channels_last", "low",
                      f"NHWC GEMM/conv kernels are {nhwc}% of GEMM/conv time, with few "
                      "layout transposes")
    return None


def infer_parallelism(ev: Evidence) -> dict | None:
    """DDP from its reducer ops and forward annotation. Other frameworks are M4b: a run
    without DDP markers is left unknown, never "none"."""
    if ev.ddp_reducer_ops and ev.ddp_forward:
        return _entry("DDP", "high", f"{ev.ddp_reducer_ops} DDP gradient-reducer CPU ops and "
                      f"{ev.ddp_forward} DistributedDataParallel.forward annotations")
    if ev.ddp_reducer_ops:
        return _entry("DDP", "medium", f"{ev.ddp_reducer_ops} DDP gradient-reducer CPU ops, "
                      "no DistributedDataParallel.forward annotation")
    if ev.ddp_forward:
        return _entry("DDP", "medium", f"{ev.ddp_forward} DistributedDataParallel.forward "
                      "annotations, no gradient-reducer CPU ops")
    return None


RULES = {
    "precision": infer_precision,
    "torch_compile": infer_torch_compile,
    "dataloader_workers": infer_dataloader_workers,
    "batch_size_per_gpu": infer_batch_size,
    "cuda_graphs": infer_cuda_graphs,
    "memory_format": infer_memory_format,
    "parallelism": infer_parallelism,
}


def infer_config(ev: Evidence | None) -> dict:
    """Config name -> {"value", "source": "inferred", "confidence", "evidence"}."""
    if ev is None:
        return {}
    return {key: entry for key, rule in RULES.items() if (entry := rule(ev)) is not None}


def infer_topology(devices_per_rank: list[set[int]], visible_gpus_per_rank: list[int],
                   world_size: int) -> dict:
    """num_nodes and gpus_per_node of a multi-rank run, or {} when the trace can't tell.

    Each rank runs its kernels on its local GPU, so the distinct device indices across ranks
    give the GPUs per node. That must match the GPUs each rank's process could see
    (``deviceProperties``) and divide the world size. When every rank sees one GPU, as with
    per-process CUDA_VISIBLE_DEVICES pinning, one node of N GPUs and N nodes of one GPU
    look the same, so nothing is inferred.
    """
    if world_size <= 1 or len(devices_per_rank) != world_size:
        return {}
    if any(len(d) != 1 for d in devices_per_rank):
        return {}
    gpus_per_node = len(set().union(*devices_per_rank))
    if gpus_per_node == 1 or world_size % gpus_per_node:
        return {}
    if any(v != gpus_per_node for v in visible_gpus_per_rank):
        return {}
    nodes = world_size // gpus_per_node
    evidence = (f"the {world_size} ranks run on {gpus_per_node} distinct local GPU indices, "
                f"and each rank's process sees {gpus_per_node} GPUs")
    return {"num_nodes": _entry(nodes, "medium", evidence),
            "gpus_per_node": _entry(gpus_per_node, "medium", evidence)}
