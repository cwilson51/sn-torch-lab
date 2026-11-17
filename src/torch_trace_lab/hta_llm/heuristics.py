"""Rule-based flags computed from normalized metrics (spec §7a). Pure functions.

Flags make findings deterministic, keep ``--no-llm`` useful, and let the LLM focus on
explaining and prioritizing rather than detecting.
"""

from dataclasses import asdict, dataclass
from statistics import median

from torch_trace_lab.hta_llm import settings as s
from torch_trace_lab.hta_llm.normalize import RunMetrics, r1, r2


@dataclass
class Flag:
    id: str
    severity: str   # high | medium | low
    evidence: str
    threshold: str

    def to_dict(self) -> dict:
        return asdict(self)


MEANINGS = {
    "GPU_IDLE_HIGH": "The GPU sits idle for a large share of the run, with no kernel executing. "
                     "Check HOST_BOUND or LAUNCH_BOUND for the reason.",
    "HOST_BOUND": "When the GPU is idle, it is mostly waiting for the CPU, usually for data "
                  "loading or Python overhead between steps. The faster the GPU work, the more "
                  "easily the CPU side becomes the limit.",
    "LAUNCH_BOUND": "Kernels are so short that launching them from the CPU takes about as long as "
                    "running them, so the GPU waits between launches.",
    "TOP_KERNEL_DOMINANT": "A single kernel takes most of the GPU time. Often expected (a large "
                           "GEMM or convolution); check that it runs as fast as it can.",
    "MEMCPY_HIGH": "A large share of GPU time is spent copying data between host and device.",
    "NO_AMP_SUSPECTED": "Matrix multiplies and convolutions run in fp32 or TF32 on a GPU whose "
                        "tensor cores are faster in fp16/bf16 (automatic mixed precision).",
    "COMM_HEAVY": "Communication (NCCL collectives) takes a large share of GPU time on a "
                  "multi-GPU run. Time a rank spends waiting for slower ranks inside a "
                  "collective also counts as communication.",
    "POOR_OVERLAP": "Communication is heavy and mostly does not run alongside compute, so the "
                    "GPUs wait for it instead of hiding it behind the backward pass.",
    "RANK_STRAGGLER": "One rank does noticeably more GPU work per step than the others; with "
                      "synchronous collectives, every other rank waits for it.",
    "RANK_IDLE_IMBALANCE": "Ranks differ a lot in how long their GPU sits idle, which points "
                           "to uneven data loading or CPU work across ranks.",
}


def thresholds() -> dict[str, str]:
    """Each flag's trigger, from the current settings."""
    return {
        "GPU_IDLE_HIGH": f"GPU idle > {s.GPU_IDLE_HIGH_PCT:g}% of the GPU timeline",
        "HOST_BOUND": f"host wait > {s.HOST_BOUND_PCT:g}% of GPU idle time, with GPU idle "
                      f"≥ {s.HOST_BOUND_MIN_IDLE_PCT:g}%",
        "LAUNCH_BOUND": f"median kernel shorter than the median CPU launch, or > "
                        f"{s.LAUNCH_BOUND_SHORT_KERNEL_PCT:g}% of kernels < "
                        f"{s.LAUNCH_BOUND_SHORT_KERNEL_US:g} µs; with GPU idle "
                        f"≥ {s.LAUNCH_BOUND_MIN_IDLE_PCT:g}% and a median launch delay "
                        f"≤ {s.LAUNCH_BOUND_MAX_DELAY_RATIO:g}× the median CPU launch "
                        "(kernels start soon after launch)",
        "TOP_KERNEL_DOMINANT": f"one kernel > {s.TOP_KERNEL_DOMINANT_PCT:g}% of GPU kernel time",
        "MEMCPY_HIGH": f"host-to-device + device-to-host copies > {s.MEMCPY_HIGH_PCT:g}% of "
                       "GPU kernel time",
        "NO_AMP_SUSPECTED": f"fp32 + TF32 > {s.NO_AMP_FP32_SHARE_PCT:g}% of GEMM/conv time, on a "
                            f"GPU with compute capability ≥ {s.NO_AMP_MIN_COMPUTE_CAPABILITY:g}",
        "COMM_HEAVY": f"communication > {s.COMM_HEAVY_PCT:g}% of GPU kernel time on the median "
                      "rank",
        "POOR_OVERLAP": f"< {s.POOR_OVERLAP_PCT:g}% of communication time overlaps compute on "
                        "the median rank, while COMM_HEAVY",
        "RANK_STRAGGLER": f"slowest rank's busy (non-communication) GPU time per step > "
                          f"{s.RANK_STRAGGLER_RATIO:g}× the median rank's",
        "RANK_IDLE_IMBALANCE": f"GPU idle % differs by > {s.RANK_IDLE_IMBALANCE_POINTS:g} "
                               "points between ranks",
    }


def describe_flags() -> str:
    """Plain-text listing of every flag, its meaning and its trigger."""
    lines = ["Rule-based flags computed by `torch-lab analyze-traces` (thresholds from settings):",
             ""]
    for flag_id, trigger in thresholds().items():
        lines += [flag_id, f"  {MEANINGS[flag_id]}", f"  Fires when: {trigger}.", ""]
    lines.append("Severity (high / medium / low) grows with how far a value exceeds its threshold.")
    return "\n".join(lines)


def _flag(flag_id: str, severity: str, evidence: str) -> "Flag":
    return Flag(flag_id, severity, evidence, thresholds()[flag_id])


def _severity(value: float, threshold: float) -> str:
    if value >= 2 * threshold:
        return "high"
    if value >= 1.25 * threshold:
        return "medium"
    return "low"


def gpu_idle_high(m: RunMetrics) -> Flag | None:
    if m.gpu_idle_pct is None or m.gpu_idle_pct <= s.GPU_IDLE_HIGH_PCT:
        return None
    return _flag("GPU_IDLE_HIGH", _severity(m.gpu_idle_pct, s.GPU_IDLE_HIGH_PCT),
                 f"GPU idle {m.gpu_idle_pct}% of the GPU timeline"
                 + (f" ({m.idle_ms_per_step} ms/step)" if m.idle_ms_per_step is not None else ""))


def host_bound(m: RunMetrics) -> Flag | None:
    if (m.host_wait_pct_of_idle is None or m.gpu_idle_pct is None
            or m.gpu_idle_pct < s.HOST_BOUND_MIN_IDLE_PCT
            or m.host_wait_pct_of_idle <= s.HOST_BOUND_PCT):
        return None
    return _flag("HOST_BOUND", _severity(m.gpu_idle_pct, s.GPU_IDLE_HIGH_PCT),
                 f"host wait is {m.host_wait_pct_of_idle}% of GPU idle time; GPU idle "
                 f"{m.gpu_idle_pct}% (the GPU waits for the CPU: data loading or Python overhead)")


def launch_bound(m: RunMetrics) -> Flag | None:
    if m.median_kernel_us is None or m.gpu_idle_pct is None:
        return None
    if m.gpu_idle_pct < s.LAUNCH_BOUND_MIN_IDLE_PCT:
        return None
    faster_than_launch = (m.median_launch_cpu_us is not None
                          and m.median_kernel_us < m.median_launch_cpu_us)
    many_short = (m.short_kernel_pct or 0) > s.LAUNCH_BOUND_SHORT_KERNEL_PCT
    if not (faster_than_launch or many_short):
        return None
    # A launch-bound GPU runs each kernel soon after it is launched. If kernels wait in the
    # queue for much longer than a launch takes, the CPU is ahead of the GPU and the idle
    # time has another cause (see HOST_BOUND).
    if (m.median_launch_delay_us is not None and m.median_launch_cpu_us is not None
            and m.median_launch_delay_us > s.LAUNCH_BOUND_MAX_DELAY_RATIO
            * max(m.median_launch_cpu_us, 1.0)):
        return None
    reasons = []
    if faster_than_launch:
        reasons.append(f"median kernel {m.median_kernel_us} µs < median CPU launch "
                       f"{m.median_launch_cpu_us} µs")
    if many_short:
        reasons.append(f"{m.short_kernel_pct}% of kernels < {s.LAUNCH_BOUND_SHORT_KERNEL_US:g} µs")
    return _flag("LAUNCH_BOUND", "high" if faster_than_launch and many_short else "medium",
                 "; ".join(reasons) + f"; {m.kernels_per_step} kernels/step; GPU idle "
                 f"{m.gpu_idle_pct}%"
                 + (f"; median launch delay {m.median_launch_delay_us} µs"
                    if m.median_launch_delay_us is not None else ""))


def top_kernel_dominant(m: RunMetrics) -> Flag | None:
    if m.top_kernel_pct is None or m.top_kernel_pct <= s.TOP_KERNEL_DOMINANT_PCT:
        return None
    return _flag("TOP_KERNEL_DOMINANT", _severity(m.top_kernel_pct, s.TOP_KERNEL_DOMINANT_PCT),
                 f"{m.top_kernel} is {m.top_kernel_pct}% of GPU kernel time")


def memcpy_high(m: RunMetrics) -> Flag | None:
    if m.memcpy_pct is None or m.memcpy_pct <= s.MEMCPY_HIGH_PCT:
        return None
    return _flag("MEMCPY_HIGH", _severity(m.memcpy_pct, s.MEMCPY_HIGH_PCT),
                 f"host<->device memcpy is {m.memcpy_pct}% of GPU kernel time")


def no_amp_suspected(m: RunMetrics, compute_capability: float | None) -> Flag | None:
    if compute_capability is None or compute_capability < s.NO_AMP_MIN_COMPUTE_CAPABILITY:
        return None
    shares = m.gemm_conv_precision_pct
    full = shares.get("fp32", 0.0) + shares.get("tf32", 0.0)
    if full <= s.NO_AMP_FP32_SHARE_PCT:
        return None
    detail = ", ".join(f"{p} {v}%" for p, v in shares.items())
    return _flag("NO_AMP_SUSPECTED", "medium",
                 f"GEMM/conv kernel time by precision: {detail}; GPU compute capability "
                 f"{compute_capability:g} has tensor cores for fp16/bf16")


# Flags that only a multi-GPU run can raise.
RANK_FLAGS = ("COMM_HEAVY", "POOR_OVERLAP", "RANK_STRAGGLER", "RANK_IDLE_IMBALANCE")


def _median(ranks: list[RunMetrics], metric: str) -> float | None:
    values = [getattr(r, metric) for r in ranks if getattr(r, metric) is not None]
    return median(values) if values else None


def _spread(ranks: list[RunMetrics], metric: str) -> tuple[RunMetrics, RunMetrics] | None:
    known = [r for r in ranks if getattr(r, metric) is not None]
    if len(known) < 2:
        return None
    return (min(known, key=lambda r: getattr(r, metric)),
            max(known, key=lambda r: getattr(r, metric)))


def comm_heavy(ranks: list[RunMetrics]) -> Flag | None:
    pct = _median(ranks, "comm_pct")
    if pct is None or pct <= s.COMM_HEAVY_PCT:
        return None
    ms = _median(ranks, "comm_ms_per_step")
    return _flag("COMM_HEAVY", _severity(pct, s.COMM_HEAVY_PCT),
                 f"communication is {r1(pct)}% of GPU kernel time on the median rank "
                 f"({r2(ms)} ms/step)")


def poor_overlap(ranks: list[RunMetrics]) -> Flag | None:
    pct, overlap = _median(ranks, "comm_pct"), _median(ranks, "overlap_pct")
    if pct is None or overlap is None or pct <= s.COMM_HEAVY_PCT \
            or overlap >= s.POOR_OVERLAP_PCT:
        return None
    exposed = _median(ranks, "exposed_comm_ms_per_step")
    return _flag("POOR_OVERLAP", "high" if overlap < s.POOR_OVERLAP_PCT / 2 else "medium",
                 f"{r1(overlap)}% of communication time overlaps compute on the median rank; "
                 f"{r2(exposed)} ms/step of communication is exposed")


def rank_straggler(ranks: list[RunMetrics]) -> Flag | None:
    busy = _median(ranks, "busy_ms_per_step")
    pair = _spread(ranks, "busy_ms_per_step")
    if not busy or pair is None:
        return None
    slow = pair[1]
    ratio = slow.busy_ms_per_step / busy
    if ratio <= s.RANK_STRAGGLER_RATIO:
        return None
    steps = _spread(ranks, "step_time_ms")
    step_note = (f"; step time ranges {steps[0].step_time_ms}–{steps[1].step_time_ms} ms "
                 "across ranks" if steps else "")
    return _flag("RANK_STRAGGLER", _severity(ratio - 1, s.RANK_STRAGGLER_RATIO - 1),
                 f"rank {slow.rank} has {slow.busy_ms_per_step} ms/step of busy GPU time, "
                 f"{r2(ratio)}× the median rank's {r2(busy)} ms/step{step_note}")


def rank_idle_imbalance(ranks: list[RunMetrics]) -> Flag | None:
    pair = _spread(ranks, "gpu_idle_pct")
    if pair is None:
        return None
    lo, hi = pair
    spread = hi.gpu_idle_pct - lo.gpu_idle_pct
    if spread <= s.RANK_IDLE_IMBALANCE_POINTS:
        return None
    return _flag("RANK_IDLE_IMBALANCE", _severity(spread, s.RANK_IDLE_IMBALANCE_POINTS),
                 f"GPU idle is {hi.gpu_idle_pct}% on rank {hi.rank} but {lo.gpu_idle_pct}% on "
                 f"rank {lo.rank} ({r1(spread)} points apart)")


def rank_flags(ranks: list[RunMetrics]) -> list[Flag]:
    """Run-level flags computed across a run's ranks (spec §7a, M3)."""
    checks = [comm_heavy(ranks), poor_overlap(ranks), rank_straggler(ranks),
              rank_idle_imbalance(ranks)]
    return [f for f in checks if f is not None]


def flags_for(m: RunMetrics, compute_capability: float | None) -> list[Flag]:
    checks = [gpu_idle_high(m), host_bound(m), launch_bound(m), top_kernel_dominant(m),
              memcpy_high(m), no_amp_suspected(m, compute_capability)]
    return [f for f in checks if f is not None]
