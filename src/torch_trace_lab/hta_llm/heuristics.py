"""Rule-based flags computed from normalized metrics (spec §7a). Pure functions.

Flags make findings deterministic, keep ``--no-llm`` useful, and let the LLM focus on
explaining and prioritizing rather than detecting.
"""

from dataclasses import asdict, dataclass

from torch_trace_lab.hta_llm import settings as s
from torch_trace_lab.hta_llm.normalize import RunMetrics


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
                        f"≥ {s.LAUNCH_BOUND_MIN_IDLE_PCT:g}%",
        "TOP_KERNEL_DOMINANT": f"one kernel > {s.TOP_KERNEL_DOMINANT_PCT:g}% of GPU kernel time",
        "MEMCPY_HIGH": f"host-to-device + device-to-host copies > {s.MEMCPY_HIGH_PCT:g}% of "
                       "GPU kernel time",
        "NO_AMP_SUSPECTED": f"fp32 + TF32 > {s.NO_AMP_FP32_SHARE_PCT:g}% of GEMM/conv time, on a "
                            f"GPU with compute capability ≥ {s.NO_AMP_MIN_COMPUTE_CAPABILITY:g}",
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
    reasons = []
    if faster_than_launch:
        reasons.append(f"median kernel {m.median_kernel_us} µs < median CPU launch "
                       f"{m.median_launch_cpu_us} µs")
    if many_short:
        reasons.append(f"{m.short_kernel_pct}% of kernels < {s.LAUNCH_BOUND_SHORT_KERNEL_US:g} µs")
    return _flag("LAUNCH_BOUND", "high" if faster_than_launch and many_short else "medium",
                 "; ".join(reasons) + f"; {m.kernels_per_step} kernels/step; GPU idle "
                 f"{m.gpu_idle_pct}%")


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


def flags_for(m: RunMetrics, compute_capability: float | None) -> list[Flag]:
    checks = [gpu_idle_high(m), host_bound(m), launch_bound(m), top_kernel_dominant(m),
              memcpy_high(m), no_amp_suspected(m, compute_capability)]
    return [f for f in checks if f is not None]
