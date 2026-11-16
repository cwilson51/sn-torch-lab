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


def _severity(value: float, threshold: float) -> str:
    if value >= 2 * threshold:
        return "high"
    if value >= 1.25 * threshold:
        return "medium"
    return "low"


def gpu_idle_high(m: RunMetrics) -> Flag | None:
    if m.gpu_idle_pct is None or m.gpu_idle_pct <= s.GPU_IDLE_HIGH_PCT:
        return None
    return Flag("GPU_IDLE_HIGH", _severity(m.gpu_idle_pct, s.GPU_IDLE_HIGH_PCT),
                f"GPU idle {m.gpu_idle_pct}% of the GPU timeline"
                + (f" ({m.idle_ms_per_step} ms/step)" if m.idle_ms_per_step is not None else ""),
                f"idle > {s.GPU_IDLE_HIGH_PCT}%")


def host_bound(m: RunMetrics) -> Flag | None:
    if (m.host_wait_pct_of_idle is None or m.gpu_idle_pct is None
            or m.gpu_idle_pct < s.HOST_BOUND_MIN_IDLE_PCT
            or m.host_wait_pct_of_idle <= s.HOST_BOUND_PCT):
        return None
    return Flag("HOST_BOUND", _severity(m.gpu_idle_pct, s.GPU_IDLE_HIGH_PCT),
                f"host wait is {m.host_wait_pct_of_idle}% of GPU idle time; GPU idle "
                f"{m.gpu_idle_pct}% (the GPU waits for the CPU: data loading or Python overhead)",
                f"host wait > {s.HOST_BOUND_PCT}% of idle, with idle ≥ {s.HOST_BOUND_MIN_IDLE_PCT}%")


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
    return Flag("LAUNCH_BOUND", "high" if faster_than_launch and many_short else "medium",
                "; ".join(reasons) + f"; {m.kernels_per_step} kernels/step; GPU idle "
                f"{m.gpu_idle_pct}%",
                f"median kernel < median launch, or > {s.LAUNCH_BOUND_SHORT_KERNEL_PCT}% of kernels "
                f"< {s.LAUNCH_BOUND_SHORT_KERNEL_US:g} µs; with idle ≥ {s.LAUNCH_BOUND_MIN_IDLE_PCT}%")


def top_kernel_dominant(m: RunMetrics) -> Flag | None:
    if m.top_kernel_pct is None or m.top_kernel_pct <= s.TOP_KERNEL_DOMINANT_PCT:
        return None
    return Flag("TOP_KERNEL_DOMINANT", _severity(m.top_kernel_pct, s.TOP_KERNEL_DOMINANT_PCT),
                f"{m.top_kernel} is {m.top_kernel_pct}% of GPU kernel time",
                f"one kernel > {s.TOP_KERNEL_DOMINANT_PCT}%")


def memcpy_high(m: RunMetrics) -> Flag | None:
    if m.memcpy_pct is None or m.memcpy_pct <= s.MEMCPY_HIGH_PCT:
        return None
    return Flag("MEMCPY_HIGH", _severity(m.memcpy_pct, s.MEMCPY_HIGH_PCT),
                f"host<->device memcpy is {m.memcpy_pct}% of GPU kernel time",
                f"H2D + D2H > {s.MEMCPY_HIGH_PCT}%")


def no_amp_suspected(m: RunMetrics, compute_capability: float | None) -> Flag | None:
    if compute_capability is None or compute_capability < s.NO_AMP_MIN_COMPUTE_CAPABILITY:
        return None
    shares = m.gemm_conv_precision_pct
    full = shares.get("fp32", 0.0) + shares.get("tf32", 0.0)
    if full <= s.NO_AMP_FP32_SHARE_PCT:
        return None
    detail = ", ".join(f"{p} {v}%" for p, v in shares.items())
    return Flag("NO_AMP_SUSPECTED", "medium",
                f"GEMM/conv kernel time by precision: {detail}; GPU compute capability "
                f"{compute_capability:g} has tensor cores for fp16/bf16",
                f"fp32 + tf32 > {s.NO_AMP_FP32_SHARE_PCT}% of GEMM/conv time, "
                f"compute capability ≥ {s.NO_AMP_MIN_COMPUTE_CAPABILITY:g}")


def flags_for(m: RunMetrics, compute_capability: float | None) -> list[Flag]:
    checks = [gpu_idle_high(m), host_bound(m), launch_bound(m), top_kernel_dominant(m),
              memcpy_high(m), no_amp_suspected(m, compute_capability)]
    return [f for f in checks if f is not None]
