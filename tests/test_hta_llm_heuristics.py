from dataclasses import replace

import pytest

from kineto import write_trace
from torch_trace_lab.hta_llm import analyses, heuristics, normalize
from torch_trace_lab.hta_llm.normalize import RunMetrics

CLEAN = RunMetrics(
    label="clean", steps=10, step_time_ms=100.0, gpu_idle_pct=3.0, gpu_compute_pct=95.0,
    gpu_non_compute_pct=2.0, idle_ms_per_step=3.0, host_wait_pct_of_idle=30.0,
    kernel_wait_pct_of_idle=70.0, other_idle_pct_of_idle=0.0, kernel_ms_per_step=97.0,
    top_kernel="conv", top_kernel_pct=12.0, memcpy_pct=1.0,
    gemm_conv_precision_pct={"fp16": 90.0, "fp32": 10.0}, kernels_per_step=300.0,
    median_kernel_us=80.0, median_launch_cpu_us=6.0, median_launch_delay_us=20.0,
    short_kernel_pct=10.0,
)


def ids(m, cc=9.0):
    return [f.id for f in heuristics.flags_for(m, cc)]


def test_clean_run_has_no_flags():
    assert ids(CLEAN) == []


@pytest.mark.parametrize("changes, flag", [
    ({"gpu_idle_pct": 35.0}, "GPU_IDLE_HIGH"),
    ({"gpu_idle_pct": 12.0, "host_wait_pct_of_idle": 80.0}, "HOST_BOUND"),
    ({"gpu_idle_pct": 15.0, "median_kernel_us": 4.0}, "LAUNCH_BOUND"),
    ({"gpu_idle_pct": 15.0, "short_kernel_pct": 70.0}, "LAUNCH_BOUND"),
    ({"top_kernel_pct": 45.0}, "TOP_KERNEL_DOMINANT"),
    ({"memcpy_pct": 15.0}, "MEMCPY_HIGH"),
    ({"gemm_conv_precision_pct": {"fp32": 100.0}}, "NO_AMP_SUSPECTED"),
    ({"gemm_conv_precision_pct": {"tf32": 80.0, "fp16": 20.0}}, "NO_AMP_SUSPECTED"),
])
def test_each_flag_fires(changes, flag):
    assert flag in ids(replace(CLEAN, **changes))


def test_host_bound_needs_real_idle():
    assert "HOST_BOUND" not in ids(replace(CLEAN, gpu_idle_pct=2.0, host_wait_pct_of_idle=99.0))


def test_launch_bound_needs_idle_gpu():
    assert "LAUNCH_BOUND" not in ids(replace(CLEAN, median_kernel_us=2.0, short_kernel_pct=90.0))


def test_no_amp_needs_tensor_cores():
    m = replace(CLEAN, gemm_conv_precision_pct={"fp32": 100.0})
    assert "NO_AMP_SUSPECTED" not in ids(m, cc=6.1)
    assert "NO_AMP_SUSPECTED" not in ids(m, cc=None)


def test_flag_fields():
    f = heuristics.flags_for(replace(CLEAN, gpu_idle_pct=45.0), 9.0)[0]
    assert f.to_dict().keys() == {"id", "severity", "evidence", "threshold"}
    assert f.severity == "high" and "45.0%" in f.evidence


def test_flags_on_hta_fixtures(tmp_path):
    """End to end through HTA: a host-bound fp32 trace and a busy fp16 one."""
    slow = tmp_path / "slow"
    write_trace(slow / "t.json", steps=5, gap_us=600,
                kernels=[("volta_sgemm_128x64_nn", 300), ("elementwise", 100)])
    fast = tmp_path / "fast"
    write_trace(fast / "t.json", steps=5, gap_us=0,
                kernels=[("ampere_fp16_s16816gemm_fp16_128x128", 300), ("elementwise", 250),
                         ("relu", 250), ("bn", 250)])
    slow_ids = ids(normalize.normalize_run("slow", analyses.run_all(slow)))
    fast_ids = ids(normalize.normalize_run("fast", analyses.run_all(fast)))
    assert {"GPU_IDLE_HIGH", "HOST_BOUND", "NO_AMP_SUSPECTED"} <= set(slow_ids)
    assert fast_ids == []
