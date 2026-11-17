from dataclasses import replace

import pytest

from kineto import ALLREDUCE, write_run, write_trace
from torch_trace_lab.hta_llm import analyses, heuristics, normalize, settings
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


LIMIT_US = settings.LAUNCH_BOUND_MAX_DELAY_RATIO * 6.0  # for a 6 µs median CPU launch


@pytest.mark.parametrize("delay_us, fires", [
    (5.0, True),             # kernels start right after launch
    (LIMIT_US, True),
    (LIMIT_US + 1, False),   # queued: the CPU is ahead of the GPU
    (30_000.0, False),
    (None, True),            # unknown: not ruled out
])
def test_launch_bound_needs_kernels_to_start_soon_after_launch(delay_us, fires):
    """Short kernels with an idle GPU are launch bound only if kernels start soon after their
    launch; kernels queued far longer than a launch takes mean the CPU is ahead."""
    m = replace(CLEAN, gpu_idle_pct=15.0, median_kernel_us=1.0, short_kernel_pct=75.0,
                median_launch_cpu_us=6.0, median_launch_delay_us=delay_us)
    assert ("LAUNCH_BOUND" in ids(m)) is fires


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


# --- Rank flags (M3) ----------------------------------------------------------------------

CLEAN_RANK = replace(CLEAN, comm_ms_per_step=10.0, comm_pct=10.0, exposed_comm_ms_per_step=2.0,
                     overlap_pct=80.0, busy_ms_per_step=87.0)


def ranks(n=4, **changes):
    """``n`` identical clean ranks; ``changes`` maps rank -> field overrides."""
    return [replace(CLEAN_RANK, rank=r, **changes.get(f"r{r}", {})) for r in range(n)]


def rank_ids(rs):
    return [f.id for f in heuristics.rank_flags(rs)]


def test_clean_ranks_have_no_rank_flags():
    assert rank_ids(ranks()) == []
    assert rank_ids(ranks(1)) == []
    assert rank_ids([CLEAN]) == []  # a single-GPU run has no comm metrics


@pytest.mark.parametrize("changes, expected", [
    ({f"r{r}": {"comm_pct": 40.0} for r in range(4)}, ["COMM_HEAVY"]),
    ({f"r{r}": {"comm_pct": 40.0, "overlap_pct": 20.0} for r in range(4)},
     ["COMM_HEAVY", "POOR_OVERLAP"]),
    ({f"r{r}": {"overlap_pct": 20.0} for r in range(4)}, []),  # poor overlap, light comm
    ({"r2": {"busy_ms_per_step": 100.0}}, ["RANK_STRAGGLER"]),
    ({"r2": {"busy_ms_per_step": 95.0}}, []),                 # 1.09x the median
    ({"r3": {"gpu_idle_pct": 15.0}}, ["RANK_IDLE_IMBALANCE"]),
    ({"r3": {"gpu_idle_pct": 13.0}}, []),                     # exactly 10 points apart
    ({"r0": {"comm_pct": 40.0}}, []),                         # one comm-heavy rank, not the median
])
def test_each_rank_flag_fires_on_its_case(changes, expected):
    assert rank_ids(ranks(**changes)) == expected


def test_rank_flag_evidence_names_ranks():
    rs = ranks(r2={"busy_ms_per_step": 130.0, "step_time_ms": 130.0},
               r3={"gpu_idle_pct": 25.0})
    flags = {f.id: f for f in heuristics.rank_flags(rs)}
    straggler = flags["RANK_STRAGGLER"]
    assert "rank 2 has 130.0 ms/step" in straggler.evidence and "1.49×" in straggler.evidence
    assert "step time ranges 100.0–130.0 ms" in straggler.evidence
    assert straggler.severity == "high"
    assert "25.0% on rank 3 but 3.0% on rank 0" in flags["RANK_IDLE_IMBALANCE"].evidence


def test_rank_flags_through_hta(tmp_path):
    """End to end: rank 1 does 50% more compute; communication is serialized."""
    write_run(tmp_path, 3, steps=5, gap_us=0, kernels=[("elementwise", 400)],
              comm_kernels=[(ALLREDUCE, 300)],
              per_rank={1: {"kernels": [("elementwise", 600)]}})
    results = analyses.run_ranks(tmp_path)
    rs = [normalize.normalize_run("r", a, rank=r) for r, a in results.items()]
    assert set(rank_ids(rs)) == {"COMM_HEAVY", "POOR_OVERLAP", "RANK_STRAGGLER"}
