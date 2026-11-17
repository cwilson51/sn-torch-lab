import pandas as pd
import pytest

from kineto import ALLREDUCE, BROADCAST, write_run, write_trace
from torch_trace_lab.hta_llm import analyses, normalize, tables
from torch_trace_lab.hta_llm.analyses import RunAnalysis


@pytest.mark.parametrize("name, precision", [
    ("volta_sgemm_128x64_nn", "fp32"),
    ("sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize128x128x16", "tf32"),
    ("cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4", "tf32"),
    ("ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_nn", "fp16"),
    ("sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc", "bf16"),
    ("void cudnn::cnn::implicit_convolve_sgemm<float, float>", "fp32"),
    ("triton_tem_fused_mm_0", None),
])
def test_kernel_precision(name, precision):
    assert normalize.kernel_precision(name) == precision


def test_shorten():
    full = ("void at::native::vectorized_elementwise_kernel<4, at::native::AddFunctor<float>, "
            "at::detail::Array<char*, 3> >(int, at::native::AddFunctor<float>, "
            "at::detail::Array<char*, 3>)")
    assert normalize.shorten(full) == "vectorized_elementwise_kernel<AddFunctor>"
    assert normalize.shorten("/home/jdoe/x/my_kernel") == "my_kernel"
    assert len(normalize.shorten("k" * 200)) == 60


# Real kernel names from PyTorch traces (cuDNN, CUTLASS, ATen, Triton; H100, torch 2.14).
THRESHOLD = ("void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, "
             "float, float, at::native::(anonymous namespace)::threshold_kernel_impl<float>("
             "at::TensorIteratorBase&, float, float)::{lambda(float, float)#1}>, "
             "std::array<char*, 3ul>, false>(int, at::native::BinaryFunctor<float, float, float>)")
CLAMP = ("void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::"
         "launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::"
         "detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() "
         "const::{lambda(float)#1}, std::array<char*, 2ul>, false>(int, float, "
         "std::array<char*, 2ul>)")
ADD = ("void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, "
       "std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, "
       "std::array<char*, 3ul>)")
CUTLASS = ("void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>("
           "cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4::Params)")
MTA = ("void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous "
       "namespace)::TensorListMetadata<2>, at::native::(anonymous namespace)::"
       "BinaryOpListAlphaFunctor<float, 2, 2, 0>, std::plus<float>, float>(at::native::"
       "(anonymous namespace)::TensorListMetadata<2>, float)")
BN = ("void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float, float, "
      "cudnnTensorStruct, float const*)")
NCHW = ("void cudnn::engines_precompiled::nchwToNhwcKernel<float, float, float, false, true, "
        "(cudnnKernelDataType_t)0>(cudnn::engines_precompiled::nchw2nhwc_params_t<float>, "
        "float const*, float*)")
XMMA_TF32 = ("sm90_xmma_wgrad_indexed_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_"
             "tilesize128x128x32_warpgroupsize1x1x1_g1_execute_segment_k_off_kernel__5x_cudnn")
XMMA_F16 = ("sm90_xmma_wgrad_indexed_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_"
            "tilesize128x64x64_warpgroupsize1x1x1_g1_execute_segment_k_off_kernel__5x_cudnn")
TRITON_BN = ("triton_red_fused__native_batch_norm_legit_functional_convolution_backward_"
             "native_batch_norm_backward_53")


@pytest.mark.parametrize("full, short", [
    (THRESHOLD, "vectorized_elementwise_kernel<threshold>"),
    (CLAMP, "vectorized_elementwise_kernel<launch_clamp_scalar>"),
    (ADD, "vectorized_elementwise_kernel<CUDAFunctor_add>"),
    (CUTLASS, "Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>"),
    (MTA, "multi_tensor_apply_kernel<BinaryOpListAlphaFunctor>"),
    (BN, "bn_bw_1C11_kernel_new"),
])
def test_shorten_real_names(full, short):
    assert normalize.shorten(full) == short


@pytest.mark.parametrize("name, category, precision", [
    (CUTLASS, "GEMM/conv", "tf32"),
    (XMMA_TF32, "GEMM/conv", "tf32"),
    (XMMA_F16, "GEMM/conv", "fp16"),
    (BN, "batch norm", None),
    (NCHW, "layout transpose", None),
    (TRITON_BN, "fused (torch.compile)", None),
    ("triton_tem_fused_mm_0", "GEMM/conv", None),
    (ADD, "elementwise/reduction", None),
    (MTA, "optimizer", None),
    ("Memcpy HtoD (Pageable -> Device)", "memcpy/memset", None),
    ("my_custom_cuda_kernel", "uncategorized", None),
])
def test_kernel_category_and_precision(name, category, precision):
    """cuDNN batch-norm/layout kernels and Triton fused kernels are not GEMM/conv."""
    assert normalize.kernel_category(name) == category
    gemm = normalize.is_gemm_or_conv(name)
    assert gemm == (category == "GEMM/conv")
    if gemm:
        assert normalize.kernel_precision(name) == precision


def test_short_name_collisions_are_unique():
    names = normalize.short_names(["void f<int>(int)", "void f<float>(float)"])
    assert names == {"void f<int>(int)": "f", "void f<float>(float)": "f #2"}


def fake_analysis(kernel_sums, steps=2):
    rows = [{"name": n, "kernel_type": "COMPUTATION", "sum_us": s, "mean_us": s / 4, "count": 4}
            for n, s in kernel_sums.items()]
    return RunAnalysis(
        steps=steps, step_time_us=10_000.0,
        temporal=pd.DataFrame([{"rank": 0, "idle_us": 4000, "compute_us": 15000,
                                "non_compute_us": 1000, "kernel_us": 16000, "idle_pct": 20.0,
                                "compute_pct": 75.0, "non_compute_pct": 5.0}]),
        idle=pd.DataFrame({"category": ["host_wait", "kernel_wait", "other"],
                           "idle_us": [3000.0, 1000.0, 0.0], "ratio": [0.75, 0.25, 0.0]}),
        kernels=pd.DataFrame(rows),
        launch=pd.DataFrame({"cpu_duration_us": [5, 5, 5, 5], "gpu_duration_us": [3, 4, 100, 200],
                             "launch_delay_us": [1, 2, 3, 4]}),
    )


def test_per_step_means_and_rounding():
    m = normalize.normalize_run("a", fake_analysis({"volta_sgemm_x": 3333.333, "k2": 1000}),
                                batch_size=64)
    assert m.step_time_ms == 10.0
    assert m.idle_ms_per_step == 2.0
    assert m.host_wait_pct_of_idle == 75.0
    assert m.host_wait_ms_per_step == 1.5   # 3000 µs over 2 steps
    assert m.kernels[0].ms_per_step == 1.67      # 3333.333 us / 2 steps
    assert m.kernels[0].pct_gpu == 76.9
    assert m.kernels[0].precision == "fp32"
    assert m.samples_per_s == 6400.0
    assert m.short_kernel_pct == 50.0
    assert m.gemm_conv_precision_pct == {"fp32": 100.0}


def test_top_n_collapses_other():
    sums = {f"k{i}": 100 * (10 - i) for i in range(10)}
    m = normalize.normalize_run("a", fake_analysis(sums), top_n=3)
    assert [k.short for k in m.kernels] == ["k0", "k1", "k2", "(remaining kernels)"]
    assert m.kernels[-1].ms_per_step == pytest.approx((sum(sums.values()) - 2700) / 2 / 1000,
                                                      abs=0.01)


def test_kernel_comparison_and_baseline():
    names = {}
    a = normalize.normalize_run("a", fake_analysis({"k1": 2000, "k2": 1000}), names=names)
    b = normalize.normalize_run("b", fake_analysis({"k1": 1000, "k3": 500}), names=names)
    rows = normalize.kernel_comparison([a, b], baseline=normalize.resolve_baseline(["a", "b"], "a"))
    k1 = next(r for r in rows if r["kernel"] == "k1")
    assert k1["a"] == 1.0 and k1["b"] == 0.5
    assert k1["Δ b"] == -0.5
    # Kernels in only one run are left to the category comparison.
    assert [r["kernel"] for r in rows] == ["k1"]
    rows_b = normalize.kernel_comparison([a, b], baseline=1)
    assert "Δ a" in rows_b[0]
    cats = normalize.category_comparison([a, b], baseline=0)
    total = next(r for r in cats if r["category"] == "total")
    assert total["a"] == 1.5 and total["b"] == 0.75 and total["Δ% b"] == -50.0
    with pytest.raises(ValueError, match="not one of the runs"):
        normalize.resolve_baseline(["a", "b"], "c")


def test_shared_kernel_outside_top_n_shows_real_time():
    """A kernel in one run's top N but below another's still shows its real ms/step."""
    names = {}
    a = normalize.normalize_run("a", fake_analysis({"big": 4000, "copy": 2000}), top_n=2,
                                names=names)
    b = normalize.normalize_run("b", fake_analysis({"x": 9000, "y": 8000, "copy": 1000}),
                                top_n=2, names=names)
    rows = normalize.kernel_comparison([a, b])
    copy = next(r for r in rows if r["kernel"] == "copy")
    assert copy["a"] == 1.0 and copy["b"] == 0.5   # "copy" is not in b's top 2
    assert [r["kernel"] for r in rows] == ["copy"]  # "big", "x", "y" occur in one run only


def test_step_time_differences_cover_every_pair():
    runs = [normalize.RunMetrics("a", 9, 200.0), normalize.RunMetrics("b", 9, 150.0),
            normalize.RunMetrics("c", 9, 100.0)]
    diffs = {(d["from"], d["to"]): d for d in normalize.step_time_differences(runs)}
    assert len(diffs) == 6
    assert diffs[("b", "c")] == {"from": "b", "to": "c", "delta_ms": -50.0, "delta_pct": -33.3}
    assert diffs[("c", "a")]["delta_ms"] == 100.0 and diffs[("c", "a")]["delta_pct"] == 100.0


def test_tables_render():
    md = tables.render([{"name": "a|b", "ms": 1.5, "n": None}])
    assert md.splitlines() == ["| name | ms | n |", "|---|---:|---:|", "| a\\|b | 1.5 | – |"]


def test_real_hta_on_fixture(tmp_path):
    write_trace(tmp_path / "t.json", memcpy_us=50, steps=4,
                kernels=[("volta_sgemm_128x64_nn", 400), ("tiny", 3)])
    m = normalize.normalize_run("a", analyses.run_all(tmp_path))
    assert m.unavailable == {}
    assert m.steps == 4  # every step is complete, the last one too
    assert m.top_kernel == "volta_sgemm_128x64_nn"
    assert m.memcpy_pct and m.memcpy_pct > 10
    assert m.gpu_idle_pct is not None and m.host_wait_pct_of_idle == 100.0


@pytest.mark.parametrize("gpu_annotations", [False, True])
def test_gpu_step_annotations_are_not_kernels_or_steps(tmp_path, gpu_annotations):
    """Real CUDA traces mirror ProfilerStep#N on the GPU; HTA would count them as kernels."""
    write_trace(tmp_path / "t.json", steps=10, gpu_annotations=gpu_annotations,
                kernels=[("volta_sgemm_128x64_nn", 400), ("elementwise", 100)])
    a = analyses.run_all(tmp_path)
    m = normalize.normalize_run("a", a)
    assert a.steps == 10  # every step once; never doubled by the GPU-side annotations
    assert [k.short for k in m.kernels] == ["volta_sgemm_128x64_nn", "elementwise"]
    assert m.kernel_ms_per_step == 0.5           # 400 + 100 µs per step
    assert m.kernels[0].calls_per_step == 1.0


def test_failed_analysis_is_recorded(tmp_path, monkeypatch):
    write_trace(tmp_path / "t.json")
    monkeypatch.setattr(analyses, "idle", lambda ta: 1 / 0)
    a = analyses.run_all(tmp_path)
    assert "ZeroDivisionError" in a.unavailable["idle"]
    assert a.kernels is not None


# --- Multi-rank (M3) ---------------------------------------------------------------------

def test_comm_overlap_merges_and_intersects():
    kernels = [("COMPUTATION", 0, 100), ("COMPUTATION", 50, 100),   # compute 0-150
               ("COMMUNICATION", 100, 100),                        # comm 100-200
               ("COMMUNICATION", 120, 30),                         # inside the first
               ("MEMORY", 180, 50)]                                # not compute
    assert analyses.comm_overlap(kernels) == {"comm_us": 100.0, "overlapped_us": 50.0}
    assert analyses.comm_overlap([("COMPUTATION", 0, 10)]) == {"comm_us": 0.0,
                                                               "overlapped_us": 0.0}


def rank_metrics(rank, **values):
    return normalize.RunMetrics("r", 5, values.pop("step_time_ms", 100.0), rank=rank, **values)


def test_aggregate_ranks():
    ranks = [rank_metrics(0, busy_ms_per_step=50.0, gpu_idle_pct=5.0),
             rank_metrics(1, busy_ms_per_step=60.0, gpu_idle_pct=25.0),
             rank_metrics(2, busy_ms_per_step=50.0, gpu_idle_pct=None)]
    rows = {r["metric"]: r for r in normalize.aggregate_ranks(ranks)}
    assert rows["busy_ms_per_step"] == {"metric": "busy_ms_per_step", "min": 50.0, "min_rank": 0,
                                        "median": 50.0, "max": 60.0, "max_rank": 1,
                                        "max_over_median": 1.2}
    assert rows["gpu_idle_pct"]["median"] == 15.0  # ranks without a value are left out
    assert "comm_pct" not in rows                  # no rank has it


def test_slowest_rank_is_most_busy_not_longest_step():
    ranks = [rank_metrics(0, busy_ms_per_step=50.0, step_time_ms=120.0),
             rank_metrics(1, busy_ms_per_step=70.0, step_time_ms=100.0),
             rank_metrics(2, busy_ms_per_step=70.0, step_time_ms=100.0)]
    assert normalize.slowest_rank(ranks).rank == 1  # ties go to the lowest rank
    no_busy = [rank_metrics(0, step_time_ms=90.0), rank_metrics(1, step_time_ms=95.0)]
    assert normalize.slowest_rank(no_busy).rank == 1  # falls back to step time


def test_real_hta_multi_rank(tmp_path):
    write_run(tmp_path, 2, steps=4, kernels=[("volta_sgemm_128x64_nn", 400)],
              comm_kernels=[(ALLREDUCE, 200)],
              per_rank={1: {"kernels": [("volta_sgemm_128x64_nn", 600)],
                            "comm_kernels": [(ALLREDUCE, 50)]}})
    results = analyses.run_ranks(tmp_path)
    assert sorted(results) == [0, 1]
    assert all(a.unavailable == {} for a in results.values())
    ms = [normalize.normalize_run("r", a, rank=r) for r, a in results.items()]
    assert [m.busy_ms_per_step for m in ms] == [0.4, 0.6]
    assert [m.comm_ms_per_step for m in ms] == [0.2, 0.05]
    assert ms[0].comm_pct == 33.3
    assert ms[0].overlap_pct == 0.0 and ms[0].exposed_comm_ms_per_step == 0.2  # serialized
    assert normalize.slowest_rank(ms).rank == 1
    # Host wait stays within the GPU's idle time, although the NCCL stream idles most of
    # the step.
    assert all(m.host_wait_ms_per_step <= m.idle_ms_per_step for m in ms)


def test_comm_overlap_matches_hta(tmp_path):
    write_run(tmp_path, 2, steps=4, kernels=[("volta_sgemm_128x64_nn", 400)],
              comm_kernels=[(ALLREDUCE, 300)], comm_overlap=True,
              per_rank={1: {"comm_overlap": False}})
    ta = analyses.load(tmp_path)
    with analyses.quiet():
        analyses.trim_to_complete_steps(ta)
        ours = analyses.comm(ta).set_index("rank")
        hta = ta.get_comm_comp_overlap(visualize=False).set_index("rank")
    for rank in (0, 1):
        pct = 100 * ours.loc[rank, "overlapped_us"] / ours.loc[rank, "comm_us"]
        assert pct == pytest.approx(hta.loc[rank, "comp_comm_overlap_pctg"], abs=0.01)
    assert hta.loc[0, "comp_comm_overlap_pctg"] > 90
    assert hta.loc[1, "comp_comm_overlap_pctg"] == 0


def test_single_gpu_run_has_no_comm(tmp_path):
    write_trace(tmp_path / "t.json", steps=4)
    m = normalize.normalize_run("a", analyses.run_all(tmp_path))
    assert m.comm_ms_per_step == 0.0 and m.comm_pct == 0.0
    assert m.overlap_pct is None and m.exposed_comm_ms_per_step == 0.0
    assert m.busy_ms_per_step == m.kernel_ms_per_step


def test_unequal_step_counts_are_noted(tmp_path):
    from torch_trace_lab.hta_llm import pipeline
    write_run(tmp_path, 2, steps=5, per_rank={1: {"steps": 7}})
    results = analyses.run_ranks(tmp_path)
    assert results[0].steps == 5 and results[1].steps == 7
    note = pipeline.step_count_note("r", results)
    assert note and "5 to 7" in note
    assert pipeline.step_count_note("r", {0: results[0]}) is None


def test_sub_microsecond_kernels_are_counted(tmp_path):
    """Kernels under 1 µs are recorded as 0 µs, so HTA's mean is 0; they must not make the
    kernel breakdown unavailable (seen in real DDP traces)."""
    write_trace(tmp_path / "t.json", steps=4,
                kernels=[("volta_sgemm_128x64_nn", 400), ("zero", 0), ("zero", 0)])
    a = analyses.run_all(tmp_path)
    assert "kernels" not in a.unavailable
    zero = a.kernels[a.kernels["name"] == "zero"].iloc[0]
    assert zero["count"] == 8 and zero["sum_us"] == 0  # 2 per step, 4 steps


def test_comm_by_collective_on_real_hta(tmp_path):
    write_run(tmp_path, 2, steps=4, kernels=[("volta_sgemm_128x64_nn", 400)],
              comm_kernels=[(ALLREDUCE, 200), (BROADCAST, 20)], comm_overlap=True)
    ms = [normalize.normalize_run("r", a, rank=r)
          for r, a in analyses.run_ranks(tmp_path).items()]
    for m in ms:
        assert m.comm_by_collective_ms_per_step == {"all-reduce": 0.2, "broadcast": 0.02}
        assert m.allreduce_ms_per_step == 0.2
        assert m.exposed_allreduce_ms_per_step == 0.0  # fully under the 400 µs GEMM
    rows = normalize.collective_table(ms, per_rank=False)
    assert rows == [{"collective": "all-reduce", "min": 0.2, "median": 0.2, "max": 0.2},
                    {"collective": "broadcast", "min": 0.02, "median": 0.02, "max": 0.02}]


def test_truncated_last_step_is_dropped(tmp_path):
    """A last step whose GPU work wasn't collected (a launch without its GPU event) is left out,
    on that rank only; complete last steps are kept."""
    write_run(tmp_path, 2, steps=4, kernels=[("volta_sgemm_128x64_nn", 400), ("elementwise", 100)],
              per_rank={1: {"truncate_last_step": True}})
    results = analyses.run_ranks(tmp_path)
    assert results[0].steps == 4 and results[1].steps == 3
    ms = [normalize.normalize_run("r", a, rank=r) for r, a in results.items()]
    assert all(m.kernel_ms_per_step == 0.5 for m in ms)  # per-step means use complete steps


def test_few_steps_are_noted(tmp_path):
    from torch_trace_lab.hta_llm import pipeline, settings
    write_trace(tmp_path / "few" / "t.json", steps=settings.FEW_STEPS - 1)
    write_trace(tmp_path / "enough" / "t.json", steps=settings.FEW_STEPS)
    few = pipeline.few_steps_note("few", analyses.run_ranks(tmp_path / "few"))
    assert few and f"only {settings.FEW_STEPS - 1} complete profiler steps" in few
    assert pipeline.few_steps_note("enough", analyses.run_ranks(tmp_path / "enough")) is None


@pytest.mark.parametrize("truncate, steps", [(False, 4), (True, 3)])
def test_launches_that_never_get_gpu_events_dont_drop_the_last_step(tmp_path, truncate, steps):
    write_trace(tmp_path / "t.json", steps=4, eventless_launches=2, truncate_last_step=truncate)
    assert analyses.run_all(tmp_path).steps == steps
