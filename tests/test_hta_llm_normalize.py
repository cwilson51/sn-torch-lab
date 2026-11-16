import pandas as pd
import pytest

from kineto import write_trace
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


def test_tables_render():
    md = tables.render([{"name": "a|b", "ms": 1.5, "n": None}])
    assert md.splitlines() == ["| name | ms | n |", "|---|---:|---:|", "| a\\|b | 1.5 | – |"]


def test_real_hta_on_fixture(tmp_path):
    write_trace(tmp_path / "t.json", memcpy_us=50, steps=4,
                kernels=[("volta_sgemm_128x64_nn", 400), ("tiny", 3)])
    m = normalize.normalize_run("a", analyses.run_all(tmp_path))
    assert m.unavailable == {}
    assert m.steps == 3  # HTA drops the last, partial step
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
    assert a.steps in (8, 9)  # complete steps only; never doubled
    assert [k.short for k in m.kernels] == ["volta_sgemm_128x64_nn", "elementwise"]
    assert m.kernel_ms_per_step == 0.5           # 400 + 100 µs per step
    assert m.kernels[0].calls_per_step == 1.0


def test_failed_analysis_is_recorded(tmp_path, monkeypatch):
    write_trace(tmp_path / "t.json")
    monkeypatch.setattr(analyses, "idle", lambda ta: 1 / 0)
    a = analyses.run_all(tmp_path)
    assert "ZeroDivisionError" in a.unavailable["idle"]
    assert a.kernels is not None
