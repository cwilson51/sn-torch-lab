"""Config inference from trace contents (spec §5.2 source 2, §14 M2)."""

import pytest

from kineto import ALLREDUCE, DDP_EVENTS, make_trace
from torch_trace_lab.hta_llm.infer import collect_evidence, infer_config, infer_topology

FP16_GEMM = ("sm90_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc", 400)
BF16_GEMM = ("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_tn", 400)
TF32_GEMM = ("sm90_xmma_wgrad_indexed_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc", 400)
FP32_GEMM = ("volta_sgemm_128x64_nn", 400)
ELEMENTWISE = ("void at::native::vectorized_elementwise_kernel<4, at::native::AddFunctor<float>>",
               100)
TRANSPOSE = ("void cudnn::engines_precompiled::nchwToNhwcKernel<float, float, float>", 100)

SCALER = {"cat": "cpu_op", "name": "aten::_amp_foreach_non_finite_check_and_unscale_"}
COMPILED_REGION = {"cat": "cpu_op", "name": "Torch-Compiled Region: 0/0"}
MULTI_LOADER = {"cat": "user_annotation",
                "name": "enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__"}
SINGLE_LOADER = {"cat": "user_annotation",
                 "name": "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__"}
GRAPH_LAUNCH = {"cat": "cuda_runtime", "name": "cudaGraphLaunch"}


def conv(dims, dtype="float"):
    return {"cat": "cpu_op", "name": "aten::conv2d",
            "args": {"Input Dims": [dims, [64, 3, 7, 7]], "Input type": [dtype, dtype]}}


def inferred(kernels=(FP32_GEMM, ELEMENTWISE), extra_events=()):
    trace = make_trace(kernels=list(kernels), extra_events=list(extra_events))
    return infer_config(collect_evidence(trace["traceEvents"]))


@pytest.mark.parametrize("kernel, extra, expected, confidence", [
    (FP16_GEMM, [SCALER], "fp16-amp", "high"),
    (FP16_GEMM, [], "fp16", "medium"),
    (BF16_GEMM, [], "bf16", "high"),
    (TF32_GEMM, [], "tf32", "high"),
    (FP32_GEMM, [], "fp32", "high"),
])
def test_precision(kernel, extra, expected, confidence):
    p = inferred([kernel, ELEMENTWISE], extra)["precision"]
    assert p["value"] == expected
    assert p["confidence"] == confidence
    assert p["source"] == "inferred"


def test_mixed_precision_lowers_confidence():
    p = inferred([FP16_GEMM, FP32_GEMM, ("ampere_bf16_s16816gemm_bf16", 300)])["precision"]
    assert p["confidence"] == "low"


def test_precision_mentions_input_dtype():
    p = inferred([FP16_GEMM], [SCALER, conv([32, 3, 224, 224], "c10::Half")])["precision"]
    assert "conv/linear inputs mostly fp16" in p["evidence"]


def test_compile_from_compiled_regions_and_triton():
    assert inferred(extra_events=[COMPILED_REGION])["torch_compile"]["value"] is True
    triton = inferred([("triton_poi_fused_add_relu_0", 50), FP32_GEMM])["torch_compile"]
    assert (triton["value"], triton["confidence"]) == (True, "medium")
    assert inferred()["torch_compile"]["value"] is False


@pytest.mark.parametrize("events, expected", [
    ([MULTI_LOADER], ">0"),
    ([SINGLE_LOADER], 0),
])
def test_dataloader_workers(events, expected):
    assert inferred(extra_events=events)["dataloader_workers"]["value"] == expected


def test_dataloader_workers_unknown_without_iterator_ops():
    assert "dataloader_workers" not in inferred()


def test_batch_size_from_shapes():
    c = inferred(extra_events=[conv([48, 3, 224, 224])])
    assert c["batch_size_per_gpu"]["value"] == 48


def test_batch_size_unknown_without_shapes():
    assert "batch_size_per_gpu" not in inferred(extra_events=[COMPILED_REGION])


def test_cuda_graphs():
    assert inferred(extra_events=[GRAPH_LAUNCH])["cuda_graphs"]["value"] is True
    assert inferred()["cuda_graphs"]["value"] is False


def test_memory_format():
    assert inferred([TF32_GEMM, TRANSPOSE])["memory_format"]["value"] == "contiguous (NCHW)"
    assert inferred([TF32_GEMM])["memory_format"]["value"] == "channels_last"
    assert "memory_format" not in inferred([FP32_GEMM])


def test_no_evidence_leaves_everything_unknown():
    trace = make_trace(kernels=[ELEMENTWISE])
    events = [e for e in trace["traceEvents"] if e["cat"] not in ("cpu_op",)]
    assert infer_config(collect_evidence(events)) == {}
    assert infer_config(None) == {}


def test_evidence_strings_hold_no_trace_text():
    c = inferred([FP16_GEMM], [SCALER, MULTI_LOADER, COMPILED_REGION, conv([8, 3, 2, 2])])
    for entry in c.values():
        assert "aten::" not in entry["evidence"]
        assert "xmma" not in entry["evidence"]


# --- M4: DDP and topology ----------------------------------------------------------------

@pytest.mark.parametrize("events, confidence", [
    (DDP_EVENTS, "high"),
    (DDP_EVENTS[:2], "medium"),   # reducer ops only
    (DDP_EVENTS[2:], "medium"),   # DistributedDataParallel.forward only
])
def test_ddp_detected(events, confidence):
    p = inferred(extra_events=events)["parallelism"]
    assert (p["value"], p["confidence"], p["source"]) == ("DDP", confidence, "inferred")
    assert "reducer::" not in p["evidence"]


def test_no_ddp_markers_leave_parallelism_unknown():
    # NCCL kernels alone don't say which framework; cuDNN's nhwcAddPaddingKernel holds "dd".
    padding = ("void cudnn::engines_precompiled::nhwcAddPaddingKernel<float>", 10)
    trace = make_trace(kernels=[FP32_GEMM, padding], comm_kernels=[(ALLREDUCE, 100)])
    assert "parallelism" not in infer_config(collect_evidence(trace["traceEvents"]))


def test_evidence_records_kernel_devices():
    ev = collect_evidence(make_trace(device=3, comm_kernels=[(ALLREDUCE, 10)])["traceEvents"])
    assert ev.devices == {3}


@pytest.mark.parametrize("devices, visible, nodes, gpus", [
    ([{0}, {1}, {2}, {3}], [4, 4, 4, 4], 1, 4),   # 1 node x 4 GPUs
    ([{0}, {1}, {0}, {1}], [2, 2, 2, 2], 2, 2),   # 2 nodes x 2 GPUs
])
def test_topology(devices, visible, nodes, gpus):
    t = infer_topology(devices, visible, 4)
    assert t["num_nodes"]["value"] == nodes and t["gpus_per_node"]["value"] == gpus
    assert t["num_nodes"]["confidence"] == "medium"


@pytest.mark.parametrize("devices, visible, world_size", [
    ([{0}, {0}, {0}, {0}], [1, 1, 1, 1], 4),       # each process pinned to one GPU
    ([{0}, {1}, {2}, {3}], [8, 8, 8, 8], 4),       # sees more GPUs than the ranks use
    ([{0}, {1}, {2}, {0}, {1}], [3, 3, 3, 3, 3], 5),  # 5 ranks don't fill 3-GPU nodes
    ([{0, 1}, {1}], [2, 2], 2),                    # a rank on two GPUs
    ([{0}, set()], [2, 2], 2),                     # a rank with no kernels
    ([{0}], [1], 1),                               # single GPU
])
def test_topology_unknown(devices, visible, world_size):
    assert infer_topology(devices, visible, world_size) == {}
