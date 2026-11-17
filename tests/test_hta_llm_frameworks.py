"""Distributed-framework helpers (spec §6a, M4)."""

import pytest

from kineto import ALLREDUCE, BROADCAST
from torch_trace_lab.hta_llm import frameworks
from torch_trace_lab.hta_llm.analyses import collective_breakdown


@pytest.mark.parametrize("name, expected", [
    (ALLREDUCE, "all-reduce"),
    (BROADCAST, "broadcast"),
    ("ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)", "all-gather"),
    ("ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
     "reduce-scatter"),
    ("ncclKernel_AllReduce_RING_LL_Sum_float(ncclWorkElem)", "all-reduce"),  # older NCCL
    ("ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)", "other"),
    ("ncclDevKernel_Generic", "other"),
    ("volta_sgemm_128x64_nn", None),
    ("void cudnn::engines_precompiled::nhwcAddPaddingKernel<float>", None),
])
def test_collective_type(name, expected):
    assert frameworks.collective_type(name) == expected


@pytest.mark.parametrize("name", [
    "torch::distributed::reducer::mul_out",
    "torch.distributed.ddp.reducer::copy_bucket_to_grad",
])
def test_ddp_cpu_ops(name):
    assert frameworks.is_ddp_cpu_op(name)


def test_padding_kernel_is_not_ddp():
    assert not frameworks.is_ddp_cpu_op("nhwcAddPaddingKernel")


@pytest.mark.parametrize("value", ["DDP", "ddp"])
def test_ddp_prompt(value):
    text = frameworks.framework_prompt(value)
    assert "parallelism = DDP" in text and "bucket_cap_mb" in text and "no_sync" in text


@pytest.mark.parametrize("value", [None, "none", "FSDP", "DeepSpeed"])
def test_no_prompt_for_other_frameworks(value):
    assert frameworks.framework_prompt(value) == ""


def test_collective_breakdown():
    kernels = [
        ("COMPUTATION", None, 0.0, 100.0),
        ("COMMUNICATION", "all-reduce", 60.0, 60.0),   # 40 µs overlapped, 20 exposed
        ("COMMUNICATION", "all-reduce", 150.0, 10.0),  # exposed
        ("COMMUNICATION", "broadcast", 200.0, 5.0),
    ]
    out = collective_breakdown(kernels)
    assert out["by_collective"] == {"all-reduce": 70.0, "broadcast": 5.0}
    assert out["allreduce_us"] == 70.0
    assert out["allreduce_overlapped_us"] == 40.0


def test_collective_breakdown_without_comm():
    out = collective_breakdown([("COMPUTATION", None, 0.0, 10.0)])
    assert out == {"by_collective": {}, "allreduce_us": 0.0, "allreduce_overlapped_us": 0.0}
