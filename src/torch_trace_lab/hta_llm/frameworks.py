"""Distributed-framework knowledge (spec §6a): detection markers, NCCL collective types and
framework prompt guidance.

M4 covers DDP. Patterns are taken from real DDP traces (torch 2.x, NCCL 2.27). Detection
markers never match on a bare "ddp" substring: cuDNN's ``nhwcAddPaddingKernel`` contains it.
"""

import re

from torch_trace_lab.hta_llm import report

# CPU ops recorded by DDP's gradient reducer, and DDP's forward annotation.
DDP_CPU_OP_PREFIXES = ("torch::distributed::reducer::", "torch.distributed.ddp.reducer::")
DDP_ANNOTATION = "DistributedDataParallel.forward"

# Collective types, in table order. NCCL kernels that are none of these are "other".
COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "broadcast", "other")
_NCCL_COLLECTIVE = re.compile(r"nccl(?:Dev)?Kernel_([A-Za-z]+)")
_TYPES = {"AllReduce": "all-reduce", "AllGather": "all-gather",
          "ReduceScatter": "reduce-scatter", "Broadcast": "broadcast"}

# parallelism value -> prompt file under prompts/frameworks/ (FSDP, DeepSpeed: M4b).
PROMPT_FILES = {"ddp": "frameworks/ddp.md"}


def is_ddp_cpu_op(name: str) -> bool:
    return name.startswith(DDP_CPU_OP_PREFIXES)


def collective_type(kernel_name: str) -> str | None:
    """The collective an NCCL kernel runs (one of COLLECTIVES), or None for other kernels.

    ``ncclDevKernel_AllReduce_Sum_f32_RING_LL(...)`` -> ``all-reduce``.
    """
    if "nccl" not in kernel_name.lower():
        return None
    m = _NCCL_COLLECTIVE.search(kernel_name)
    return _TYPES.get(m.group(1), "other") if m else "other"


def framework_prompt(parallelism) -> str:
    """Prompt guidance for the run's framework, or "" if there is none for it."""
    path = PROMPT_FILES.get(str(parallelism or "").strip().lower())
    return report.prompt_file(path).strip() if path else ""
