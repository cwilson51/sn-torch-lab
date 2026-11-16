"""Pick a training batch size that fills the available GPU memory.

Rules:

* On CUDA, try powers of 2 from ``MIN_BATCH`` up to ``max_batch`` (default 512). Each
  candidate runs one real training step (forward, backward, SGD with momentum) on
  synthetic data. The result is the largest candidate whose peak reserved memory stays
  within ``headroom`` (default 90%) of the memory that was free before probing.
* On CPU, memory is rarely the limit; run time is. Return a fixed ``CPU_BATCH`` (32).

Use it from your own scripts::

    from torch_trace_lab.batch_size import find_batch_size
    batch_size = find_batch_size(my_model_fn, (3, 224, 224), num_classes=1000)

or print the size for the demo's ResNet50 with ``python -m torch_trace_lab.batch_size``.
"""

import warnings
from collections.abc import Callable, Iterable

import torch

GPU_MAX_BATCH = 512
CPU_BATCH = 32
MIN_BATCH = 8
HEADROOM = 0.9


def powers_of_two(low: int, high: int) -> list[int]:
    """Powers of 2 in ``[low, high]``."""
    sizes, n = [], 1
    while n <= high:
        if n >= low:
            sizes.append(n)
        n *= 2
    return sizes


def largest_fitting(
    try_step: Callable[[int], int], budget_bytes: float, candidates: Iterable[int]
) -> int | None:
    """Return the largest candidate whose peak memory fits in ``budget_bytes``.

    ``try_step(batch_size)`` returns the peak bytes used by one training step, or raises
    ``torch.OutOfMemoryError``. Candidates are tried in increasing order and the search
    stops at the first one that runs out of memory or exceeds the budget.

    Peak memory grows roughly linearly with batch size, so once two sizes have been
    measured, the next one's peak is extrapolated and the search stops without trying it
    if the estimate is over budget. That avoids deliberately running out of memory (and
    the allocator warnings PyTorch prints when that happens). Catching OOM remains as a
    fallback in case the estimate is too low.
    """
    best = None
    measured: list[tuple[int, int]] = []  # (batch_size, peak_bytes) for sizes that fit
    for batch_size in sorted(candidates):
        if len(measured) >= 2:
            (n1, p1), (n2, p2) = measured[-2:]
            estimate = p2 + (p2 - p1) / (n2 - n1) * (batch_size - n2)
            if estimate > budget_bytes:
                break
        try:
            peak = try_step(batch_size)
        except torch.OutOfMemoryError:
            break
        if peak > budget_bytes:
            break
        best = batch_size
        measured.append((batch_size, peak))
    return best


def default_device() -> torch.device:
    """Same choice as ``train.py``: CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_batch_size(
    make_model: Callable[[], torch.nn.Module],
    input_shape: tuple[int, ...],
    num_classes: int,
    device: torch.device | str | None = None,
    *,
    max_batch: int = GPU_MAX_BATCH,
    headroom: float = HEADROOM,
) -> int:
    """Largest power-of-2 batch size that fits on ``device`` for a classification model.

    ``make_model`` builds the model, ``input_shape`` is one sample's shape (no batch
    dimension), and ``num_classes`` sizes the synthetic labels.
    """
    device = torch.device(device) if device is not None else default_device()
    if device.type != "cuda":
        return min(CPU_BATCH, max_batch)

    free, _total = torch.cuda.mem_get_info(device)
    budget = free * headroom

    model = make_model().to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    def try_step(batch_size: int) -> int:
        inputs = labels = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            inputs = torch.randn(batch_size, *input_shape, device=device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            return torch.cuda.max_memory_reserved(device)
        finally:
            optimizer.zero_grad(set_to_none=True)
            del inputs, labels, loss

    try:
        best = largest_fitting(try_step, budget, powers_of_two(MIN_BATCH, max_batch))
    finally:
        del model, optimizer
        torch.cuda.empty_cache()

    if best is None:
        warnings.warn(
            f"Even batch size {MIN_BATCH} does not fit in {budget / 2**30:.1f} GiB; "
            f"using {MIN_BATCH} anyway.",
            stacklevel=2,
        )
        return MIN_BATCH
    return best


def resnet50_batch_size(device: torch.device | str | None = None) -> int:
    """Batch size for the demo's ResNet50 on ImageNet-sized (3x224x224) images."""
    from torchvision.models import resnet50

    return find_batch_size(resnet50, (3, 224, 224), num_classes=1000, device=device)


if __name__ == "__main__":
    print(resnet50_batch_size())
