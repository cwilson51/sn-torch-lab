import pytest
import torch

from torch_trace_lab import batch_size as bs


def peak_per_image(bytes_per_image):
    return lambda n: n * bytes_per_image


def test_powers_of_two():
    assert bs.powers_of_two(8, 512) == [8, 16, 32, 64, 128, 256, 512]
    assert bs.powers_of_two(8, 100) == [8, 16, 32, 64]


def test_stops_at_budget():
    candidates = bs.powers_of_two(8, 512)
    assert bs.largest_fitting(peak_per_image(100), 7000, candidates) == 64


def test_stops_at_oom():
    def step(n):
        if n > 32:
            raise torch.OutOfMemoryError("out of memory")
        return n

    assert bs.largest_fitting(step, float("inf"), bs.powers_of_two(8, 512)) == 32


def test_respects_cap():
    candidates = bs.powers_of_two(bs.MIN_BATCH, bs.GPU_MAX_BATCH)
    assert bs.largest_fitting(peak_per_image(1), float("inf"), candidates) == 512


def test_skips_size_predicted_over_budget():
    tried = []

    def step(n):
        tried.append(n)
        if n > 64:
            raise torch.OutOfMemoryError("should have been predicted")
        return 1000 + 100 * n

    budget = 1000 + 100 * 64 + 50
    assert bs.largest_fitting(step, budget, bs.powers_of_two(8, 512)) == 64
    assert tried == [8, 16, 32, 64]


def test_low_estimate_falls_back_to_oom():
    def step(n):
        if n > 64:
            raise torch.OutOfMemoryError("out of memory")
        return 100 * n

    assert bs.largest_fitting(step, float("inf"), bs.powers_of_two(8, 512)) == 64


def test_nothing_fits():
    assert bs.largest_fitting(peak_per_image(100), 10, bs.powers_of_two(8, 512)) is None


def test_cpu_uses_fixed_size():
    assert bs.find_batch_size(lambda: pytest.fail("no probe on CPU"), (3, 224, 224), 1000,
                              device="cpu") == bs.CPU_BATCH


def test_cpu_respects_smaller_cap():
    assert bs.find_batch_size(torch.nn.Identity, (3,), 10, device="cpu", max_batch=16) == 16
