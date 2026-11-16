import json

import pytest

from torch_trace_lab import train


@pytest.mark.parametrize("flags, device, precision", [
    ([], "cuda", "fp32"),
    (["--compile"], "cuda", "tf32"),
    (["--compile", "--amp"], "cuda", "fp16-amp"),
    (["--amp"], "cpu", "bf16-amp"),
    (["--compile"], "cpu", "fp32"),
])
def test_precision(flags, device, precision):
    args = train.parse_args(["64", *flags])
    assert train.torch_lab_metadata(args, device)["precision"] == precision


def test_metadata_fields():
    args = train.parse_args(["128", "--nWorkers", "4", "--compile",
                             "--skip_first", "10", "--warmup", "10", "--active", "10"])
    meta = train.torch_lab_metadata(args, "cuda")
    assert meta["batch_size_per_gpu"] == 128
    assert meta["dataloader_workers"] == 4
    assert meta["torch_compile"] is True
    assert meta["profiler_schedule"] == {"skip_first": 10, "wait": 0, "warmup": 10, "active": 10}


def test_embed_metadata_keeps_trace(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"traceEvents": [{"ph": "X"}], "schemaVersion": 1}))
    train.embed_metadata(path, {"precision": "fp32"})
    trace = json.loads(path.read_text())
    assert trace["torch_lab"] == {"precision": "fp32"}
    assert trace["traceEvents"] == [{"ph": "X"}]
