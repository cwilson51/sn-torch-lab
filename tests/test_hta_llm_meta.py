"""--meta FILE.toml loading and source merging (spec §5.2 sources 4-5, conflicts; §14 M2)."""

import pytest

from torch_trace_lab.hta_llm.context import resolve_config, same_value
from torch_trace_lab.hta_llm.meta import as_config, load_meta

LABELS = ["bs64-ddp", "bs64-fsdp"]


def write(tmp_path, text):
    path = tmp_path / "meta.toml"
    path.write_text(text)
    return path


def test_top_level_applies_to_all_and_runs_override(tmp_path):
    path = write(tmp_path, '''
model = "ResNet-50"
precision = "fp32"
[runs."bs64-ddp"]
precision = "bf16-amp"
parallelism = "ddp"
  [runs."bs64-ddp".ddp]
  bucket_cap_mb = 25
  gradient_as_bucket_view = false
''')
    per_run, warnings = load_meta(path, LABELS)
    assert warnings == []
    assert per_run["bs64-fsdp"] == {"model": "ResNet-50", "precision": "fp32"}
    assert per_run["bs64-ddp"] == {"model": "ResNet-50", "precision": "bf16-amp",
                                   "parallelism": "DDP", "ddp.bucket_cap_mb": 25,
                                   "ddp.gradient_as_bucket_view": False}


def test_unknown_keys_and_bad_types_are_dropped_with_warnings(tmp_path):
    path = write(tmp_path, '''
account = "def-someprof"
batch_size_per_gpu = "sixty-four"
[fsdp]
sharding_strategy = "EVERYTHING"
secret_path = "/home/jdoe"
[runs."bs64-ddp"]
torch_compile = "yes"
''')
    per_run, warnings = load_meta(path, LABELS)
    assert per_run == {"bs64-ddp": {}, "bs64-fsdp": {}}
    text = "\n".join(warnings)
    for key in ["account", "batch_size_per_gpu", "fsdp.sharding_strategy", "fsdp.secret_path",
                "torch_compile"]:
        assert key in text
    # Warnings name keys, never values.
    assert "def-someprof" not in text and "/home/jdoe" not in text


def test_unknown_run_label_warns(tmp_path):
    path = write(tmp_path, '[runs."nope"]\nprecision = "fp32"\n')
    per_run, warnings = load_meta(path, LABELS)
    assert per_run == {"bs64-ddp": {}, "bs64-fsdp": {}}
    assert "does not match any run label" in warnings[0]


def test_invalid_toml_raises(tmp_path):
    with pytest.raises(ValueError, match="not valid TOML"):
        load_meta(write(tmp_path, "precision = \n"), LABELS)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="can't read"):
        load_meta(tmp_path / "missing.toml", LABELS)


def test_notes_are_split_off_and_text_is_collapsed(tmp_path):
    path = write(tmp_path, 'notes = """switched to\n   channels_last"""\ntask = "  cls  "\n')
    per_run, _ = load_meta(path, ["a"])
    config, notes = as_config(per_run["a"])
    assert notes == "switched to channels_last"
    assert config == {"task": {"value": "cls"}}


INFERRED_BF16 = {"value": "bf16", "source": "inferred", "confidence": "high",
                 "evidence": "bf16 kernels are 99.0% of GEMM/conv time"}


def test_user_value_wins_and_conflict_is_noted():
    config, conflicts = resolve_config("run-a", {
        "inferred": {"precision": INFERRED_BF16},
        "embedded": {"precision": {"value": "fp32"}},
        "user": {"precision": {"value": "fp32"}},
    })
    assert config["precision"] == {"value": "fp32", "source": "user"}
    assert len(conflicts) == 1
    assert "run-a: precision: using fp32 from --meta" in conflicts[0]
    assert "bf16 (inferred, high confidence" in conflicts[0]


def test_inferred_fills_gaps_and_agreement_is_not_a_conflict():
    config, conflicts = resolve_config("r", {
        "inferred": {"precision": INFERRED_BF16,
                     "dataloader_workers": {**INFERRED_BF16, "value": ">0"}},
        "embedded": {"precision": {"value": "bf16-amp"}, "dataloader_workers": {"value": 8}},
    })
    assert conflicts == []
    assert config["precision"]["source"] == "embedded"


@pytest.mark.parametrize("key, a, b, same", [
    ("precision", "fp32 (TF32 convs)", "tf32", True),
    ("precision", "fp32", "tf32", False),
    ("precision", "fp16-amp", "fp16", True),
    ("precision", "bf16-amp", "fp16-amp", False),
    ("dataloader_workers", ">0", 4, True),
    ("dataloader_workers", ">0", 0, False),
    ("torch_compile", True, False, False),
    ("model", "ResNet-50", "resnet-50", True),
])
def test_same_value(key, a, b, same):
    assert same_value(key, a, b) is same
