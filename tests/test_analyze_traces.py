import json
import os

import pytest

from kineto import write_trace
from test_hta_llm_brine import KEY, FakeSession, completion, models
from torch_trace_lab import cli
from torch_trace_lab.hta_llm import brine, ingest, sanitize

CMD = "analyze-traces"
OUTPUTS = ["report.md", "bundle.md", "bundle.json", "prompt.md", "run.log"]


@pytest.fixture
def two_runs(tmp_path):
    parent = tmp_path / "traces"
    write_trace(parent / "serial" / "t.json", steps=5, gap_us=600,
                kernels=[("volta_sgemm_128x64_nn", 300), ("elementwise", 100)],
                torch_lab={"precision": "fp32", "batch_size_per_gpu": 64,
                           "dataloader_workers": 1})
    write_trace(parent / "amp" / "t.json", steps=5, gap_us=20,
                kernels=[("ampere_fp16_s16816gemm_fp16", 150), ("elementwise", 100)],
                torch_lab={"precision": "fp16-amp", "batch_size_per_gpu": 64,
                           "dataloader_workers": 4})
    return parent


@pytest.fixture
def fake_brine(monkeypatch):
    """Patch requests.Session used by the Brine client; returns the fake session."""
    monkeypatch.setenv("BRINE_BASE_URL", BRINE_URL)
    session = FakeSession(models("DeepSeek-V4-Flash-0731"), completion("## Summary\nLLM says hi"))
    monkeypatch.setattr(brine.requests, "Session", lambda: session)
    return session


BRINE_URL = "https://brine.sharcnet.ca/v1"


def test_missing_base_url_writes_outputs_and_exits_1(monkeypatch, two_runs, tmp_path, fake_brine):
    monkeypatch.delenv("BRINE_BASE_URL")
    monkeypatch.setenv("BRINE_API_KEY", KEY)
    out = tmp_path / "out"
    assert run(two_runs, "--output-dir", out) == 1
    assert "BRINE_BASE_URL is not set" in (out / "report.md").read_text()
    assert (out / "bundle.md").exists() and (out / "prompt.md").exists()
    assert fake_brine.calls == []


def test_disallowed_base_url_is_refused(monkeypatch, two_runs, tmp_path, fake_brine):
    monkeypatch.setenv("BRINE_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("BRINE_API_KEY", KEY)
    out = tmp_path / "out"
    assert run(two_runs, "--output-dir", out) == 1
    assert "Refusing to send" in (out / "report.md").read_text()
    assert fake_brine.calls == []


def run(*args):
    return cli.main([CMD, *map(str, args)])


def test_list_flags(monkeypatch, capsys):
    monkeypatch.setenv("CC_CLUSTER", "nibi")  # cheap, so it works on a login node too
    assert run("--list-flags") == 0
    out = capsys.readouterr().out
    for flag_id in ["GPU_IDLE_HIGH", "HOST_BOUND", "LAUNCH_BOUND", "TOP_KERNEL_DOMINANT",
                    "MEMCPY_HIGH", "NO_AMP_SUSPECTED"]:
        assert flag_id in out
    assert "Fires when: GPU idle > 20% of the GPU timeline." in out


def test_list_flags_matches_flag_thresholds():
    from torch_trace_lab.hta_llm import heuristics
    assert heuristics.MEANINGS.keys() == heuristics.thresholds().keys()


def test_paths_required_without_list_flags(capsys):
    assert run() == 2
    assert "PATH" in capsys.readouterr().err


def test_placement_login_node_refuses_before_loading(monkeypatch, two_runs, tmp_path, capsys):
    monkeypatch.setenv("CC_CLUSTER", "nibi")
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: pytest.fail("traces were loaded"))
    assert run(two_runs, "--output-dir", tmp_path / "out") == 1
    err = capsys.readouterr().err
    assert "salloc" in err and "Open OnDemand" in err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("env", [{}, {"CC_CLUSTER": "nibi", "SLURM_JOB_ID": "77"}])
def test_placement_runs_locally_or_in_job(monkeypatch, env, two_runs, tmp_path):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert run(two_runs, "--no-llm", "--output-dir", tmp_path / "out") == 0


def test_no_llm_writes_everything(two_runs, tmp_path, fake_brine):
    out = tmp_path / "out"
    assert run(two_runs, "--no-llm", "--output-dir", out) == 0
    for name in OUTPUTS:
        assert (out / name).exists(), name
    assert fake_brine.calls == []
    b = json.loads((out / "bundle.json").read_text())
    assert b["scenario"] == "S2" and b["baseline"] == "amp"
    serial = next(r for r in b["runs"] if r["label"] == "serial")
    assert {f["id"] for f in serial["flags"]} >= {"GPU_IDLE_HIGH", "HOST_BOUND", "NO_AMP_SUSPECTED"}
    assert {d["field"] for d in b["config_diff"]} == {"precision", "dataloader_workers"}
    serial_derived = next(r for r in b["derived"] if r["run"] == "serial")
    amp_derived = next(r for r in b["derived"] if r["run"] == "amp")
    assert serial_derived["step_time_vs_baseline_ms"] == round(
        serial_derived["step_time_ms"] - amp_derived["step_time_ms"], 2)
    report = (out / "report.md").read_text()
    assert "flags only" in report and "HOST_BOUND" in report


def test_single_run_and_baseline_option(two_runs, tmp_path):
    out = tmp_path / "out"
    assert run(two_runs / "serial", "--no-llm", "--output-dir", out) == 0
    assert json.loads((out / "bundle.json").read_text())["scenario"] == "S1"
    assert "How can I make this faster" in (out / "prompt.md").read_text()
    assert run(two_runs, "--baseline", "serial", "--no-llm", "--output-dir", out) == 0
    assert json.loads((out / "bundle.json").read_text())["baseline"] == "serial"
    assert "Runtime differences" in (out / "prompt.md").read_text()
    assert run(two_runs, "--baseline", "nope", "--no-llm", "--output-dir", out) == 1


def test_llm_report(monkeypatch, two_runs, tmp_path, fake_brine):
    monkeypatch.setenv("BRINE_API_KEY", KEY)
    out = tmp_path / "out"
    assert run(two_runs, "--output-dir", out) == 0
    assert "LLM says hi" in (out / "report.md").read_text()
    sent = fake_brine.calls[1][2]["json"]["messages"][1]["content"]
    assert "## 3. Rule-based flags" in sent


def test_missing_key_writes_outputs_and_exits_1(two_runs, tmp_path, fake_brine, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/torch-lab/brine_key here
    monkeypatch.setattr(brine.settings, "BRINE_KEY_FILE", str(tmp_path / "missing"))
    out = tmp_path / "out"
    assert run(two_runs, "--output-dir", out) == 1
    for name in OUTPUTS:
        assert (out / name).exists()
    assert "no Brine API key" in (out / "report.md").read_text()
    assert fake_brine.calls == []


def test_unreachable_brine_writes_outputs_and_exits_1(monkeypatch, two_runs, tmp_path):
    import requests
    monkeypatch.setenv("BRINE_API_KEY", KEY)
    monkeypatch.setenv("BRINE_BASE_URL", BRINE_URL)
    session = FakeSession(*[requests.ConnectionError()] * 3)
    monkeypatch.setattr(brine.requests, "Session", lambda: session)
    monkeypatch.setattr(brine.time, "sleep", lambda s: None)
    out = tmp_path / "out"
    assert run(two_runs, "--output-dir", out) == 1
    assert "could not connect" in (out / "report.md").read_text()
    assert (out / "bundle.md").exists()


def test_cpu_only_trace_fails(tmp_path, capsys):
    write_trace(tmp_path / "r" / "t.json", gpu=False)
    assert run(tmp_path / "r", "--no-llm", "--output-dir", tmp_path / "out") == 1
    assert "contain only CPU events" in capsys.readouterr().err


def test_no_metadata_says_not_distinguishable(tmp_path):
    write_trace(tmp_path / "p" / "a" / "t.json", gap_us=500)
    write_trace(tmp_path / "p" / "b" / "t.json", gap_us=50)
    out = tmp_path / "out"
    assert run(tmp_path / "p", "--no-llm", "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    assert b["configs_distinguishable"] is False
    assert "not distinguishable" in (out / "bundle.md").read_text()


# --- Allowlist, leak check and key hygiene (spec §8a, §14) ---

SECRETS = ["jdoe", "/home/jdoe", "4471234", "gpu-node-17", "def-someprof", "site-packages"]


@pytest.fixture
def leaky_env(monkeypatch, tmp_path):
    monkeypatch.setenv("USER", "jdoe")
    monkeypatch.setenv("HOME", "/home/jdoe")
    monkeypatch.setenv("SLURM_JOB_ID", "4471234")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "def-someprof")
    monkeypatch.setattr(sanitize, "_hostnames", lambda: {"gpu-node-17"})
    parent = tmp_path / "exp"
    extra = {"traceName": "/home/jdoe/scratch/4471234_train_trace.json",
             "trace_id": "ABCDEF", "host_name": "gpu-node-17",
             "notes_from_tool": "def-someprof"}
    pyfuncs = ["/home/jdoe/.venv/lib/python3.12/site-packages/torch/nn/x.py(12): forward",
               "threading.py(995): _bootstrap"]
    for name in ["4471234_jdoe_run", "clean-run"]:
        write_trace(parent / name / "4471234_trace.json", extra_top=extra,
                    python_functions=pyfuncs,
                    torch_lab={"precision": "fp32", "account": "def-someprof",
                               "path": "/home/jdoe/data"})
    return parent


def test_nothing_sensitive_reaches_outputs(leaky_env, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(leaky_env, "--no-llm", "--output-dir", out) == 0
    for name in ["prompt.md", "bundle.md", "bundle.json"]:
        text = (out / name).read_text()
        for secret in SECRETS:
            assert secret not in text, f"{secret!r} leaked into {name}"
        assert "traceName" not in text and "ABCDEF" not in text


def test_run_note_with_path_blocks_brine(leaky_env, tmp_path, monkeypatch, fake_brine):
    monkeypatch.setenv("BRINE_API_KEY", KEY)
    out = tmp_path / "out"
    code = run(leaky_env, "--run-note", "clean-run=data in /home/jdoe/imagenet",
               "--output-dir", out)
    assert code == 1
    assert fake_brine.calls == []
    for name in ["bundle.md", "bundle.json", "prompt.md", "report.md"]:
        assert (out / name).exists()
    assert "leak check" in (out / "report.md").read_text()


def test_key_never_written(monkeypatch, two_runs, tmp_path, fake_brine, capsys):
    monkeypatch.setenv("BRINE_API_KEY", KEY)
    out = tmp_path / "out"
    assert run(two_runs, "--output-dir", out) == 0
    captured = capsys.readouterr()
    for name in os.listdir(out):
        assert KEY not in (out / name).read_text(), name
    assert KEY not in captured.out + captured.err


def test_run_note_needs_label_for_comparisons(two_runs, tmp_path, capsys):
    assert run(two_runs, "--run-note", "no label", "--no-llm", "--output-dir", tmp_path) == 1
    assert "LABEL=" in capsys.readouterr().err


def test_run_note_reaches_bundle(two_runs, tmp_path):
    out = tmp_path / "out"
    assert run(two_runs, "--run-note", "amp=bs 64, 4 workers", "--no-llm",
               "--output-dir", out) == 0
    assert "bs 64, 4 workers" in (out / "bundle.md").read_text()


BF16_GEMM = ("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_tn", 300)
MULTI_LOADER = {"cat": "user_annotation",
                "name": "enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__"}


@pytest.fixture
def bf16_run(tmp_path):
    run_dir = tmp_path / "bf16-run"
    write_trace(run_dir / "t.json", steps=5, kernels=[BF16_GEMM, ("elementwise", 100)],
                extra_events=[MULTI_LOADER])
    return run_dir


def test_inferred_values_reach_bundle_with_confidence(bf16_run, tmp_path):
    out = tmp_path / "out"
    assert run(bf16_run, "--no-llm", "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    config = b["runs"][0]["config"]
    assert config["precision"]["value"] == "bf16"
    assert config["precision"]["source"] == "inferred"
    assert config["precision"]["confidence"] == "high"
    assert config["dataloader_workers"]["value"] == ">0"
    md = (out / "bundle.md").read_text()
    assert "bf16 (inferred, high)" in md
    assert "Evidence for inferred values" in md


def test_meta_conflict_is_noted_and_user_value_wins(bf16_run, tmp_path):
    meta = tmp_path / "meta.toml"
    meta.write_text('precision = "fp32"\ntask = "image classification"\nshoe_size = 44\n')
    out = tmp_path / "out"
    assert run(bf16_run, "--meta", meta, "--no-llm", "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    config = b["runs"][0]["config"]
    assert config["precision"] == {"value": "fp32", "source": "user"}
    assert config["task"]["value"] == "image classification"
    assert any("precision: using fp32 from --meta" in n and "bf16" in n
               for n in b["data_quality"])
    for name in ["bundle.md", "prompt.md", "bundle.json"]:
        text = (out / name).read_text()
        assert "shoe_size" not in text
    assert "precision: using fp32 from --meta" in (out / "prompt.md").read_text()
    assert "shoe_size is not a known key" in (out / "run.log").read_text()


def test_meta_per_run_overrides_in_comparison(two_runs, tmp_path):
    meta = tmp_path / "meta.toml"
    meta.write_text('model = "ResNet-50"\n[runs."amp"]\nnotes = "channels_last"\n')
    out = tmp_path / "out"
    assert run(two_runs, "--meta", meta, "--run-note", "amp=lr 0.1", "--no-llm",
               "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    runs = {r["label"]: r for r in b["runs"]}
    assert runs["amp"]["notes"] == "channels_last; lr 0.1"
    assert runs["serial"]["config"]["model"] == {"value": "ResNet-50", "source": "user"}


def test_invalid_meta_exits_1(two_runs, tmp_path, capsys):
    meta = tmp_path / "meta.toml"
    meta.write_text("precision = \n")
    assert run(two_runs, "--meta", meta, "--no-llm", "--output-dir", tmp_path / "o") == 1
    assert "not valid TOML" in capsys.readouterr().err


def test_meta_notes_with_path_block_brine(leaky_env, tmp_path, monkeypatch, fake_brine):
    monkeypatch.setenv("BRINE_API_KEY", KEY)
    meta = tmp_path / "meta.toml"
    meta.write_text('notes = "data in /home/jdoe/imagenet"\n')
    out = tmp_path / "out"
    assert run(leaky_env, "--meta", meta, "--output-dir", out) == 1
    assert fake_brine.calls == []
    assert "leak check" in (out / "report.md").read_text()
