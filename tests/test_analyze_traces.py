import json
import os

import pytest

from kineto import (ALLREDUCE, BROADCAST, DDP_EVENTS, DEVICE_PROPERTIES, write_run,
                    write_trace)
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


def test_budget_plan_shrinks_in_order():
    from torch_trace_lab.hta_llm import pipeline
    assert pipeline.budget_plan(15) == [
        (15, ()), (10, ()), (5, ()),
        (5, ("launch stats",)),
        (5, ("launch stats", "category comparison")),
        (5, ("launch stats", "category comparison", "kernel-name appendix")),
    ]
    assert pipeline.budget_plan(3)[0] == (3, ())
    assert all(n == 3 for n, _ in pipeline.budget_plan(3))


def many_kernel_runs(tmp_path):
    parent = tmp_path / "runs"
    kernels = [(f"void some_namespace::kernel_number_{i}<float, 128, true>", 50 + i)
               for i in range(16)]
    for name in ["a", "b"]:
        write_trace(parent / name / "t.json", steps=2, kernels=kernels)
    return parent


def test_budget_cascade_reduces_top_n_first(monkeypatch, tmp_path):
    from torch_trace_lab.hta_llm import pipeline, settings
    parent = many_kernel_runs(tmp_path)
    collected = pipeline.collect(pipeline.Options([parent], tmp_path / "o"), pipeline.Log())
    sizes = {}
    for n, drop in pipeline.budget_plan(15):
        b, md = pipeline.assemble(collected, n, drop, top_n_requested=15)
        sizes[(n, drop)] = pipeline.estimate_tokens(*pipeline.build_prompt(b, md))
    # Budget between the top-10 and top-15 sizes: only top-n shrinks.
    monkeypatch.setattr(settings, "MAX_PROMPT_TOKENS", sizes[(10, ())])
    out = tmp_path / "out"
    assert run(parent, "--no-llm", "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    assert b["budget"]["top_n_used"] == 10 and b["budget"]["dropped_sections"] == []
    assert len(b["runs"][0]["kernels"]) == 11  # top 10 + remaining
    assert "To fit the prompt budget, kernel tables show the top 10 kernels instead of 15" \
        in (out / "bundle.md").read_text()


def test_budget_cascade_drops_sections_then_warns(monkeypatch, tmp_path, capsys):
    from torch_trace_lab.hta_llm import settings
    parent = many_kernel_runs(tmp_path)
    monkeypatch.setattr(settings, "MAX_PROMPT_TOKENS", 10)
    out = tmp_path / "out"
    assert run(parent, "--no-llm", "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    assert b["budget"]["top_n_used"] == 5
    assert b["budget"]["dropped_sections"] == list(settings.OPTIONAL_SECTIONS)
    assert b["category_comparison"] == [] and b["kernel_names"] == {}
    md = (out / "bundle.md").read_text()
    assert "CUDA kernel launch stats" not in md and "Appendix" not in md
    assert "still larger than" in capsys.readouterr().err


def test_within_budget_bundle_is_unchanged(two_runs, tmp_path):
    out = tmp_path / "out"
    assert run(two_runs, "--no-llm", "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    assert b["budget"]["dropped_sections"] == [] and b["budget"]["top_n_used"] == 15
    assert b["budget"]["estimated_tokens"] < b["budget"]["max_prompt_tokens"]
    assert "To fit the prompt budget" not in (out / "bundle.md").read_text()


def test_prompt_lists_meta_keys_and_has_no_placeholders(two_runs, tmp_path):
    out = tmp_path / "out"
    assert run(two_runs, "--no-llm", "--output-dir", out) == 0
    prompt = (out / "prompt.md").read_text()
    assert "{{" not in prompt
    assert "`batch_size_per_gpu`" in prompt and "`deepspeed.zero_stage`" in prompt
    assert "allow_tf32" in prompt  # remediation note on cuDNN's default TF32 convolutions


# --- S3: one run on several GPUs (M3) ------------------------------------------------------

@pytest.fixture
def ddp_run(tmp_path):
    """4 ranks: rank 2 does more compute (a straggler), rank 3 waits on its host."""
    return write_run(tmp_path / "ddp-4gpu", 4, steps=6, gap_us=50,
                     kernels=[("volta_sgemm_128x64_nn", 400), ("elementwise", 100)],
                     comm_kernels=[(ALLREDUCE, 300)],
                     per_rank={2: {"kernels": [("volta_sgemm_128x64_nn", 600),
                                               ("elementwise", 100)]},
                               3: {"gap_us": 400}})


def test_s3_no_llm(ddp_run, tmp_path):
    out = tmp_path / "out"
    assert run(ddp_run, "--no-llm", "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    assert b["scenario"] == "S3" and b["schema"] == "torch-lab-hta-bundle/4"
    r = b["runs"][0]
    assert r["ranks_analyzed"] == 4 and r["slowest_rank"] == 2
    assert [row["rank"] for row in r["ranks"]] == [0, 1, 2, 3]
    busy = next(row for row in r["rank_aggregate"] if row["metric"] == "busy_ms_per_step")
    assert busy["max_rank"] == 2 and busy["max_over_median"] == 1.4
    ids = {f["id"] for f in r["flags"]}
    assert {"COMM_HEAVY", "POOR_OVERLAP", "RANK_STRAGGLER", "RANK_IDLE_IMBALANCE"} <= ids
    assert all(f["evidence"].startswith("rank 2: ") for f in r["flags"]
               if f["id"] in {"TOP_KERNEL_DOMINANT", "NO_AMP_SUSPECTED"})
    md = (out / "bundle.md").read_text()
    for heading in ["## 4. Rank comparison", "### Every rank",
                    "## 5. Communication (slowest rank: rank 2)",
                    "## 6. Temporal breakdown (slowest rank: rank 2)"]:
        assert heading in md
    report = (out / "report.md").read_text()
    assert "slowest rank (most busy GPU time per step) is rank 2" in report
    assert "### RANK_STRAGGLER" in report and "DistributedSampler" in report  # known fixes
    prompt = (out / "prompt.md").read_text()
    assert "single training run on 4 GPUs" in prompt
    assert "{{" not in prompt and "<!--" not in prompt
    assert "Runtime differences" not in prompt and "one GPU" not in prompt
    assert "## RANK_STRAGGLER" in prompt and "## HOST_BOUND" in prompt


def test_single_gpu_prompt_has_no_multi_gpu_remediation(two_runs, tmp_path):
    out = tmp_path / "out"
    assert run(two_runs, "--no-llm", "--output-dir", out) == 0
    prompt = (out / "prompt.md").read_text()
    assert "## HOST_BOUND" in prompt and "## NO_AMP_SUSPECTED" in prompt
    assert not any(f"## {f}" in prompt for f in
                   ["COMM_HEAVY", "POOR_OVERLAP", "RANK_STRAGGLER", "RANK_IDLE_IMBALANCE"])


def test_s3_more_than_8_ranks_has_no_every_rank_table(tmp_path):
    out = tmp_path / "out"
    assert run(write_run(tmp_path / "big", 9, steps=3, comm_kernels=[(ALLREDUCE, 100)]),
               "--no-llm", "--output-dir", out) == 0
    r = json.loads((out / "bundle.json").read_text())["runs"][0]
    assert r["ranks_analyzed"] == 9 and r["ranks"] == [] and r["rank_aggregate"]
    assert "### Every rank" not in (out / "bundle.md").read_text()


def test_comparing_multi_gpu_runs_exits_1(ddp_run, tmp_path, capsys):
    write_trace(tmp_path / "single" / "t.json")
    assert run(ddp_run, tmp_path / "single", "--no-llm", "--output-dir", tmp_path / "out") == 1
    assert "Comparing multi-GPU runs is not yet supported" in capsys.readouterr().err


def test_budget_plan_drops_all_ranks_tables_first():
    from torch_trace_lab.hta_llm import pipeline
    plan = pipeline.budget_plan(15, multi_rank=True)
    assert plan[:4] == [(15, ()), (10, ()), (5, ()), (5, ("all-ranks tables",))]
    assert plan[-1] == (5, ("all-ranks tables", "launch stats", "category comparison",
                            "kernel-name appendix"))


def test_s3_budget_never_drops_rank_comparison_or_slowest_rank(ddp_run, tmp_path,
                                                               monkeypatch, capsys):
    from torch_trace_lab.hta_llm import settings
    monkeypatch.setattr(settings, "MAX_PROMPT_TOKENS", 10)
    out = tmp_path / "out"
    assert run(ddp_run, "--no-llm", "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    assert b["budget"]["dropped_sections"] == [*settings.RANK_SECTIONS,
                                               *settings.OPTIONAL_SECTIONS]
    r = b["runs"][0]
    assert r["ranks"] == []                        # dropped first
    assert r["rank_aggregate"] and r["kernels"]    # never dropped
    md = (out / "bundle.md").read_text()
    assert "Rank comparison" in md and "GPU kernel breakdown (slowest rank: rank 2)" in md
    assert "### Every rank" not in md
    assert "still larger than" in capsys.readouterr().err


def test_list_flags_includes_rank_flags(capsys):
    assert run("--list-flags") == 0
    out = capsys.readouterr().out
    for flag_id in ["COMM_HEAVY", "POOR_OVERLAP", "RANK_STRAGGLER", "RANK_IDLE_IMBALANCE"]:
        assert flag_id in out


# --- M4: DDP ---------------------------------------------------------------------------------

def _gpus(n):
    return [{**DEVICE_PROPERTIES[0], "id": i} for i in range(n)]


@pytest.fixture
def ddp_markers_run(tmp_path):
    """2 nodes x 2 GPUs of DDP: reducer ops, all-reduce overlapping compute, a broadcast."""
    return write_run(tmp_path / "ddp-2x2", 4, steps=4, gap_us=50,
                     kernels=[("volta_sgemm_128x64_nn", 400), ("elementwise", 100)],
                     comm_kernels=[(ALLREDUCE, 300), (BROADCAST, 20)], comm_overlap=True,
                     extra_events=DDP_EVENTS, device_properties=_gpus(2),
                     per_rank={r: {"device": r % 2} for r in range(4)})


def test_ddp_run_bundle_and_prompt(ddp_markers_run, tmp_path):
    out = tmp_path / "out"
    assert run(ddp_markers_run, "--no-llm", "--output-dir", out) == 0
    r = json.loads((out / "bundle.json").read_text())["runs"][0]
    assert r["config"]["parallelism"]["value"] == "DDP"
    assert r["config"]["parallelism"]["confidence"] == "high"
    assert r["config"]["num_nodes"]["value"] == 2 and r["config"]["gpus_per_node"]["value"] == 2
    assert [c["collective"] for c in r["collectives"]] == ["all-reduce", "broadcast"]
    assert all(f"rank {i}" in r["collectives"][0] for i in range(4))
    m = r["metrics"]
    assert set(m["comm_by_collective_ms_per_step"]) == {"all-reduce", "broadcast"}
    assert m["allreduce_ms_per_step"] == 0.3
    assert 0 <= m["exposed_allreduce_ms_per_step"] < m["allreduce_ms_per_step"]
    assert any(row["metric"] == "exposed_allreduce_ms_per_step" for row in r["rank_aggregate"])
    md = (out / "bundle.md").read_text()
    assert "### Communication by collective" in md and "DDP (inferred, high)" in md
    assert "samples/s per GPU" in md and "samples/s, all GPUs" in md
    prompt = (out / "prompt.md").read_text()
    assert "parallelism = DDP" in prompt and "broadcast_buffers" in prompt
    assert "Do not assume one" not in prompt and "{{" not in prompt


def test_multi_rank_without_ddp_markers_assumes_no_framework(ddp_run, tmp_path):
    out = tmp_path / "out"
    assert run(ddp_run, "--no-llm", "--output-dir", out) == 0
    r = json.loads((out / "bundle.json").read_text())["runs"][0]
    assert "parallelism" not in r["config"]
    assert "num_nodes" not in r["config"]  # every rank on device 0: can't tell
    prompt = (out / "prompt.md").read_text()
    assert "Do not assume one" in prompt and "parallelism = DDP" not in prompt


def test_meta_parallelism_without_guidance(ddp_run, tmp_path):
    meta = tmp_path / "meta.toml"
    meta.write_text('parallelism = "FSDP"\n')
    out = tmp_path / "out"
    assert run(ddp_run, "--meta", meta, "--no-llm", "--output-dir", out) == 0
    r = json.loads((out / "bundle.json").read_text())["runs"][0]
    assert r["config"]["parallelism"] == {"value": "FSDP", "source": "user"}
    prompt = (out / "prompt.md").read_text()
    assert "uses FSDP. There is no FSDP-specific guidance" in prompt
    assert "parallelism = DDP" not in prompt


def test_meta_parallelism_conflict_is_noted(ddp_markers_run, tmp_path):
    meta = tmp_path / "meta.toml"
    meta.write_text('parallelism = "none"\nnum_nodes = 1\n')
    out = tmp_path / "out"
    assert run(ddp_markers_run, "--meta", meta, "--no-llm", "--output-dir", out) == 0
    b = json.loads((out / "bundle.json").read_text())
    assert b["runs"][0]["config"]["parallelism"]["value"] == "none"
    notes = " ".join(b["data_quality"])
    assert "parallelism: using none from --meta" in notes and "DDP" in notes
    assert "num_nodes: using 1 from --meta" in notes
    assert "parallelism = DDP" not in (out / "prompt.md").read_text()


def test_throughput_is_per_gpu_and_total(tmp_path):
    out = tmp_path / "out"
    run_dir = write_run(tmp_path / "ddp", 2, steps=4, comm_kernels=[(ALLREDUCE, 100)],
                        torch_lab={"batch_size_per_gpu": 64})
    assert run(run_dir, "--no-llm", "--output-dir", out) == 0
    row = json.loads((out / "bundle.json").read_text())["derived"][0]
    assert row["samples_per_s_per_gpu"] and row["samples_per_s_total"] == round(
        2 * row["samples_per_s_per_gpu"], 1)


def test_single_gpu_throughput_has_no_total(two_runs, tmp_path):
    out = tmp_path / "out"
    assert run(two_runs, "--no-llm", "--output-dir", out) == 0
    rows = json.loads((out / "bundle.json").read_text())["derived"]
    assert all(r["samples_per_s_per_gpu"] for r in rows)
    assert not any("samples_per_s_total" in r for r in rows)
