"""M4 acceptance on real DDP traces (spec §14 M4).

The traces are not committed: they live in the repository's gitignored ``traces/`` directory
(``traces/multi-gpu/<run>/``, one file per rank). Each test is skipped when its run is absent.
"""

import json
from pathlib import Path

import pytest

from torch_trace_lab import cli

TRACES = Path(__file__).resolve().parents[1] / "traces"

# run directory -> (num_nodes, gpus_per_node)
DDP_RUNS = {"1_nodes_4_gpus": (1, 4), "2_nodes_4_gpus": (2, 2)}


def _require(path: Path) -> Path:
    if not path.is_dir():
        pytest.skip(f"real traces not present: {path.relative_to(TRACES.parent)}")
    return path


def _analyze(path: Path, out: Path) -> dict:
    assert cli.main(["analyze-traces", str(path), "--no-llm", "--output-dir", str(out)]) == 0
    return json.loads((out / "bundle.json").read_text())


@pytest.mark.parametrize("run_name", sorted(DDP_RUNS))
def test_real_ddp_run(run_name, tmp_path):
    b = _analyze(_require(TRACES / "multi-gpu" / run_name), tmp_path)
    assert b["scenario"] == "S3"
    r = b["runs"][0]
    assert r["ranks_analyzed"] == 4
    assert r["unavailable"] == {}
    config = r["config"]
    assert (config["parallelism"]["value"], config["parallelism"]["confidence"]) == ("DDP", "high")
    nodes, gpus = DDP_RUNS[run_name]
    assert config["num_nodes"]["value"] == nodes
    assert config["gpus_per_node"]["value"] == gpus
    assert {c["collective"] for c in r["collectives"]} == {"all-reduce", "broadcast"}
    assert r["metrics"]["exposed_allreduce_ms_per_step"] is not None
    assert "parallelism = DDP" in (tmp_path / "prompt.md").read_text()


def test_real_single_gpu_run_is_not_ddp(tmp_path):
    single = _require(TRACES / "single-gpu")
    runs = sorted(p for p in single.iterdir() if p.is_dir())
    if not runs:
        pytest.skip("no single-GPU runs in traces/single-gpu")
    r = _analyze(runs[0], tmp_path)["runs"][0]
    assert "parallelism" not in r["config"]
    assert "num_nodes" not in r["config"]
    assert r["collectives"] == []


def test_real_ddp_runs_compared(tmp_path):
    """M5 (S4) on the two real DDP runs: same GPU count, different node layout."""
    multi = _require(TRACES / "multi-gpu")
    runs = [_require(multi / name) for name in sorted(DDP_RUNS)]
    out = tmp_path / "out"
    assert cli.main(["analyze-traces", *map(str, runs), "--diff-rank", "slowest", "--no-llm",
                     "--output-dir", str(out)]) == 0
    b = json.loads((out / "bundle.json").read_text())
    assert b["scenario"] == "S4" and b["comparison_kind"] == "same framework"
    assert [r["ranks_analyzed"] for r in b["runs"]] == [4, 4]
    assert "num_nodes" in {d["field"] for d in b["config_diff"]}
    assert not any("scaling_efficiency_pct" in r for r in b["derived"])  # same GPU count
    (diff,) = b["trace_diff"].values()
    assert "unavailable" not in diff and diff["rows"]
    assert diff["rank"] == b["runs"][1]["slowest_rank"]
    assert {c["collective"] for c in b["collectives_by_run"]} == {"all-reduce", "broadcast"}
    assert b["budget"]["estimated_tokens"] <= b["budget"]["max_prompt_tokens"]
