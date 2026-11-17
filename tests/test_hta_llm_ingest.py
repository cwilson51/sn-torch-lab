import json

import pytest

from kineto import write_run, write_trace
from torch_trace_lab.hta_llm import ingest
from torch_trace_lab.hta_llm.ingest import IngestError

ENV = {"USER": "jdoe", "HOME": "/home/jdoe"}


def run_ingest(tmp_path, *paths):
    stage = tmp_path / "_stage"
    stage.mkdir()
    return ingest.ingest(list(paths), stage, environ=ENV)


def test_single_file_is_run_1(tmp_path):
    f = write_trace(tmp_path / "anything.json")
    runs, warnings = run_ingest(tmp_path, f)
    assert [r.label for r in runs] == ["run-1"]
    assert warnings == []
    assert (runs[0].staged_dir / "anything.json").exists()


def test_parent_directory_is_a_comparison(tmp_path):
    write_trace(tmp_path / "traces" / "b-run" / "x.json")
    write_trace(tmp_path / "traces" / "a-run" / "y.json.gz")
    (tmp_path / "traces" / "a-run" / "slurm-1.out").write_text("log")
    runs, _ = run_ingest(tmp_path, tmp_path / "traces")
    assert [r.label for r in runs] == ["a-run", "b-run"]


def test_explicit_run_directories_keep_order(tmp_path):
    b = write_trace(tmp_path / "zz" / "t.json").parent
    a = write_trace(tmp_path / "aa" / "t.json").parent
    runs, _ = run_ingest(tmp_path, b, a)
    assert [r.label for r in runs] == ["zz", "aa"]


def test_non_trace_files_are_ignored(tmp_path):
    run = tmp_path / "run"
    write_trace(run / "trace.json")
    (run / "config.json").write_text(json.dumps({"lr": 0.1}))
    (run / "broken.json").write_text("{not json")
    runs, _ = run_ingest(tmp_path, run)
    assert [t.path.name for t in runs[0].traces] == ["trace.json"]
    assert sorted(p.name for p in runs[0].staged_dir.iterdir()) == ["trace.json"]


def test_parent_with_trace_files_is_an_error(tmp_path):
    write_trace(tmp_path / "p" / "stray.json")
    write_trace(tmp_path / "p" / "run" / "t.json")
    with pytest.raises(IngestError, match="both trace files and run subdirectories"):
        run_ingest(tmp_path, tmp_path / "p")


def test_several_runs_in_one_directory_without_rank_info(tmp_path):
    write_trace(tmp_path / "r" / "a.json")
    write_trace(tmp_path / "r" / "b.json")
    with pytest.raises(IngestError, match="several runs in one directory"):
        run_ingest(tmp_path, tmp_path / "r")


def test_duplicate_ranks(tmp_path):
    write_trace(tmp_path / "r" / "a.json", rank=0, world_size=2)
    write_trace(tmp_path / "r" / "b.json", rank=0, world_size=2)
    with pytest.raises(IngestError, match="Duplicate ranks"):
        run_ingest(tmp_path, tmp_path / "r")


def test_missing_ranks_are_listed(tmp_path):
    write_trace(tmp_path / "r" / "a.json", rank=0, world_size=4)
    write_trace(tmp_path / "r" / "b.json", rank=2, world_size=4)
    with pytest.raises(IngestError, match=r"ranks \[1, 3\]"):
        run_ingest(tmp_path, tmp_path / "r")


def test_valid_multi_rank_run_is_sorted_by_rank(tmp_path):
    # Misleading file names: ranks come from distributedInfo only.
    write_trace(tmp_path / "r" / "rank3.json", rank=0, world_size=2)
    write_trace(tmp_path / "r" / "rank0.json", rank=1, world_size=2)
    runs, warnings = run_ingest(tmp_path, tmp_path / "r")
    assert warnings == []
    assert runs[0].ranks == 2
    assert [t.path.name for t in runs[0].traces] == ["rank3.json", "rank0.json"]
    assert sorted(p.name for p in runs[0].staged_dir.iterdir()) == ["rank0.json", "rank3.json"]


def test_ranks_beyond_world_size_are_an_error(tmp_path):
    for rank in range(3):
        write_trace(tmp_path / "r" / f"{rank}.json", rank=rank, world_size=2)
    with pytest.raises(IngestError, match="don't match world_size 2"):
        run_ingest(tmp_path, tmp_path / "r")


@pytest.mark.parametrize("multi", [["a"], ["a", "b"]])
def test_comparing_multi_gpu_runs_is_accepted(tmp_path, multi):
    """S4 (M5): multi-GPU runs can be compared, also with single-GPU runs."""
    for label in ["a", "b"]:
        if label in multi:
            write_run(tmp_path / "exp" / label, 2)
        else:
            write_trace(tmp_path / "exp" / label / "t.json")
    runs, _ = run_ingest(tmp_path, tmp_path / "exp")
    assert [(r.label, r.ranks) for r in runs] == [
        (label, 2 if label in multi else 1) for label in ["a", "b"]]
    assert all(r.staged_dir and len(list(r.staged_dir.iterdir())) == r.ranks for r in runs)


def test_stage_traces_links_only_the_given_files(tmp_path):
    run_dir = write_run(tmp_path / "r", 3)
    runs, _ = run_ingest(tmp_path, run_dir)
    target = ingest.stage_traces(runs[0].traces[2:3], tmp_path / "one")
    files = list(target.iterdir())
    assert len(files) == 1 and json.loads(files[0].read_text())["distributedInfo"]["rank"] == 2


def test_misleading_filename_single_rank(tmp_path):
    f = write_trace(tmp_path / "r" / "rank3.json", rank=0, world_size=1)
    runs, warnings = run_ingest(tmp_path, f.parent)
    assert len(runs[0].traces) == 1 and warnings == []


def test_incomplete_single_file_warns(tmp_path):
    f = write_trace(tmp_path / "r" / "t.json", rank=3, world_size=8)
    runs, warnings = run_ingest(tmp_path, f.parent)
    assert any("8-rank run" in w for w in warnings)


def test_cpu_only_trace_fails_with_hint(tmp_path):
    f = write_trace(tmp_path / "r" / "t.json", gpu=False)
    with pytest.raises(IngestError, match="contain only CPU events"):
        run_ingest(tmp_path, f)


def test_missing_profiler_steps_warns(tmp_path):
    f = write_trace(tmp_path / "r" / "t.json", profiler_steps=False)
    _, warnings = run_ingest(tmp_path, f.parent)
    assert any("ProfilerStep" in w for w in warnings)


def test_sensitive_labels_are_replaced(tmp_path):
    write_trace(tmp_path / "p" / "jdoe-baseline" / "t.json")
    write_trace(tmp_path / "p" / "12345678_amp" / "t.json")
    write_trace(tmp_path / "p" / "fine" / "t.json")
    runs, warnings = run_ingest(tmp_path, tmp_path / "p")
    # Sorted by directory name: 12345678_amp, fine, jdoe-baseline.
    assert [r.label for r in runs] == ["run-1", "fine", "run-3"]
    assert len(warnings) == 2


def test_duplicate_labels_get_suffixes(tmp_path):
    a = write_trace(tmp_path / "x" / "same" / "t.json").parent
    b = write_trace(tmp_path / "y" / "same" / "t.json").parent
    runs, _ = run_ingest(tmp_path, a, b)
    assert [r.label for r in runs] == ["same-1", "same-2"]


def test_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(IngestError, match="No trace files"):
        run_ingest(tmp_path, tmp_path / "empty")
