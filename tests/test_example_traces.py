import re
import sys
from types import SimpleNamespace

import pytest

from torch_trace_lab import cli, example_traces as et

CMD = "generate-example-traces"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ["CC_CLUSTER", "SLURM_JOB_ID", "SLURM_ACCOUNT", "SBATCH_ACCOUNT",
                "SLURM_CPUS_PER_TASK"]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def calls(monkeypatch):
    record = SimpleNamespace(run_traces=0, subprocess=[])
    monkeypatch.setattr(et, "run_traces", lambda args: setattr(
        record, "run_traces", record.run_traces + 1) or 0)

    def fake_run(cmd, **kwargs):
        record.subprocess.append(cmd)
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 42\n", stderr="")

    monkeypatch.setattr(et.subprocess, "run", fake_run)
    return record


def answers(monkeypatch, *replies):
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def test_local_runs_traces(calls):
    assert cli.main([CMD]) == 0
    assert calls.run_traces == 1
    assert calls.subprocess == []


def test_alliance_in_job_runs_traces(monkeypatch, calls):
    monkeypatch.setenv("CC_CLUSTER", "nibi")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert cli.main([CMD]) == 0
    assert calls.run_traces == 1


def test_login_node_submits(monkeypatch, calls, tmp_path):
    monkeypatch.setenv("CC_CLUSTER", "nibi")
    answers(monkeypatch, "y", "def-test")
    assert cli.main([CMD, "--output-dir", str(tmp_path)]) == 0
    assert calls.run_traces == 0
    assert calls.subprocess == [["sbatch", str(tmp_path / "generate_example_traces_job.sh")]]

    script = (tmp_path / "generate_example_traces_job.sh").read_text()
    assert "#SBATCH --account=def-test" in script
    assert f"--output-dir \"{tmp_path.resolve()}\"" in script
    assert f"python/{sys.version_info.major}.{sys.version_info.minor}" in script
    if et.in_venv():
        assert f'source "{sys.prefix}/bin/activate"' in script


def test_login_node_declined(monkeypatch, calls, tmp_path):
    monkeypatch.setenv("CC_CLUSTER", "nibi")
    answers(monkeypatch, "n")
    assert cli.main([CMD, "--output-dir", str(tmp_path)]) == 0
    assert calls.subprocess == []
    assert calls.run_traces == 0


def test_account_defaults_from_env(monkeypatch, calls, tmp_path):
    monkeypatch.setenv("CC_CLUSTER", "nibi")
    monkeypatch.setenv("SLURM_ACCOUNT", "def-env")
    answers(monkeypatch, "y", "")
    assert cli.main([CMD, "--output-dir", str(tmp_path)]) == 0
    assert "--account=def-env" in (tmp_path / "generate_example_traces_job.sh").read_text()


def test_sbatch_failure_returns_error(monkeypatch, calls, tmp_path, capsys):
    monkeypatch.setenv("CC_CLUSTER", "nibi")
    monkeypatch.setattr(et.subprocess, "run", lambda cmd, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="sbatch: error: Invalid account"))
    answers(monkeypatch, "y", "def-bad")
    assert cli.main([CMD, "--output-dir", str(tmp_path)]) == 1
    assert "Invalid account" in capsys.readouterr().err


@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
def test_cancel_at_prompt(monkeypatch, calls, tmp_path, capsys, exc):
    monkeypatch.setenv("CC_CLUSTER", "nibi")

    def interrupt(prompt=""):
        raise exc

    monkeypatch.setattr("builtins.input", interrupt)
    assert cli.main([CMD, "--output-dir", str(tmp_path)]) == 130
    assert "Cancelled." in capsys.readouterr().out
    assert calls.subprocess == []


def test_workers_from_slurm_cpus(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    runs = dict(et.trace_runs(64, tmp_path))
    assert runs["Serial"][-1] == "1"
    for name in ["Threaded", "Threaded, compiled", "Threaded, compiled, AMP"]:
        args = runs[name]
        assert args[args.index("--nWorkers") + 1] == "6"


def test_workers_default_without_slurm(monkeypatch, tmp_path):
    monkeypatch.setattr(et.os, "cpu_count", lambda: 16)
    args = dict(et.trace_runs(64, tmp_path))["Threaded"]
    assert args[args.index("--nWorkers") + 1] == "4"


def test_rendered_template_has_no_placeholders(tmp_path):
    args = cli.build_parser().parse_args([CMD, "--output-dir", str(tmp_path)])
    script = et.render_job_script("def-test", args)
    assert not re.search(r"@[A-Z_]+@", script)
    assert f"torch-lab {CMD} --output-dir" in script


def test_no_subcommand_is_an_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


@pytest.fixture
def train_runs(monkeypatch):
    """Fake subprocess.run: the batch-size probe prints `probe_output`, train runs succeed."""
    record = SimpleNamespace(train=[], probes=0, probe_output="256\n", probe_code=0)

    def fake_run(cmd, **kwargs):
        if cmd[-1] == "torch_trace_lab.batch_size":
            record.probes += 1
            return SimpleNamespace(returncode=record.probe_code, stdout=record.probe_output)
        record.train.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(et.subprocess, "run", fake_run)
    return record


def batch_sizes(train_cmds):
    return {cmd[cmd.index("torch_trace_lab.train") + 1] for cmd in train_cmds}


def test_auto_batch_size_probes_once(train_runs, tmp_path, capsys):
    assert cli.main([CMD, "--output-dir", str(tmp_path)]) == 0
    assert train_runs.probes == 1
    assert len(train_runs.train) == 4
    assert batch_sizes(train_runs.train) == {"256"}
    assert "Batch size: 256 (auto-detected)" in capsys.readouterr().out


def test_explicit_batch_size_skips_probe(train_runs, tmp_path):
    assert cli.main([CMD, "--output-dir", str(tmp_path), "--batch-size", "64"]) == 0
    assert train_runs.probes == 0
    assert batch_sizes(train_runs.train) == {"64"}


@pytest.mark.parametrize("code, output", [(1, ""), (0, "not a number\n")])
def test_failed_probe_returns_error(train_runs, tmp_path, capsys, code, output):
    train_runs.probe_code, train_runs.probe_output = code, output
    assert cli.main([CMD, "--output-dir", str(tmp_path)]) == 1
    assert train_runs.train == []
    assert "--batch-size N" in capsys.readouterr().err


def test_job_script_defers_auto_to_compute_node(monkeypatch, train_runs, tmp_path, capsys):
    monkeypatch.setenv("CC_CLUSTER", "nibi")
    assert cli.main([CMD, "--output-dir", str(tmp_path), "--account", "def-x", "--dry-run"]) == 0
    assert "--batch-size auto" in capsys.readouterr().out
    assert train_runs.probes == 0


@pytest.mark.parametrize("value", ["0", "-4", "foo"])
def test_invalid_batch_size(value, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([CMD, "--batch-size", value])
    assert exc.value.code == 2
