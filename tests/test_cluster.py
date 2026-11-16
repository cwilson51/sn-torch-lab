import pytest

from torch_trace_lab import cluster


@pytest.mark.parametrize("env, login", [
    ({}, False),
    ({"CC_CLUSTER": "nibi", "SLURM_JOB_ID": "1"}, False),
    ({"CC_CLUSTER": "nibi"}, True),
])
def test_on_login_node(monkeypatch, env, login):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cluster.on_login_node() is login


def test_threaded_workers(monkeypatch):
    monkeypatch.setattr(cluster.os, "cpu_count", lambda: 2)
    assert cluster.threaded_workers() == 2
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert cluster.threaded_workers() == 8
