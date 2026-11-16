from torch_trace_lab.hta_llm import sanitize

ENV = {"USER": "jdoe", "HOME": "/Users/jdoe", "SLURM_JOB_ID": "4471234",
       "SLURM_NTASKS": "1", "SLURM_CPU_BIND": "none"}


def leaks(text, **kw):
    return sanitize.find_leaks(text, environ=ENV, **kw)


def test_clean_text():
    assert leaks("volta_sgemm took 40.1% of GPU time across 1 rank; none flagged") == []


def test_finds_each_kind(monkeypatch):
    monkeypatch.setattr(sanitize, "_hostnames", lambda: {"gpu-node-17"})
    assert leaks("user jdoe ran it")
    assert leaks("saved under /Users/jdoe/x")
    assert leaks("at /scratch/x/trace.json")
    assert leaks("on gpu-node-17")
    assert leaks("job 4471234 finished")
    assert leaks("account def-someprof")
    assert leaks("key sk-abc123", key="sk-abc123")


def test_slurm_finding_names_the_variable():
    found = leaks("job 4471234 finished")
    assert found == ["the value of $SLURM_JOB_ID"]
    assert "4471234" not in found[0]


def test_cluster_name_is_allowed():
    env = {**ENV, "CC_CLUSTER": "nibi", "SLURM_CLUSTER_NAME": "nibi"}
    assert sanitize.find_leaks("- Cluster: nibi", environ=env) == []


def test_username_needs_word_boundary():
    assert leaks("jdoexyz") == []


def test_unsafe_label():
    assert sanitize.unsafe_label("jdoe-run", environ=ENV)
    assert sanitize.unsafe_label("20261116-141100", environ=ENV)
    assert not sanitize.unsafe_label("threaded-compiled-amp", environ=ENV)
