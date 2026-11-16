"""Scoring in scripts/eval_models.py (the manual Brine model eval). No network."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_models.py"
spec = importlib.util.spec_from_file_location("eval_models", SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

BUNDLE = {
    "runs": [
        {"label": "serial", "metrics": {"step_time_ms": 412.37, "gpu_idle_pct": 41.2},
         "flags": [{"id": "HOST_BOUND", "evidence": "host wait 83.5% of idle"},
                   {"id": "NO_AMP_SUSPECTED", "evidence": "fp32 share 99.7%"}]},
        {"label": "amp", "metrics": {"step_time_ms": 120.0, "gpu_idle_pct": 3.1},
         "flags": []},
    ],
    "step_time_differences": [{"from": "serial", "to": "amp", "delta_ms": -292.37,
                               "delta_pct": -70.9}],
}
PROMPT = ("# System prompt\n\nrules\n\n# User prompt\n\nUse `pin_memory=True` ... "
          "volta_sgemm_128x64_nn ... bundle")


def test_grounded_numbers_pass_and_rounding_is_allowed():
    report = "Step time is 412.37 ms (412.4 ms rounded), idle 41.2%, amp is -292.37 ms faster."
    bad, checked = ev.ungrounded_numbers(report, BUNDLE)
    assert (bad, checked) == ([], 4)


def test_made_up_numbers_are_caught():
    bad, _ = ev.ungrounded_numbers("Idle drops by 38.1% and saves 300 ms.", BUNDLE)
    assert bad == ["38.1%", "300 ms"]


def test_numbers_inside_bundle_strings_count():
    bad, _ = ev.ungrounded_numbers("host wait is 83.5% of idle", BUNDLE)
    assert bad == []


def test_invented_names():
    report = ("Set `pin_memory=True`. The `volta_sgemm_128x64_nn` kernel dominates, "
              "then `ampere_fake_gemm_kernel` and sm80_xmma_madeup.")
    assert ev.invented_names(report, PROMPT) == ["ampere_fake_gemm_kernel", "sm80_xmma_madeup"]


def test_flag_mismatches():
    report = "HOST_BOUND fired. LAUNCH_BOUND explains the rest."
    assert ev.flag_mismatches(report, BUNDLE) == (["LAUNCH_BOUND"], ["NO_AMP_SUSPECTED"])


def test_split_prompt_round_trips():
    from torch_trace_lab.hta_llm.pipeline import prompt_markdown
    assert ev.split_prompt(prompt_markdown("sys\ntext", "user # text")) == ("sys\ntext",
                                                                          "user # text\n")
    with pytest.raises(ValueError):
        ev.split_prompt("not a prompt")


def test_summary_ranks_fewest_issues_first():
    good = ev.score("good", "b", "Step time 412.37 ms; HOST_BOUND and NO_AMP_SUSPECTED.",
                    PROMPT, BUNDLE)
    bad = ev.score("bad", "b", "Step time 999 ms; `made_up`.", PROMPT, BUNDLE)
    good.latency_s = bad.latency_s = 1.0
    table = ev.summary_table([bad, good])
    assert table.index("| good |") < table.index("| bad |")
