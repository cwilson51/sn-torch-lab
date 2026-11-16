"""``torch-lab analyze-traces``: HTA analysis of profiler traces with an LLM report from Brine.

* Not on an Alliance cluster -> run here.
* Inside a Slurm job (salloc, Open OnDemand) -> run here.
* On an Alliance login node -> refuse; trace analysis is too heavy for login nodes.
"""

import argparse
import os
import sys
from pathlib import Path

from torch_trace_lab import cluster
from torch_trace_lab.hta_llm import settings

LAYOUT = """\
trace layout (file names don't matter):
  one run:       torch-lab analyze-traces my-run/          (or my-run/trace.json)
  compare runs:  torch-lab analyze-traces experiments/     (each subdirectory is a run)
                 torch-lab analyze-traces runA/ runB/ ...
  Each run directory holds the trace file(s) of one run, one file per rank.
  The output of `torch-lab generate-example-traces` already follows this layout.

Only aggregated tables are sent to SHARCNET Brine, never raw traces. Keep sensitive
information (paths, usernames, job IDs, account names) out of --run-note text and
run directory names; prompt.md shows exactly what is sent.
"""


def login_node_message() -> str:
    return (
        f"You are on a login node of the Alliance cluster '{os.environ.get('CC_CLUSTER')}'. "
        "Trace analysis needs several GB of memory and minutes of CPU time, which is too heavy "
        "for a login node.\n"
        "Start an interactive job and run the command there, for example:\n"
        f"  salloc --time=0:30:0 --cpus-per-task=4 --mem={settings.SUGGESTED_MEM} "
        "--account=<your-account>\n"
        "A GPU is not needed. You can also use an Open OnDemand desktop or JupyterLab session.")


def positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {n}")
    return n


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "analyze-traces",
        help="Analyze profiler traces with HTA and get an LLM report from SHARCNET Brine.",
        description=(
            "Run Holistic Trace Analysis on PyTorch Profiler traces, compute rule-based "
            "performance flags, and ask an LLM on SHARCNET Brine for a ranked report. One run "
            "answers 'how can I make this faster?'; several runs answer 'why do they differ?'. "
            "Single-GPU runs only for now."),
        epilog=LAYOUT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", type=Path, metavar="PATH",
                        help="A run directory, a trace file, a directory of run directories, "
                             "or two or more run directories.")
    parser.add_argument("--output-dir", type=Path, default=Path("hta-report"),
                        help="Where to write the report and bundle (default: ./hta-report).")
    parser.add_argument("--run-note", action="append", default=[], metavar="[LABEL=]TEXT",
                        help="Free-text context for a run, e.g. baseline=\"bs 64, 8 workers\". "
                             "Repeatable. Sent to Brine: no sensitive information.")
    parser.add_argument("--baseline", metavar="LABEL",
                        help="Baseline run for comparisons (default: the first run).")
    parser.add_argument("--top-n", type=positive_int, default=settings.TOP_N,
                        help=f"Kernels per table (default: {settings.TOP_N}).")
    parser.add_argument("--model", help="Brine model (default: $TORCH_LAB_BRINE_MODEL, "
                                        f"$BRINE_MODEL, or {settings.BRINE_DEFAULT_MODEL}).")
    parser.add_argument("--no-llm", action="store_true",
                        help="Write the bundle, prompt and a flags-only report; don't call Brine.")
    parser.add_argument("--list-flags", action="store_true",
                        help="Print the rule-based flags, what each means and when it fires, "
                             "then exit. No PATH needed.")
    parser.set_defaults(func=run)


def run(args) -> int:
    if args.list_flags:
        from torch_trace_lab.hta_llm import heuristics
        print(heuristics.describe_flags())
        return 0
    if not args.paths:
        print("torch-lab analyze-traces: error: the following arguments are required: PATH "
              "(or use --list-flags)", file=sys.stderr)
        return 2
    if cluster.on_login_node():
        print(login_node_message(), file=sys.stderr)
        return 1
    if cluster.in_slurm_job():
        os.environ.setdefault("OMP_NUM_THREADS", str(cluster.threaded_workers()))

    from torch_trace_lab.hta_llm import pipeline  # heavy imports only once placement is OK

    opts = pipeline.Options(paths=args.paths, output_dir=args.output_dir,
                            run_notes=args.run_note, baseline=args.baseline, top_n=args.top_n,
                            model=args.model, no_llm=args.no_llm)
    try:
        return pipeline.run(opts)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
