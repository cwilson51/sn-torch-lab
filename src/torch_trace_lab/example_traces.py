"""``torch-lab generate-example-traces``: produce the example profiler traces used in the demo.

These are fixed ResNet50 examples, not a tool for profiling your own code.

* Not on an Alliance cluster -> run the profiling runs locally.
* On an Alliance cluster inside a Slurm job (interactive or batch) -> run them.
* On an Alliance login node -> offer to submit a batch job that runs them.
"""

import argparse
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

from torch_trace_lab.cluster import in_slurm_job, on_alliance_cluster, threaded_workers

TEMPLATE = "templates/generate_example_traces.sh"
PROFILE_ARGS = ["--skip_first", "10", "--warmup", "10", "--active", "10"]


# Each config writes to its own run directory, the layout `torch-lab analyze-traces` expects.
RUN_SUBDIRS = {
    "Main process": "main-process",
    "Serial": "serial",
    "Threaded": "threaded",
    "Threaded, compiled": "threaded-compiled",
    "Threaded, compiled, AMP": "threaded-compiled-amp",
}


def trace_runs(batch_size: int, output_dir: Path) -> list[tuple[str, list[str]]]:
    n = str(threaded_workers())
    configs = [
        ("Main process", ["--nWorkers", "0"]),  # PyTorch default: no worker processes
        ("Serial", ["--nWorkers", "1"]),
        ("Threaded", ["--nWorkers", n]),
        ("Threaded, compiled", ["--nWorkers", n, "--compile"]),
        ("Threaded, compiled, AMP", ["--nWorkers", n, "--compile", "--amp"]),
    ]
    return [
        (name, [str(batch_size), "--output_dir", str(output_dir / RUN_SUBDIRS[name]),
                *PROFILE_ARGS, *extra])
        for name, extra in configs
    ]


def batch_size_arg(value: str) -> str | int:
    """argparse type for --batch-size: ``auto`` or a positive integer."""
    if value == "auto":
        return value
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be 'auto' or a positive integer, not {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {n}")
    return n


def resolve_batch_size(value: str | int) -> int | None:
    """Return the batch size to use, probing the GPU when ``value`` is ``auto``.

    The probe runs in a subprocess so this process never creates a CUDA context, which
    would hold GPU memory while the training runs need it. Returns None if probing fails.
    """
    if value != "auto":
        return int(value)
    proc = subprocess.run(
        [sys.executable, "-m", "torch_trace_lab.batch_size"], stdout=subprocess.PIPE, text=True
    )
    lines = (proc.stdout or "").strip().splitlines()
    if proc.returncode != 0 or not lines or not lines[-1].strip().isdigit():
        return None
    return int(lines[-1])


def run_traces(args) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = resolve_batch_size(args.batch_size)
    if batch_size is None:
        print("Could not detect a batch size automatically; pass --batch-size N instead.",
              file=sys.stderr)
        return 1
    if args.batch_size == "auto":
        print(f"Batch size: {batch_size} (auto-detected)")

    results = []
    for name, run_args in trace_runs(batch_size, output_dir):
        print("=" * 53)
        print(f"{name} (batch size {batch_size}):")
        print("=" * 53, flush=True)
        proc = subprocess.run([sys.executable, "-m", "torch_trace_lab.train", *run_args])
        results.append((name, proc.returncode == 0))

    print("=" * 53)
    for name, ok in results:
        print(f"  {'OK    ' if ok else 'FAILED'} {name}")
    print(f"Traces written to {output_dir}")
    return 0 if all(ok for _, ok in results) else 1


def in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def render_job_script(account: str, args) -> str:
    output_dir = str(Path(args.output_dir).resolve())
    if in_venv():
        activate = f'source "{sys.prefix}/bin/activate"'
    else:
        activate = "# No virtual environment detected; using the loaded python module."
    replacements = {
        "@ACCOUNT@": account,
        "@TIME@": args.time,
        "@GPUS@": args.gpus,
        "@OUTPUT_DIR@": output_dir,
        "@PYVER@": f"{sys.version_info.major}.{sys.version_info.minor}",
        "@ACTIVATE@": activate,
        "@BATCH_SIZE@": str(args.batch_size),
    }
    script = (files("torch_trace_lab") / TEMPLATE).read_text()
    for key, value in replacements.items():
        script = script.replace(key, value)
    return script


def ask_yes_no(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


def ask_account(default: str | None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        account = input(f"Account to submit under (e.g. def-someuser){suffix}: ").strip()
        account = account or (default or "")
        if account:
            return account
        print("An account is required.")


def submit_job(args) -> int:
    print(
        f"You are on a login node of the Alliance cluster '{os.environ.get('CC_CLUSTER')}'.\n"
        "Generating traces needs a GPU, so it should run as a Slurm job."
    )
    if not args.yes and not args.dry_run and not ask_yes_no("Submit a batch job to generate traces?"):
        print("Not submitting. Run `torch-lab generate-example-traces` inside an interactive job (salloc) instead.")
        return 0

    account = args.account or ask_account(
        os.environ.get("SLURM_ACCOUNT") or os.environ.get("SBATCH_ACCOUNT")
    )
    script = render_job_script(account, args)

    if args.dry_run:
        print(script)
        return 0

    if not in_venv():
        print("Warning: not running from a virtual environment; the job will use the "
              "system python module, which must have torch_trace_lab installed.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "generate_example_traces_job.sh"
    script_path.write_text(script)

    proc = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"sbatch failed:\n{proc.stderr.strip()}", file=sys.stderr)
        return proc.returncode
    print(proc.stdout.strip())
    print(f"Job script: {script_path}")
    print(f"Log and traces will be written to {output_dir}")
    return 0


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "generate-example-traces",
        help="Generate the example ResNet50 traces used in the demo.",
        description=(
            "Generate the example PyTorch profiler traces used in the demo by profiling "
            "ResNet50 training in several configurations. Runs locally, inside a Slurm "
            "job, or offers to submit a job from an Alliance login node."
        ),
    )
    parser.add_argument("--output-dir", default="traces",
                        help="Directory for traces (default: ./traces).")
    parser.add_argument("--batch-size", type=batch_size_arg, default="auto",
                        help="Training batch size, or 'auto' to pick the largest power of 2 "
                             "that fits in GPU memory, up to 512 (32 on CPU). Default: auto.")
    job = parser.add_argument_group("batch job options (Alliance login nodes only)")
    job.add_argument("--account", help="Slurm account, e.g. def-someuser.")
    job.add_argument("--time", default="0-00:30:00", help="Job time limit (default: 0-00:30:00).")
    job.add_argument("--gpus", default="h100_1g.10gb:1",
                     help="GPU request (default: h100_1g.10gb:1).")
    job.add_argument("-y", "--yes", action="store_true", help="Submit without asking.")
    job.add_argument("--dry-run", action="store_true",
                     help="Print the job script instead of submitting it.")
    parser.set_defaults(func=run)


def run(args) -> int:
    if not on_alliance_cluster() or in_slurm_job():
        return run_traces(args)
    try:
        return submit_job(args)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 130
