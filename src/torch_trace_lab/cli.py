"""``generate-traces``: produce the example profiler traces used in the demo.

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

TEMPLATE = "templates/generate_traces.sh"
PROFILE_ARGS = ["--skip_first", "10", "--warmup", "10", "--active", "10"]


def on_alliance_cluster() -> bool:
    """Every Alliance (Compute Canada) cluster sets CC_CLUSTER."""
    return bool(os.environ.get("CC_CLUSTER"))


def in_slurm_job() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID"))


def threaded_workers() -> int:
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus:
        return int(cpus)
    return min(os.cpu_count() or 1, 4)


def trace_runs(batch_size: int, output_dir: Path) -> list[tuple[str, list[str]]]:
    n = str(threaded_workers())
    base = [str(batch_size), "--output_dir", str(output_dir), *PROFILE_ARGS]
    return [
        ("Serial", [*base, "--nWorkers", "1"]),
        ("Threaded", [*base, "--nWorkers", n]),
        ("Threaded, compiled", [*base, "--nWorkers", n, "--compile"]),
        ("Threaded, compiled, AMP", [*base, "--nWorkers", n, "--compile", "--amp"]),
    ]


def run_traces(args) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, run_args in trace_runs(args.batch_size, output_dir):
        print("=" * 53)
        print(f"{name} (batch size {args.batch_size}):")
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
        print("Not submitting. Run `generate-traces` inside an interactive job (salloc) instead.")
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
    script_path = output_dir / "generate_traces_job.sh"
    script_path.write_text(script)

    proc = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"sbatch failed:\n{proc.stderr.strip()}", file=sys.stderr)
        return proc.returncode
    print(proc.stdout.strip())
    print(f"Job script: {script_path}")
    print(f"Log and traces will be written to {output_dir}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="generate-traces",
        description="Generate the example PyTorch profiler traces used in the demo.",
    )
    parser.add_argument("--output-dir", default="traces",
                        help="Directory for traces (default: ./traces).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training batch size (default: 128).")
    job = parser.add_argument_group("batch job options (Alliance login nodes only)")
    job.add_argument("--account", help="Slurm account, e.g. def-someuser.")
    job.add_argument("--time", default="0-00:30:00", help="Job time limit (default: 0-00:30:00).")
    job.add_argument("--gpus", default="h100_1g.10gb:1",
                     help="GPU request (default: h100_1g.10gb:1).")
    job.add_argument("-y", "--yes", action="store_true", help="Submit without asking.")
    job.add_argument("--dry-run", action="store_true",
                     help="Print the job script instead of submitting it.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not on_alliance_cluster() or in_slurm_job():
        return run_traces(args)
    try:
        return submit_job(args)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
