"""Where ``torch-lab`` is running: locally, inside a Slurm job, or on an Alliance login node."""

import os


def on_alliance_cluster() -> bool:
    """Every Alliance (Compute Canada) cluster sets CC_CLUSTER."""
    return bool(os.environ.get("CC_CLUSTER"))


def in_slurm_job() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID"))


def on_login_node() -> bool:
    return on_alliance_cluster() and not in_slurm_job()


def threaded_workers() -> int:
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus:
        return int(cpus)
    return min(os.cpu_count() or 1, 4)
