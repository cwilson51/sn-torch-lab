import os

import pytest

PREFIXES = ("SLURM_", "SBATCH_", "BRINE_", "TORCH_LAB_")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests never see the cluster, Slurm job, or Brine settings of the machine running them."""
    monkeypatch.delenv("CC_CLUSTER", raising=False)
    for var in list(os.environ):
        if var.startswith(PREFIXES):
            monkeypatch.delenv(var)
