"""``--meta FILE.toml``: optional structured run metadata from the user (spec §5.2, source 5).

Top-level keys apply to every run; ``[runs."<label>"]`` tables override them for one run.
Only keys in ``SCHEMA`` are kept. Unknown keys, wrong types and unknown run labels are
warned about and dropped; they never reach the bundle. Warnings name the key, never its value.

Everything kept here is sent to Brine. The free-text keys (``notes``, ``model``, ``task``,
``interconnect``) pass the same leak check as ``--run-note``.
"""

import tomllib
from pathlib import Path

TEXT_LIMIT = 120
NOTES_LIMIT = 500


def _text(limit):
    def check(value):
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:limit]
        return None
    return check


def _int(lo=0, hi=None):
    def check(value):
        if isinstance(value, int) and not isinstance(value, bool) and value >= lo \
                and (hi is None or value <= hi):
            return value
        return None
    return check


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _bool(value):
    return value if isinstance(value, bool) else None


def _choice(*options):
    def check(value):
        if isinstance(value, str):
            for option in options:
                if value.lower() == option.lower():
                    return option
        return None
    return check


SCHEMA = {
    "model": _text(TEXT_LIMIT),
    "task": _text(TEXT_LIMIT),
    "interconnect": _text(TEXT_LIMIT),
    "notes": _text(NOTES_LIMIT),
    "precision": _text(40),
    "batch_size_per_gpu": _int(1),
    "dataloader_workers": _int(0),
    "torch_compile": _bool,
    "num_nodes": _int(1),
    "gpus_per_node": _int(1),
    "cpu_cores_per_gpu": _int(1),
    "parallelism": _choice("DDP", "FSDP", "DeepSpeed", "none"),
}
NESTED = {
    "ddp": {
        "bucket_cap_mb": _number,
        "gradient_as_bucket_view": _bool,
        "find_unused_parameters": _bool,
        "static_graph": _bool,
        "broadcast_buffers": _bool,
    },
    "fsdp": {
        "sharding_strategy": _choice("FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "NO_SHARD"),
        "backward_prefetch": _choice("BACKWARD_PRE", "BACKWARD_POST", "none"),
        "forward_prefetch": _bool,
        "limit_all_gathers": _bool,
        "cpu_offload": _bool,
    },
    "deepspeed": {
        "zero_stage": _int(0, 3),
        "offload_optimizer": _choice("none", "cpu", "nvme"),
        "offload_param": _choice("none", "cpu", "nvme"),
        "overlap_comm": _bool,
        "contiguous_gradients": _bool,
        "reduce_bucket_size": _number,
        "allgather_bucket_size": _number,
        "gradient_accumulation_steps": _int(1),
    },
}


def _validate(table: dict, where: str, warnings: list[str]) -> dict:
    """Flatten one table to {key or "framework.key": value}, keeping schema keys only."""
    kept = {}
    for key, value in table.items():
        if key in SCHEMA:
            checked = SCHEMA[key](value)
            if checked is None:
                warnings.append(f"--meta: {where}{key} has an invalid value; ignored.")
            else:
                kept[key] = checked
        elif key in NESTED and isinstance(value, dict):
            for sub, subvalue in value.items():
                rule = NESTED[key].get(sub)
                checked = rule(subvalue) if rule else None
                if rule is None:
                    warnings.append(f"--meta: {where}{key}.{sub} is not a known key; ignored.")
                elif checked is None:
                    warnings.append(f"--meta: {where}{key}.{sub} has an invalid value; ignored.")
                else:
                    kept[f"{key}.{sub}"] = checked
        else:
            warnings.append(f"--meta: {where}{_safe_key(key)} is not a known key; ignored.")
    return kept


def _safe_key(key: str) -> str:
    """Unknown keys are user text: show them only if they look like plain identifiers."""
    return key if key.replace("_", "").replace("-", "").isalnum() and len(key) <= 40 \
        else "(a key)"


def load_meta(path: Path, labels: list[str]) -> tuple[dict[str, dict], list[str]]:
    """Read ``--meta``. Returns ({label: {key: value}}, warnings).

    Raises ValueError if the file can't be read or isn't valid TOML.
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except OSError as e:
        raise ValueError(f"--meta: can't read {Path(path).name}: {e.strerror}.")
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"--meta: {Path(path).name} is not valid TOML: {e}.")

    warnings: list[str] = []
    runs = data.pop("runs", {})
    if not isinstance(runs, dict):
        warnings.append("--meta: 'runs' must be a table of [runs.\"<label>\"] tables; ignored.")
        runs = {}
    shared = _validate(data, "", warnings)
    per_run = {label: dict(shared) for label in labels}
    for label, table in runs.items():
        if label not in per_run:
            warnings.append(f"--meta: [runs.\"{_safe_key(label)}\"] does not match any run "
                            f"label ({', '.join(labels)}); ignored.")
            continue
        if not isinstance(table, dict):
            warnings.append(f"--meta: runs.\"{label}\" must be a table; ignored.")
            continue
        per_run[label].update(_validate(table, f"runs.\"{label}\".", warnings))
    return per_run, warnings


def key_list() -> str:
    """The accepted keys, for the prompt: model, task, ..., ddp.bucket_cap_mb, ..."""
    keys = [*SCHEMA, *(f"{table}.{key}" for table, keys in NESTED.items() for key in keys)]
    return ", ".join(f"`{k}`" for k in keys)


def as_config(values: dict) -> tuple[dict, str | None]:
    """Split one run's --meta values into (config entries with source=user, notes)."""
    values = dict(values)
    notes = values.pop("notes", None)
    return {k: {"value": v} for k, v in values.items()}, notes
