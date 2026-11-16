"""Leak check: find personal or job-identifying values in text before it is sent to Brine.

This is the last line of defence; the bundle is already built from an allowlist (spec §8a).
"""

import os
import re
import socket

CLUSTER_PATH = re.compile(r"/(home|project|scratch|nearline)/")
ACCOUNT = re.compile(r"\b(def|rrg|ctb)-[A-Za-z0-9_]+")
JOB_ID = re.compile(r"\d{6,}")

# SLURM_* values that are too short or too generic to identify a job or a person.
GENERIC_VALUES = {"none", "cpu", "gpu", "cores", "quiet", "block", "cyclic", "nodes",
                  "true", "false", "yes", "no", "default"}


def _slurm_values(environ) -> set[str]:
    values = set()
    for name, value in environ.items():
        if not name.startswith("SLURM_"):
            continue
        value = value.strip()
        if value.lower() in GENERIC_VALUES:
            continue
        if value.isdigit() and len(value) < 6:
            continue
        if len(value) < 4:
            continue
        values.add(value)
    return values


def _hostnames() -> set[str]:
    names = set()
    try:
        full = socket.gethostname()
    except OSError:
        return names
    for name in (full, full.split(".")[0]):
        if len(name) >= 3:
            names.add(name)
    return names


def find_leaks(text: str, *, key: str | None = None, environ=None) -> list[str]:
    """Return a description of each sensitive value found in ``text``; empty if clean."""
    environ = os.environ if environ is None else environ
    found = []

    def word(value: str) -> re.Pattern:
        return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])")

    if key and key in text:
        found.append("the Brine API key")
    user = environ.get("USER", "")
    if len(user) >= 2 and word(user).search(text):
        found.append(f"the username ${{USER}}")
    home = environ.get("HOME", "")
    if len(home) > 1 and home in text:
        found.append("the home directory $HOME")
    if m := CLUSTER_PATH.search(text):
        found.append(f"a cluster path ({m.group(0)}...)")
    for host in sorted(_hostnames()):
        if word(host).search(text):
            found.append("the local hostname")
            break
    for value in sorted(_slurm_values(environ)):
        if word(value).search(text):
            found.append("a SLURM_* environment value")
            break
    if m := ACCOUNT.search(text):
        found.append(f"an allocation account name ({m.group(1)}-...)")
    return found


def unsafe_label(label: str, environ=None) -> bool:
    """Run labels are directory names; reject ones that look like paths, users, hosts, or job IDs."""
    return bool(find_leaks(label, environ=environ) or JOB_ID.search(label)
                or "/" in label or "\\" in label)
