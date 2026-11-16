"""Turn HTA results into per-step, rounded, compact metrics (spec §7). Pure functions."""

import re
from dataclasses import dataclass, field
from statistics import median

from torch_trace_lab.hta_llm import settings
from torch_trace_lab.hta_llm.analyses import RunAnalysis

MEMCPY = re.compile(r"^Memcpy (HtoD|DtoH)")
GEMM_CONV = re.compile(r"gemm|conv|fprop|dgrad|wgrad|cudnn|cutlass|xmma|winograd|implicit"
                       r"|s1688|s16816|h1688|h16816|884", re.I)


def kernel_precision(name: str) -> str | None:
    """Best guess at a GEMM/conv kernel's math precision from its name; None if unknown."""
    n = name.lower()
    if "bf16" in n:
        return "bf16"
    if re.search(r"fp16|f16|half|h1688|h16816|h884|884gemm", n):
        return "fp16"
    if re.search(r"tf32|s1688|s16816", n):
        return "tf32"
    if re.search(r"sgemm|f32f32|fp32|scudnn|convolve_sgemm|float", n):
        return "fp32"
    return None


def is_gemm_or_conv(name: str) -> bool:
    return bool(GEMM_CONV.search(name))


def shorten(name: str, limit: int = 60) -> str:
    """A readable short kernel name: no template arguments, parameters, or namespaces."""
    s = name.strip()
    if s.startswith("void "):
        s = s[5:]
    # Drop template arguments, innermost first.
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"<[^<>]*>", "", s)
    s = re.sub(r"\(.*\)$", "", s)  # parameter list
    s = re.sub(r"\b(at::native::|at::|c10::|cudnn::|cutlass::|std::)", "", s)
    s = re.sub(r"\(anonymous namespace\)::", "", s)
    s = s.split("/")[-1]  # path-like names keep only the basename
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return s or name[:limit]


def add_short_name(names: dict[str, str], full: str) -> str:
    """Add ``full`` to the full -> short mapping, keeping short names unique."""
    if full in names:
        return names[full]
    short = shorten(full)
    taken = set(names.values())
    if short in taken:
        i = 2
        while f"{short} #{i}" in taken:
            i += 1
        short = f"{short} #{i}"
    names[full] = short
    return short


def short_names(fulls: list[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for full in fulls:
        add_short_name(names, full)
    return names


def r2(x):
    return None if x is None else round(float(x), 2)


def r1(x):
    return None if x is None else round(float(x), 1)


@dataclass
class KernelRow:
    short: str
    kernel_type: str
    ms_per_step: float
    pct_gpu: float
    calls_per_step: float
    mean_us: float
    precision: str | None


@dataclass
class RunMetrics:
    label: str
    steps: int | None
    step_time_ms: float | None
    gpu_idle_pct: float | None = None
    gpu_compute_pct: float | None = None
    gpu_non_compute_pct: float | None = None
    idle_ms_per_step: float | None = None
    host_wait_pct_of_idle: float | None = None
    kernel_wait_pct_of_idle: float | None = None
    other_idle_pct_of_idle: float | None = None
    kernel_ms_per_step: float | None = None
    kernels: list[KernelRow] = field(default_factory=list)   # top-N + "other"
    top_kernel_pct: float | None = None
    top_kernel: str | None = None
    memcpy_pct: float | None = None
    gemm_conv_precision_pct: dict[str, float] = field(default_factory=dict)
    kernels_per_step: float | None = None
    median_kernel_us: float | None = None
    median_launch_cpu_us: float | None = None
    median_launch_delay_us: float | None = None
    short_kernel_pct: float | None = None
    samples_per_s: float | None = None
    unavailable: dict[str, str] = field(default_factory=dict)


def normalize_run(label: str, a: RunAnalysis, *, top_n: int = settings.TOP_N,
                  names: dict[str, str] | None = None,
                  batch_size: int | None = None) -> RunMetrics:
    """``names`` is the shared full->short mapping; missing names are added to it."""
    names = {} if names is None else names
    steps = a.steps
    div = steps or 1  # without ProfilerStep annotations, report totals
    m = RunMetrics(label, steps, r2(a.step_time_us / 1000) if a.step_time_us else None,
                   unavailable=dict(a.unavailable))

    if a.temporal is not None and not a.temporal.empty:
        t = a.temporal.iloc[0]
        m.gpu_idle_pct = r1(t["idle_pct"])
        m.gpu_compute_pct = r1(t["compute_pct"])
        m.gpu_non_compute_pct = r1(t["non_compute_pct"])
        m.idle_ms_per_step = r2(t["idle_us"] / div / 1000)

    if a.idle is not None and not a.idle.empty:
        ratios = dict(zip(a.idle["category"], a.idle["ratio"]))
        m.host_wait_pct_of_idle = r1(100 * ratios.get("host_wait", 0.0))
        m.kernel_wait_pct_of_idle = r1(100 * ratios.get("kernel_wait", 0.0))
        m.other_idle_pct_of_idle = r1(100 * ratios.get("other", 0.0))

    if a.kernels is not None and not a.kernels.empty:
        k = a.kernels.sort_values("sum_us", ascending=False).reset_index(drop=True)
        total = float(k["sum_us"].sum())
        m.kernel_ms_per_step = r2(total / div / 1000)
        for full in k["name"]:
            add_short_name(names, full)
        rows = []
        for _, row in k.head(top_n).iterrows():
            rows.append(KernelRow(
                short=names[row["name"]], kernel_type=row["kernel_type"],
                ms_per_step=r2(row["sum_us"] / div / 1000),
                pct_gpu=r1(100 * row["sum_us"] / total) if total else 0.0,
                calls_per_step=r1(row["count"] / div), mean_us=r2(row["mean_us"]),
                precision=kernel_precision(row["name"]) if is_gemm_or_conv(row["name"]) else None,
            ))
        rest = k.iloc[top_n:]
        if not rest.empty:
            rows.append(KernelRow("other", "-", r2(rest["sum_us"].sum() / div / 1000),
                                  r1(100 * rest["sum_us"].sum() / total) if total else 0.0,
                                  r1(rest["count"].sum() / div), r2(rest["mean_us"].mean()), None))
        m.kernels = rows
        if total:
            m.top_kernel = names[k.iloc[0]["name"]]
            m.top_kernel_pct = r1(100 * k.iloc[0]["sum_us"] / total)
            memcpy = k[k["name"].str.match(MEMCPY)]["sum_us"].sum()
            m.memcpy_pct = r1(100 * memcpy / total)
        gc = k[k["name"].map(is_gemm_or_conv)]
        gc_total = float(gc["sum_us"].sum())
        if gc_total:
            by_prec: dict[str, float] = {}
            for _, row in gc.iterrows():
                p = kernel_precision(row["name"]) or "unknown"
                by_prec[p] = by_prec.get(p, 0.0) + float(row["sum_us"])
            m.gemm_conv_precision_pct = {p: r1(100 * v / gc_total)
                                         for p, v in sorted(by_prec.items())}

    if a.launch is not None and not a.launch.empty:
        l = a.launch
        m.kernels_per_step = r1(len(l) / div)
        m.median_kernel_us = r2(median(l["gpu_duration_us"]))
        m.median_launch_cpu_us = r2(median(l["cpu_duration_us"]))
        m.median_launch_delay_us = r2(median(l["launch_delay_us"]))
        short = (l["gpu_duration_us"] < settings.LAUNCH_BOUND_SHORT_KERNEL_US).mean()
        m.short_kernel_pct = r1(100 * short)

    if batch_size and m.step_time_ms:
        m.samples_per_s = r1(batch_size / (m.step_time_ms / 1000))
    return m


def kernel_comparison(runs: list[RunMetrics], baseline: int = 0) -> list[dict]:
    """Join per-run kernel tables on short name, with deltas vs. the baseline run (ms/step)."""
    order: list[str] = []
    for r in runs:
        for k in r.kernels:
            if k.short != "other" and k.short not in order:
                order.append(k.short)
    order.append("other")
    base = runs[baseline]
    rows = []
    for short in order:
        values = {r.label: next((k.ms_per_step for k in r.kernels if k.short == short), None)
                  for r in runs}
        if all(v is None for v in values.values()):
            continue
        b = values[base.label] or 0.0
        row = {"kernel": short, **values}
        for r in runs:
            if r is base:
                continue
            v = values[r.label] or 0.0
            row[f"Δ {r.label}"] = r2(v - b)
            row[f"Δ% {r.label}"] = r1(100 * (v - b) / b) if b else None
        rows.append(row)
    return rows


def resolve_baseline(labels: list[str], baseline: str | None) -> int:
    if baseline is None:
        return 0
    if baseline not in labels:
        raise ValueError(f"--baseline {baseline!r} is not one of the runs: {', '.join(labels)}")
    return labels.index(baseline)
