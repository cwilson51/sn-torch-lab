"""Turn HTA results into per-step, rounded, compact metrics (spec §7). Pure functions."""

import re
from dataclasses import dataclass, field
from statistics import median

from torch_trace_lab.hta_llm import settings
from torch_trace_lab.hta_llm.analyses import RunAnalysis

MEMCPY = re.compile(r"^Memcpy (HtoD|DtoH)")
# Matrix-multiply and convolution kernels (cuBLAS, cuDNN, CUTLASS, Triton mm templates).
# Not batch-norm or layout kernels that merely live in cuDNN, and not Triton pointwise or
# reduction kernels whose fused-op names mention "convolution".
GEMM_CONV = re.compile(r"gemm|xmma|fprop|dgrad|wgrad|winograd|implicit_convolve|cutlass"
                       r"|s1688|s16816|h1688|h16816|^triton_tem_", re.I)
NOT_GEMM_CONV = re.compile(r"^(void )?(cudnn::)?bn_|nchwToNhwc|nhwcToNchw|^triton_(poi|red|per)_",
                           re.I)


def kernel_precision(name: str) -> str | None:
    """Best guess at a GEMM/conv kernel's math precision from its name; None if unknown."""
    n = name.lower()
    if "bf16" in n:
        return "bf16"
    if re.search(r"fp16|f16f16|_f16_|half|h1688|h16816|h884|884gemm", n):
        return "fp16"
    if re.search(r"tf32|s1688|s16816", n):
        return "tf32"
    if re.search(r"sgemm|f32f32|fp32|scudnn|convolve_sgemm", n):
        return "fp32"
    return None


def is_gemm_or_conv(name: str) -> bool:
    return bool(GEMM_CONV.search(name)) and not NOT_GEMM_CONV.search(name)


# The top-N table's row for all kernels below the top N. Deliberately not "other", which
# would be confused with a kernel category.
REMAINING = "(remaining kernels)"

CATEGORIES = ("GEMM/conv", "fused (torch.compile)", "batch norm", "layout transpose",
              "elementwise/reduction", "optimizer", "memcpy/memset", "uncategorized")


def kernel_category(name: str) -> str:
    """A coarse kernel category that lines up across eager and compiled runs."""
    if re.match(r"Mem(cpy|set)", name):
        return "memcpy/memset"
    if is_gemm_or_conv(name):
        return "GEMM/conv"
    if name.startswith("triton_"):
        return "fused (torch.compile)"
    if re.search(r"nchwToNhwc|nhwcToNchw", name):
        return "layout transpose"
    if re.search(r"(^|::)bn_|batch_?norm", name, re.I):
        return "batch norm"
    if re.search(r"multi_tensor_apply|fused_adam|fused_sgd|_amp_", name, re.I):
        return "optimizer"
    if re.search(r"elementwise|reduce|reduction", name, re.I):
        return "elementwise/reduction"
    return "uncategorized"


# Kernel names that say nothing without their template arguments.
GENERIC_KERNELS = {"vectorized_elementwise_kernel", "unrolled_elementwise_kernel",
                   "elementwise_kernel", "reduce_kernel", "Kernel", "Kernel2",
                   "multi_tensor_apply_kernel", "index_elementwise_kernel"}
TEMPLATE_STOPWORDS = {"float", "double", "half", "char", "bool", "void", "true", "false", "int",
                      "long", "unsigned", "const", "std", "array", "native", "detail", "Array",
                      "TensorIteratorBase", "lambda", "operator", "Scalar", "TensorListMetadata",
                      "anonymous", "namespace", "BFloat16", "Half", "c10", "OffsetCalculator"}


def _template_tag(args: str) -> str | None:
    """The most telling identifier in a kernel's template arguments."""
    words = [w for w in re.findall(r"[A-Za-z_]\w{3,}", args) if w not in TEMPLATE_STOPWORDS]
    for pattern in (r"\w+_kernel_impl$|^launch_\w+|\w+_kernel_cuda$",
                    r"Functor_\w+|\w+_\w*gemm\w*", r"Functor"):
        for w in words:
            if re.search(pattern, w):
                return w.removesuffix("_kernel_impl").removesuffix("_kernel_cuda")
    return words[0] if words else None


def shorten(name: str, limit: int = 60) -> str:
    """A readable short kernel name: no template arguments, parameters, or namespaces.

    Generic kernels (``vectorized_elementwise_kernel``, CUTLASS ``Kernel2``, ...) keep their
    most telling template argument, e.g. ``vectorized_elementwise_kernel<CUDAFunctor_add>``.
    """
    s = name.strip()
    if s.startswith("void "):
        s = s[5:]
    s = re.sub(r"\(anonymous namespace\)::", "", s)
    first_template = s.find("<")
    template_args = ""
    if first_template != -1:
        depth, end = 0, len(s)
        for i in range(first_template, len(s)):
            depth += {"<": 1, ">": -1}.get(s[i], 0)
            if depth == 0:
                end = i
                break
        template_args = s[first_template + 1:end]
    # Drop template arguments, innermost first.
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"<[^<>]*>", "", s)
    s = re.sub(r"\(.*\)$", "", s)  # parameter list
    s = re.sub(r"\b(at::native::|at::|c10::|cudnn::|cutlass::|std::)", "", s)
    s = s.split("/")[-1]  # path-like names keep only the basename
    s = re.sub(r"\s+", " ", s).strip()
    if s in GENERIC_KERNELS and (tag := _template_tag(template_args)):
        s = f"{s}<{tag}>"
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
    host_wait_ms_per_step: float | None = None
    kernel_wait_pct_of_idle: float | None = None
    other_idle_pct_of_idle: float | None = None
    kernel_ms_per_step: float | None = None
    kernels: list[KernelRow] = field(default_factory=list)   # top-N + REMAINING
    # ms/step of every kernel in the run, by short name. Used to join runs; not rendered,
    # so a kernel outside one run's top N still shows its real time in comparisons.
    all_kernels: dict[str, float] = field(default_factory=dict)
    top_kernel_pct: float | None = None
    top_kernel: str | None = None
    memcpy_pct: float | None = None
    gemm_conv_precision_pct: dict[str, float] = field(default_factory=dict)
    categories: dict[str, float] = field(default_factory=dict)    # ms/step per kernel category
    category_pct: dict[str, float] = field(default_factory=dict)  # % of GPU kernel time
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
        host_us = a.idle.loc[a.idle["category"] == "host_wait", "idle_us"].sum()
        m.host_wait_ms_per_step = r2(host_us / div / 1000)
        m.kernel_wait_pct_of_idle = r1(100 * ratios.get("kernel_wait", 0.0))
        m.other_idle_pct_of_idle = r1(100 * ratios.get("other", 0.0))

    if a.kernels is not None and not a.kernels.empty:
        k = a.kernels.sort_values("sum_us", ascending=False).reset_index(drop=True)
        total = float(k["sum_us"].sum())
        m.kernel_ms_per_step = r2(total / div / 1000)
        for full in k["name"]:
            add_short_name(names, full)
        m.all_kernels = {names[n]: r2(s / div / 1000) for n, s in zip(k["name"], k["sum_us"])}
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
            rows.append(KernelRow(REMAINING, "-", r2(rest["sum_us"].sum() / div / 1000),
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
                                         for p, v in sorted(by_prec.items())
                                         if r1(100 * v / gc_total)}
        by_cat = k.groupby(k["name"].map(kernel_category))["sum_us"].sum()
        m.categories = {c: r2(by_cat[c] / div / 1000) for c in CATEGORIES if c in by_cat.index}
        m.category_pct = {c: r1(100 * by_cat[c] / total) for c in CATEGORIES
                          if c in by_cat.index and total}

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
    """Join kernels across runs on short name, with ms/step deltas vs. the baseline.

    Rows are the kernels in any run's top N that occur in at least two runs; kernels unique
    to one run (e.g. after torch.compile replaces eager kernels) are left to the category
    comparison. Values come from each run's full kernel list, so None means the kernel does
    not occur in that run at all, not merely that it fell outside the top N.
    """
    order: list[str] = []
    for r in runs:
        for k in r.kernels:
            if k.short != REMAINING and k.short not in order:
                order.append(k.short)
    base = runs[baseline]
    rows = []
    for short in order:
        values = {r.label: r.all_kernels.get(short) for r in runs}
        if sum(v is not None for v in values.values()) < 2:
            continue
        b = values[base.label] or 0.0
        row = {"kernel": short, **values}
        for r in runs:
            if r is not base:
                row[f"Δ {r.label}"] = r2((values[r.label] or 0.0) - b)
        rows.append(row)
    return rows


def step_time_differences(runs: list[RunMetrics]) -> list[dict]:
    """Step-time difference for every ordered pair of runs: ``to`` minus ``from``."""
    rows = []
    for a in runs:
        for b in runs:
            if a is b or not a.step_time_ms or not b.step_time_ms:
                continue
            delta = b.step_time_ms - a.step_time_ms
            rows.append({"from": a.label, "to": b.label, "delta_ms": r2(delta),
                         "delta_pct": r1(100 * delta / a.step_time_ms)})
    return rows


def category_comparison(runs: list[RunMetrics], baseline: int = 0) -> list[dict]:
    """GPU kernel time per category (ms/step) across runs, with deltas vs. the baseline."""
    base = runs[baseline]
    rows = []
    for cat in [*CATEGORIES, "total"]:
        if cat == "total":
            values = {r.label: r.kernel_ms_per_step for r in runs}
        else:
            values = {r.label: r.categories.get(cat) for r in runs}
        if all(not v for v in values.values()):
            continue
        b = values[base.label] or 0.0
        row = {"category": cat, **values}
        for r in runs:
            if r is not base:
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
