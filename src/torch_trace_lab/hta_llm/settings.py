"""Constants for ``torch-lab analyze-traces``: thresholds, sizes, and Brine defaults."""

# Tables
TOP_N = 15

# Rule-based flag thresholds (spec §7a). General starting points for any training trace;
# `torch-lab analyze-traces --list-flags` shows them.
GPU_IDLE_HIGH_PCT = 20.0          # GPU idle > this % of the GPU timeline
HOST_BOUND_PCT = 50.0             # host-wait > this % of idle time ...
HOST_BOUND_MIN_IDLE_PCT = 5.0     # ... and the GPU is idle at least this much; below that,
                                  # idle time is too small to be worth fixing
LAUNCH_BOUND_SHORT_KERNEL_US = 10.0
LAUNCH_BOUND_SHORT_KERNEL_PCT = 50.0  # > this % of kernels shorter than the cutoff above
LAUNCH_BOUND_MIN_IDLE_PCT = 10.0  # a busy GPU is not launch bound
LAUNCH_BOUND_MAX_DELAY_RATIO = 5.0  # median launch delay <= this × median CPU launch time;
                                    # longer, and kernels sit queued: the CPU is ahead
TOP_KERNEL_DOMINANT_PCT = 30.0    # one kernel > this % of GPU kernel time
MEMCPY_HIGH_PCT = 10.0            # H2D + D2H > this % of GPU kernel time
NO_AMP_MIN_COMPUTE_CAPABILITY = 7.0
NO_AMP_FP32_SHARE_PCT = 50.0      # fp32/tf32 GEMM+conv time > this % of GEMM+conv time
# Multi-rank runs (S3). Comm flags use the median rank; rank flags compare ranks.
COMM_HEAVY_PCT = 30.0             # NCCL kernels > this % of GPU kernel time
POOR_OVERLAP_PCT = 50.0           # < this % of comm time overlaps compute, while COMM_HEAVY
RANK_STRAGGLER_RATIO = 1.10       # max/median busy (non-comm) GPU time per step across ranks
RANK_IDLE_IMBALANCE_POINTS = 10.0  # max - min GPU idle % across ranks
# Per-rank tables are included only up to this many ranks (spec §7.2).
MAX_RANKS_IN_TABLES = 8
# Fewer complete profiler steps than this get a data-quality note: per-step means of so few
# steps carry step-to-step noise (e.g. collective wait times that vary from step to step).
FEW_STEPS = 5

# Config inference from trace contents (hta_llm/infer.py)
INFER_HIGH_SHARE_PCT = 80.0       # dominant precision >= this % of GEMM/conv time -> high confidence
INFER_MIN_SHARE_PCT = 50.0        # >= this -> medium; below -> low. Also the NHWC-kernel share
                                  # for inferring channels_last
INFER_TRANSPOSE_PCT = 5.0         # NCHW<->NHWC transposes >= this % of GEMM/conv time ->
                                  # tensors are NCHW and cuDNN converts them

# Prompt size. Brine chat models allow 131,072 tokens in + out.
CONTEXT_TOKENS = 131_072
MAX_PROMPT_TOKENS = 24_000
# Over budget, shrink the prompt in this order (spec §7.9): smaller kernel tables, then drop
# optional bundle sections one by one; only then warn and send it anyway.
BUDGET_TOP_N_STEPS = (10, 5)
OPTIONAL_SECTIONS = ("launch stats", "category comparison", "kernel-name appendix")
# Multi-rank bundles drop the all-ranks tables before OPTIONAL_SECTIONS. The rank aggregation
# and the slowest rank's tables are never dropped: they are the bottleneck evidence.
RANK_SECTIONS = ("all-ranks tables",)
CHARS_PER_TOKEN = 2.5  # conservative: markdown tables and kernel names tokenize densely
                       # (a measured bundle prompt ran ~2.6 chars/token on Brine)

# Brine. The base URL has no default: users set BRINE_BASE_URL, and it must use https to
# one of ALLOWED_HOSTS (the egress guard).
BRINE_DEFAULT_MODEL = "DeepSeek-V4-Flash-0731"
BRINE_KEY_FILE = "~/.config/torch-lab/brine_key"
ALLOWED_HOSTS = frozenset({"brine.sharcnet.ca"})
TEMPERATURE = 0.2
MAX_TOKENS = 6144
REQUEST_TIMEOUT_S = 300
RETRIES = 3
RETRY_BACKOFF_S = 2.0

# Suggested memory for `salloc`. Memory grows with the total size of the traces analyzed;
# a few traces of 35-75 MB each peaked at about 5 GB, so 8G leaves headroom for typical
# single-GPU runs. Larger or more traces need more.
SUGGESTED_MEM = "8G"
