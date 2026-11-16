"""Constants for ``torch-lab analyze-traces``: thresholds, sizes, and Brine defaults."""

# Tables
TOP_N = 15

# Rule-based flag thresholds (spec §7a). Tuned on the generate-example-traces demo traces.
GPU_IDLE_HIGH_PCT = 20.0          # GPU idle > this % of step time
HOST_BOUND_PCT = 50.0             # host-wait > this % of idle time ...
HOST_BOUND_MIN_IDLE_PCT = 10.0    # ... and the GPU is idle at least this much
LAUNCH_BOUND_SHORT_KERNEL_US = 10.0
LAUNCH_BOUND_SHORT_KERNEL_PCT = 50.0  # > this % of kernels shorter than the cutoff above
LAUNCH_BOUND_MIN_IDLE_PCT = 10.0  # a busy GPU is not launch bound
TOP_KERNEL_DOMINANT_PCT = 30.0    # one kernel > this % of GPU kernel time
MEMCPY_HIGH_PCT = 10.0            # H2D + D2H > this % of GPU kernel time
NO_AMP_MIN_COMPUTE_CAPABILITY = 7.0
NO_AMP_FP32_SHARE_PCT = 50.0      # fp32/tf32 GEMM+conv time > this % of GEMM+conv time

# Prompt size. Brine chat models allow 131,072 tokens in + out.
CONTEXT_TOKENS = 131_072
MAX_PROMPT_TOKENS = 24_000
CHARS_PER_TOKEN = 2.5  # conservative: the demo prompt measured ~2.6 chars/token on Brine

# Brine
BRINE_BASE_URL = "https://brine.sharcnet.ca/v1"
BRINE_DEFAULT_MODEL = "DeepSeek-V4-Flash-0731"
BRINE_KEY_FILE = "~/.config/torch-lab/brine_key"
ALLOWED_HOSTS = frozenset({"brine.sharcnet.ca"})
TEMPERATURE = 0.2
MAX_TOKENS = 6144
REQUEST_TIMEOUT_S = 300
RETRIES = 3
RETRY_BACKOFF_S = 2.0

# Suggested memory for `salloc` when analyzing the four demo traces (measured, see README).
SUGGESTED_MEM = "16G"
