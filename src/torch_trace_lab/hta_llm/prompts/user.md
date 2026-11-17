<!-- single -->
This is a single training run on one GPU. The user's question: "How can I make this faster?"

Lead with the rule-based flags. Under **Findings**, give the top 3–5 bottlenecks. For each one, give:
- **Evidence:** the flag(s) and numbers from the bundle;
- **Likely cause:** marked as a hypothesis when the data cannot confirm it;
- **Fix:** chosen from the remediation guide;
- **How to verify:** which number in a re-run bundle should change, and in which direction.
<!-- /single -->
<!-- comparison -->
These are {{N_RUNS}} single-GPU training runs. The baseline is **{{BASELINE}}**. Answer both questions, and state findings that serve both once, then cross-reference them:

1. **Runtime differences:** why do step time and throughput differ between the runs? Attribute each difference to the config diff and the flags, ranked by how much of the step-time difference it explains (using the kernel and temporal tables). Point out any differences the config diff does not explain.
2. **Bottleneck comparison:** how does the bottleneck mix shift across runs? Say which flags appear or clear between runs, how the idle and kernel shares move, and name each run's dominant bottleneck.

If the bundle says the configs are not distinguishable from the available data, explain the differences from the trace evidence alone. Suggest what the user should record next time, naming only config fields that are still "unknown" or inferred with low confidence.

Under **Recommended next experiments**, focus on the best-performing run: what would make it faster still?
<!-- /comparison -->
<!-- distributed -->
This is a single training run on {{N_RANKS}} GPUs (ranks). The user's question: "Where are the distributed-training bottlenecks, and how can I make this faster?"

The rank comparison covers every rank. The communication, temporal, idle, kernel and launch sections show only the slowest rank (the one with the most busy GPU time); say so when you quote them.

Lead with the rule-based flags. Under **Findings**, give the top 3–5 bottlenecks, covering, where the bundle has evidence for them:
- **Rank imbalance and stragglers:** whether one rank does more work than the others (RANK_STRAGGLER) or idles more (RANK_IDLE_IMBALANCE), and which rank;
- **Communication vs. compute:** how much of the GPU time is communication (COMM_HEAVY), and how much of it is exposed rather than overlapped with compute (POOR_OVERLAP);
- **Single-GPU bottlenecks** on the slowest rank (the other flags), which slow every rank down through the collectives.

For each one, give:
- **Evidence:** the flag(s) and numbers from the bundle, naming the rank where a number is per rank;
- **Likely cause:** marked as a hypothesis when the data cannot confirm it. Remember that ranks waiting for a slower rank inside a collective show that wait as communication time;
- **Fix:** chosen from the remediation guide;
- **How to verify:** which number in a re-run bundle should change, and in which direction.

The bundle does not say which distributed framework (DDP, FSDP, DeepSpeed) the run uses unless the context table shows `parallelism`. Do not assume one; when a fix depends on the framework, say so.
<!-- /distributed -->

Whenever you suggest recording run settings for next time, note that a `--meta` file accepts only these keys (anything else is ignored): {{META_KEYS}}. For any other setting, suggest a free-text `notes` entry instead of inventing a key.

# Remediation guide (known fixes per flag)

{{REMEDIATION}}

# Analysis bundle

{{BUNDLE}}
