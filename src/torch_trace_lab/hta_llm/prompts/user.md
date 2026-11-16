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

If the bundle says the configs are not distinguishable from the available data, explain the differences from the trace evidence alone. Suggest what the user should record next time, naming only config fields that are still "unknown" or inferred with low confidence (for example in a `--meta` file).

Under **Recommended next experiments**, focus on the best-performing run: what would make it faster still?
<!-- /comparison -->

# Remediation guide (known fixes per flag)

{{REMEDIATION}}

# Analysis bundle

{{BUNDLE}}
