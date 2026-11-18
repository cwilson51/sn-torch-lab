# sn-torch-lab

This package installs the `torch-lab` command, which can:

- `generate-example-traces`: profile ResNet-50 training in a few configurations, giving you traces to explore;
- `analyze-traces`: analyze any [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) traces with [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) (HTA) and, optionally, an LLM.

Run `torch-lab <command> --help` for all options.

This package was presented in the SHARCNET General Interest Seminar on November 18, 2026.

## Installation

On an Alliance cluster (from a login node):

```
module load StdEnv/2023 python/3.11 scipy-stack/2026a
virtualenv --no-download ~/torch-lab-env && source ~/torch-lab-env/bin/activate
pip install --no-index --upgrade pip

git clone https://github.com/cwilson51/sn-torch-lab.git
pip install --no-index ./sn-torch-lab
```

Anywhere else (Python 3.11 or newer):

```
git clone https://github.com/cwilson51/sn-torch-lab.git
pip install ./sn-torch-lab
```

## Usage

### Generating traces

```
torch-lab generate-example-traces
```

This trains ResNet-50 on a synthetic image dataset in five configurations: no DataLoader workers, one worker, one worker plus `torch.compile`, several workers plus `torch.compile`, and the same with AMP. Each configuration's trace goes in its own directory under `./demo/example-traces/single-gpu` (change with `--output-dir`). The batch size is picked automatically to fit the GPU. What happens depends on where you run it:

1. **On your own machine.** The runs execute locally. With a CUDA GPU you get GPU traces. Without one, the runs fall back to CPU-only profiling: useful for looking at the profiler, but `analyze-traces` needs GPU traces.
2. **On an Alliance login node.** You are asked whether to submit a batch job and which account to use (e.g. `def-someuser`). The job requests one `h100_1g.10gb` MIG slice (1/7 of an H100) for 30 minutes, reuses your current virtual environment, and writes the traces, job script and Slurm log to the output directory. Change the request with `--gpus` and `--time`; skip the prompts with `--account def-someuser --yes`.
3. **In an interactive job.** The runs execute directly on the allocated node:

   ```
   salloc --time=0:30:0 --gpus=h100_1g.10gb:1 --cpus-per-task=2 --mem=16G --account=def-someuser
   torch-lab generate-example-traces
   ```

### Analyzing traces

`analyze-traces` takes a directory and infers the scenario from its layout. File names don't matter:

```
one run             several runs to compare        one multi-GPU run
my-run/             experiments/                   ddp-run/
  trace.json          baseline/trace.json            rank0.json  rank1.json ...
                      bs128-amp/trace.json
```

- **One run:** where is the time going, and how can I make it faster?
- **Several runs:** why do they differ, and how does the bottleneck shift between them? Each subdirectory is one run; the first alphabetically is the baseline (change with `--baseline`).
- **One multi-GPU run:** each file is one rank (read from the trace, not the file name). The report looks at communication, overlap and straggling ranks.
- **Several multi-GPU runs:** as above, plus scaling efficiency across GPU counts.

Traces must be recorded on a CUDA GPU. Analysis needs a few GB of memory, so on a cluster run it in a job (e.g. `salloc --time=0:30:0 --cpus-per-task=4 --mem=8G`), not on a login node.

#### Without an LLM

The examples below use the traces from `generate-example-traces`, but any traces in the layout above work.

```
torch-lab analyze-traces ./demo/example-traces/single-gpu --no-llm
```

HTA computes per-step time breakdowns, idle time, top kernels and, for comparisons, a trace diff. Rule-based heuristics then raise flags such as `HOST_BOUND` (GPU waiting on the CPU or data loading) or `NO_AMP_SUSPECTED` (matrix math running in fp32); `--list-flags` lists them all. Results go to `./hta-report`:

- `report.md`: the flags raised for each run, with known fixes;
- `bundle.md`: all the tables;
- `prompt.md`: the bundle wrapped in instructions for an LLM.

You can give `prompt.md` to any AI assistant you like. For an agent that reads skills (e.g. Claude Code), a minimal skill:

```markdown
---
name: analyze-pytorch-traces
description: Analyze PyTorch Profiler traces with torch-lab. Use when asked why training is slow or how to speed it up.
---

1. Run `torch-lab analyze-traces <trace dir> --no-llm --output-dir hta-report`.
2. Read `hta-report/prompt.md` and follow its instructions to write the report.
3. Quote only numbers that appear in `hta-report/bundle.md`.
```

#### With Brine

[Brine](https://brine.sharcnet.ca) is SHARCNET's AI-as-a-service platform. Without `--no-llm`, `analyze-traces` sends `prompt.md` to a model on Brine, which writes a ranked, actionable report. You need an API key: email [help@sharcnet.ca](mailto:help@sharcnet.ca) to request one. Then:

```
export BRINE_BASE_URL=https://brine.sharcnet.ca/v1
export BRINE_API_KEY=...        # or save it in ~/.config/torch-lab/brine_key (chmod 600)
torch-lab analyze-traces ./demo/example-traces/single-gpu
```

The report is in `hta-report/report.md`. Its numbers come from `bundle.md`, so check them there, and open the traces in [Perfetto](https://ui.perfetto.dev) to confirm. Only aggregated tables and run facts are sent, never raw traces, paths or usernames; `prompt.md` shows exactly what was sent. If Brine can't be reached, you still get the flags-only report.

## Caveat

LLMs are not perfect oracles for PyTorch trace analysis: they can misread tables, overstate small differences or suggest fixes that don't apply to your code. This package is under development. The heuristics are incomplete and their thresholds still need tuning, so results may vary. Treat the report as a starting point and verify its claims in the traces.
