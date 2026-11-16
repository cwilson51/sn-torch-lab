# PyTorch Profiler and Holistic Trace Analysis
SHARCNET General Interest Seminar

This package will allow you to run the presentation as well as run some examples for generating traces on Digital Research Alliance of Canada clusters.

## Installation on Alliance Clusters

1. Download the repoository on a login node:
   ```
   git clone git@github.com:cwilson51/sn-torch-lab.git
   ```

2. You can install with pip on the login node or within your jobs:
   ```
   cd /path/to/sn-torch-lab
   pip install --no-index .
   ```

## General installation

Download the repository and install with pip:
   ```
   git clone git@github.com:cwilson51/sn-torch-lab.git
   pip install ./sn-torch-lab
   ```
## Usage

### Presentation

The presentation is a Jupyter notebook using RISE to create the presentation mode. You can run the presentation as a normal notebook in JupyterLab.

### Command line

Installing the package provides the `torch-lab` command. Run `torch-lab --help` to list its subcommands, and `torch-lab <command> --help` for a subcommand's options.

### Generating the example traces

Generate the example traces used in the demo with

```
torch-lab generate-example-traces
```

This produces the fixed ResNet50 examples shown in the presentation; it is not a tool for profiling your own code. It profiles ResNet50 training in five configurations (main process, i.e. the DataLoader default `num_workers=0`; serial, one worker process; threaded, threaded + `torch.compile`, threaded + `torch.compile` + AMP) and writes each configuration's trace to its own run directory under `./traces` (change with `--output-dir`):

```
traces/
  main-process/        serial/        threaded/
  threaded-compiled/   threaded-compiled-amp/
```

Each trace also records its run configuration (batch size, DataLoader workers, `torch.compile`, precision) in a top-level `torch_lab` object. This layout and metadata are what `torch-lab analyze-traces` expects of any traces, so `torch-lab analyze-traces ./traces` compares the example configurations directly. What happens depends on where you run it:

- **Your own machine:** the runs execute locally. Without a CUDA GPU they fall back to CPU-only profiling.
- **An Alliance cluster, inside a job** (e.g. after `salloc --gpus=1 ...`): the runs execute directly on the allocated node.
- **An Alliance cluster login node:** you are asked whether to submit a batch job and which account to use (e.g. `def-someuser`). The job reuses your current virtual environment, and the job script and Slurm log are saved next to the traces.

#### Training data

The runs train on a synthetic ImageNet-like dataset: real JPEG files on disk (about 500 px on the long side and 100–150 KB each on average, with a few larger images), loaded through `ImageFolder` with the standard `RandomResizedCrop` and horizontal-flip augmentation. Each sample is opened, read, decoded and augmented, so the DataLoader cost in the traces looks like real training. With `torchvision.datasets.FakeData`, images are made in memory and no file is ever read.

- The dataset is created on first use, with enough images for the profiled steps (batch size × 35). Inside a Slurm job it goes in node-local `$SLURM_TMPDIR/synthetic-imagenet`. Elsewhere it goes in `~/.cache/torch-trace-lab/synthetic-imagenet`, and later runs reuse it.
- On Linux, each run first evicts the files from the page cache, so every configuration starts with cold reads like a first epoch. On macOS reads may come from memory.

To create a dataset by hand, run `python -m torch_trace_lab.synthetic_data ROOT N_IMAGES`. Pass `--data_dir` to `python -m torch_trace_lab.train` to use it.

#### Batch size

By default (`--batch-size auto`) the batch size is chosen on the machine that runs the training, and all five configurations use the same size so their traces are comparable:

- **With a CUDA GPU:** the largest power of 2 (from 8 up to 512) whose peak memory for one fp32 ResNet50 training step stays within 90% of free GPU memory. Expect about 64 on an `h100_1g.10gb` MIG slice and 512 on a full H100.
- **CPU only:** a fixed 32, since run time rather than memory is the limit.

Pass `--batch-size N` to use a fixed size instead. When a batch job is submitted from a login node, detection happens in the job on the compute node.

To see the size that would be picked, run `python -m torch_trace_lab.batch_size`. To use the same logic in your own scripts:

```python
from torch_trace_lab.batch_size import find_batch_size

batch_size = find_batch_size(make_model, input_shape=(3, 224, 224), num_classes=1000)
```

Useful options for submitting jobs:

```
torch-lab generate-example-traces --account def-someuser --yes      # submit without prompts
torch-lab generate-example-traces --account def-someuser --dry-run  # print the job script only
torch-lab generate-example-traces --time 0-01:00:00 --gpus h100:1   # change the resource request
```

### Analyzing traces with HTA and an LLM

```
torch-lab analyze-traces ./traces
```

`analyze-traces` runs [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) on PyTorch Profiler traces, computes rule-based performance flags, and asks an LLM on [SHARCNET Brine](https://brine.sharcnet.ca) for a ranked, actionable report. With one run it answers "how can I make this faster?"; with several runs it answers "why do they differ, and how does the bottleneck mix shift?". Only single-GPU runs are supported for now.

**Trace layout.** File names don't matter; the structure comes from directories:

```
# one run                                # several runs to compare
torch-lab analyze-traces my-run/         torch-lab analyze-traces experiments/
my-run/                                  experiments/
  trace.json                               baseline/trace.json
                                           bs128-amp/trace.json
```

You can also pass a single trace file, or two or more run directories (`torch-lab analyze-traces runA/ runB/`). Each run directory holds one run. The traces must be recorded on a CUDA GPU: traces from a CPU-only machine (including any Mac) have no GPU kernels to analyze.

**Outputs** (in `./hta-report`, change with `--output-dir`):

- `report.md`: the LLM report, or a flags-only report with `--no-llm` or when Brine can't be used;
- `bundle.md` / `bundle.json`: the aggregated tables, flags, and run context sent to the LLM;
- `prompt.md`: exactly what is (or would be) sent to Brine;
- `run.log`.

**Options:**

| Option | Meaning |
|---|---|
| `--run-note [LABEL=]TEXT` | Free-text context for a run, e.g. `--run-note baseline="bs 64, 8 workers"`. Repeatable. For a single run, the label can be left out. |
| `--baseline LABEL` | Baseline run for comparisons (default: the first run). |
| `--top-n N` | Kernels per table (default: 15). |
| `--model NAME` | Brine model (default: `$TORCH_LAB_BRINE_MODEL`, `$BRINE_MODEL`, or `DeepSeek-V4-Flash-0731`). |
| `--no-llm` | Write the bundle, prompt, and a flags-only report without calling Brine. |
| `--list-flags` | Print the rule-based flags, what each means, and the threshold that triggers it, then exit. No `PATH` needed. |

**Brine settings.** Set the Brine endpoint and your own API key before running without `--no-llm`:

```
export BRINE_BASE_URL=https://brine.sharcnet.ca/v1
export BRINE_API_KEY=...   # or store it in ~/.config/torch-lab/brine_key with chmod 600
```

There is no built-in default URL. For safety, the URL must use https to an allowed host (`brine.sharcnet.ca`); anything else is refused. The key is never written to any output. If the URL or key is missing, Brine can't be reached, or the call fails, the bundle, prompt and a flags-only report are still written and the command exits with status 1.

**What is sent to Brine.** Only aggregated tables (per-step times, top kernels, idle breakdown, flags) and allowlisted run facts (GPU model, precision, batch size, ...). Raw traces, file paths, usernames, hostnames, job IDs and Slurm data are never sent. Before every call a leak check scans the prompt for your username, home directory, cluster paths, hostname, Slurm values and account names, and blocks the call if it finds any. **`--run-note` text and run directory names are sent as written, so keep sensitive information out of them.** Check `prompt.md` to see exactly what was sent.

**Where to run it.** On your own machine, or inside a job on an Alliance cluster. A GPU is not needed. On a login node the command refuses to run, because loading the traces needs several GB of memory. Memory grows with the total size of the traces: a few traces of 35–75 MB each peak at about 5 GB, so `--mem=8G` suits typical single-GPU runs; request more for larger or more traces. Start an interactive job instead:

```
salloc --time=0:30:0 --cpus-per-task=4 --mem=8G --account=def-someuser
torch-lab analyze-traces ./traces
```

Open OnDemand desktop and JupyterLab sessions work too. If the compute nodes can't reach `brine.sharcnet.ca`, run with `--no-llm` there, or copy the traces to a machine that can reach Brine.

#### Profiling your own code

To produce traces that `analyze-traces` can compare, profile a fixed number of steps in each run and give each run its own directory:

```python
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

run_dir = Path("experiments/baseline")  # one directory per run; file names don't matter
run_dir.mkdir(parents=True, exist_ok=True)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(skip_first=10, wait=0, warmup=10, active=10, repeat=1),
    record_shapes=True,  # larger traces and some overhead, but more detail
) as prof:
    for step, (inputs, labels) in enumerate(loader):
        train_step(inputs, labels)
        prof.step()  # once per training iteration
        if step >= 35:
            break

trace = run_dir / "trace.json"
prof.export_chrome_trace(str(trace))

# Optional: record the run config so the report can explain differences between runs.
data = json.loads(trace.read_text())
data["torch_lab"] = {
    "model": "my model",
    "batch_size_per_gpu": 64,
    "dataloader_workers": 8,
    "torch_compile": False,
    "precision": "bf16-amp",  # fp32 | fp32 (TF32 convs) | tf32 | fp16-amp | bf16-amp
}
trace.write_text(json.dumps(data))
```

For distributed runs, every rank writes its own trace file into the same run directory. Multi-GPU analysis is planned but not supported yet.
