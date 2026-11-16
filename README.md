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

This produces the fixed ResNet50 examples shown in the presentation; it is not a tool for profiling your own code. It profiles ResNet50 training in five configurations (main process, i.e. the DataLoader default `num_workers=0`; serial, one worker process; threaded, threaded + `torch.compile`, threaded + `torch.compile` + AMP) and writes the trace JSON files to `./traces` (change with `--output-dir`). What happens depends on where you run it:

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
