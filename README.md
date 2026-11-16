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

This produces the fixed ResNet50 examples shown in the presentation; it is not a tool for profiling your own code. It profiles ResNet50 training in four configurations (serial, threaded, threaded + `torch.compile`, threaded + `torch.compile` + AMP) and writes the trace JSON files to `./traces` (change with `--output-dir`). What happens depends on where you run it:

- **Your own machine:** the runs execute locally. Without a CUDA GPU they fall back to CPU-only profiling.
- **An Alliance cluster, inside a job** (e.g. after `salloc --gpus=1 ...`): the runs execute directly on the allocated node.
- **An Alliance cluster login node:** you are asked whether to submit a batch job and which account to use (e.g. `def-someuser`). The job reuses your current virtual environment, and the job script and Slurm log are saved next to the traces.

Useful options for submitting jobs:

```
torch-lab generate-example-traces --account def-someuser --yes      # submit without prompts
torch-lab generate-example-traces --account def-someuser --dry-run  # print the job script only
torch-lab generate-example-traces --time 0-01:00:00 --gpus h100:1   # change the resource request
```
