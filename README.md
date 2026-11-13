# PyTorch Profiler and Holistic Trace Analysis
SHARCNET General Interest Seminar

This package will allow you to run the presentation as well as run some examples for generating traces on Digital Research Alliance of Canada clusters.

## Installation on Alliance Clusters

1. Download the repoository on a login node:
   ```
   git clone git@github.com:cwilson51/GIS-PP-and-HTA.git
   ```

2. You can install with pip on the login node or within your jobs:
   ```
   cd /path/to/GIS-PP-and-HTA
   pip install --no-index .
   ```

## General installation

Download the repository and install with pip:
   ```
   git clone git@github.com:cwilson51/GIS-PP-and-HTA.git
   pip install ./GIS-PP-and-HTA
   ```
## Usage

### Presentation

The presentation is a Jupyter notebook using RISE to create the presentation mode. You can run the presentation as a normal notebook in JupyterLab.

### Generating traces on Alliance Clusters

Sample job scripts are provided in `scripts/`. Simply edit the job script to match your account details and submit with `sbatch`.

### Generating traces from the command line

The package includes a command line interface for quickly generating traces. After installing the package, one can begin generating traces with

```
trace-lab profile
```
