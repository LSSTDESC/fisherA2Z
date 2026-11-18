#!/bin/bash
#SBATCH --job-name=lsst_lookup
#SBATCH --partition=kipac
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/lookup_%j.out

# 21 depths x 2 probes computed once each (~1 min per depth), 18 areas by exact
# covariance rescaling, 3 photo-z prior scenarios -> 2268 rows.
unset PYTHONPATH            # Sherlock's jupyterlab module leaks into the env otherwise
PY=$HOME/miniforge3/envs/fisher_env/bin/python
$PY -c "import numpy, scipy, pyccl, fisherA2Z; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pyccl', pyccl.__version__)"
cd "$SLURM_SUBMIT_DIR"
$PY run_lookup.py lookup_results.csv
