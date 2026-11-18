#!/bin/bash
#SBATCH --job-name=lsst_lookup_tables
#SBATCH --partition=kipac
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/tables_%j.out
unset PYTHONPATH
cd "$SLURM_SUBMIT_DIR"
$HOME/miniforge3/envs/fisher_env/bin/python make_tables.py lookup_results.csv .
