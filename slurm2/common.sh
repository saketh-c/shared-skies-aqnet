# Shared environment for the AQNet v2 Slurm stage scripts. Sourced, not run.
# Identical to v1 slurm/common.sh except ART, which points at the v2
# namespace (research/aqnet2/artifacts/v2 — DESIGN §12: per-config subdirs
# fix the v1 flat-namespace overwrite hazard).
module load python/3.12.5
export REPO=$HOME/scratch/aqnet/repo
export VENV=$HOME/venvs/aqnet
export PIP_CACHE_DIR=$HOME/scratch/aqnet/pipcache
export ART=$REPO/research/aqnet2/artifacts/v2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
