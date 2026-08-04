# Shared environment for the AQNet Slurm stage scripts. Sourced, not run.
module load python/3.12.5
export REPO=$HOME/scratch/aqnet/repo
export VENV=$HOME/venvs/aqnet
export PIP_CACHE_DIR=$HOME/scratch/aqnet/pipcache
export ART=$REPO/research/aqnet/artifacts
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
