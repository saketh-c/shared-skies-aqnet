# Shared environment for the AQNet v2 Slurm stage scripts. Sourced, not run.
# Identical to v1 slurm/common.sh except ART and the domain block: the run
# domain (config2 / EXPANSION.md) comes from the submitting environment via
# AQNET2_DOMAIN, default tx — sbatch exports the caller's environment into
# every job (--export=ALL is the default), so `AQNET2_DOMAIN=west7 bash
# submit.sh` reaches every stage's python through plain inheritance — and
# ART points at that domain's artifact namespace (research/aqnet2/artifacts/
# v2 for tx, v3 for west7 — DESIGN §12: per-config subdirs fix the v1
# flat-namespace overwrite hazard). The case below MUST stay in sync with
# config2._DOMAINS[*]["artifacts"]; an unknown domain fails the job loudly
# rather than writing into a namespace config2 would refuse.
module load python/3.12.5
export REPO=$HOME/scratch/aqnet/repo
export VENV=$HOME/venvs/aqnet
export PIP_CACHE_DIR=$HOME/scratch/aqnet/pipcache
export AQNET2_DOMAIN=${AQNET2_DOMAIN:-tx}
case "$AQNET2_DOMAIN" in
  tx)    AQ2_NS=v2 ;;
  west7) AQ2_NS=v3 ;;
  *) echo "[common.sh] unknown AQNET2_DOMAIN '$AQNET2_DOMAIN' (known: tx west7)" >&2
     exit 1 ;;
esac
export ART=$REPO/research/aqnet2/artifacts/$AQ2_NS
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
