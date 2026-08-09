#!/bin/bash
#SBATCH -J aq2-colocate
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1 -n 1 -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
set -euo pipefail
source "$HOME/scratch/aqnet/repo/slurm2/common.sh"
[ -f "$VENV/.ok" ] || { echo "venv not built — run aq2-audit first"; exit 1; }
source "$VENV/bin/activate"
cd "$REPO"
if [ "${AQNET2_PA_SOURCE:-v2}" = "v4" ]; then
  PAIRS="$ART/colocation_pairs_v4.parquet"
else
  PAIRS="$ART/colocation_pairs.parquet"
fi
if [ "${FORCE:-0}" != "1" ] && [ -f "$PAIRS" ]; then
  echo "colocate outputs exist ($(basename "$PAIRS")): skipping (FORCE=1 to re-run)"; exit 0
fi
python research/aqnet2/pipeline2.py colocate
