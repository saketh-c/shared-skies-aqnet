#!/bin/bash
#SBATCH -J aq2-statics
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1 -n 1 -c 8
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
set -euo pipefail
source "$HOME/scratch/aqnet/repo/slurm2/common.sh"
[ -f "$VENV/.ok" ] || { echo "venv not built — run aq2-audit first"; exit 1; }
source "$VENV/bin/activate"
cd "$REPO"
# Sentinel is the committed statics parquet (DESIGN §12.6), not an $ART file
# — it mirrors pipeline2.stage_statics exactly.
if [ "${FORCE:-0}" != "1" ] && [ -f "$REPO/pipeline/static_covariates.parquet" ]; then
  echo "statics outputs exist — skipping (FORCE=1 to re-run)"; exit 0
fi
python research/aqnet2/pipeline2.py statics
