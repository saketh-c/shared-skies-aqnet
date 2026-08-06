#!/bin/bash
#SBATCH -J aq2-features
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1 -n 1 -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
set -euo pipefail
source "$HOME/scratch/aqnet/repo/slurm2/common.sh"
[ -f "$VENV/.ok" ] || { echo "venv not built — run aq2-audit first"; exit 1; }
source "$VENV/bin/activate"
cd "$REPO"
if [ "${FORCE:-0}" != "1" ] && [ -f "$ART/frame_truth.parquet" ] && [ -f "$ART/folds2.json" ] && [ -f "$ART/nbr_overrides_outer.npz" ] && [ -f "$ART/oof_tier0.npz" ]; then
  echo "features outputs exist — skipping (FORCE=1 to re-run)"; exit 0
fi
python research/aqnet2/pipeline2.py features
