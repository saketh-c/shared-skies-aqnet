#!/bin/bash
#SBATCH -J aq-fuse
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1 -n 1 -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
set -euo pipefail
source "$HOME/scratch/aqnet/slurm/common.sh"
[ -f "$VENV/.ok" ] || { echo "venv not built — run aq-data first"; exit 1; }
source "$VENV/bin/activate"
cd "$REPO"
if [ "${FORCE:-0}" != "1" ] && [ -f "$ART/oof_meta.npz" ]; then
  echo "fuse outputs exist — skipping (FORCE=1 to re-run)"; exit 0
fi
python research/aqnet/pipeline_colab.py fuse
