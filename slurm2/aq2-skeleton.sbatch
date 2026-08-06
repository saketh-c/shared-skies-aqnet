#!/bin/bash
#SBATCH -J aq2-skeleton
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1 -n 1 -c 24
#SBATCH --mem=128G
#SBATCH -t 16:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
# DESIGN §13 budgeted skeleton at up to 12 h for tx; the link is sized at
# 16 h / 128G so the west7 domain (~1.6x rows, wider grids) fits in one
# link too. Per-fold checkpoints stand either way — if it times out or is
# preempted, resubmit submit.sh (or this file): completed folds are
# checkpointed and the stage resumes (the documented kriging fallback /
# candidate-B escape also cuts the budget).
set -euo pipefail
source "$HOME/scratch/aqnet/repo/slurm2/common.sh"
[ -f "$VENV/.ok" ] || { echo "venv not built — run aq2-audit first"; exit 1; }
source "$VENV/bin/activate"
cd "$REPO"
if [ "${FORCE:-0}" != "1" ] && [ -f "$ART/oof_tier1.npz" ]; then
  echo "skeleton outputs exist — skipping (FORCE=1 to re-run)"; exit 0
fi
python research/aqnet2/pipeline2.py skeleton --resume
