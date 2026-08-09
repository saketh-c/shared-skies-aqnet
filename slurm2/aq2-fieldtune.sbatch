#!/bin/bash
#SBATCH -J aq2-fieldtune
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
# Registered v4 T3 hyperparameter search (EXPANSION.md; tune_deep.py):
# 16 candidates fit on the inner SELECTION folds of outer contexts 0-2
# and scored on their CONFIRMATION folds. Sentinel = the COMPLETED
# tune_t3.json; in-flight progress lives in tune_t3_partial.json, so a
# preempted link resumes mid-search on resubmission (afterany self-chain
# of 3 x 24 h links, see aq2-graphpre.sbatch).
# gpu-l40s primary; rtx6000 fallback by swapping the partition line:
##SBATCH -p gpu-rtx6000
set -euo pipefail
source "$HOME/scratch/aqnet/repo/slurm2/common.sh"
[ -f "$VENV/.ok" ] || { echo "venv not built -- run aq2-audit first"; exit 1; }
source "$VENV/bin/activate"
cd "$REPO"
if [ "${FORCE:-0}" != "1" ] && [ -f "$ART/tune_t3.json" ]; then
  echo "fieldtune outputs exist -- skipping (FORCE=1 to re-run)"; exit 0
fi
nvidia-smi -L || true
python research/aqnet2/pipeline2.py fieldtune --resume
