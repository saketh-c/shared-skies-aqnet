#!/bin/bash
#SBATCH -J aq2-graphres
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
# afterany self-chain of 3 x 8 h links (see aq2-graphpre.sbatch).
set -euo pipefail
source "$HOME/scratch/aqnet/repo/slurm2/common.sh"
[ -f "$VENV/.ok" ] || { echo "venv not built — run aq2-audit first"; exit 1; }
source "$VENV/bin/activate"
cd "$REPO"
if [ "${FORCE:-0}" != "1" ] && [ -f "$ART/oof_tier2.npz" ]; then
  echo "graphres outputs exist — skipping (FORCE=1 to re-run)"; exit 0
fi
nvidia-smi -L || true
python research/aqnet2/pipeline2.py graphres --resume
