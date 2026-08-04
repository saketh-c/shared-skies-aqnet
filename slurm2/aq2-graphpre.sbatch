#!/bin/bash
#SBATCH -J aq2-graphpre
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
# Long GPU training: submitted by submit.sh as an afterany self-chain of
# 3 x 8 h links. Each link exits 0 fast on the sentinel; otherwise it
# resumes from last.pt (atomic checkpoints every epoch AND every 30 min,
# first inside the 1-h protected window — DESIGN §13).
set -euo pipefail
source "$HOME/scratch/aqnet/repo/slurm2/common.sh"
[ -f "$VENV/.ok" ] || { echo "venv not built — run aq2-audit first"; exit 1; }
source "$VENV/bin/activate"
cd "$REPO"
if [ "${FORCE:-0}" != "1" ] && [ -f "$ART/graph_pretrain.json" ]; then
  echo "graphpre outputs exist — skipping (FORCE=1 to re-run)"; exit 0
fi
nvidia-smi -L || true
python research/aqnet2/pipeline2.py graphpre --resume
