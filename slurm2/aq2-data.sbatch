#!/bin/bash
#SBATCH -J aq2-data
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
# No sentinel by design (v1 aq-data precedent): the data stage is
# cache-driven (month chunks + .failed.json sidecars), so re-running after a
# preemption RESUMES the fetch instead of trusting a partial one.
python research/aqnet2/pipeline2.py data
