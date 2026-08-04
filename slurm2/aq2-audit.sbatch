#!/bin/bash
#SBATCH -J aq2-audit
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1 -n 1 -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
set -euo pipefail
source "$HOME/scratch/aqnet/repo/slurm2/common.sh"

# First job in the v2 chain: owns the venv bootstrap (v1 aq-data pattern).
# .ok marks a fully completed install so a half-built venv from a preempted
# run is rebuilt instead of trusted.
if [ ! -f "$VENV/.ok" ]; then
  echo "building venv at $VENV"
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r "$REPO/research/aqnet/requirements.txt"
  pip install -r "$REPO/research/aqnet2/requirements2.txt"
  # The requirements install pulls PyPI's default torch build (cu13x), which
  # CANNOT initialize on Phoenix GPU nodes (NVIDIA driver 12.9 < CUDA 13.0:
  # torch.cuda.is_available() returns False and training silently runs on
  # CPU — observed 2026-08-04). Force-replace with cu126 wheels, which the
  # 12.9 driver supports on the RTX 6000s / L40S.
  pip uninstall -y torch
  pip install torch --index-url https://download.pytorch.org/whl/cu126
  touch "$VENV/.ok"
else
  source "$VENV/bin/activate"
  # v2 deps on top of an existing v1 venv (idempotent, fast when satisfied).
  pip install -q -r "$REPO/research/aqnet2/requirements2.txt"
fi

cd "$REPO"
mkdir -p "$ART"
git rev-parse HEAD > "$ART/git_sha.txt"
pip freeze > "$ART/versions.txt"

if [ "${FORCE:-0}" != "1" ] && [ -f "$ART/power_analysis.json" ] && [ -f "$ART/audit_report.json" ]; then
  echo "audit outputs exist — skipping (FORCE=1 to re-run)"; exit 0
fi
python research/aqnet2/pipeline2.py audit
