#!/bin/bash
#SBATCH -J aq-data
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1 -n 1 -c 8
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
set -euo pipefail
source "$HOME/scratch/aqnet/slurm/common.sh"

# Build the Python env once; .ok marks a fully completed install so a
# half-built venv from a preempted run is rebuilt instead of trusted.
if [ ! -f "$VENV/.ok" ]; then
  echo "building venv at $VENV"
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r "$REPO/research/aqnet/requirements.txt"
  # The requirements install pulls PyPI's default torch build (cu13x), which
  # CANNOT initialize on Phoenix GPU nodes (NVIDIA driver 12.9 < CUDA 13.0;
  # torch.cuda.is_available() returns False and training silently runs on
  # CPU — observed 2026-08-04). Force-replace it with cu126 wheels, which
  # the 12.9 driver supports on the RTX 6000s.
  pip uninstall -y torch
  pip install torch --index-url https://download.pytorch.org/whl/cu126
  touch "$VENV/.ok"
else
  source "$VENV/bin/activate"
fi

cd "$REPO"
mkdir -p "$ART"
git rev-parse HEAD > "$ART/git_sha.txt"
pip freeze > "$ART/versions.txt"

# The data stage is cache-driven (per-month chunks, failed-month sidecars),
# so re-running after a preemption resumes rather than restarts.
python research/aqnet/pipeline_colab.py data
