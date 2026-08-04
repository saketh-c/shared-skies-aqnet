#!/bin/bash
#SBATCH -J aq2-data-pa
#SBATCH -A gts-ar70
#SBATCH -q embers
#SBATCH -p cpu-small
#SBATCH -N 1 -n 1 -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o /storage/home/hcoda1/8/ntan68/scratch/aqnet/logs/%x-%j.out
set -euo pipefail
source "$HOME/scratch/aqnet/repo/slurm2/common.sh"
[ -f "$VENV/.ok" ] || { echo "venv not built — run aq2-audit first"; exit 1; }
source "$VENV/bin/activate"
cd "$REPO"
# BUILD scope decision #1: the cf_1 refetch is SKIPPED — this stage records
# the decision artifact (reconstruction fallback lives in calibrate), so the
# budget is minutes, not the multi-day resumable chain DESIGN §12 reserved.
if [ "${FORCE:-0}" != "1" ] && [ -f "$ART/data_pa_decision.json" ]; then
  echo "data-pa outputs exist — skipping (FORCE=1 to re-run)"; exit 0
fi
python research/aqnet2/pipeline2.py data-pa
