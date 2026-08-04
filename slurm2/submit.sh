#!/bin/bash
# Submit the full AQNet v2 pipeline as a dependency chain on embers.
#
# Safe to re-run after a preemption or timeout: completed stages detect
# their sentinels and exit 0 immediately, so only unfinished work repeats.
# embers preemption is CANCEL-not-requeue — a killed chain head means
# resubmitting THIS script is the recovery path (idempotent by design).
#
# Long GPU trainings run as afterany self-chains of 3 x 8 h links: link N+1
# starts whenever link N ends (success, timeout, or preemption mid-link),
# exits 0 fast if the stage sentinel already exists, and otherwise resumes
# from last.pt. Cross-stage edges stay afterok so a genuinely failed stage
# holds the chain visibly (DependencyNeverSatisfied) instead of cascading.
#
# FORCE=1 is a per-job env var read inside each sbatch, never handled here.
set -euo pipefail
S=$HOME/scratch/aqnet/repo/slurm2
mkdir -p $HOME/scratch/aqnet/logs

chain3 () {  # chain3 <dep-jobid> <sbatch-file> -> echoes last link's jobid
  local dep=$1 file=$2
  local a b c
  a=$(sbatch --parsable --dependency=afterok:$dep "$file")
  b=$(sbatch --parsable --dependency=afterany:$a "$file")
  c=$(sbatch --parsable --dependency=afterany:$b "$file")
  echo "$c"
}

AUDIT=$(sbatch --parsable "$S/aq2-audit.sbatch")
DATAPA=$(sbatch --parsable --dependency=afterok:$AUDIT "$S/aq2-data-pa.sbatch")
DATA=$(sbatch --parsable --dependency=afterok:$DATAPA "$S/aq2-data.sbatch")
STAT=$(sbatch --parsable --dependency=afterok:$DATA "$S/aq2-statics.sbatch")
COLO=$(sbatch --parsable --dependency=afterok:$STAT "$S/aq2-colocate.sbatch")
CAL=$(sbatch --parsable --dependency=afterok:$COLO "$S/aq2-calibrate.sbatch")
PRI=$(sbatch --parsable --dependency=afterok:$CAL "$S/aq2-priors.sbatch")
FEAT=$(sbatch --parsable --dependency=afterok:$PRI "$S/aq2-features.sbatch")
SKEL=$(sbatch --parsable --dependency=afterok:$FEAT "$S/aq2-skeleton.sbatch")
GPRE=$(chain3 "$SKEL" "$S/aq2-graphpre.sbatch")
GRES=$(chain3 "$GPRE" "$S/aq2-graphres.sbatch")
FPRE=$(chain3 "$GRES" "$S/aq2-fieldpre.sbatch")
FRES=$(chain3 "$FPRE" "$S/aq2-fieldres.sbatch")
GATE=$(sbatch --parsable --dependency=afterok:$FRES "$S/aq2-gates.sbatch")
EXC=$(sbatch --parsable --dependency=afterok:$GATE "$S/aq2-exceed.sbatch")
UQ=$(sbatch --parsable --dependency=afterok:$EXC "$S/aq2-uq.sbatch")
VAL=$(sbatch --parsable --dependency=afterok:$UQ "$S/aq2-validate.sbatch")
# export is OPTIONAL — uncomment after the configuration is frozen and the
# vault has been consumed (serving bundle + demo surface, gpu-rtx6000):
# EXP=$(sbatch --parsable --dependency=afterok:$VAL "$S/aq2-export.sbatch")

echo "submitted: audit=$AUDIT data-pa=$DATAPA data=$DATA statics=$STAT" \
     "colocate=$COLO calibrate=$CAL priors=$PRI features=$FEAT" \
     "skeleton=$SKEL graphpre..=$GPRE graphres..=$GRES fieldpre..=$FPRE" \
     "fieldres..=$FRES gates=$GATE exceed=$EXC uq=$UQ validate=$VAL"
