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
#
# Domain: `AQNET2_DOMAIN=west7 bash submit.sh` runs the whole chain against
# the west7 namespace (artifacts/v3). The export below pins the caller's
# value (default tx) so every sbatch here — the chain3 self-chain links
# included — records the SAME domain in its job environment (--export=ALL
# is sbatch's default); common.sh re-defaults inside each job, so a bare
# resubmission of a single sbatch still resolves to tx. Re-running THIS
# script after a preemption must repeat the same AQNET2_DOMAIN.
#
# Run-generation pinning: ALL FOUR AQNET2_* knobs (domain, PA source,
# artifacts tag, seed offset) are exported with their frozen defaults and
# echoed, then recorded once in $ART/launch_env.json at first submission.
# A resubmission whose environment disagrees with that record is refused
# loudly: a fresh shell that forgets one knob would otherwise run the
# remaining stages with v2 defaults against this namespace's artifacts,
# mixing run generations mid-chain (the exact failure this guard exists
# to prevent). Fix the environment, or point AQNET2_ARTIFACTS_TAG at a
# fresh namespace; never edit launch_env.json to match a wrong shell.
set -euo pipefail
S=$HOME/scratch/aqnet/repo/slurm2
export AQNET2_DOMAIN=${AQNET2_DOMAIN:-tx}
export AQNET2_PA_SOURCE=${AQNET2_PA_SOURCE:-v2}
export AQNET2_ARTIFACTS_TAG=${AQNET2_ARTIFACTS_TAG:-}
export AQNET2_SEED_OFFSET=${AQNET2_SEED_OFFSET:-0}
# common.sh maps (domain, tag) to the SAME artifacts dir config2 resolves
# in-process; sourced here so the launch_env guard below sees $ART.
source "$S/common.sh"
mkdir -p $HOME/scratch/aqnet/logs

echo "run env: AQNET2_DOMAIN=$AQNET2_DOMAIN" \
     "AQNET2_PA_SOURCE=$AQNET2_PA_SOURCE" \
     "AQNET2_ARTIFACTS_TAG=${AQNET2_ARTIFACTS_TAG:-<unset>}" \
     "AQNET2_SEED_OFFSET=$AQNET2_SEED_OFFSET (ART=$ART)"

LAUNCH_ENV=$(printf '{"AQNET2_DOMAIN": "%s", "AQNET2_PA_SOURCE": "%s", "AQNET2_ARTIFACTS_TAG": "%s", "AQNET2_SEED_OFFSET": "%s"}' \
  "$AQNET2_DOMAIN" "$AQNET2_PA_SOURCE" "$AQNET2_ARTIFACTS_TAG" "$AQNET2_SEED_OFFSET")
if [ -f "$ART/launch_env.json" ]; then
  RECORDED=$(cat "$ART/launch_env.json")
  if [ "$RECORDED" != "$LAUNCH_ENV" ]; then
    echo "REFUSING TO SUBMIT: this shell's AQNET2_* env disagrees with the" >&2
    echo "run generation recorded at $ART/launch_env.json" >&2
    echo "  recorded: $RECORDED" >&2
    echo "  current:  $LAUNCH_ENV" >&2
    echo "Re-export the recorded values and resubmit (or use a fresh" >&2
    echo "AQNET2_ARTIFACTS_TAG for a new generation); submitting as-is" >&2
    echo "would mix run generations mid-chain." >&2
    exit 1
  fi
else
  mkdir -p "$ART"
  printf '%s\n' "$LAUNCH_ENV" > "$ART/launch_env.json"
  echo "recorded launch env -> $ART/launch_env.json"
fi

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

echo "submitted (domain=$AQNET2_DOMAIN pa_source=$AQNET2_PA_SOURCE" \
     "tag=${AQNET2_ARTIFACTS_TAG:-<unset>} seed_offset=$AQNET2_SEED_OFFSET):" \
     "audit=$AUDIT data-pa=$DATAPA data=$DATA statics=$STAT" \
     "colocate=$COLO calibrate=$CAL priors=$PRI features=$FEAT" \
     "skeleton=$SKEL graphpre..=$GPRE graphres..=$GRES fieldpre..=$FPRE" \
     "fieldres..=$FRES gates=$GATE exceed=$EXC uq=$UQ validate=$VAL"
