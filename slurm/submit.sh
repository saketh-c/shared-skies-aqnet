#!/bin/bash
# Submit the full AQNet pipeline as a dependency chain on embers.
# Safe to re-run after a preemption or timeout: completed stages detect
# their outputs and exit immediately, so only unfinished work repeats.
set -euo pipefail
S=$HOME/scratch/aqnet/slurm
mkdir -p $HOME/scratch/aqnet/logs

DATA=$(sbatch --parsable "$S/aq-data.sbatch")
FEAT=$(sbatch --parsable --dependency=afterok:$DATA "$S/aq-features.sbatch")
TAB=$(sbatch --parsable --dependency=afterok:$FEAT "$S/aq-tabular.sbatch")
DEEP=$(sbatch --parsable --dependency=afterok:$FEAT "$S/aq-deep.sbatch")
ABL=$(sbatch --parsable --dependency=afterok:$FEAT "$S/aq-ablation.sbatch")
FUSE=$(sbatch --parsable --dependency=afterok:$TAB:$DEEP "$S/aq-fuse.sbatch")
VAL=$(sbatch --parsable --dependency=afterok:$FUSE:$ABL "$S/aq-validate.sbatch")

echo "submitted: data=$DATA features=$FEAT tabular=$TAB deep=$DEEP ablation=$ABL fuse=$FUSE validate=$VAL"
